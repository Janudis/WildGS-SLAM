import os
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
from munch import munchify

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval, align_full_traj, traj_eval_and_plot, save_traj, full_traj_eval_dpvo_stream
from src.utils.datasets import BaseDataset

# from src.tracker import Tracker
from src.tracker import make_tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.utils.datasets import RGB_NoPose
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

from typing import Any, Optional
from scipy.spatial.transform import Rotation

from src.dpvo.config import cfg as dpvo_default_cfg
from src.dpvo.dpvo import DPVO
from src.dpvo.net import VONet
from torch.cuda.amp import autocast

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]
        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        # Printer is shared across both modes
        self.printer = Printer(len(stream))

        # mode switch
        self.tracker_type = self.cfg["tracking"].get("tracker_type", "droid").lower()
        if self.tracker_type not in ["droid", "dpvo"]:
            raise ValueError(f"Unknown tracker_type: {self.tracker_type}")
        self.is_droid = (self.tracker_type == "droid")
        self.is_dpvo = (self.tracker_type == "dpvo")

        # -------------------------
        # Uncertainty MLP (mapping)
        # -------------------------
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            n_features = self.cfg["mapping"]["uncertainty_params"]["feature_dim"]
            self.uncer_network = generate_uncertainty_mlp(n_features)
            self.uncer_network.share_memory()
        else:
            self.uncer_network = None
            if self.cfg["tracking"]["uncertainty_params"]["activate"]:
                raise ValueError(
                    "uncertainty estimation cannot be activated on tracking while not on mapping"
                )

        # -------------------------
        # Initialize tracking backend
        # -------------------------
        self.droid_net = None
        self.dpvo = None
        # These exist in droid mode only (for now)
        self.ba = None
        self.traj_filler = None

        if self.is_droid:
            # ---- DROID net ----
            self.droid_net: DroidNet = DroidNet()
            self.load_pretrained(cfg)
            self.droid_net.to(self.device).eval()
            self.droid_net.share_memory()

            # ---- DepthVideo (mapper-facing buffer) ----
            self.video = DepthVideo(cfg, self.printer, uncer_network=self.uncer_network)

            # ---- DROID backend BA ----
            self.ba = Backend(self.droid_net, self.video, self.cfg)

            # ---- DROID pose filler (non-keyframes) ----
            self.traj_filler = PoseTrajectoryFiller(
                cfg=cfg,
                net=self.droid_net,
                video=self.video,
                printer=self.printer,
                device=self.device,
            )

        else:
            # ---- DPVO ----
            dpvo_weights = self.cfg["tracking"]["dpvo"].get("weights", None)
            if dpvo_weights is None:
                raise ValueError("Missing cfg['tracking']['dpvo']['weights'] for DPVO mode")

            # Build DPVO yacs config from defaults + YAML overrides
            from src.dpvo.config import cfg as dpvo_cfg_default  # yacs CfgNode
            dpvo_cfg = dpvo_cfg_default.clone()

            # Your YAML structure: tracking.dpvo.opts contains overrides
            dpvo_opts = self.cfg["tracking"]["dpvo"].get("opts", {})
            if dpvo_opts is not None:
                for k, v in dpvo_opts.items():
                    # only set keys that exist; helps catch typos early
                    if not hasattr(dpvo_cfg, k):
                        raise KeyError(f"Unknown DPVO option '{k}' in cfg['tracking']['dpvo']['opts']")
                    setattr(dpvo_cfg, k, v)

            # Important: DPVO wants ht/wd full-res for image tensor shape
            self.dpvo = DPVO(dpvo_cfg, dpvo_weights, ht=self.H, wd=self.W, viz=False)
            # Mapper-facing buffer
            self.video = self.dpvo.pg
            self.ba = None
            self.traj_filler = None

        # -------------------------
        # process sync counters
        # -------------------------
        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        # DPVO-only: signal that dpvo.terminate() (final BA) finished in tracker process
        self.dpvo_final_ba_done = torch.zeros((1)).int()
        self.dpvo_final_ba_done.share_memory_()
        # DPVO: mapping has finished GPU-heavy work (final_refine + rendering)
        self.dpvo_mapping_gpu_done = torch.zeros((1)).int()
        self.dpvo_mapping_gpu_done.share_memory_()

        # misc
        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream

    def load_pretrained(self, cfg):
        if not self.is_droid:
            return

        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )


    def tracking(self, pipe):
        """Spawned in its own process."""
        self.tracker = make_tracker(self, pipe)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass

        self.printer.print("Tracking Starts!", FontColor.TRACKER)
        self.printer.pbar_ready()

        # --- run tracking loop ---
        self.tracker.run(self.stream)

        # --- DPVO final BA ---
        if self.is_dpvo:
            try:
                self.printer.print(
                    "DPVO: waiting for mapping to finish GPU-heavy stage...",
                    FontColor.INFO
                )
                while int(self.dpvo_mapping_gpu_done[0].item()) == 0:
                    time.sleep(0.05)

                self.printer.print("DPVO Final BA (dpvo.terminate) Triggered!", FontColor.TRACKER)

                # Important: clear as much as possible *in this process*
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Pi3MOS-style: AMP + no_grad
                with torch.no_grad():
                    with autocast(enabled=bool(self.dpvo.cfg.MIXED_PRECISION)):
                        poses, tstamps = self.dpvo.terminate()
                
                # print("poses shape:", np.asarray(poses).shape, "dtype:", np.asarray(poses).dtype)
                # print("tstamps shape:", np.asarray(tstamps).shape)
                traj_path = os.path.join(self.save_dir, "dpvo_traj.npz")
                np.savez(traj_path, poses=poses, tstamps=tstamps)
                self.printer.print(f"DPVO Final BA Done! Saved: {traj_path}", FontColor.TRACKER)

            except Exception as e:
                self.printer.print(f"DPVO Final BA failed: {e}", FontColor.ERROR)

            finally:
                with torch.no_grad():
                    self.dpvo_final_ba_done[0] = 1

        self.printer.print("Tracking Done!", FontColor.TRACKER)


    def mapping(self, pipe, q_main2vis, q_vis2main):
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        else:
            self.mapper = Mapper(self, pipe, None, q_main2vis, q_vis2main)

        self.printer.print("Mapping Triggered!", FontColor.MAPPER)
        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass

        self.printer.print("Mapping Starts!", FontColor.MAPPER)

        # --- run mapping loop ---
        self.mapper.run()

        # IMPORTANT (DPVO): mapping finished the stage that would collide with dpvo.terminate()
        # Signal the tracker NOW (not at the end of terminate()).
        if self.is_dpvo:
            with torch.no_grad():
                self.dpvo_mapping_gpu_done[0] = 1
            torch.cuda.empty_cache()

        self.printer.print("Mapping Done!", FontColor.MAPPER)

        # OPTIONAL but recommended: if you want WildGS-like log order, wait for DPVO BA to finish
        # before starting evaluation / refine (so dpvo_traj exists).
        if self.is_dpvo and self.cfg["tracking"]["backend"]["final_ba"]:
            self.printer.print("Waiting for DPVO Final BA (dpvo.terminate)...", FontColor.INFO)
            while int(self.dpvo_final_ba_done[0].item()) == 0:
                time.sleep(0.01)

        self.terminate()


    def backend(self):
        # DROID-only BA backend
        if self.is_droid:
            self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

            metric_depth_reg_activated = self.video.metric_depth_reg
            if metric_depth_reg_activated:
                self.video.metric_depth_reg = False

            self.ba = Backend(self.droid_net, self.video, self.cfg)
            torch.cuda.empty_cache()
            self.ba.dense_ba(7)
            torch.cuda.empty_cache()
            self.ba.dense_ba(12)

            self.printer.print("Final Global BA Done!", FontColor.TRACKER)

            if metric_depth_reg_activated:
                self.video.metric_depth_reg = True

        else:
            return

    def terminate(self):
        """Fill poses for non-keyframes + evaluate."""
        # -------------------------
        # (A) optional eval before final BA
        # -------------------------
        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            if hasattr(self.video, "save_video"):
                self.video.save_video(f"{self.save_dir}/video.npz")

                if not isinstance(self.stream, RGB_NoPose):
                    try:
                        ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                            f"{self.save_dir}/video.npz",
                            f"{self.save_dir}/traj/before_final_ba",
                            "kf_traj",
                            self.stream,
                            self.logger,
                            self.printer,
                        )
                    except Exception as e:
                        self.printer.print(e, FontColor.ERROR)
            else:
                self.printer.print(
                    "WARNING: video.save_video() not available in current mode; skipping before_final_ba traj eval.",
                    FontColor.INFO
                )

            self.mapper.save_all_kf_figs(self.save_dir, iteration="before_refine")

        # -------------------------
        # (B) final BA (droid only)
        # -------------------------
        if self.cfg["tracking"]["backend"]["final_ba"] and self.is_droid:
            self.backend()

        # -------------------------
        # (C) keyframe trajectory eval
        # -------------------------
        if hasattr(self.video, "save_video"):
            self.video.save_video(f"{self.save_dir}/video.npz")

            if not isinstance(self.stream, RGB_NoPose):
                try:
                    ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)
        else:
            self.printer.print(
                "WARNING: video.save_video() not available; skipping kf_traj_eval.",
                FontColor.INFO
            )

        # -------------------------
        # (D) final refine (mapping)
        # -------------------------
        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.mapper.final_refine(iters=self.cfg["mapping"]["final_refine_iters"])
        # Evaluate / dump renders
        self.mapper.save_all_kf_figs(self.save_dir, iteration="after_refine")

        # -------------------------
        # (E) full trajectory eval (DROID-only unless you implement a DPV filler)
        # -------------------------
        if self.is_droid and (self.traj_filler is not None):
            self.traj_filler.setup_feature_extractor()
            full_traj_eval(
                self.traj_filler,
                self.mapper,
                f"{self.save_dir}/traj",
                "full_traj",
                self.stream,
                self.logger,
                self.printer,
                self.cfg.get("fast_mode", False),
            )

        elif self.is_dpvo:
            self.printer.print("Waiting for DPVO final trajectory...", FontColor.INFO)
            while int(self.dpvo_final_ba_done[0].item()) == 0:
                time.sleep(0.01)

            dpvo_npz = os.path.join(self.save_dir, "dpvo_traj.npz")
            if not os.path.exists(dpvo_npz):
                self.printer.print(f"DPVO trajectory file missing: {dpvo_npz}", FontColor.ERROR)
            else:
                try:
                    stats = full_traj_eval_dpvo_stream(
                        dpvo_npz_path=dpvo_npz,
                        stream=self.stream,
                        printer=self.printer,
                        save_dir=f"{self.save_dir}/traj",
                        plot_name="full_traj_dpvo",
                        do_plot=True,
                        correct_scale=True,
                        dpvo_pose_is_Twc=True,   # try True first
                    )
                    self.printer.print(f"DPVO full traj stats: {stats}", FontColor.EVAL)
                except Exception as e:
                    self.printer.print(f"DPVO full traj eval failed: {e}", FontColor.ERROR)

        else:
            self.printer.print(
                "Full trajectory evaluation skipped (no supported trajectory source).",
                FontColor.INFO
            )

        # Save final map
        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")

        # Save uncertainty network
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            torch.save(
                self.mapper.uncer_network.state_dict(),
                os.path.join(self.save_dir, "uncertainty_mlp_weight.pth"),
            )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def run(self):
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,q_main2vis,q_vis2main)),
        ]
        self.num_running_thread += len(processes)
        if self.cfg['gui']:
            self.num_running_thread += 1
        for p in processes:
            p.start()

        if self.cfg['gui']:
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            self.all_trigered += 1


        for p in processes:
            p.join()

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()

    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        """From Splat-SLAM. Not used in WildGS-SLAM evaluation, but might be useful in the future."""
        # Evaluate depth error
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,  # How much of each frame uses depth from droid (the rest from Omnidata)
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        # Write to the file
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
