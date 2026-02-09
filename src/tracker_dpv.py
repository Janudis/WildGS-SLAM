# src/tracker_dpv.py

import torch
import numpy as np
import cv2

from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer,FontColor
from src.utils.datasets import load_metric_depth
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.mono_priors.img_feature_extractors import get_feature_extractor, predict_img_features

class DPVTracker:
    def __init__(self, slam, pipe):
        self.cfg = slam.cfg
        self.device = torch.device(self.cfg["device"])
        self.verbose = slam.verbose
        self.pipe = pipe
        self.printer = slam.printer
        self.save_dir = slam.save_dir

        # DPVO instance (created in slam.py)
        self.dpvo = slam.dpvo

        # mapper-facing "video" object: PatchGraph with mapper I/O enabled
        # expected: slam.video is slam.dpvo.pg (or equivalent PatchGraph)
        self.video = slam.video

        # If your dataset uses distortion, we can mirror dpvo/stream.py’s undistortion
        cam_cfg = self.cfg.get("cam", {})
        self.distortion = np.array(cam_cfg.get("distortion", []), dtype=np.float32) \
            if ("distortion" in cam_cfg and cam_cfg["distortion"] is not None) else None
        # Original intrinsics (before resize/crop) from cfg — consistent with BaseDataset
        self.fx0 = float(cam_cfg.get("fx", 0.0))
        self.fy0 = float(cam_cfg.get("fy", 0.0))
        self.cx0 = float(cam_cfg.get("cx", 0.0))
        self.cy0 = float(cam_cfg.get("cy", 0.0))

        # ------------------------------------------------------------
        # Metric depth priors (Metric3D / DPT2) — mirror WildGS behavior
        # ------------------------------------------------------------
        self.metric_depth_estimator = get_metric_depth_estimator(self.cfg)
        # If mapping needs features (uncertainty-aware mapping), produce them here
        self.need_features = bool(self.cfg["mapping"]["uncertainty_params"]["activate"])
        self.feat_extractor = None
        if self.need_features:
            if get_feature_extractor is None or predict_img_features is None:
                raise ImportError(
                    "mapping uncertainty is active but feature extractor utilities "
                    "could not be imported (src/utils/mono_priors/feature_extractors.py)"
                )
            self.feat_extractor = get_feature_extractor(self.cfg)

    def _make_cv_bgr_uint8(self, image_torch: torch.Tensor):
        """
        Convert WildGS dataset image -> DPVO image format.

        Input:  image_torch: [1,3,H,W], RGB, float32 in [0,1]
        Output: image_cv:    [H',W',3], BGR, uint8 in [0,255], cropped to /16
                intrinsics:  np.array([fx,fy,cx,cy]) adjusted for the crop (same fx,fy; cx,cy shift)
        """
        # --- to CPU numpy, RGB float [0,1] ---
        if image_torch.ndim != 4 or image_torch.shape[0] != 1 or image_torch.shape[1] != 3:
            raise ValueError(f"Expected image [1,3,H,W], got {tuple(image_torch.shape)}")

        img = image_torch[0].detach()
        if img.is_cuda:
            img = img.cpu()

        img = img.permute(1, 2, 0).numpy()  # H,W,3 RGB float

        # clamp + scale to uint8
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0 + 0.5).astype(np.uint8)  # RGB uint8

        # RGB -> BGR (cv2 convention)
        img = img[..., ::-1].copy()

        # Optional undistort (like dpvo/stream.py)
        fx, fy, cx, cy = self.fx0, self.fy0, self.cx0, self.cy0
        if self.distortion is not None and self.distortion.size > 0:
            K = np.eye(3, dtype=np.float32)
            K[0, 0], K[1, 1] = fx, fy
            K[0, 2], K[1, 2] = cx, cy
            img = cv2.undistort(img, K, self.distortion)

        # Crop to multiples of 16 (like dpvo/stream.py)
        h, w, _ = img.shape
        h16 = h - (h % 16)
        w16 = w - (w % 16)
        img = img[:h16, :w16]

        # If we crop only bottom/right, principal point stays the same.
        # (If you ever center-crop, you must shift cx,cy accordingly.)
        intr = np.array([fx, fy, cx, cy], dtype=np.float32)
        return img, intr
    
    @torch.no_grad()
    def run(self, stream):
        prev_initialized = False
        prev_dpvo_n = self.dpvo.n

        for i in range(len(stream)):
            frame_idx, image, _, _ = stream[i]
            starting_count = self.video.counter.value

            # --- convert image to DPVO expected format ---
            image_cv, intr_np = self._make_cv_bgr_uint8(image)
            image_t = torch.from_numpy(image_cv).to(device="cuda", dtype=torch.uint8)
            image_t = image_t.permute(2, 0, 1).contiguous()  # (3,H,W)
            intrinsic_vec4 = torch.as_tensor(intr_np, device="cuda", dtype=torch.float32)

            # DPVO step
            self.dpvo(int(frame_idx), image_t, intrinsic_vec4)

            # Detect init transition (DPVO-specific: becomes True at n==8)
            just_initialized = (self.dpvo.is_initialized and not prev_initialized)
            prev_initialized = self.dpvo.is_initialized

            # Did DPVO accept a new frame?
            dpvo_n_now = self.dpvo.n
            accepted_new_frame = (dpvo_n_now > prev_dpvo_n)
            prev_dpvo_n = dpvo_n_now

            if accepted_new_frame:
                # last accepted frame index inside DPVO
                k = self.dpvo.n - 1
                pose_w2c_7 = self.dpvo.pg.poses_[k].detach()

                # Produce Metric3D prior and save to disk (WildGS-compatible)
                metric_depth = predict_metric_depth(
                    self.metric_depth_estimator,
                    int(frame_idx),
                    image.to(self.device),
                    self.cfg,
                    str(self.device),
                    save_depth=True,
                )

                # If mapping uncertainty is active, also save features
                if self.need_features:
                    _ = predict_img_features(
                        self.feat_extractor,
                        int(frame_idx),
                        image.to(self.device),
                        self.cfg,
                        str(self.device),
                    )

                # Append into mapper-facing buffer EVEN BEFORE DPVO init
                self.video.append(
                    float(frame_idx),
                    None,
                    pose_w2c_7,
                    None,
                    metric_depth,
                    intrinsic_vec4.to(self.device),
                )

            # Notify mapper only after DPVO has initialized
            if starting_count < self.video.counter.value:
                curr_kf_idx = self.video.counter.value - 1

                if just_initialized:
                    # This should happen when buffer already contains ~8 KFs (0..7)
                    self.pipe.send(
                        {
                            "is_keyframe": True,
                            "video_idx": int(curr_kf_idx),
                            "timestamp": int(frame_idx),
                            "just_initialized": True,
                            "end": False,
                        }
                    )
                    self.pipe.recv()

                elif self.dpvo.is_initialized:
                    # Normal mapping updates
                    self.pipe.send(
                        {
                            "is_keyframe": True,
                            "video_idx": int(curr_kf_idx),
                            "timestamp": int(frame_idx),
                            "just_initialized": False,
                            "end": False,
                        }
                    )
                    self.pipe.recv()
                else:
                    # DPVO not initialized yet: we are only filling the buffer
                    pass

            self.printer.update_pbar()

        self.pipe.send(
            {
                "is_keyframe": True,
                "video_idx": None,
                "timestamp": None,
                "just_initialized": False,
                "end": True,
            }
        )
