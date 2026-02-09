import numpy as np
import torch
from einops import asnumpy, reduce, repeat
import lietorch

from . import projective_ops as pops
from .loop_closure.optim_utils import reduce_edges
from .utils import *
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F


class PatchGraph:
    """ Dataclass for storing variables """

    # def __init__(self, cfg, P, DIM, pmem, **kwargs):
    def __init__(self, cfg, P, DIM, pmem, ht=None, wd=None, device="cuda", **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        dev = torch.device(device) if isinstance(device, str) else device
        dtype = kwargs.get("dtype", torch.float32)

        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        ### edge information ###
        # self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.net = torch.zeros(1, 0, DIM, device=dev, dtype=dtype)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        ### inactive edge information (i.e., no longer updated, but useful for BA) ###
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.weight_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")
        self.target_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")

        ### depth priors from Metric3D
        self.prior_inv_depth_ = torch.zeros(self.N, self.M, 1, dtype=torch.float, device="cuda") # prior inverse depth.
        self.confidences_ = torch.zeros(self.N, self.M, 1, dtype=torch.float, device="cuda") # confidence score of depth prediction. 
        self.var_ = torch.zeros(self.N, self.M, 1, dtype=torch.float, device="cuda") # variance of estimated inverse depth
        # confidences_ == 0 means "ignore prior"

        # ============================================================
        # Mapper-facing shared buffers (DepthVideo-like, but inside PatchGraph)
        # IMPORTANT: these are indexed by a monotonic video_idx and are NOT
        # affected by DPVO keyframe removal/compaction.
        # ============================================================
        # Enable flag so other code can check existence
        self.enable_mapper_io = (ht is not None) and (wd is not None)
        self.metric_depth_reg = True  # we always provide metric depth (Metric3D)

        if self.enable_mapper_io:
            self._lock = Lock()

            self.ht = int(ht)
            self.wd = int(wd)
            self.device = device  # e.g. "cuda" or "cuda:0"

            # Use DPVO buffer length as the keyframe/video buffer length.
            self.video_buffer = int(self.N)

            # Monotonic keyframe counter for mapper indexing (video_idx)
            self.counter = Value("i", 0)

            # ---- Minimal per-keyframe state (shared memory) ----
            self.timestamp = torch.zeros(
                self.video_buffer, device=self.device, dtype=torch.float
            ).share_memory_()

            # Pose storage (SE3 as 7D: [tx,ty,tz,qx,qy,qz,qw] or dpvo convention)
            self.poses_video = torch.zeros(
                self.video_buffer, 7, device=self.device, dtype=torch.float
            ).share_memory_()
            self.poses_video[:] = torch.as_tensor(
                [0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device
            )

            # Intrinsics per keyframe (optional, but useful / consistent)
            self.intrinsics_video = torch.zeros(
                self.video_buffer, 4, device=self.device, dtype=torch.float
            ).share_memory_()

            # Full-res metric depth in meters
            self.depth_video = torch.zeros(
                self.video_buffer, self.ht, self.wd, device=self.device, dtype=torch.float
            ).share_memory_()

            # Valid depth mask (full-res)
            self.depth_mask_video = torch.zeros(
                self.video_buffer, self.ht, self.wd, device=self.device, dtype=torch.bool
            ).share_memory_()

            # (Optional) store RGB on CPU if you want debug / future use; mapper does NOT read it
            # Keep it off by default to save RAM:
            self.store_mapper_images = False
            if self.store_mapper_images:
                self.images = torch.zeros(
                    self.video_buffer, 3, self.ht, self.wd, device="cpu", dtype=torch.float32
                ).share_memory_()

    def edges_loop(self):
        """ Adding edges from old patches to new frames """
        lc_range = self.cfg.MAX_EDGE_AGE
        l = self.n - self.cfg.REMOVAL_WINDOW # l is the upper bound for "old" patches

        if l <= 0:
            return torch.empty(2, 0, dtype=torch.long, device='cuda')

        # create candidate edges
        jj, kk = flatmeshgrid(
            torch.arange(self.n - self.cfg.GLOBAL_OPT_FREQ, self.n - self.cfg.KEYFRAME_INDEX, device="cuda"),
            torch.arange(max(l - lc_range, 0) * self.M, l * self.M, device="cuda"), indexing='ij')
        ii = self.ix[kk]

        # Remove edges which have too large flow magnitude
        flow_mg, val = pops.flow_mag(SE3(self.poses), self.patches[...,1,1].view(1,-1,3,1,1), self.intrinsics, ii, jj, kk, beta=0.5)
        flow_mg_sum = reduce(flow_mg * val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).float()
        num_val = reduce(val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).clamp(min=1)
        flow_mag = torch.where(num_val > (self.M * 0.75), flow_mg_sum / num_val, torch.inf)

        mask = (flow_mag < self.cfg.BACKEND_THRESH)
        es = reduce_edges(asnumpy(flow_mag[mask]), asnumpy(ii[::self.M][mask]), asnumpy(jj[::self.M][mask]), max_num_edges=1000, nms=1)

        edges = torch.as_tensor(es, device=ii.device)
        ii, jj = repeat(edges, 'E ij -> ij E M', M=self.M, ij=2)
        kk = ii.mul(self.M) + torch.arange(self.M, device=ii.device)
        return kk.flatten(), jj.flatten()

    def normalize(self):
        """ normalize depth and poses """
        s = self.patches_[:self.n,:,2].mean()
        self.patches_[:self.n,:,2] /= s
        self.poses_[:self.n,:3] *= s
        for t, (t0, dP) in self.delta.items():
            self.delta[t] = (t0, dP.scale(s))
        self.poses_[:self.n] = (SE3(self.poses_[:self.n]) * SE3(self.poses_[[0]]).inv()).data

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
    
    # ============================================================
    # Mapper-facing methods
    # ============================================================
    def get_lock(self):
        if not self.enable_mapper_io:
            raise RuntimeError("Mapper I/O buffers disabled.")
        return self._lock

    def __item_setter(self, index, item):
        """
        Match DepthVideo.append() layout so Tracker code can stay WildGS-like:

            item[0] timestamp (scalar)
            item[1] image (unused for mapper; can be None)
            item[2] pose7 (required)  (w2c in dpvo convention)
            item[3] disps (unused; can be None)
            item[4] mono_depth OR metric depth prior (required for dpv) (H,W) meters
            item[5] intrinsics4 (optional but recommended)

        Anything after item[5] is ignored.
        """
        if not self.enable_mapper_io:
            raise RuntimeError("Mapper I/O buffers disabled.")

        # index / counter
        index = int(index)
        if index >= self.video_buffer:
            raise RuntimeError(f"PatchGraph mapper buffer overflow: {index} >= {self.video_buffer}")

        # Maintain monotonic counter like DepthVideo
        if index >= self.counter.value:
            self.counter.value = index + 1

        # Unpack DepthVideo-style tuple
        timestamp = item[0]
        pose7 = item[2] if len(item) > 2 else None
        depth = item[4] if len(item) > 4 else None
        intr = item[5] if len(item) > 5 else None

        # timestamp
        if torch.is_tensor(timestamp):
            self.timestamp[index] = timestamp.float()
        else:
            self.timestamp[index] = float(timestamp)

        # pose
        if pose7 is None:
            raise ValueError("pose7 is required for mapper I/O")
        if not torch.is_tensor(pose7):
            raise ValueError("pose7 must be a torch.Tensor")

        pose7 = pose7.to(self.device).float()
        self.poses_video[index] = pose7

        # depth (meters)
        if depth is None:
            raise ValueError("depth (metric prior) is required for mapper I/O")
        if not torch.is_tensor(depth):
            raise ValueError("depth must be a torch.Tensor")

        depth = depth.to(self.device).float()

        # Safety: ensure expected shape (H,W)
        if depth.ndim != 2:
            raise ValueError(f"depth must be HxW, got shape {tuple(depth.shape)}")
        if depth.shape[-2] != self.ht or depth.shape[-1] != self.wd:
            raise ValueError(
                f"depth shape mismatch: got {tuple(depth.shape)} expected ({self.ht},{self.wd})"
            )

        self.depth_video[index] = depth
        # More faithful than all-ones: keep only positive depths valid
        self.depth_mask_video[index] = (depth > 0)

        # intrinsics
        if intr is not None:
            if not torch.is_tensor(intr):
                intr = torch.as_tensor(intr)
            self.intrinsics_video[index] = intr.to(self.device).float()

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(int(self.counter.value), item)

    def get_pose(self, index, device):
        w2c = lietorch.SE3(self.poses_video[index].clone().to(device))
        c2w = w2c.inv().matrix()
        return c2w

    def get_depth_and_pose(self, index, device):
        """
        This matches what Mapper.get_w2c_and_depth() expects:
        est_depth, valid_depth_mask, c2w
        """
        with self.get_lock():
            est_depth = self.depth_video[index].clone().to(device)
            depth_mask = self.depth_mask_video[index].clone().to(device)
            c2w = self.get_pose(index, device)
        return est_depth, depth_mask, c2w
    
    def save_video(self, path: str):
        """
        Save mapper-facing keyframe data to npz, compatible with kf_traj_eval().
        Fields:
            poses: (K,4,4) c2w
            depths: (K,H,W)
            timestamps: (K,)
            valid_depth_masks: (K,H,W)
        """
        if not self.enable_mapper_io:
            raise RuntimeError("Mapper I/O buffers disabled; cannot save video.")
        with self.get_lock():
            K = int(self.counter.value)

        poses = []
        depths = []
        timestamps = []
        valid_masks = []
        for i in range(K):
            depth_i, mask_i, pose_i = self.get_depth_and_pose(i, device="cpu")
            ts_i = self.timestamp[i].detach().cpu()
            poses.append(pose_i)
            depths.append(depth_i)
            timestamps.append(ts_i)
            valid_masks.append(mask_i)
        poses = torch.stack(poses, dim=0).numpy()
        depths = torch.stack(depths, dim=0).numpy()
        timestamps = torch.stack(timestamps, dim=0).numpy()
        valid_masks = torch.stack(valid_masks, dim=0).numpy()
        np.savez(
            path,
            poses=poses,
            depths=depths,
            timestamps=timestamps,
            valid_depth_masks=valid_masks,
        )
        # PatchGraph doesn't have printer; print only if exists
        if hasattr(self, "printer") and self.printer is not None:
            self.printer.print(f"Saved final depth video: {path}", FontColor.INFO)
        else:
            print(f"[PatchGraph] Saved final depth video: {path}")
