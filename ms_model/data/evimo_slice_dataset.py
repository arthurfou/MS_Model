import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ms_model.io.loaders import load_events
from ms_model.representations.event_slice import make_slice_sequence


@dataclass
class _SequenceData:
    slices: torch.Tensor    # (N_s, 3, H, W)
    depth: np.ndarray       # (N_f, H, W) float32 — NaN = no GT depth (background)
    mask: np.ndarray        # (N_f, H, W) int64  — 0=bg, 1..C=objects
    ego_vel: np.ndarray     # (N_f, 6) float32 [vx, vy, vz, wx, wy, wz]
    K: np.ndarray           # (3, 3) float32 — camera intrinsics
    slice_ts: np.ndarray    # (N_s,) centre timestamps of each slice
    frame_ts: np.ndarray    # (N_f,) GT frame timestamps


def _parse_npz(path: Path, slice_dt: float) -> _SequenceData:
    data = np.load(path, allow_pickle=True)
    meta = data["meta"].item()
    frames = meta["frames"]
    N_f = len(frames)

    K = data["K"].astype(np.float32)

    frame_ts = np.array([f["ts"] for f in frames], dtype=np.float64)

    # depth: NaN at background pixels, mm at object pixels
    depth = data["depth"].astype(np.float32)

    # mask: remap 0, 1000, 2000, ... → 0, 1, 2, ...
    mask = np.round(data["mask"] / 1000).astype(np.int64)

    # ego velocity: (N_f, 6)
    ego_vel = np.zeros((N_f, 6), dtype=np.float32)
    for i, f in enumerate(frames):
        vel = f["cam"]["vel"]
        t_vel = vel["t"]
        rpy = vel["rpy"]
        ego_vel[i] = [t_vel["x"], t_vel["y"], t_vel["z"],
                      rpy["r"], rpy["p"], rpy["y"]]

    # Event slices with disk cache
    ea = load_events(path)
    cache = Path(path).with_suffix(f".slices_{int(slice_dt * 1000)}ms.pt")
    if cache.exists():
        slices = torch.load(cache)
    else:
        slices = make_slice_sequence(ea, float(ea.t.min()), float(ea.t.max()), slice_dt)
        torch.save(slices, cache)

    N_s = slices.shape[0]
    t0 = float(ea.t.min())
    slice_ts = np.array([t0 + (i + 0.5) * slice_dt for i in range(N_s)])

    return _SequenceData(
        slices=slices,
        depth=depth,
        mask=mask,
        ego_vel=ego_vel,
        K=K,
        slice_ts=slice_ts,
        frame_ts=frame_ts,
    )


class EvimoSliceDataset(Dataset):
    """Dataset for the EV-IMO SfM pipeline.

    Each sample contains:
      slices   (n_ctx, 3, H, W) — consecutive 25 ms event slices
      depth_gt (H, W) float32   — sparse GT depth in mm, NaN = background
      mask_gt  (H, W) int64     — 0=bg, 1..C=objects
      ego_vel  (6,) float32     — GT camera velocity [vx,vy,vz,wx,wy,wz]
      K        (3, 3) float32   — camera intrinsics

    The depth network takes slices[:, half] (middle slice).
    The pose network takes slices concatenated to (n_ctx*3, H, W).
    GT supervision is aligned to the GT frame nearest the middle slice timestamp.
    """

    def __init__(
        self,
        npz_paths: list[Union[str, Path]],
        slice_dt: float = 0.025,
        n_slices_context: int = 5,
        max_objects: int = 3,
    ):
        self.n_ctx = n_slices_context
        self.half = n_slices_context // 2
        self.seqs: list[_SequenceData] = []
        self.index: list[tuple[int, int]] = []  # (seq_idx, centre_slice_idx)

        for i, path in enumerate(npz_paths):
            path = Path(path)
            t0 = time.time()
            print(f"[EvimoSliceDataset] ({i + 1}/{len(npz_paths)}) {path.name} ...", flush=True)

            seq = _parse_npz(path, slice_dt)
            N_s = seq.slices.shape[0]
            seq_idx = len(self.seqs)

            for c in range(self.half, N_s - self.half):
                self.index.append((seq_idx, c))

            self.seqs.append(seq)
            print(
                f"[EvimoSliceDataset]   {N_s} slices, "
                f"{seq.depth.shape[0]} GT frames, {time.time() - t0:.1f}s",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        seq_idx, centre = self.index[idx]
        seq = self.seqs[seq_idx]

        slices = seq.slices[centre - self.half : centre + self.half + 1]  # (n_ctx, 3, H, W)

        # Nearest GT frame to centre slice
        gt_idx = int(np.argmin(np.abs(seq.frame_ts - seq.slice_ts[centre])))

        return {
            "slices": slices,
            "depth_gt": torch.from_numpy(seq.depth[gt_idx].copy()),
            "mask_gt": torch.from_numpy(seq.mask[gt_idx].copy()),
            "ego_vel": torch.from_numpy(seq.ego_vel[gt_idx].copy()),
            "K": torch.from_numpy(seq.K.copy()),
        }
