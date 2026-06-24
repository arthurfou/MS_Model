import numpy as np
import torch

from ms_model.io.events import EventArray


def to_event_slice(ea: EventArray, t_lo: float, t_hi: float) -> torch.Tensor:
    """3-channel event slice from events in [t_lo, t_hi).

    Returns (3, H, W) float32:
      ch0: positive event count per pixel
      ch1: negative event count per pixel
      ch2: mean normalised timestamp in [0, 1], 0 where no events
    """
    H, W = ea.H, ea.W
    mask = (ea.t >= t_lo) & (ea.t < t_hi)

    if not mask.any():
        return torch.zeros(3, H, W, dtype=torch.float32)

    t = ea.t[mask]
    x = ea.x[mask].astype(np.int32)
    y = ea.y[mask].astype(np.int32)
    p = ea.p[mask]

    # Normalise timestamps within the window to [0, 1]
    t_norm = ((t - t_lo) / (t_hi - t_lo + 1e-10)).astype(np.float32)

    pos_count = np.zeros((H, W), dtype=np.float32)
    neg_count = np.zeros((H, W), dtype=np.float32)
    ts_sum = np.zeros((H, W), dtype=np.float32)
    ts_cnt = np.zeros((H, W), dtype=np.float32)

    pos_mask = p == 1
    np.add.at(pos_count, (y[pos_mask], x[pos_mask]), 1.0)
    np.add.at(neg_count, (y[~pos_mask], x[~pos_mask]), 1.0)
    np.add.at(ts_sum, (y, x), t_norm)
    np.add.at(ts_cnt, (y, x), 1.0)

    ts_agg = np.where(ts_cnt > 0, ts_sum / np.where(ts_cnt > 0, ts_cnt, 1.0), 0.0)

    return torch.from_numpy(np.stack([pos_count, neg_count, ts_agg]))


def make_slice_sequence(
    ea: EventArray,
    t_start: float,
    t_end: float,
    slice_dt: float = 0.025,
) -> torch.Tensor:
    """Cut [t_start, t_end) into fixed-width windows of slice_dt seconds.

    Returns (N, 3, H, W). N = number of complete windows.
    """
    slices = []
    t = t_start
    while t + slice_dt <= t_end + 1e-9:
        slices.append(to_event_slice(ea, t, t + slice_dt))
        t += slice_dt

    if not slices:
        return torch.zeros(1, 3, ea.H, ea.W, dtype=torch.float32)

    return torch.stack(slices)
