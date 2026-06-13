from ms_model.io.events import EventArray
import numpy as np


def render_event_frame(
            ea: EventArray,
            t_lo: float,
            t_hi: float,
            pos_color=(255, 0, 0),
            neg_color=(0, 0, 255)
        ) -> np.ndarray:
    """Rend les events de ea dans [t_lo, t_hi) en image (H, W, 3) uint8."""
    mask = (ea.t >= t_lo) & (ea.t < t_hi)
    xs = ea.x[mask].astype(np.int64)
    ys = ea.y[mask].astype(np.int64)
    pols = ea.p[mask] > 0

    frame = np.zeros((ea.H, ea.W, 3), dtype=np.uint8)
    frame[ys[pols], xs[pols]] = pos_color
    frame[ys[~pols], xs[~pols]] = neg_color
    return frame


def make_frames(
            ea: EventArray,
            frame_ts: np.ndarray | None = None,
            n_frames: int | None = None
        ) -> list[np.ndarray]:
    """Découpe le flux d'events en une liste de frames rendues.

    - Si frame_ts est fourni (sinon ea.frame_ts), il sert de bornes temporelles.
    - Sinon, n_frames doit être fourni : bornes uniformes entre ea.t.min() et ea.t.max().
    """
    if frame_ts is None:
        if n_frames is None:
            if ea.frame_ts is None:
                raise ValueError("frame_ts ou n_frames doit être fourni")
            frame_ts = ea.frame_ts
        else:
            frame_ts = np.linspace(ea.t.min(), ea.t.max(), n_frames + 1)

    return [
        render_event_frame(ea, frame_ts[i - 1], frame_ts[i]) for i in range(1, len(frame_ts))
    ]
