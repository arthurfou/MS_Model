import numpy as np

from ms_model.io.events import EventArray
from ms_model.io.masks import FrameMasks
from ms_model.viz.render import make_frames


def _id_to_color(obj_id: float) -> tuple[int, int, int]:
    """Couleur déterministe pour un id d'objet (0 = noir/fond)."""
    if obj_id == 0:
        return (0, 0, 0)
    rng = np.random.default_rng(int(obj_id))
    return tuple(int(c) for c in rng.integers(50, 256, size=3))


def render_mask_frame(mask: np.ndarray) -> np.ndarray:
    """Rend un masque (H, W) en image (H, W, 3) uint8, une couleur par id d'objet."""
    frame = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue
        frame[mask == obj_id] = _id_to_color(obj_id)
    return frame


def render_mask_overlay(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """Surligne en `color` les pixels où mask != 0, mélangés avec `frame`."""
    out = frame.copy()
    region = mask != 0
    out[region] = (1 - alpha) * frame[region] + alpha * np.array(color)
    return out


def make_mask_frames(fm: FrameMasks) -> list[np.ndarray]:
    """Rend chaque masque de fm en image (H, W, 3) uint8."""
    return [render_mask_frame(m) for m in fm.masks]


def make_overlay_frames(ea: EventArray, fm: FrameMasks, color=(0, 255, 0), alpha: float = 0.4) -> list[np.ndarray]:
    """Rend les events alignés sur les timestamps de fm, avec overlay du masque correspondant.

    len(frame_ts) == len(fm.ts), donc make_frames produit len(fm.ts) - 1 frames ;
    on overlay fm.masks[i] (masque au début de l'intervalle) sur frames[i].
    """
    frames = make_frames(ea, frame_ts=fm.ts)
    return [render_mask_overlay(f, fm.masks[i], color=color, alpha=alpha) for i, f in enumerate(frames)]
