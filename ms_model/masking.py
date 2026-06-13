import cv2
import numpy as np

from ms_model.io.events import EventArray
from ms_model.io.masks import FrameMasks


def thicken_mask(fm: FrameMasks, radius: int = 3) -> FrameMasks:
    """Dilate chaque masque de `fm` de `radius` pixels (dans toutes les directions).

    Sert à compenser l'imprécision du masque GT : un pixel à moins de
    `radius` pixels d'une zone masquée est lui aussi considéré comme masqué.
    """
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dtype = fm.masks.dtype
    thickened = np.stack([cv2.dilate(m.astype(np.float32), kernel) for m in fm.masks]).astype(dtype)
    return FrameMasks(masks=thickened, ts=fm.ts)


def events_in_mask(ea: EventArray, fm: FrameMasks) -> np.ndarray:
    """Retourne un booléen par event : True si l'event tombe dans une zone masquée.

    Pour chaque event, on trouve le masque correspondant via son timestamp
    (le dernier masque dont ts <= t), puis on regarde le pixel (y, x).
    """
    bin_idx = np.searchsorted(fm.ts, ea.t, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(fm.masks) - 1)
    return fm.masks[bin_idx, ea.y.astype(np.int64), ea.x.astype(np.int64)] != 0


def remove_events_in_mask(ea: EventArray, fm: FrameMasks) -> EventArray:
    """Retourne un nouvel EventArray sans les events tombant dans une zone masquée."""
    hit = events_in_mask(ea, fm)
    return EventArray(events=ea.events[~hit], H=ea.H, W=ea.W, frame_ts=ea.frame_ts)
