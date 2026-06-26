from pathlib import Path
from typing import Optional, Union

import numpy as np

from .events import EventArray
from .masks import FrameMasks

# Résolution DVS346 utilisée par EVIMO (cf. DEVO/scripts/preprocess_evimo.py)
EVIMO_H, EVIMO_W = 260, 346


def load_evimo_npz(path: Union[str, Path]) -> EventArray:
    """Charge un fichier EVIMO `*.npz`.

    Contient une clé "events" (N,4) float32 [t_sec, x, y, p] et une clé
    "index" donnant, pour chaque frame, l'indice du premier event qui lui
    correspond dans le tableau "events".
    """
    data = np.load(path)
    events = data["events"].astype(np.float64)

    frame_ts = None
    if "index" in data.files:
        idx = np.clip(data["index"], 0, len(events) - 1)
        frame_ts = events[idx, 0]

    return EventArray(events=events, H=EVIMO_H, W=EVIMO_W, frame_ts=frame_ts)


def load_npy_canonical(
    path: Union[str, Path],
    H: int = EVIMO_H,
    W: int = EVIMO_W,
    frame_ts_path: Optional[Union[str, Path]] = None,
) -> EventArray:
    """Charge un fichier `evs.npy` "canonique" (N,4) [t_µs, x, y, p].

    Format produit par `EVOwithMS/app/evs_reader.py`. Les timestamps sont
    convertis de microsecondes en secondes pour rejoindre la représentation
    pivot `EventArray`.
    """
    raw = np.load(path).astype(np.float64)
    events = raw.copy()
    events[:, 0] = raw[:, 0] / 1e6

    frame_ts = None
    if frame_ts_path is not None:
        frame_ts = np.loadtxt(frame_ts_path, dtype=np.float64) / 1e6

    return EventArray(events=events, H=H, W=W, frame_ts=frame_ts)


_LOADERS = {
    "npz": load_evimo_npz,
    "npy": load_npy_canonical,
}


def load_evimo_mask(path: Union[str, Path]) -> FrameMasks:
    """Charge le masque de segmentation GT depuis un npz EVIMO.

    "ts" vient de meta["frames"][i]["ts"] — différent de frame_ts (issu de
    "index"), qui lui correspond aux frames de la caméra event.
    """
    data = np.load(path, allow_pickle=True)
    masks = data["mask"]
    ts = np.array([f["ts"] for f in data["meta"].item()["frames"]], dtype=np.float64)
    return FrameMasks(masks=masks, ts=ts)


def save_events_npz(ea: EventArray, path: Union[str, Path]) -> None:
    """Sauvegarde un EventArray au format npz compatible avec load_evimo_npz.

    Si ea.frame_ts est disponible, l'index par frame est recompté depuis
    les events filtrés (les indices ont bougé après filtrage).
    """
    path = Path(path)
    out: dict = {"events": ea.events.astype(np.float32)}
    if ea.frame_ts is not None and len(ea) > 0:
        idx = np.searchsorted(ea.t, ea.frame_ts, side="left")
        out["index"] = np.clip(idx, 0, len(ea) - 1).astype(np.int64)
    np.savez_compressed(path, **out)


def load_events(path: Union[str, Path], format: Optional[str] = None, **kwargs) -> EventArray:
    """Charge un fichier d'events et retourne un `EventArray`.

    Le format est déduit de l'extension du fichier (`format=` permet de le
    forcer). Formats supportés pour l'instant : "npz" (EVIMO), "npy"
    (canonique, cf. `load_npy_canonical`).
    """
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt not in _LOADERS:
        raise ValueError(
            f"Format d'events non supporté: '{fmt}' ({path}). "
            f"Formats disponibles: {sorted(_LOADERS)}"
        )

    return _LOADERS[fmt](path, **kwargs)
