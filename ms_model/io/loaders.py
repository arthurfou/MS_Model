from pathlib import Path
from typing import Optional, Union

import numpy as np

from .events import EventArray
from .masks import FrameMasks

# Résolution DVS346 utilisée par EVIMO1 (cf. DEVO/scripts/preprocess_evimo.py)
EVIMO_H, EVIMO_W = 260, 346

# Résolution samsung_mono EVIMO2
EVIMO2_H, EVIMO2_W = 480, 640


def _is_evimo2_dir(path: Path) -> bool:
    """True si path est un répertoire au format EVIMO2 (dataset_info.npz présent)."""
    return path.is_dir() and (path / "dataset_info.npz").exists()


def _load_evimo2_events(path: Path) -> "EventArray":
    """Charge les events depuis un répertoire EVIMO2 (3 fichiers séparés)."""
    ev_t  = np.load(path / "dataset_events_t.npy").astype(np.float64)   # (N,) sec
    ev_xy = np.load(path / "dataset_events_xy.npy").astype(np.float64)  # (N, 2)
    ev_p  = np.load(path / "dataset_events_p.npy").astype(np.float64)   # (N,)
    fi    = np.load(path / "dataset_info.npz", allow_pickle=True)
    idx   = fi["index"].astype(np.int64)                                 # (N_frames,)
    events = np.stack([ev_t, ev_xy[:, 0], ev_xy[:, 1], ev_p], axis=1)
    frame_ts = ev_t[np.clip(idx, 0, len(ev_t) - 1)]
    return EventArray(events=events, H=EVIMO2_H, W=EVIMO2_W, frame_ts=frame_ts)


def _load_evimo2_mask(path: Path) -> "FrameMasks":
    """Charge les masques GT depuis un répertoire EVIMO2 (dataset_mask.npz par frame)."""
    fm_file = np.load(path / "dataset_mask.npz", allow_pickle=True)
    mask_keys = sorted(fm_file.keys())
    masks = np.stack([fm_file[k] for k in mask_keys])   # (N_frames, H, W)
    fi = np.load(path / "dataset_info.npz", allow_pickle=True)
    meta = fi["meta"].item()
    ts = np.array([f["ts"] for f in meta["frames"]], dtype=np.float64)
    N = min(len(mask_keys), len(ts))
    return FrameMasks(masks=masks[:N], ts=ts[:N])


def load_evimo_npz(path: Union[str, Path]) -> "EventArray":
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
    """Charge le masque de segmentation GT — supporte EVIMO1 (.npz) et EVIMO2 (répertoire).

    "ts" vient de meta["frames"][i]["ts"] — différent de frame_ts (issu de
    "index"), qui lui correspond aux frames de la caméra event.
    """
    path = Path(path)
    if _is_evimo2_dir(path):
        return _load_evimo2_mask(path)
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

    Supporte EVIMO1 (.npz ou .npy) et EVIMO2 (répertoire avec dataset_events_*.npy).
    Le format est déduit de l'extension ou du type de chemin.
    """
    path = Path(path)
    if _is_evimo2_dir(path):
        return _load_evimo2_events(path)

    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt not in _LOADERS:
        raise ValueError(
            f"Format d'events non supporté: '{fmt}' ({path}). "
            f"Formats disponibles: {sorted(_LOADERS)}"
        )

    return _LOADERS[fmt](path, **kwargs)


# ---------------------------------------------------------------------------
# Loaders spécifiques aux datasets DEVO (hors EVIMO)
# ---------------------------------------------------------------------------

def load_events_rpg(evs_txt_path: Union[str, Path], H: int = 180, W: int = 240) -> EventArray:
    """Charge les events RPG depuis `evs_{side}.txt` [t_µs, x, y, p] → EventArray (t en secondes)."""
    evs = np.loadtxt(evs_txt_path, delimiter=" ").astype(np.float64)
    evs[:, 0] /= 1e6  # µs → secondes
    return EventArray(events=evs, H=H, W=W)


def load_events_fpv(scenedir: Union[str, Path], H: int = 260, W: int = 346) -> EventArray:
    """Charge les events FPV depuis `events.txt` [t_sec, x, y, p] avec offset → EventArray (t en secondes)."""
    scenedir = Path(scenedir)
    evs = np.loadtxt(str(scenedir / "events.txt"), delimiter=" ").astype(np.float64)
    # t déjà en secondes ; on soustrait l'offset converti en secondes
    t_offset_path = scenedir / "t_offset_us.txt"
    if t_offset_path.exists():
        t_offset_s = float(np.loadtxt(str(t_offset_path))) / 1e6
        evs[:, 0] -= t_offset_s
    return EventArray(events=evs, H=H, W=W)


def load_events_prophesee_h5(h5_path: Union[str, Path], H: int = 260, W: int = 346) -> EventArray:
    """Charge les events depuis un fichier H5 au format Prophesee/EventSlicer → EventArray (t en secondes).

    Couvre HKU (`evs_{side}.h5`), EDS (`events.h5`), VECtor (`*.hdf5`), TUM-VIE (`*events_{side}.h5`).
    Les timestamps sont en µs relatifs (+ t_offset absolu si présent).
    """
    import h5py
    with h5py.File(str(h5_path), 'r') as f:
        prefix = 'events/' if 'events/x' in f else ''
        t = f[f'{prefix}t'][:].astype(np.float64)
        x = f[f'{prefix}x'][:].astype(np.float64)
        y = f[f'{prefix}y'][:].astype(np.float64)
        p = f[f'{prefix}p'][:].astype(np.float64)
        t_offset = int(f['t_offset'][()]) if 't_offset' in f else 0
    t_abs_s = (t + t_offset) / 1e6  # µs → secondes
    events = np.stack([t_abs_s, x, y, p], axis=1)
    return EventArray(events=events, H=H, W=W)


def load_events_mvsec(h5_path: Union[str, Path], side: str = "left", H: int = 260, W: int = 346) -> EventArray:
    """Charge les events MVSEC depuis `*_data.hdf5` → EventArray (t en secondes).

    Format HDF5 MVSEC : `davis/{side}/events` [x, y, t_sec, p].
    Les timestamps y sont en secondes (cf. `pp_mvsec.py` : `image_raw_ts * 1e6`).
    """
    import h5py
    with h5py.File(str(h5_path), 'r') as f:
        all_evs = f["davis"][side]["events"][:]  # [x, y, t_sec, p]
    events = np.zeros((len(all_evs), 4), dtype=np.float64)
    events[:, 0] = all_evs[:, 2]   # t (secondes)
    events[:, 1] = all_evs[:, 0]   # x
    events[:, 2] = all_evs[:, 1]   # y
    events[:, 3] = all_evs[:, 3]   # p
    return EventArray(events=events, H=H, W=W)
