"""Exporte des triplets d'images (events / GT / prédiction) depuis EVIMO1 ou EVIMO2.

Pour chaque frame échantillonnée :
  {k}_a_events.png : events rendus rouge (ON) / bleu (OFF) sur fond noir
  {k}_b_gt.png     : la même image, objets **en mouvement** surlignés en jaune (GT)
  {k}_c_pred.png   : idem avec le masque prédit par un checkpoint (si --ckpt)

Sans --ckpt, seules les deux premières images sont écrites.

⚠️ `dataset_mask.npz` n'est PAS un masque de mouvement : il segmente *tous* les objets
annotés VICON, y compris le décor statique (tapis à motifs, plateaux, boîtes posées).
Sur `scene14_dyn_test_03`, 78 % des pixels sont couverts par deux props immobiles.

Les objets réellement mobiles sont donc identifiés géométriquement (`moving_object_ids`) :
les poses objet de `meta["frames"][i][obj_id]` sont exprimées **dans le repère caméra**
(vérifié : projeter t par K tombe sur le centroïde du masque). On les ramène dans le
repère monde via la pose caméra, et un objet est déclaré mobile si sa position monde
s'écarte de plus de `--motion-thresh` de sa médiane.

Usage:
    python scripts/export_gt_pairs.py \\
        --seq ../datasets/evimo2/samsung_mono/imo/eval/scene14_dyn_test_03_000000 \\
        --out ../datasets/evimo2/_pairs \\
        --ckpt checkpoints/evimo2-run1/best.pt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.models import build_model
from ms_model.representations.voxel import make_voxel_sequence
from ms_model.viz.render import render_event_frame

YELLOW = (255, 255, 0)   # RGB
NON_OBJ_KEYS = {"cam", "gt_frame", "classical_frame", "id", "ts"}


def _meta_frames(seq: Path) -> list:
    """Frames de la meta, que la séquence soit un répertoire EVIMO2 ou un .npz EVIMO1."""
    if seq.is_dir():
        return np.load(seq / "dataset_info.npz", allow_pickle=True)["meta"].item()["frames"]
    with np.load(seq, allow_pickle=True) as data:
        return data["meta"].item()["frames"]


def _out_name(seq: Path) -> str:
    """Nom de sous-dossier unique : les .npz EVIMO1 s'appellent tous `seq_00`."""
    if seq.is_dir():
        return seq.name
    return f"{seq.parent.parent.name}_{seq.stem}"


def _t(pos) -> np.ndarray:
    return np.array([pos["t"]["x"], pos["t"]["y"], pos["t"]["z"]], dtype=np.float64)


def _rotate(pos, v: np.ndarray) -> np.ndarray:
    """Applique la rotation du quaternion de `pos` au vecteur v (formule de Rodrigues)."""
    q = pos["q"]
    u = np.array([q["x"], q["y"], q["z"]], dtype=np.float64)
    w = float(q["w"])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def moving_object_ids(seq: Path, thresh: float = 0.02, verbose: bool = False) -> set[int]:
    """Ids d'objets (valeur du masque = id × 1000) réellement mobiles dans le repère monde."""
    frames = [f for f in _meta_frames(seq) if "cam" in f]
    obj_keys = sorted({k for f in frames for k in f} - NON_OBJ_KEYS)

    moving = set()
    for oid in obj_keys:
        world = np.array([
            _rotate(f["cam"]["pos"], _t(f[oid]["pos"])) + _t(f["cam"]["pos"])
            for f in frames if oid in f
        ])
        if len(world) < 2:
            continue
        spread = np.linalg.norm(world - np.median(world, axis=0), axis=1).max()
        if verbose:
            print(f"  obj {oid:>3} : écart max au repos = {spread * 100:6.2f} cm"
                  f"  → {'MOBILE' if spread > thresh else 'statique'}")
        if spread > thresh:
            moving.add(int(oid))
    return moving


def predict_masks(ea, ts, ckpt_path: Path, yaml_path: Path | None = None,
                  threshold: float = 0.5) -> np.ndarray:
    """Masques prédits (N, H, W) uint8 pour les N = len(ts) - 1 intervalles.

    Inférence par chunks de `seq_len` frames, comme à l'entraînement : une séquence
    EVIMO2 entière d'un bloc ferait ~2 Go de voxels.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
    elif yaml_path:
        config = yaml.safe_load(Path(yaml_path).read_text())
    else:
        raise SystemExit(
            f"{ckpt_path} n'embarque pas de config — passe --yaml <config.yaml>."
        )

    data_cfg, model_cfg = config["data"], dict(config["model"])
    model_name = model_cfg.pop("name")
    nb_bins = data_cfg.get("nb_time_bins", 5)
    seq_len = int(data_cfg.get("seq_len", 16))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, in_channels=nb_bins,
                        patch_size=data_cfg.get("patch_size", 4), **model_cfg)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval().to(device)

    out = []
    with torch.no_grad():
        for start in range(0, len(ts) - 1, seq_len):
            end = min(start + seq_len, len(ts) - 1)
            voxels = make_voxel_sequence(ea, ts[start:end + 1], nb_of_time_bins=nb_bins)
            probs = torch.sigmoid(model(voxels.unsqueeze(0).to(device))[0, :, 0])
            full = F.interpolate(probs.unsqueeze(1).float().cpu(),
                                 size=(ea.H, ea.W), mode="nearest")
            out.append((full[:, 0] > threshold).numpy().astype(np.uint8))
    return np.concatenate(out, axis=0)


def overlay(events: np.ndarray, region: np.ndarray, tint: float = 0.35) -> np.ndarray:
    """Surligne `region` en jaune sur une image d'events déjà rendue."""
    img = events.copy()
    img[region] = (1 - tint) * img[region] + tint * np.array(YELLOW)
    img[events.any(axis=2) & region] = YELLOW
    return img


def make_montage(panels: list[tuple[str, np.ndarray]], pad: int = 8) -> np.ndarray:
    """Assemble les volets côte à côte, chacun titré, séparés par une bande grise."""
    H, W = panels[0][1].shape[:2]
    band = 28
    out = np.full((H + band, len(panels) * W + (len(panels) - 1) * pad, 3), 40, np.uint8)
    for j, (label, img) in enumerate(panels):
        x = j * (W + pad)
        out[band:, x:x + W] = img
        cv2.putText(out, label, (x + 6, band - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_pair(ea, mask, moving, t_lo, t_hi, tint=0.35):
    """(events RGB, events + objets mobiles en jaune RGB) pour l'intervalle [t_lo, t_hi)."""
    events = render_event_frame(ea, t_lo, t_hi)

    region = np.isin(mask, [oid * 1000 for oid in moving]) if moving \
        else np.zeros(mask.shape, dtype=bool)

    return events, overlay(events, region, tint)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True,
                    help="séquence : répertoire EVIMO2 ou fichier .npz EVIMO1")
    ap.add_argument("--out", required=True, help="répertoire de sortie")
    ap.add_argument("--num-pairs", type=int, default=10,
                    help="nombre de frames échantillonnées (défaut: 10)")
    ap.add_argument("--ckpt", help="checkpoint .pt du modèle MS — ajoute le volet prédiction")
    ap.add_argument("--yaml", help="config .yaml (si le checkpoint n'embarque pas la sienne)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="seuil de binarisation des probas prédites (défaut: 0.5)")
    ap.add_argument("--motion-thresh", type=float, default=0.02,
                    help="seuil de mobilité en mètres (défaut: 0.02)")
    ap.add_argument("--all-objects", action="store_true",
                    help="surligner tous les objets annotés, mobiles ou non (masque brut)")
    ap.add_argument("--montage", action="store_true",
                    help="écrire aussi une image unique avec les volets côte à côte")
    ap.add_argument("--tint", type=float, default=0.35,
                    help="opacité de la teinte jaune sur la silhouette (défaut: 0.35)")
    ap.add_argument("--render-ms", type=float, default=None,
                    help="durée d'accumulation des events affichés, en ms, à partir de "
                         "l'instant de la frame GT (défaut: tout l'intervalle inter-frame, "
                         "~30 ms sur EVIMO1 — d'où le flou de mouvement)")
    args = ap.parse_args()

    seq = Path(args.seq)
    out = Path(args.out) / _out_name(seq)
    out.mkdir(parents=True, exist_ok=True)

    ea = load_events(seq)
    fm = load_evimo_mask(seq)

    if args.all_objects:
        moving = {int(v) // 1000 for v in np.unique(fm.masks) if v != 0}
        print(f"{_out_name(seq)} : tous les objets annotés → {sorted(moving)}")
    else:
        print(f"{_out_name(seq)} : classification statique/mobile")
        moving = moving_object_ids(seq, args.motion_thresh, verbose=True)
        print(f"  objets mobiles retenus : {sorted(moving) or 'aucun'}")

    n_frames = len(fm.ts) - 1
    if n_frames < 1:
        raise SystemExit(f"{seq} : pas assez de frames GT ({len(fm.ts)})")
    n = min(args.num_pairs, n_frames)
    # écarté des bords : les toutes premières/dernières frames sont souvent creuses
    idx = np.linspace(0.05 * n_frames, 0.95 * n_frames - 1, n).round().astype(int)

    pred = None
    if args.ckpt:
        print(f"\ninférence {Path(args.ckpt).name} sur {n_frames} frames...")
        pred = predict_masks(ea, fm.ts, Path(args.ckpt),
                             Path(args.yaml) if args.yaml else None, args.threshold)

    print()
    for k, i in enumerate(idx):
        t_lo, t_hi = fm.ts[i], fm.ts[i + 1]
        if args.render_ms is not None:
            t_hi = min(t_hi, t_lo + args.render_ms * 1e-3)
        events, gt = make_pair(ea, fm.masks[i], moving, t_lo, t_hi, tint=args.tint)
        images = [("a_events", events), ("b_gt", gt)]
        if pred is not None:
            images.append(("c_pred", overlay(events, pred[i] != 0, args.tint)))
        for suffix, img in images:
            cv2.imwrite(str(out / f"{k:02d}_{suffix}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if args.montage:
            labels = {"a_events": "events", "b_gt": "GT (objets mobiles)", "c_pred": "prediction"}
            mont = make_montage([(labels[sfx], img) for sfx, img in images])
            cv2.imwrite(str(out / f"{k:02d}_montage.png"), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))

        n_ev = int(((ea.t >= t_lo) & (ea.t < t_hi)).sum())
        gt_region = np.isin(fm.masks[i], [o * 1000 for o in moving]) if moving \
            else np.zeros(fm.masks[i].shape, dtype=bool)
        line = (f"[{k:02d}] frame {i:4d}  t={t_lo:7.3f}s  {1000 * (t_hi - t_lo):5.1f}ms  "
                f"{n_ev:6d} events  "
                f"GT {100 * gt_region.mean():5.1f}%")
        if pred is not None:
            pr = pred[i] != 0
            union = (gt_region | pr).sum()
            iou = (gt_region & pr).sum() / union if union else float("nan")
            line += f"  pred {100 * pr.mean():5.1f}%  IoU {iou:.3f}"
        print(line)

    per_frame = (3 if pred is not None else 2) + (1 if args.montage else 0)
    print(f"\n{len(idx) * per_frame} images écrites dans {out}")


if __name__ == "__main__":
    main()
