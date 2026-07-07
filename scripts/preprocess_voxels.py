"""Prétraitement offline des voxel grids EVIMO.

Calcule et sauvegarde les tenseurs voxelisés (.voxels_bins<N>.npy) à côté de
chaque .npz, sans lancer d'entraînement. À exécuter une seule fois sur un
nœud CPU avant de soumettre les jobs GPU.

Le format .npy permet un chargement lazy via numpy memmap (pas d'OOM même sur
de grands datasets). Les anciens caches .pt sont migrés automatiquement.

Usage :
    python scripts/preprocess_voxels.py --data-root ../datasets/evimo_full
    python scripts/preprocess_voxels.py --data-root ../datasets/evimo_full --nb-time-bins 5
"""

import argparse
import time
from pathlib import Path

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.representations.voxel import make_voxel_sequence

import numpy as np

try:
    import torch  # noqa: F401 — vérifie la présence de torch pour make_voxel_sequence
except ImportError:
    raise SystemExit("torch n'est pas installé dans l'environnement courant.")


def preprocess(data_root: str, nb_time_bins: int, force: bool) -> None:
    root = Path(data_root)
    npz_files = sorted(root.rglob("*.npz"))

    if not npz_files:
        print(f"Aucun fichier .npz trouvé dans {root}")
        return

    print(f"{len(npz_files)} fichiers .npz trouvés dans {root}\n")

    for i, npz_path in enumerate(npz_files, 1):
        cache_path = npz_path.with_suffix(f".voxels_bins{nb_time_bins}.npy")
        old_pt     = npz_path.with_suffix(f".voxels_bins{nb_time_bins}.pt")

        if cache_path.exists() and not force:
            print(f"[{i}/{len(npz_files)}] {npz_path.name} — cache déjà présent, ignoré")
            continue

        t0 = time.time()
        print(f"[{i}/{len(npz_files)}] {npz_path.name} ...", end=" ", flush=True)

        if old_pt.exists() and not force:
            import torch as _torch
            voxel_seq = _torch.load(old_pt)
            arr = voxel_seq.numpy()
        else:
            ea = load_events(npz_path)
            fm = load_evimo_mask(npz_path)
            import torch as _torch
            voxel_seq = make_voxel_sequence(ea, fm.ts, nb_of_time_bins=nb_time_bins)
            arr = voxel_seq.numpy()

        np.save(cache_path, arr)
        print(f"{arr.shape[0]} frames, {time.time() - t0:.1f}s")

    print("\nPrétraitement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",     required=True, help="Racine du dataset EVIMO")
    parser.add_argument("--nb-time-bins",  type=int, default=5)
    parser.add_argument("--force",         action="store_true", help="Recalculer même si le cache existe")
    args = parser.parse_args()

    preprocess(args.data_root, args.nb_time_bins, args.force)
