#!/usr/bin/env python3
"""Balaie le seuil de sigmoïde et rapporte précision/rappel/F1/IoU sur le split d'éval.

Reconstruit exactement le split `data.val_sequences` défini dans le yaml
d'entraînement (le split "eval EVIMO", utilisé comme val/test par
ms_model/training/train.py), fait tourner le modèle dessus, et pour une grille
de seuils compare la prédiction (au niveau patch, résolution native du modèle)
au masque GT downsamplé — mêmes conventions que `run_epoch` dans train.py.

Usage:
    python scripts/sweep_threshold.py \\
        --weights checkpoints/convlstm-v4b-bidir/best_iou.pt \\
        --config  configs/convlstm_v4b_full.yaml

Nécessite matplotlib pour le plot PR/F1/IoU (`pip install -e .[eval]`) ; le
CSV et la table stdout sont produits même sans matplotlib installé.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ms_model.inference import load_model_from_checkpoint
from ms_model.training.train import build_dataset


def parse_thresholds(spec: str) -> np.ndarray:
    """Parse 'start:stop:n' -> np.linspace(start, stop, n)."""
    start, stop, n = spec.split(":")
    return np.linspace(float(start), float(stop), int(n))


def sweep(model, loader, thresholds: np.ndarray, device: torch.device) -> dict:
    """Accumule TP/FP/FN par seuil sur tout le loader, sans stocker les logits."""
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)

    model.eval()
    with torch.no_grad():
        for voxel_seq, mask_seq in loader:
            voxel_seq = voxel_seq.to(device)
            mask_seq = mask_seq.to(device).unsqueeze(2)  # (B,T,1,Hp,Wp)

            logits = model(voxel_seq)
            probs = torch.sigmoid(logits).flatten()
            gt = (mask_seq > 0.5).flatten()

            for k, t in enumerate(thresholds):
                pred = probs > t
                tp[k] += (pred & gt).sum().item()
                fp[k] += (pred & ~gt).sum().item()
                fn[k] += (~pred & gt).sum().item()

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), np.nan)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)

    return {
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def print_table(results: dict) -> None:
    thresholds = results["threshold"]
    best_f1_idx = int(np.nanargmax(results["f1"]))
    best_iou_idx = int(np.nanargmax(results["iou"]))

    print(f"{'seuil':>6} {'précision':>10} {'rappel':>8} {'F1':>8} {'IoU':>8}")
    for i in range(len(thresholds)):
        marker = ""
        if i == best_f1_idx:
            marker += " <- best F1"
        if i == best_iou_idx:
            marker += " <- best IoU"
        print(
            f"{thresholds[i]:6.3f} {results['precision'][i]:10.4f} "
            f"{results['recall'][i]:8.4f} {results['f1'][i]:8.4f} {results['iou'][i]:8.4f}{marker}"
        )

    print(
        f"\nMeilleur seuil (F1)  : {thresholds[best_f1_idx]:.3f} "
        f"(F1={results['f1'][best_f1_idx]:.4f})"
    )
    print(
        f"Meilleur seuil (IoU) : {thresholds[best_iou_idx]:.3f} "
        f"(IoU={results['iou'][best_iou_idx]:.4f})"
    )


def write_csv(results: dict, output_csv: Path) -> None:
    thresholds = results["threshold"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall", "f1", "iou"])
        for i in range(len(thresholds)):
            writer.writerow([
                thresholds[i],
                results["precision"][i],
                results["recall"][i],
                results["f1"][i],
                results["iou"][i],
            ])
    print(f"CSV sauvegardé : {output_csv}")


def write_plot(results: dict, output_plot: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # pas d'affichage interactif requis, juste un PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib non installé — plot ignoré (le CSV/table restent disponibles). "
            "Installer avec: pip install -e .[eval]"
        )
        return

    thresholds = results["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(results["recall"], results["precision"], "o-")
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
        ax.annotate(f"{thresholds[i]:.2f}", (results["recall"][i], results["precision"][i]), fontsize=7)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe PR")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(thresholds, results["f1"], "o-", label="F1")
    ax.plot(thresholds, results["iou"], "s-", label="IoU")
    ax.set_xlabel("Seuil")
    ax.set_title("F1 / IoU vs seuil")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    print(f"Plot sauvegardé : {output_plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Chemin vers best.pt, best_iou.pt ou last.pt")
    parser.add_argument("--config", required=True, help="Chemin vers le yaml d'entraînement")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-détect si omis)")
    parser.add_argument("--thresholds", default="0.05:0.95:19",
                         help="Grille de seuils 'start:stop:n' (défaut: 0.05:0.95:19)")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Défaut: training.batch_size du yaml")
    parser.add_argument("--num-workers", type=int, default=None,
                         help="Défaut: training.num_workers du yaml")
    parser.add_argument("--limit-sequences", type=int, default=None,
                         help="Limiter aux N premières séquences de val_sequences (pour un test rapide)")
    parser.add_argument("--output-csv", default="threshold_sweep.csv")
    parser.add_argument("--output-plot", default="threshold_sweep.png")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, data_cfg = load_model_from_checkpoint(args.weights, args.config, device)

    full_cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = full_cfg["training"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)

    root = Path(data_cfg.get("root", "."))
    val_sequences = data_cfg["val_sequences"]
    if args.limit_sequences is not None:
        val_sequences = val_sequences[: args.limit_sequences]

    print(f"== Chargement du split de validation ({len(val_sequences)} séquences) ==", flush=True)
    val_set = build_dataset(val_sequences, data_cfg, root)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    thresholds = parse_thresholds(args.thresholds)
    print(f"== Balayage de {len(thresholds)} seuils sur {len(val_set)} exemples ==", flush=True)
    results = sweep(model, val_loader, thresholds, device)

    print()
    print_table(results)
    write_csv(results, Path(args.output_csv))
    write_plot(results, Path(args.output_plot))


if __name__ == "__main__":
    main()
