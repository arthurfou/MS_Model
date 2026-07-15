#!/usr/bin/env python3
"""Filter events from a npz file using a trained motion segmentation model.

Usage:
    python scripts/filter_events.py \\
        --weights checkpoints/convlstm-v4-unet/best_iou.pt \\
        --config  configs/convlstm_v4_unet.yaml \\
        --input   path/to/seq.npz \\
        --output  path/to/seq_filtered.npz
"""
import argparse
from pathlib import Path

from ms_model.inference import Predictor
from ms_model.io.loaders import save_events_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Chemin vers best.pt ou last.pt")
    parser.add_argument("--config",  required=True, help="Chemin vers le yaml d'entraînement")
    parser.add_argument("--input",   required=True, help="Fichier .npz d'events en entrée")
    parser.add_argument("--output",  required=True, help="Fichier .npz filtré en sortie")
    parser.add_argument("--device",  default=None,  help="cuda / cpu (auto-détect si omis)")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Seuil de sigmoïde pour binariser le masque dynamique (défaut: 0.5)")
    args = parser.parse_args()

    predictor = Predictor(args.weights, args.config, device=args.device, threshold=args.threshold)
    ea_filtered = predictor.filter_events(args.input)
    save_events_npz(ea_filtered, args.output)
    print(f"Sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
