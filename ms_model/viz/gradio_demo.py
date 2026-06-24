"""Démo Gradio — visualisation des prédictions de masques sur des enregistrements EVIMO.

Usage :
    python -m ms_model.viz.gradio_demo \
        --data-root ../datasets/EV-IMO/eval \
        --ckpt-root checkpoints \
        --port 7860
"""

import argparse
from pathlib import Path

import numpy as np

from ms_model.inference import Predictor
from ms_model.viz.mask import render_mask_overlay


def _find_files(root: str, pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(root).rglob(pattern))


def _show_frame(results: dict | None, frame_idx: int):
    if results is None:
        return None, None, None
    i = int(frame_idx)
    event_frame = results["event_frames"][i]
    gt_overlay   = render_mask_overlay(event_frame, results["gt_masks"][i].astype(np.uint8),   color=(0, 255, 0))
    pred_overlay = render_mask_overlay(event_frame, results["pred_masks"][i].astype(np.uint8), color=(255, 120, 0))
    return event_frame, gt_overlay, pred_overlay


def build_demo(data_root: str, ckpt_root: str, config_root: str = "configs"):
    import gradio as gr

    npz_files  = _find_files(data_root, "*.npz")
    ckpt_files = _find_files(ckpt_root, "best.pt")
    cfg_files  = _find_files(config_root, "*.yaml")

    with gr.Blocks(title="MS-Model — Motion Segmentation") as demo:
        gr.Markdown("## Motion Segmentation — visualisation des prédictions")

        state = gr.State(None)

        with gr.Row():
            dd_seq  = gr.Dropdown(npz_files,  label="Séquence (.npz)",  value=npz_files[0]  if npz_files  else None)
            dd_ckpt = gr.Dropdown(ckpt_files, label="Poids (best.pt)",  value=ckpt_files[0] if ckpt_files else None)
            dd_cfg  = gr.Dropdown(cfg_files,  label="Config (.yaml)",   value=cfg_files[0]  if cfg_files  else None)

        btn    = gr.Button("Lancer l'inférence", variant="primary")
        status = gr.Textbox(label="Statut", interactive=False)
        slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Frame", interactive=True)

        with gr.Row():
            img_ev   = gr.Image(label="Events",      type="numpy")
            img_gt   = gr.Image(label="Masque GT",   type="numpy")
            img_pred = gr.Image(label="Prédiction",  type="numpy")

        def run_inference(npz_path, weights_path, config_path):
            if not all([npz_path, weights_path, config_path]):
                return None, "Sélectionne une séquence, un checkpoint et une config.", gr.update()
            try:
                results = Predictor(weights_path, config_path).predict_sequence(npz_path)
                n = results["n_frames"]
                return results, f"{n} frames chargées.", gr.update(maximum=n - 1, value=0)
            except Exception as e:
                return None, f"Erreur : {e}", gr.update()

        btn.click(
            run_inference,
            inputs=[dd_seq, dd_ckpt, dd_cfg],
            outputs=[state, status, slider],
        ).then(
            _show_frame,
            inputs=[state, slider],
            outputs=[img_ev, img_gt, img_pred],
        )

        slider.change(
            _show_frame,
            inputs=[state, slider],
            outputs=[img_ev, img_gt, img_pred],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",   default="../datasets/EV-IMO/eval")
    parser.add_argument("--ckpt-root",   default="checkpoints")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--port",        type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(args.data_root, args.ckpt_root, args.config_root)
    demo.launch(server_port=args.port)
