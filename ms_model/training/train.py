import argparse
from pathlib import Path
from typing import Union

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from ms_model.data.evimo_dataset import EvimoSegDataset
from ms_model.data.evimo_slice_dataset import EvimoSliceDataset
from ms_model.models import build_model
from ms_model.training.losses import build_loss
from ms_model.training.evimo_losses import EvimoLoss


def build_dataset(paths: list[str], data_cfg: dict, root: Path) -> EvimoSegDataset:
    return EvimoSegDataset(
        npz_paths=[root / p for p in paths],
        seq_len=data_cfg["seq_len"],
        nb_time_bins=data_cfg["nb_time_bins"],
        patch_size=data_cfg["patch_size"],
        mask_thicken_radius=data_cfg.get("mask_thicken_radius", 0),
    )


def build_evimo_dataset(
    paths: list[str], data_cfg: dict, root: Path
) -> EvimoSliceDataset:
    return EvimoSliceDataset(
        npz_paths=[root / p for p in paths],
        slice_dt=data_cfg.get("slice_dt", 0.025),
        n_slices_context=data_cfg.get("n_slices_context", 5),
        max_objects=data_cfg.get("max_objects", 3),
    )


def run_epoch_evimo(
    model, loader, loss_fn: EvimoLoss, device, optimizer=None
) -> dict:
    """Training/validation epoch for the EV-IMO pipeline."""
    train_mode = optimizer is not None
    model.train(train_mode)

    totals = {"loss": 0.0, "warp": 0.0, "depth": 0.0, "mask": 0.0, "iou": 0.0}
    n_batches = 0

    for raw_batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in raw_batch.items()
        }

        with torch.set_grad_enabled(train_mode):
            output = model(batch)
            total_loss, components = loss_fn(output, batch)

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            # IoU: foreground = any object channel stronger than background
            masks = output["masks"]                                    # (B, C+1, H, W)
            fg_pred = (masks[:, 1:].sum(1) > masks[:, 0]).float()      # (B, H, W)
            fg_gt = (batch["mask_gt"] > 0).float()
            inter = (fg_pred * fg_gt).sum()
            union = (fg_pred + fg_gt).clamp(max=1).sum()
            totals["iou"] += (inter / union.clamp(min=1)).item()

        totals["loss"] += total_loss.item()
        totals["warp"] += components["warp"].item()
        totals["depth"] += components["depth"].item()
        totals["mask"] += components["mask"].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    for voxel_seq, mask_seq in loader:
        voxel_seq = voxel_seq.to(device)
        mask_seq = mask_seq.to(device).unsqueeze(2)  # (B,T,1,Hp,Wp)

        with torch.set_grad_enabled(train_mode):
            logits = model(voxel_seq)
            loss = loss_fn(logits, mask_seq)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            pred = (logits.detach() > 0).float()
            gt   = (mask_seq > 0.5).float()
            intersection = (pred * gt).sum()
            union        = (pred + gt).clamp(max=1).sum()
            total_iou   += (intersection / union.clamp(min=1)).item()

        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / n_batches, "iou": total_iou / n_batches}


def load_resume_checkpoint(checkpoint_dir: Path) -> Union[dict, None]:
    """Charge last.pt si present et au format de reprise (dict avec model/optimizer/epoch/...).
    Retourne None si rien a reprendre (pas de fichier, ou ancien format sans ces infos)."""
    last_ckpt_path = checkpoint_dir / "last.pt"
    if not last_ckpt_path.exists():
        return None

    ckpt = torch.load(last_ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "epoch" in ckpt and "optimizer" in ckpt):
        print(f"[resume] {last_ckpt_path} existe mais dans un ancien format (pas de reprise possible).")
        return None
    return ckpt


def confirm_overwrite_dir(checkpoint_dir: Path) -> None:
    """Si le dossier de checkpoints existe deja (et n'est pas repris), demande confirmation."""
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        try:
            answer = input(f"Le dossier {checkpoint_dir} existe deja et contient des fichiers "
                            f"(non repris). Continuer et ecraser ? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes", "o", "oui"):
            raise SystemExit("Run annule.")


def main(config_path: str, data_root_override: str = None, wandb_name_override: str = None):
    config = yaml.safe_load(Path(config_path).read_text())
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]
    data_cfg = config["data"]

    if data_root_override:
        data_cfg["root"] = data_root_override
        print(f"[data] root override: {data_root_override}", flush=True)

    fixed_name = wandb_name_override or wandb_cfg.get("name") or None
    checkpoint_base = Path(train_cfg.get("checkpoint_dir", "checkpoints"))

    resume_ckpt = None
    if fixed_name:
        checkpoint_dir = checkpoint_base / fixed_name
        resume_ckpt = load_resume_checkpoint(checkpoint_dir)
        if resume_ckpt is not None:
            print(f"[resume] reprise du run '{fixed_name}' a partir de l'epoch "
                  f"{resume_ckpt['epoch'] + 1} (best_val_loss={resume_ckpt['best_val_loss']:.4f})", flush=True)
        else:
            confirm_overwrite_dir(checkpoint_dir)

    root = Path(data_cfg.get("root", "."))

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    is_evimo = model_name == "evimo"

    print("== Chargement du dataset d'entrainement ==", flush=True)
    if is_evimo:
        train_set = build_evimo_dataset(data_cfg["train_sequences"], data_cfg, root)
    else:
        train_set = build_dataset(data_cfg["train_sequences"], data_cfg, root)
    print("== Chargement du dataset de validation ==", flush=True)
    if is_evimo:
        val_set = build_evimo_dataset(data_cfg["val_sequences"], data_cfg, root)
    else:
        val_set = build_dataset(data_cfg["val_sequences"], data_cfg, root)

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True,
                               num_workers=train_cfg.get("num_workers", 0))
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg.get("num_workers", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_evimo:
        model = build_model(model_name, **model_cfg).to(device)
        loss_fn = EvimoLoss(**train_cfg.get("evimo_loss_kwargs", {}))
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    else:
        model = build_model(
            model_name,
            in_channels=data_cfg["nb_time_bins"],
            patch_size=data_cfg["patch_size"],
            **model_cfg,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
        scheduler = None
        loss_fn = build_loss(
            train_cfg.get("loss", "bce"),
            pos_weight=train_cfg.get("loss_pos_weight"),
            focal_gamma=train_cfg.get("loss_focal_gamma", 2.0),
            bce_weight=train_cfg.get("loss_bce_weight", 0.5),
        ).to(device)

    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_val_loss"]
        wandb_run_id = resume_ckpt.get("wandb_run_id")

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        name=fixed_name,
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
        config=config,
    )

    checkpoint_dir = checkpoint_base / run.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if start_epoch >= train_cfg["epochs"]:
        print(f"[resume] entrainement deja termine ({start_epoch}/{train_cfg['epochs']} epochs), rien a faire.")
        wandb.finish()
        return

    for epoch in range(start_epoch, train_cfg["epochs"]):
        if is_evimo:
            train_metrics = run_epoch_evimo(model, train_loader, loss_fn, device, optimizer)
            val_metrics   = run_epoch_evimo(model, val_loader, loss_fn, device)
            if scheduler is not None:
                scheduler.step()
        else:
            train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_metrics   = run_epoch(model, val_loader, loss_fn, device)

        log_dict = {
            "epoch":      epoch,
            "train/loss": train_metrics["loss"],
            "train/iou":  train_metrics["iou"],
            "val/loss":   val_metrics["loss"],
            "val/iou":    val_metrics["iou"],
        }
        if is_evimo:
            log_dict.update({
                "train/warp":  train_metrics["warp"],
                "train/depth": train_metrics["depth"],
                "train/mask":  train_metrics["mask"],
                "val/warp":    val_metrics["warp"],
                "val/depth":   val_metrics["depth"],
                "val/mask":    val_metrics["mask"],
            })
        wandb.log(log_dict)
        if is_evimo:
            print(
                f"epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f} (warp={train_metrics['warp']:.4f} "
                f"depth={train_metrics['depth']:.4f} mask={train_metrics['mask']:.4f}) "
                f"train_iou={train_metrics['iou']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f}",
                flush=True,
            )
        else:
            print(f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['iou']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f}", flush=True)

        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": config,
            }, checkpoint_dir / "best.pt")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "wandb_run_id": run.id,
            "config": config,
        }, checkpoint_dir / "last.pt")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None,
                        help="Ecrase data.root du yaml (utile sur cluster)")
    parser.add_argument("--wandb-name", default=None,
                        help="Ecrase wandb.name du yaml")
    args = parser.parse_args()
    main(args.config, data_root_override=args.data_root, wandb_name_override=args.wandb_name)
