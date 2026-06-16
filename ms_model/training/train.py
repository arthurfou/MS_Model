import argparse
from pathlib import Path
from typing import Union

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from ms_model.data.evimo_dataset import EvimoSegDataset
from ms_model.models import build_model
from ms_model.training.losses import build_loss


def build_dataset(paths: list[str], data_cfg: dict, root: Path) -> EvimoSegDataset:
    return EvimoSegDataset(
        npz_paths=[root / p for p in paths],
        seq_len=data_cfg["seq_len"],
        nb_time_bins=data_cfg["nb_time_bins"],
        patch_size=data_cfg["patch_size"],
        mask_thicken_radius=data_cfg.get("mask_thicken_radius", 0),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_batches = 0.0, 0
    for voxel_seq, mask_seq in loader:
        voxel_seq = voxel_seq.to(device)
        mask_seq = mask_seq.to(device).unsqueeze(2)  # (B,T,1,Hp,Wp), match model output

        with torch.set_grad_enabled(train_mode):
            logits = model(voxel_seq)
            loss = loss_fn(logits, mask_seq)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


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


def main(config_path: str, data_root_override: str = None):
    config = yaml.safe_load(Path(config_path).read_text())
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]
    data_cfg = config["data"]

    if data_root_override:
        data_cfg["root"] = data_root_override
        print(f"[data] root override: {data_root_override}", flush=True)

    fixed_name = wandb_cfg.get("name") or None
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

    print("== Chargement du dataset d'entrainement ==", flush=True)
    train_set = build_dataset(data_cfg["train_sequences"], data_cfg, root)
    print("== Chargement du dataset de validation ==", flush=True)
    val_set = build_dataset(data_cfg["val_sequences"], data_cfg, root)

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True,
                               num_workers=train_cfg.get("num_workers", 0))
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg.get("num_workers", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = build_model(
        model_name,
        in_channels=data_cfg["nb_time_bins"],
        patch_size=data_cfg["patch_size"],
        **model_cfg,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    loss_fn = build_loss(train_cfg.get("loss", "bce"))

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
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)

        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "wandb_run_id": run.id,
        }, checkpoint_dir / "last.pt")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None,
                        help="Ecrase data.root du yaml (utile sur cluster)")
    args = parser.parse_args()
    main(args.config, data_root_override=args.data_root)
