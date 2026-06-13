import argparse
from pathlib import Path

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


def main(config_path: str):
    config = yaml.safe_load(Path(config_path).read_text())

    root = Path(config["data"].get("root", "."))
    data_cfg = config["data"]
    train_set = build_dataset(data_cfg["train_sequences"], data_cfg, root)
    val_set = build_dataset(data_cfg["val_sequences"], data_cfg, root)

    train_cfg = config["training"]
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

    run = wandb.init(project=config["wandb"]["project"], entity=config["wandb"].get("entity"), config=config)

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / run.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)

        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
