# MS_Model — Motion Segmentation sur données événementielles

Modèle PyTorch de segmentation dynamique/statique par patch, conçu pour s'intégrer
dans le pipeline DEVO (voir `../DEVO/`).

Environnement conda : `devo`.

---

## Lancer un entraînement

### Test rapide (CPU, 1 séquence, 2 epochs)

Utile pour vérifier que tout le pipeline tourne avant un vrai run.

```bash
cd MS_Model
conda run --no-capture-output -n devo python -m ms_model.training.train \
    --config configs/convlstm_evimo_smoke.yaml
```

### Run complet (local, toutes séquences)

```bash
conda run --no-capture-output -n devo python -m ms_model.training.train \
    --config configs/convlstm_evimo.yaml
```

Le chargement initial des séquences prend **2 à 5 minutes** avant le premier affichage
(voxelisation de toutes les séquences en RAM). C'est normal — les prints
`[EvimoSegDataset] (i/N) chargement ...` confirment que ça avance.

### Sur le cluster SoC (SLURM)

```bash
# Depuis la machine locale, copier les données sur le cluster d'abord :
rsync -av --progress EVOwithMS/datasets/EV-IMO/ <user>@<cluster>:<chemin_data>/EV-IMO/

# Sur le cluster :
cd MS_Model
sbatch scripts/train_hpc.sh configs/config_lstm_yogya.yaml /chemin/vers/evimo/eval
```

`data.root` du yaml est écrasé par le second argument — pas besoin de modifier le yaml
entre la machine locale et le cluster.

Adapter dans `scripts/train_hpc.sh` avant le premier lancement :
- `--partition` et `--gres` selon les ressources disponibles (`sinfo`)
- le chemin `source "$HOME/miniconda3/..."` si conda est installé ailleurs

Suivre le job :
```bash
squeue -u <user>          # statut du job
tail -f logs/ms_model_<jobid>.out   # logs en direct
```

---

## Reprendre un entraînement interrompu

Si le run a un **`wandb.name` fixe** dans le yaml (ex: `name: "convlstm_yogya"`),
la reprise est **automatique** : relancer la même commande repart exactement là où
le run s'est arrêté (poids + état optimizer Adam + epoch + best_val_loss).

```bash
# même commande qu'au départ, pas de flag supplémentaire
conda run --no-capture-output -n devo python -m ms_model.training.train \
    --config configs/config_lstm_yogya.yaml
```

Au démarrage, le script affiche :
```
[resume] reprise du run 'convlstm_yogya' a partir de l'epoch 42 (best_val_loss=0.4123)
```

Si `wandb.name` est vide (`""`), wandb génère un nom unique à chaque run — pas de
reprise possible dans ce cas, mais pas de collision non plus.

---

## Checkpoints

Sauvegardés dans `checkpoints/<nom_du_run>/` après chaque epoch :

| Fichier | Contenu | Usage |
|---|---|---|
| `last.pt` | poids + optimizer + epoch + best_val_loss + wandb run id | reprise d'entraînement |
| `best.pt` | poids du modèle uniquement (meilleure val_loss) | inférence / évaluation finale |

Charger `best.pt` pour l'inférence :

```python
from ms_model.models import build_model
import torch

model = build_model("convlstm", in_channels=5, patch_size=4, hidden_dim=64)
model.load_state_dict(torch.load("checkpoints/convlstm_yogya/best.pt", map_location="cpu"))
model.eval()
```

---

## Configs disponibles

| Fichier | Usage |
|---|---|
| `configs/convlstm_evimo_smoke.yaml` | smoke test rapide (1 seq, 2 epochs, CPU) |
| `configs/convlstm_evimo.yaml` | run de référence local (6 seq train, 50 epochs) |
| `configs/config_lstm_yogya.yaml` | run cluster complet (15 seq train, 100 epochs, GPU) |

---

## Suivi avec wandb

Les métriques (`train/loss`, `val/loss`) sont loggées sur
[wandb.ai/arthurfou-cole-polytechnique/ms-model](https://wandb.ai/arthurfou-cole-polytechnique/ms-model).

Chaque run = une ligne dans le dashboard, avec l'ensemble des hyperparamètres (le yaml
entier) et les courbes de loss. Deux runs peuvent avoir le même `wandb.name` sans
perdre de données côté wandb (IDs distincts), mais ça écraserait les checkpoints
sur disque — d'où la confirmation demandée au démarrage si le dossier existe déjà.
