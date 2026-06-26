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

## Filtrer des events avec un modèle entraîné

`scripts/filter_events.py` prend un checkpoint, un yaml de config et un fichier
`.npz` EVIMO, et produit un nouveau `.npz` dont les events tombant dans les zones
prédites comme dynamiques ont été retirés.

```bash
conda run --no-capture-output -n devo python scripts/filter_events.py \
    --weights checkpoints/convlstm-v4-unet/best_iou.pt \
    --config  configs/convlstm_evimo.yaml \
    --input   path/to/seq.npz \
    --output  path/to/seq_filtered.npz
```

Le script affiche le nombre d'events avant/après et le pourcentage retiré.

**Pipeline interne :**
1. `Predictor.predict_sequence()` — voxelise la séquence et prédit un masque (H×W bool) par frame.
2. `FrameMasks(pred_masks, ts=fm.ts[:-1])` — aligne chaque masque sur son intervalle temporel `[ts_i, ts_{i+1})`.
3. `remove_events_in_mask()` — pour chaque event, lookup du masque actif via `searchsorted`, suppression si le pixel est masqué.
4. `save_events_npz()` — sauvegarde en `.npz` compatible `load_evimo_npz`, avec l'index par frame recompté.

Utilisation depuis Python :

```python
from ms_model.inference import Predictor
from ms_model.io.loaders import save_events_npz

predictor = Predictor("checkpoints/.../best_iou.pt", "configs/convlstm_evimo.yaml")
ea_filtered = predictor.filter_events("seq.npz")
save_events_npz(ea_filtered, "seq_filtered.npz")
```

---

## Suivi avec wandb

Les métriques (`train/loss`, `val/loss`) sont loggées sur
[wandb.ai/arthurfou-cole-polytechnique/ms-model](https://wandb.ai/arthurfou-cole-polytechnique/ms-model).

Chaque run = une ligne dans le dashboard, avec l'ensemble des hyperparamètres (le yaml
entier) et les courbes de loss. Deux runs peuvent avoir le même `wandb.name` sans
perdre de données côté wandb (IDs distincts), mais ça écraserait les checkpoints
sur disque — d'où la confirmation demandée au démarrage si le dossier existe déjà.
