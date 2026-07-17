# RÉCAP — Implémentation M0 → M4 (task-driven event suppression pour VO apprise)

Récapitulatif complet du travail d'implémentation des jalons du [`PLAN.md`](PLAN.md).
**État : tout le code est écrit, compile, et les sous-composants sont unit-testés sur CPU.
Aucun run end-to-end n'a encore eu lieu (exige le GPU + les extensions CUDA de DEVO).**

- Jobs SLURM prêts : [`DEVO/scripts/slurm/`](DEVO/scripts/slurm/) (voir §SLURM en bas).
- Environnement conda : `devo` (DEVO + MS_Model partagent le même env ; `ms_model` doit être
  `pip install -e MS_Model`).

---

## Idée générale (rappel)

Le papier RSS 2026 de RPG (Pellerito et al., « Motion-aware Event Suppression ») prédit un
masque d'objets dynamiques mais **ne boucle pas sur l'odométrie** → c'est notre **baseline**,
pas notre concurrent. Notre contribution : **coupler** la suppression à la VO apprise (DEVO),
l'objectif de pose façonne le masque, au niveau patch (sparsité préservée), supervisé par la
**cohérence au mouvement rigide** (déployable **sans GT**).

Deux points d'injection (déjà documentés dans `DEVO/CLAUDE.md`) :
- **score map** (`selector.py`/`enet.py`) → `scores *= (1 - p_dyn)` (jalon 1 / inférence).
- **poids ω de la DBA** (`ba.py`) → `weights *= (1 - p_dyn)` (jalon 2 / entraînement, différentiable).

**Toutes les modifications de DEVO sont additives : `None` = DEVO vanilla strictement identique.**

---

## M0 — Oracle (le gate décisif)

**But** : brancher un masque GT EVIMO sur la score map et mesurer le delta ATE. Si l'oracle
améliore l'ATE → le « plafond » existe, tout le plan tient. Sinon → changer de données avant
d'investir. **À lancer en premier.**

**Fichiers**
- `DEVO/devo/enet.py` — `Patchifier.forward(dyn_score=)` : atténue la score map après le sigmoid.
- `DEVO/devo/devo.py` — `DEVO.__call__(dyn_score=)` transmis à `patchify`.
- `DEVO/utils/eval_utils.py` — `run_voxel(dyn_mask_provider=)`.
- `DEVO/evals/eval_evs/eval_evimo_evs.py` — `evaluate(dyn_mask_provider_factory=)`.
- `DEVO/evals/eval_evs/eval_evimo_m0_oracle.py` — **driver M0** (passes vanilla vs oracle + delta ATE).
- `MS_Model/ms_model/oracle.py` — `OracleDynMaskProvider` (GT EVIMO → score map, aligné timestamp).

**Vérifié (CPU)** : provider sur vrai GT `box/seq_00.npz` (68 frames, score map (65,86), alignement
timestamp exact) ; injection (None = no-op, mismatch de résolution géré, zone dynamique annulée).

**Commande**
```bash
conda activate devo && cd DEVO
python evals/eval_evs/eval_evimo_m0_oracle.py \
    --datapath <eval_EVIMO_preprocessé> \
    --weights DEVO.pth \
    --val_split splits/evimo/evimo_val.txt \
    --mask_root <racine_npz_EVIMO> \
    --thicken_radius 2
```
Prérequis : scènes preprocessées (`preprocess_evimo.py` → `evs.npy`, `gt_stamped.txt`) ;
convention scène→npz dans `resolve_npz()` (`box/raw/seq_00` → `.../box/npz/seq_00.npz`).

---

## M1 — Suppression apprise découplée (baseline « art antérieur »)

**But** : masque prédit par un modèle MS entraîné **séparément** (`convlstm_v4`, déjà entraîné dans
`MS_Model/checkpoints/`), branché en préprocesseur. Réutilise **toute** la plomberie de M0.

**Clarification GPL** : pas besoin de ré-implémenter le modèle RPG (GPLv3) — `convlstm_v4` est
**ton** modèle (IP propre) et fait déjà office de masque appris découplé.

**Fichiers**
- `MS_Model/ms_model/oracle.py` — refactor : base `TimestampMaskProvider` + **`LearnedDynMaskProvider`**
  (fait tourner le modèle MS → score dynamique par frame, servi comme l'oracle ; continu ou binaire).
- `DEVO/evals/eval_evs/eval_evimo_m1_decoupled.py` — **driver M1** (3 passes vanilla / appris / oracle).

**Vérifié (CPU)** : provider e2e avec le vrai checkpoint `v4-full-run1/best.pt` — 67 frames,
scores (67,65,86) dans [0,1], fraction dynamique 1,45 % ≈ oracle → masques cohérents.

**Commande**
```bash
conda activate devo && cd DEVO
python evals/eval_evs/eval_evimo_m1_decoupled.py \
    --datapath <eval_EVIMO_preprocessé> \
    --weights DEVO.pth \
    --val_split splits/evimo/evimo_val.txt \
    --mask_root <racine_npz_EVIMO> \
    --ms_weights <MS_Model>/checkpoints/v4-full-run1/best.pt \
    --ms_config  <MS_Model>/configs/convlstm_v4_full.yaml \
    --with_oracle          # ajoute la passe plafond
    # --threshold 0.5      # optionnel : masque binaire dur (défaut = score continu doux)
```

---

## M2 — Entraînement couplé (supervisé)

**But** : fine-tuner conjointement MS + DEVO. `images → MS → masque → eVONet(dyn_mask) → poses`,
`loss = pose + flow (+ scores) [+ λ·masque GT]`. Le gradient de la pose-loss remonte **jusqu'au
modèle de masque**. DEVO gelé d'abord → dégel progressif. Sauvegarde **séparée** DEVO/MS.

**Fichiers**
- `DEVO/devo/enet.py` — `_sample_patch_scores()` + `eVONet.forward(dyn_mask=)` : échantillonne le
  masque aux coords des patches → `dyn_weights` par arête → **BA différentiable** (`ba.py`).
- `DEVO/devo/ba.py` — `BA(dyn_weights=)` : atténue ω (différentiable).
- `DEVO/train_coupled.py` — **trainer couplé mono-GPU** (gel/dégel, loss masque supervisée, sauvegarde).
- `MS_Model/ms_model/data/evimo_clip_dataset.py` — **`EvimoClipDataset`** : npz EVIMO → clips format
  DEVO `(images, poses, disps, intrinsics, gt_dyn, scene_id)`.

**Vérifié (CPU)** : sampling par patch différentiable (patch sur cellule dynamique → 0.9999,
gradient → dyn_mask) ; dataset EVIMO-clip sur vrai npz (53 clips, shapes correctes, ‖q‖≈1.0,
disps finies, masque GT 2,8 %).

**⚠️ Risque connu** : la **convention de pose EVIMO** (piège Sim3 documenté dans `error_explained.md`)
est isolée dans `evimo_cam_to_pose7()` — si `pose` explose/NaN au run, c'est là qu'il faut corriger.

**Commande**
```bash
conda activate devo && cd DEVO
python train_coupled.py \
    --dataset evimo --datapath <racine_npz_EVIMO> \
    --devo_weights DEVO.pth \
    --ms_weights <MS_Model>/checkpoints/v4-full-run1/best.pt \
    --ms_config  <MS_Model>/configs/convlstm_v4_full.yaml \
    --provide_gt_mask --n_frames 15 --steps 20000 --freeze_devo_steps 5000
```

---

## M3 — Entraînement couplé auto-supervisé (SANS GT — la thèse déployable)

**But** : superviser le masque par le **résidu de la DBA** (incohérence au mouvement rigide =
dynamique), sans aucun GT. La BA fournit le signal pendant l'entraînement ; le masque, qui ne voit
que les events, généralise au déploiement (ni GT ni BA requis).

**Fichiers**
- `DEVO/devo/enet.py` — `eVONet.forward(return_selfsup=True)` : renvoie le **résidu par patch**
  (‖cible observée − reprojection rigide‖) via `scatter_mean`. Zéro GT.
- `DEVO/train_coupled.py` — mode `--selfsup` : cible robuste (médiane + k·MAD sur le résidu) → BCE
  sur le masque échantillonné aux patches.

**Vérifié (CPU)** : la cible robuste étiquette 10/10 patches à résidu anormal comme dynamiques ;
loss finie ; gradient → dyn_mask → MS model.

**Commande** (pas de `--provide_gt_mask` : c'est tout l'intérêt)
```bash
conda activate devo && cd DEVO
python train_coupled.py \
    --dataset evimo --datapath <racine_npz_EVIMO> \
    --devo_weights DEVO.pth \
    --ms_weights <MS_Model>/checkpoints/v4-full-run1/best.pt \
    --ms_config  <MS_Model>/configs/convlstm_v4_full.yaml \
    --selfsup --selfsup_k 3.0 \
    --n_frames 15 --steps 20000 --freeze_devo_steps 5000
```

---

## M4 — Tableau central + ablations (infra)

**But** : les 6 lignes du tableau s'évaluent identiquement (`DEVO + un fournisseur de masque`),
ne diffèrent que par l'origine du masque. Le run couplé M2/M3 produit un **checkpoint MS** rebranché
dans `LearnedDynMaskProvider`.

**Fichier**
- `DEVO/evals/eval_evs/eval_evimo_central_table.py` — lance les lignes disponibles, agrège l'ATE en
  **Markdown + CSV**, calcule Δ vs vanilla, marque « — » les lignes sans checkpoint.

**Vérifié (CPU)** : agrégation md/csv (formatage, Δ vs vanilla, lignes manquantes). **Aucun chiffre
n'est fabriqué** — le script exécute les évals réelles (GPU) et agrège leurs sorties.

**Commande**
```bash
conda activate devo && cd DEVO
python evals/eval_evs/eval_evimo_central_table.py \
    --datapath <eval_EVIMO_preprocessé> \
    --weights DEVO.pth \
    --mask_root <racine_npz_EVIMO> \
    --ms_config <MS_Model>/configs/convlstm_v4_full.yaml \
    --ms_decoupled     <MS_Model>/checkpoints/v4-full-run1/best.pt \
    --ms_coupled_sup     results_coupled/m2/ms_final.pt \
    --ms_coupled_selfsup results_coupled/m3/ms_final.pt
```

---

## État global

| Jalon | Code | Unit-testé CPU | Run end-to-end / chiffres |
|---|---|---|---|
| M0 oracle | ✅ | ✅ | ❌ (GPU) |
| M1 appris découplé | ✅ | ✅ | ❌ (GPU) |
| M2 couplé supervisé | ✅ | ✅ | ❌ (GPU) |
| M3 couplé auto-sup | ✅ | ✅ | ❌ (GPU) |
| M4 tableau/ablations | ✅ | ✅ | ❌ (GPU) |

**Risques à surveiller au premier run** : convention de pose EVIMO (M2/M3, cf. `evimo_cam_to_pose7`) ;
shift de distribution du voxel donné au MS model ; stabilité gel/dégel + poids de loss ; calibrage `selfsup_k`.

## Fichiers modifiés (working tree, non commité)

- **DEVO** : `devo/{enet,devo,ba}.py`, `utils/eval_utils.py`, `evals/eval_evs/eval_evimo_evs.py`
  + nouveaux : `evals/eval_evs/{eval_evimo_m0_oracle,eval_evimo_m1_decoupled,eval_evimo_central_table}.py`,
  `train_coupled.py`, `scripts/slurm/*.sh`.
- **MS_Model** : `ms_model/oracle.py`, `ms_model/data/evimo_clip_dataset.py`.
- **arthur_ipal** : `PLAN.md`, `RECAP_M0_M4.md`.

---

## SLURM (NUS SoC HPC)

Jobs dans [`DEVO/scripts/slurm/`](DEVO/scripts/slurm/). **Éditer le bloc `# === À ÉDITER ===`** en
tête de chaque script (chemins cluster : données, poids, repo) avant le premier lancement, et vérifier
`--partition` / `--gres` avec `sinfo`.

```bash
# depuis la racine du repo DEVO sur le cluster
cd DEVO
sbatch scripts/slurm/slurm_m0_oracle.sh          # gate décisif — À LANCER EN PREMIER
sbatch scripts/slurm/slurm_m1_decoupled.sh       # baseline découplé
sbatch scripts/slurm/slurm_m2_coupled_sup.sh     # entraînement couplé supervisé (long)
sbatch scripts/slurm/slurm_m3_coupled_selfsup.sh # entraînement couplé auto-supervisé (long)
sbatch scripts/slurm/slurm_m4_central_table.sh   # tableau final (après M1/M2/M3)

squeue -u $USER            # statut
tail -f DEVO/logs/*.log    # logs en direct
```

Chaque chemin peut aussi être surchargé sans éditer le fichier :
```bash
sbatch --export=ALL,DATAPATH=/scratch/evimo/eval,MASK_ROOT=/scratch/evimo_full/eval \
       scripts/slurm/slurm_m0_oracle.sh
```
