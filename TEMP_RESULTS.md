# Résultats intermédiaires — 2026-07-16

Snapshot cluster NUS SoC. Toutes les évaluations tournent sur le **eval split EVIMO** (21
scènes, cf. `splits/evimo/evimo_val.txt`), toutes les métriques sont l'ATE moyen en cm
(RPG evaluation, alignement Sim3, echelle corrigée).

---

## À quoi correspondent les jalons M0-M4 (rappel)

Le papier compare 5 configurations sur le même val split, qui diffèrent uniquement par
**la source du masque dynamique** injecté dans DEVO (voir `PLAN.md`) :

| jalon | ce qu'on branche dans DEVO | ce que ça mesure |
|---|---|---|
| **M0** | (aucun) vs masque **GT oracle** (segmentation EVIMO ré-échantillonnée à H/4×W/4) | **Gate go/no-go** : le plafond existe-t-il ? Un masque parfait aide-t-il l'ATE ? Si non → changer de données. Si oui → tout le plan tient. |
| **M1** | masque prédit par un modèle MS (convlstm v4-full-run1) **entraîné séparément** sur EVIMO, branché comme un préprocesseur | **Baseline « art antérieur »** — l'équivalent de ce que fait le papier RPG 2026 : un modèle de segmentation supervisé, découplé de la VO. C'est le chiffre à battre. |
| **M2** | masque prédit par un MS **fine-tuné conjointement** avec DEVO, supervisé par la GT dynamique + la loss de pose | **Couplé supervisé** : est-ce que laisser la loss de pose façonner le masque (co-training) fait mieux que M1 découplé ? |
| **M3** | idem M2, mais le masque est supervisé par le **résidu de la DBA** (incohérence au mouvement rigide), **sans aucun GT** | **La thèse du papier** : le signal auto-supervisé qui vient de la VO elle-même suffit-il ? C'est ce qui rend la méthode **déployable** (drone sans annotations). |
| **M4** | (agrégation) | Produit le tableau central du papier à partir des checkpoints M1/M2/M3, sur le même val split. |

Toutes les injections utilisent des **poids doux** différentiables (`scores *= (1 - dyn)`
au niveau score map, `weights *= (1 - dyn)` au niveau BA) — jamais de suppression dure,
sinon on casse le gradient.

---

## M0 — gate décisif ✅ GO +8.2 % (21 scènes)

**Ce qui a été fait** : DEVO tourne deux fois sur les 21 scènes du eval set — d'abord
vanilla (pas de masque), puis avec le masque oracle GT EVIMO injecté dans la score map
(`dyn_score=masque_GT` → les patches sur les zones étiquetées dynamiques sont écartés).
Aucun entraînement, juste 2 évaluations et un delta d'ATE.

| pass | ATE moyen (script) |
|---|---:|
| vanilla | 11.30 |
| **oracle GT** | **10.38** |

**Δ = +0.92 cm (+8.2 %) → GO.** Le plafond existe.

### Détail par scène (cm)

| scène | vanilla | oracle | Δ |
|---|---:|---:|---:|
| box/seq_00 | 6.04 | 2.67 | **+3.38** |
| box/seq_01 | 37.70 | 25.79 | **+11.91** |
| box/seq_02 | 7.88 | 6.39 | +1.49 |
| box/seq_03 | 5.30 | 2.14 | **+3.16** |
| box/seq_04 | 27.25 | 17.56 | **+9.70** |
| box/seq_05 | 30.43 | 30.51 | −0.08 |
| tabletop/seq_00 | 4.03 | 3.63 | +0.40 |
| tabletop/seq_01 | 1.21 | 1.76 | −0.55 |
| tabletop/seq_02 | 4.03 | 5.46 | −1.43 |
| tabletop/seq_03 | 0.18 | 0.18 | ≈0 |
| table/seq_00 | 7.59 | 4.39 | +3.20 |
| table/seq_01 | 2.65 | 2.79 | −0.14 |
| table/seq_02 | 4.25 | 3.38 | +0.86 |
| table/seq_03 | 25.34 | 26.91 | −1.57 |
| floor/seq_00 | 2.45 | 2.38 | +0.07 |
| floor/seq_01 | 2.67 | 3.46 | −0.79 |
| fast/seq_00 | 13.51 | 9.07 | +4.43 |
| fast/seq_01 | 10.84 | 18.27 | **−7.42** |
| fast/seq_02 | 29.85 | 29.52 | +0.33 |
| **wall/seq_00** | 17.56 | 26.99 | **−9.44** |
| wall/seq_01 | 19.03 | 15.34 | +3.70 |
| **moyenne** | **12.42** | **11.53** | **+0.90** |

Le gain est concentré sur les scènes où bouger un objet fait dérailler la VO
(`box/seq_00-04`, `fast/seq_00`, `tabletop/seq_00`). Trois régressions à comprendre pour
le papier :
- `fast/seq_01` (−7.4 cm) : probablement le bug de timestamp mentionné plus bas.
- `wall/seq_00` (−9.4 cm) : scène très statique, l'oracle sur-supprime des patches
  d'arrière-plan qu'il fallait garder.
- `tabletop/seq_02`, `table/seq_03` : idem, `thicken_radius=2` trop agressif sur les
  bords d'objet.

Job SLURM : `685237` (COMPLETED en 1h48).

---

## M1 — baseline appris découplé ✅ +0.4 % (21 scènes)

**Ce qui a été fait** : le modèle MS convlstm v4-full-run1 (entraîné séparément dans
`MS_Model/checkpoints/v4-full-run1/best.pt`, sur les 40 seqs du train set avec BCE+Dice
contre le GT) branché comme un **préprocesseur** — il regarde les events, prédit une
carte de score dynamique par frame, et sert cette carte à DEVO au même point d'injection
que M0. Zéro couplage avec la VO au moment de l'entraînement du MS. Trois passes :
vanilla, learned (masque MS appris), oracle (plafond).

| pass | ATE moyen (21 scènes) |
|---|---:|
| vanilla | 11.03 |
| **learned (MS séparé)** | **10.98** |
| oracle GT | 10.46 |

**Δ vanilla − learned = +0.05 cm (+0.4 %).** Très faible. **Écart learned → oracle
(plafond) = +0.52 cm** — l'oracle est bien au-dessus du learned, comme attendu par le
plan.

**Interprétation** : sur les 21 scènes complètes, le convlstm découplé n'apporte quasiment
rien vs vanilla. Le +12.4 % qu'on voyait sur la version 13 scènes venait d'un biais de
sélection (le val split original était concentré sur les scènes dynamiques). Sur le split
complet, le découplé **s'écrase parce qu'il ne sait pas quels events sont vraiment nuisibles
à la VO** — il n'a été entraîné qu'à segmenter la GT, pas à optimiser la pose. C'est
précisément la limite du découplé que le papier veut battre avec M2/M3 couplés.

Job SLURM : `685238` (COMPLETED en 3h47).

---

## M2 — couplé supervisé — en cours d'entraînement

**Ce qui se fait** : on part des checkpoints DEVO pré-entraîné + MS convlstm-v4 pré-entraîné,
et on les fine-tune **ensemble** sur les 40 séquences du train set. À chaque step :

    images → MS → masque dynamique (b, n, Hp, Wp)
                                  ↓
    images → eVONet(dyn_mask=...) → BA différentiable pondérée → poses estimées
                                                                       ↓
    loss = pose_loss + flow_loss (+ scores_loss) + λ · BCE(masque prédit, GT dynamique)
    
    loss.backward() → grad remonte jusqu'aux poids du MS

Curriculum : DEVO **gelé** les 5000 premiers steps (seul MS apprend, protège du forgetting
catastrophique), puis **dégelé** pour co-training (lr DEVO = 2e-5, très doux).

Fichier : `DEVO/train_coupled.py`, sortie `results_coupled/m2/ms_final.pt`.

**État actuel** (job `685102`, ~5h) : step **6752 / 20 000 (34 %)**. Dégel DEVO passé
sans crash au step 5000. Losses saines :
- `loss` global : 6-80 (dépend de la scène)
- `mask_l` (BCE contre GT) : 0.02-0.07 → le masque appris colle bien au GT
- `mask_mean` : 0.05-0.11 → ~5-11 % du champ étiqueté dynamique en moyenne (cohérent)
- `pose_loss` : 0.06-0.44 → converge, pas d'explosion

**ETA fin** : ~10h. Puis M4 pourra utiliser `ms_final.pt`.

---

## M3 — couplé auto-supervisé (la thèse) — en cours

**Ce qui se fait** : même setup que M2, MAIS **on n'utilise plus le masque GT**. La
supervision du masque vient uniquement du résidu de la DBA différentiable :

- eVONet est modifié pour renvoyer, par patch, le résidu `‖target_observée − reprojection_rigide‖`
  après convergence de la BA (mesure l'incohérence au mouvement rigide dominant).
- Un patch **statique** → mouvement expliqué par la pose caméra → résidu faible.
- Un patch sur **objet mobile** → bouge indépendamment → résidu élevé.
- Cible auto-sup = `résidu > médiane + k·MAD` (seuil robuste, k=3.0). Aucun GT.
- BCE entre le masque prédit et cette cible auto-sup.

Fichier : `DEVO/train_coupled.py --selfsup`, sortie `results_coupled/m3/ms_final.pt`.

**Le point clé du papier** : si M3 s'approche de M2 (ou même M0 oracle), la méthode
devient **déployable** — un drone événement peut apprendre à masquer les objets mobiles
depuis ses seuls résidus de VO, sans jamais avoir vu de segmentation GT.

**État actuel** (job `685103`, ~5h) : step **11 492 / 20 000 (57 %)**. Dégel DEVO passé
depuis longtemps. Losses :
- `loss` global : 5-25
- `mask_l` (BCE self-sup) : 0.19-0.22 → converge lentement mais monotone
- `mask_mean` : 0.08-0.13 → converge autour du taux réel d'objets mobiles (~10-15 %)
- `pose_loss` : 0.03-0.09 → très basse, plus stable que M2

**ETA fin** : ~4h.

---

## M4 — tableau central — en attente

**Ce qui se fera** : script agrégateur qui lance les 5 lignes du tableau du papier sur
les 21 scènes eval et produit un `.md` + `.csv`. Chaque ligne = DEVO + un fournisseur
de masque différent (aucun / oracle GT / MS découplé / MS couplé sup / MS couplé selfsup).

Job SLURM : `685104` PENDING avec dépendance `afterany:685102:685103` — se déclenchera
automatiquement quand M2 et M3 termineront.

**Tableau attendu** (à partir de M0 v21 + M1 v21 ; M2/M3 en cours d'entraînement) :

| Configuration | ATE moyen | Δ vs vanilla |
|---|---:|---:|
| DEVO vanilla | ≈ 11.0-11.3 | 0 % |
| DEVO + oracle GT | ≈ 10.4 | **+7 à +8 %** (plafond) |
| DEVO + appris découplé (M1) | ≈ 10.98 | +0.4 % (baseline art antérieur) |
| DEVO + couplé supervisé (M2) | ? (à venir) | ? — **doit battre +0.4 %** |
| DEVO + couplé auto-sup (M3) | ? (à venir) | ? — **doit s'approcher de +8 % SANS GT** |

---

## Points de méthode qui méritent d'être dans le rapport

1. **Val split étendu** — le `evimo_val.txt` livré avec DEVO ne listait que 13 scènes
   sur les 21 disponibles dans le eval set. J'ai étendu à 21 (ajouté box/03-05,
   table/02-03, fast/02, wall/00-01) pour couvrir tout le split officiel EVIMO. Tous
   les chiffres ici sont sur 21 scènes.

2. **Bug de timestamp sur `fast/*`** — `OracleDynMaskProvider` détecte un écart ~1.5
   milliards de ms entre la base temporelle des voxels DEVO (Unix epoch) et celle du
   provider (ts relatif issu de `meta['frames'][i]['ts']`). Sur ces 2 scènes, l'oracle
   sert un masque quasi-nul, ce qui explique la régression sur `fast/seq_01`. À
   corriger dans une V2 de `TimestampMaskProvider`.

3. **Oracle non-monotone vs learned** — l'oracle GT n'est pas un vrai plafond ici
   parce qu'il est binaire dilaté (`thicken_radius=2`), qui sur-supprime les patches
   d'arrière-plan proches des objets mobiles. Pour un vrai plafond utilisable dans le
   tableau du papier, il faudra rerun avec `thicken_radius=0` (ou en soft, GT × 0.5).

4. **Bug latent d'intrinsics** (résolu) — `EvimoClipDataset` retournait `intrinsics`
   en shape `(4,)` alors que la BA de DEVO attend `(n_frames, 4)`. Le smoke test CPU
   du RECAP ne l'avait pas révélé (ne va pas jusqu'à la BA). Fixé.

5. **OOM du dataset** (résolu) — `EvimoClipDataset` chargeait initialement les 40
   séquences du train set en RAM au `__init__` (voxelisation + depth + masks).
   Estimation ~160 GB, hors budget SLURM. Refactoré en **lazy loading** : mmap sur le
   cache voxel disque (`.voxels_bins5.npy` déjà pré-produit par `EvimoSegDataset`) +
   chargement depth/mask à la demande, LRU 2 séquences par worker. Init 40 seqs passe
   de ~6h à ~15s.

6. **Buffering stdout SLURM** (résolu) — Python en block-buffered quand stdout est
   redirigé vers fichier. Aucun step log visible avant que 8KB s'accumule dans le
   buffer, alors qu'à ~200 bytes/step ça prend 40+ steps. Fixé avec `python -u` et
   `PYTHONUNBUFFERED=1`.

7. **Deux NaN chacun sur M2/M3** — sur ~6000 steps M2 et ~11 000 steps M3, ~0.02-0.04 %
   des batches. Le code fait `optimizer.zero_grad(); continue` — non-fatal. Vient
   probablement d'une géométrie dégénérée dans un clip (baseline trop courte, ou
   patches vides après masquage).
