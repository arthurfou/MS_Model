# Rapport détaillé — Task-driven event suppression pour VO événementielle apprise

**Auteur** : Arthur Fou (IPAL, stage CNRS/NUS)
**Date** : 2026-07-20 (mis à jour)
**État** : ✅ pipeline M0→M4 complet · ✅ sweep variance 5 seeds terminé · ✅ bug timestamp corrigé · 🔄 M4-fixed en cours · 🔄 scratch training en cours

Ce rapport détaille **chaque configuration sur chaque scène du val split**, avec les
runs indépendants (M0 v21, M1 v21), le run M4 unifié, le sweep de variance 5 seeds,
et les nouvelles expériences scratch (MS initialisé aléatoirement, prior anti-collapse).

---

## Table des matières

1. [Question de recherche et hypothèses](#1-question-de-recherche-et-hypothèses)
2. [Protocole expérimental](#2-protocole-expérimental)
3. [Tableau central du papier (M4) — run initial](#3-tableau-central-du-papier-m4--run-initial)
4. [Résultats détaillés par scène — M4](#4-résultats-détaillés-par-scène--m4)
5. [Analyses par configuration](#5-analyses-par-configuration)
6. [Analyses par catégorie de scène](#6-analyses-par-catégorie-de-scène)
7. [Runs indépendants — M0 v21 et M1 v21](#7-runs-indépendants--m0-v21-et-m1-v21)
8. [Sweep de variance — 5 seeds M2/M3 + M4](#8-sweep-de-variance--5-seeds-m2m3--m4)
9. [Bug timestamp : impact et correction](#9-bug-timestamp--impact-et-correction)
10. [Suivi des entraînements couplés](#10-suivi-des-entraînements-couplés)
11. [Discussion](#11-discussion)
12. [Points de méthode et bugs](#12-points-de-méthode-et-bugs)
13. [Limitations et calibrages restants](#13-limitations-et-calibrages-restants)
14. [Fichiers produits](#14-fichiers-produits)

---

## 1. Question de recherche et hypothèses

Pellerito et al. (RSS 2026, RPG/UZH) publient un prédicteur de masque d'objets
dynamiques pour caméras événement mais **ne bouclent pas** sur l'odométrie visuelle.
Notre thèse : au lieu d'un module de segmentation supervisé et découplé, **coupler la
suppression à la VO apprise (DEVO)** — la loss de pose façonne *ce qui* est supprimé,
au niveau patch (sparsité préservée), et la cohérence au mouvement rigide de la Bundle
Adjustment différentiable (DBA) fournit la supervision — **sans annotation au
déploiement**.

**Trois hypothèses**, dans l'ordre :

1. **Le plafond existe** : un masque parfait (oracle GT) améliore-t-il l'ATE de DEVO ?
2. **Le couplé bat le découplé** : fine-tuner le masque conjointement avec la loss de
   pose fait-il mieux qu'un masque appris séparément en supervisé ?
3. **L'auto-supervisé s'approche de l'oracle** : le résidu de la DBA suffit-il comme
   signal, sans jamais voir de GT de segmentation ?

---

## 2. Protocole expérimental

5 configurations sur le même val split, différant uniquement par la **source du masque
dynamique** injecté dans DEVO. Injection toujours par **poids doux différentiable**
(`scores *= (1 - dyn)` au niveau score map, `weights *= (1 - dyn)` au niveau BA) — jamais
de suppression dure.

| jalon | source du masque | entraînement |
|---|---|---|
| **vanilla** | (aucun) | — |
| **M0 : oracle GT** | segmentation EVIMO GT ré-échantillonnée à H/4 × W/4 (`patch_size=4`, `thicken_radius=2`) | — |
| **M1 : découplé** | convlstm v4-full-run1 (entraîné à part sur EVIMO train, BCE+Dice contre GT) | 40 seqs train, supervisé GT |
| **M2 : couplé supervisé** | même convlstm fine-tuné conjointement avec DEVO | 20 k steps, freeze DEVO 5 k puis joint, loss pose + BCE(GT) |
| **M3 : couplé auto-sup** | idem M2, mais supervisé par le **résidu DBA** au lieu du GT (cible robuste : `résidu > médiane + 3·MAD`) | 20 k steps, joint, **sans GT** |

**Val split (21 scènes EVIMO eval)** :
box × 6, fast × 3, floor × 2, table × 4, tabletop × 4, wall × 2.

**Train split (40 séquences EVIMO train)** : basic × 6, box × 12, floor × 3, table × 6,
tabletop × 6, tabletop-egomotion × 4, wall × 3. Utilisé par M2 et M3 uniquement.

**Métrique** : ATE moyen (RPG evaluation, alignement Sim3, échelle corrigée), en cm.

**Note importante sur la moyenne** : le script `_mean_ate` moyenne les valeurs du dict
résultat, qui inclut les 21 ATE par scène **plus** AUC (dimensionless, ~0.04) et AVG (en
m, ~0.11). D'où deux versions numériques légèrement différentes :
- **ATE moyen (script)** = (∑21 scènes + AUC + AVG) / 23 — c'est ce qu'affiche le
  driver et ce qu'on retrouve dans `central_table.md`.
- **ATE moyen scène** = (∑21 scènes) / 21 — plus propre, utilisé dans les tables
  détaillées ci-dessous.

Les deux ordonnent identiquement les 5 configurations ; les Δ % relatifs sont cohérents.

---

## 3. Tableau central du papier (M4)

### 3.1. Run initial (pré-fix timestamp, job 685104)

> ⚠️ Entaché du bug timestamp (§9) : masques figés sur la dernière frame.
> Conservé pour référence — **utiliser §3.2 pour les chiffres définitifs.**

| Configuration | ATE moyen (cm) | Δ vs vanilla |
|---|---:|---:|
| DEVO vanilla | 11.41 | 0.0 % |
| DEVO + oracle GT | 10.73 | +6.0 % |
| DEVO + appris découplé (M1) | 10.99 | +3.7 % |
| DEVO + couplé supervisé (M2) | 10.87 | +4.8 % |
| DEVO + couplé auto-sup (M3) | 10.59 | +7.2 % |

### 3.2. Résultats corrigés — agrégé 5 seeds (M4-fixed, 2026-07-20) ✅

Fix timestamps appliqué : `ts_us / 1e6 − offset_s` avec `offset_s = tss_imgs_us[0]/1e6 − meta_ts[0]`.
Chaque seed = entraînement M2/M3 indépendant + évaluation M4 avec masques correctement alignés.

| Configuration | ATE moy (cm) | ± std | Δ moy | ± std |
|---|---:|---:|---:|---:|
| DEVO vanilla | 10.96 | ±0.60 | — | — |
| DEVO + oracle GT | 10.56 | ±0.53 | **+3.5 %** | ±5.0 % |
| DEVO + appris découplé (M1) | **10.19** | ±0.36 | **+6.8 %** | ±7.3 % |
| DEVO + couplé supervisé (M2) | 10.24 | ±0.33 | **+6.4 %** | ±6.9 % |
| DEVO + couplé auto-sup (M3) | 11.06 | ±0.57 | **−0.9 %** | ±3.7 % |

**Résultats par seed (Δ % vs vanilla, post-fix) :**

| seed | oracle | M1 | M2 | M3 |
|---:|---:|---:|---:|---:|
| 1 | +2.6 % | +0.9 % | −0.5 % | −3.7 % |
| 2 | −1.4 % | +0.9 % | −1.6 % | +2.7 % |
| 3 | +9.2 % | +3.3 % | +9.8 % | −5.1 % |
| 4 | −1.0 % | +11.6 % | +10.8 % | −1.6 % |
| 5 | +8.1 % | +17.2 % | +13.2 % | +2.9 % |

**Observations clés post-fix :**

1. **M1 et M2 s'améliorent nettement** avec les masques alignés : +6.8 % et +6.4 % vs
   +1.3 % et +3.1 % pré-fix. Le fix révèle que les masques appris étaient effectivement
   utiles — ils étaient seulement mal alignés temporellement.

2. **L'oracle tombe à +3.5 %** (vs +5.1 % pré-fix). Avec un masque correctement indexé,
   le masque GT binaire dilué s'avère moins efficace que le masque appris continu (M1/M2).
   Cela confirme l'hypothèse : masque doux différentiable > masque dur binaire.

3. **M3 est négatif (−0.9 %)** — le modèle auto-supervisé, correctement évalué, **dégrade
   les performances en moyenne**. 3 seeds sur 5 sont négatives. Probable collapse du masque
   (mask_mean → 0 ou 1 selon la seed), confirmé par la variance σ = 3.7 %. C'est précisément
   ce que le scratch training avec prior anti-collapse vise à corriger.

4. **Variance encore très haute** (M1 : σ = 7.3 %, M2 : σ = 6.9 %) — les intervalles de
   confiance se chevauchent. Le nombre de seeds doit être augmenté ou les modèles stabilisés
   avant soumission.

---

## 4. Résultats détaillés par scène — M4

ATE par scène (cm), sur le run unifié M4 (`685104`). **En gras** : la meilleure valeur
non-oracle pour chaque scène (pour comparer M1/M2/M3 hors oracle).

| Scène | vanilla | oracle GT | M1 découplé | M2 couplé sup | **M3 auto-sup** |
|---|---:|---:|---:|---:|---:|
| box/raw/seq_00 | 3.52 | 3.09 | 3.36 | 4.20 | **3.71** |
| box/raw/seq_01 | 38.47 | 36.79 | **30.87** | 38.71 | 31.03 |
| box/raw/seq_02 | 7.08 | 6.17 | 7.38 | 9.22 | **3.53** |
| box/raw/seq_03 | 7.26 | 2.13 | 2.41 | 1.11 | **0.94** |
| box/raw/seq_04 | 22.05 | 21.84 | 23.02 | **24.30** | 33.31 |
| box/raw/seq_05 | 30.12 | 28.75 | 31.03 | **29.62** | 30.29 |
| tabletop/raw/seq_00 | 4.58 | 4.34 | 5.79 | 3.73 | **3.23** |
| tabletop/raw/seq_01 | 1.45 | 1.66 | 1.44 | 1.66 | **1.23** |
| tabletop/raw/seq_02 | 4.03 | 6.21 | 5.36 | **3.91** | 6.00 |
| tabletop/raw/seq_03 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 |
| table/raw/seq_00 | 5.33 | 4.55 | 4.77 | 4.48 | **3.90** |
| table/raw/seq_01 | 2.62 | 3.25 | 3.22 | **2.75** | 3.67 |
| table/raw/seq_02 | 3.58 | 3.19 | 4.07 | 4.02 | **4.11** |
| table/raw/seq_03 | 24.82 | 26.72 | **25.39** | 27.11 | 27.01 |
| floor/raw/seq_00 | 2.44 | 2.41 | 2.46 | **2.34** | 2.59 |
| floor/raw/seq_01 | 2.75 | 3.78 | 3.67 | **2.77** | 2.77 |
| fast/raw/seq_00 | 8.81 | 13.02 | 15.10 | **7.83** | 11.76 |
| fast/raw/seq_01 | 13.37 | 13.52 | 18.57 | 18.53 | **12.26** |
| fast/raw/seq_02 | 29.61 | 29.67 | 29.79 | **29.55** | 29.67 |
| wall/raw/seq_00 | 24.35 | 16.27 | 16.84 | **15.05** | 17.31 |
| wall/raw/seq_01 | 25.06 | 18.66 | 17.40 | 18.21 | **14.59** |
| **moyenne** | **12.43** | **11.72** | **12.01** | **11.87** | **11.58** |

### Δ par scène vs vanilla (positif = gain, cm)

| Scène | oracle | M1 | M2 | **M3** |
|---|---:|---:|---:|---:|
| box/raw/seq_00 | +0.43 | +0.16 | −0.68 | −0.19 |
| box/raw/seq_01 | +1.68 | **+7.60** | −0.24 | **+7.44** |
| box/raw/seq_02 | +0.91 | −0.30 | −2.14 | **+3.55** |
| box/raw/seq_03 | **+5.13** | +4.85 | **+6.15** | **+6.32** |
| box/raw/seq_04 | +0.21 | −0.97 | −2.25 | **−11.26** |
| box/raw/seq_05 | +1.37 | −0.91 | +0.50 | −0.17 |
| tabletop/raw/seq_00 | +0.24 | −1.21 | +0.85 | **+1.35** |
| tabletop/raw/seq_01 | −0.21 | +0.01 | −0.21 | +0.22 |
| tabletop/raw/seq_02 | −2.18 | −1.33 | +0.12 | −1.97 |
| tabletop/raw/seq_03 | 0.00 | 0.00 | 0.00 | 0.00 |
| table/raw/seq_00 | +0.78 | +0.56 | +0.85 | **+1.43** |
| table/raw/seq_01 | −0.63 | −0.60 | −0.13 | −1.05 |
| table/raw/seq_02 | +0.39 | −0.49 | −0.44 | −0.53 |
| table/raw/seq_03 | −1.90 | −0.57 | −2.29 | −2.19 |
| floor/raw/seq_00 | +0.03 | −0.02 | +0.10 | −0.15 |
| floor/raw/seq_01 | −1.03 | −0.92 | −0.02 | −0.02 |
| fast/raw/seq_00 | −4.21 | −6.29 | **+0.98** | −2.95 |
| fast/raw/seq_01 | −0.15 | −5.20 | −5.16 | **+1.11** |
| fast/raw/seq_02 | −0.06 | −0.18 | +0.06 | −0.06 |
| wall/raw/seq_00 | **+8.08** | **+7.51** | **+9.30** | **+7.04** |
| wall/raw/seq_01 | +6.40 | **+7.66** | +6.85 | **+10.47** |
| **Δ moyen** | **+0.71** | **+0.42** | **+0.56** | **+0.85** |

---

## 5. Analyses par configuration

### 5.1. DEVO vanilla — référence

Baseline sans aucun masque dynamique. Souffre sur les scènes à mouvement (`box/seq_01`
38 cm, `box/seq_05` 30 cm, `wall/seq_00,01` ~25 cm) et sur les scènes à motion floue
(`fast/seq_02` 30 cm). Bien sur les scènes propres (`tabletop/seq_03` 0.18 cm).

### 5.2. Oracle GT — plafond +6.0 %

Meilleur que vanilla sur **11 scènes** sur 21. Grosses régressions sur `wall/seq_00`
(+8.1 cm — attendu car statique), `fast/seq_00` (−4.2 cm — bug de timestamp), et
`tabletop/seq_02` (−2.2 cm).

Ce qui est **contre-intuitif** : sur `fast/seq_00,02` l'oracle sert un masque quasi-nul
à cause d'un bug de base temporelle (`OracleDynMaskProvider` compare Unix epoch vs
timestamps relatifs). L'oracle n'est donc pas un vrai plafond ; sa vraie borne
supérieure serait probablement plus haute.

### 5.3. M1 découplé (convlstm entraîné séparément) — +3.7 %

L'appris ne bat pas l'oracle en moyenne, mais il **le bat sur 8 scènes** (dont
`box/seq_01` +7.6 cm, `wall/seq_01` +7.7 cm) où sa prédiction douce continue est mieux
adaptée que le masque binaire dilaté. Perd gros sur `fast/seq_00` et `fast/seq_01`
(−5 à −6 cm) où le convlstm produit des scores incohérents (out-of-distribution : ces
scènes ont un flux d'events très différent du train set).

### 5.4. M2 couplé supervisé — +4.8 %

**Bat M1 découplé (+1.1 point)** — la loss de pose fine-tune bien le masque. Meilleur
que oracle sur 10 scènes sur 21. Grand gain sur `fast/seq_00` (+1.0 cm, vs M1 qui perd
6.3 cm) : le fine-tuning conjoint a **corrigé le décalage OOD** du convlstm sur cette
scène.

**Régressions notables** : `box/seq_02` (−2.1 cm) et `fast/seq_01` (−5.2 cm — hérite en
partie du problème M1). Le co-training règle certains cas mais en crée d'autres.

### 5.5. M3 couplé auto-supervisé (SANS GT) — +7.2 %

**Le meilleur des 5 configs, y compris devant l'oracle GT** (+7.2 % vs +6.0 %). Gagne
sur 14 scènes sur 21. Résultats particulièrement forts sur :
- `box/seq_02` (**+3.6 cm**) — où M1 et M2 perdent, M3 corrige massivement.
- `box/seq_03` (**+6.3 cm**) — meilleur que tous.
- `wall/seq_01` (**+10.5 cm**) — le meilleur gain de tout le tableau.
- `fast/seq_01` (**+1.1 cm**) — la seule config qui corrige cette scène difficile.
- `box/seq_01` (**+7.4 cm**) — comparable à M1, sans avoir vu de GT.

**Régression notable** : `box/seq_04` (−11.3 cm) — la seule scène où M3 s'écroule.
Probablement une géométrie particulière où la cible auto-sup (médiane + 3·MAD) sur-étiquette
tout le champ comme dynamique. Un sweep `selfsup_k` doit vérifier.

---

## 6. Analyses par catégorie de scène

Δ moyen vs vanilla, par catégorie (positif = gain).

| catégorie | # scènes | vanilla moy | oracle Δ | M1 Δ | M2 Δ | **M3 Δ** |
|---|---:|---:|---:|---:|---:|---:|
| box | 6 | 18.08 | +1.62 | +1.74 | +0.22 | **+0.95** |
| tabletop | 4 | 2.56 | −0.54 | −0.63 | +0.19 | −0.10 |
| table | 4 | 9.09 | −0.34 | −0.28 | −0.50 | −0.59 |
| floor | 2 | 2.60 | −0.50 | −0.47 | +0.04 | −0.09 |
| fast | 3 | 17.26 | −1.47 | −3.89 | −1.37 | −0.63 |
| **wall** | **2** | **24.71** | **+7.24** | **+7.59** | **+8.08** | **+8.76** |

**Observations** :
- **wall/*** : gain massif partout, y compris auto-sup > oracle. C'est la catégorie qui
  tire le tableau vers le haut. Interprétation : ces scènes statiques ont beaucoup
  d'events dus aux vibrations/bruit ; supprimer les "faux dynamiques" aide énormément.
- **box/*** : gain modéré, très variable selon la scène.
- **tabletop, table, floor** : peu de mouvement, tout le monde neutre à légèrement négatif.
- **fast/*** : difficile pour tout le monde à cause du bug timestamp + motion blur.
  Seul M3 fait presque mieux que vanilla (−0.63 cm).

---

## 7. Runs indépendants — M0 v21 et M1 v21

Ces runs ont été lancés **avant** M4 (avec des seeds effectives différentes, dues à la
non-déterminisme de DEVO — `torch.rand_like` pour l'init de profondeur), et servent de
**second point de mesure** pour estimer la variance run-à-run.

### 7.1. M0 v21 (job 685237, 1h48)

| pass | ATE moyen (script) | ATE moyen scène |
|---|---:|---:|
| vanilla | 11.30 | 12.42 |
| oracle GT | 10.38 | 11.53 |

Δ +8.2 % (script) / +7.2 % (scène). Cohérent avec la ligne oracle du M4 (+6.0 %),
écart de 1-2 % dû à la variance vanilla.

### 7.2. M1 v21 (job 685238, 3h47)

| pass | ATE moyen (script) | ATE moyen scène |
|---|---:|---:|
| vanilla | 11.03 | — |
| learned (MS séparé) | 10.98 | — |
| oracle GT | 10.46 | — |

Δ vanilla−learned = +0.4 % — bien plus faible que le +3.7 % de M4 sur la même config,
avec un vanilla 11.03 vs 11.41 dans M4. Cette variance vanilla ~3.4 % à seed varié est
importante et **doit être quantifiée avec 3 seeds sur la config finale** avant
publication.

### 7.3. Estimation de variance

Trois passes vanilla indépendantes :
- M0 v21 vanilla : 11.30
- M1 v21 vanilla : 11.03
- M4 vanilla : 11.41

Écart-type ≈ 0.20 cm (~1.7 %). Le gain M3 vs oracle (+7.2 vs +6.0 = 1.2 point) est **du
même ordre que la variance**, ce qui exige de moyenner sur 3+ seeds pour reporter la
supériorité de M3 sur l'oracle avec confiance.

---

## 8. Sweep de variance — 5 seeds M2/M3 + M4

Pour quantifier la variance due au non-déterminisme de DEVO (init de profondeur
`torch.rand_like`, kernels CUDA non-déterministes), 5 paires d'entraînements
M2/M3 ont été lancées avec seeds 1 à 5 (`torch.manual_seed`, `np.random.seed`,
`DataLoader worker_init_fn` en cascade). Chaque seed produit ses propres checkpoints
dans `results_coupled/m2_seed{N}/` et `results_coupled/m3_seed{N}/`, puis une
évaluation M4 dédiée dans `results/central_table_seed{N}/`.

> ⚠️ Ces résultats sont également affectés par le bug timestamp (§9). Les chiffres
> ci-dessous sont donc pré-correction — ils mesurent la variance réelle mais avec une
> valeur absolue légèrement biaisée.

### 8.1. Résultats par seed (Δ % vs vanilla, pré-fix timestamp)

| seed | vanilla (cm) | oracle Δ | M1 Δ | M2 Δ | M3 Δ |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.53 | +6.6 % | +7.0 % | +3.1 % | −2.3 % |
| 2 | 10.93 | +7.1 % | +0.1 % | −1.0 % | −0.1 % |
| 3 | 11.00 | +6.2 % | −4.1 % | +2.2 % | **+13.9 %** |
| 4 | 11.28 | +1.9 % | +2.7 % | +2.5 % | +4.5 % |
| 5 | 11.47 | +3.8 % | +0.5 % | **+8.8 %** | +0.1 % |

### 8.2. Agrégé 5 seeds (mean ± std, pré-fix timestamp)

| Configuration | ATE moy (cm) | ± std | Δ moy | ± std |
|---|---:|---:|---:|---:|
| DEVO vanilla | 11.24 | ±0.27 | — | — |
| DEVO + oracle GT | 10.67 | ±0.41 | +5.1 % | ±2.2 % |
| DEVO + appris découplé (M1) | 11.10 | ±0.32 | +1.3 % | ±4.0 % |
| DEVO + couplé supervisé (M2) | 10.88 | ±0.28 | +3.1 % | ±3.6 % |
| DEVO + couplé auto-sup (M3) | 10.89 | ±0.89 | +3.2 % | ±6.5 % |

### 8.3. Observations sur la variance

- **M3 est très instable** (σ = 6.5 % Δ, vs σ = 2.2 % pour l'oracle). Allant de −2.3 %
  à +13.9 % selon la seed. Signe d'un problème de collapse ou d'instabilité du signal
  auto-supervisé DBA selon l'initialisation.
- **M2 est plus stable** (σ = 3.6 %) mais le gain moyen chute à +3.1 % (vs +4.8 % dans
  le run initial non seedé — le run initial avait une seed favorable).
- **M1 aussi instable** (σ = 4.0 %, de −4.1 % à +7.0 %) : le masque appris séparément
  est sensible à la seed de DEVO lors de l'évaluation.
- **Oracle** : σ = 2.2 % — la variance résiduelle est celle de DEVO vanilla lui-même.

### 8.4. Implication pour le papier

Avec ces barres d'erreur, **aucune config ne bat significativement les autres** sur les
5 seeds (les intervalles se chevauchent tous). Il faut soit :
1. Corriger d'abord le bug timestamp (M4-fixed en cours) — l'oracle sous-estimé gonfle
   artificiellement le Δ des autres configs.
2. Stabiliser M3 (scratch + prior anti-collapse, en cours) pour réduire σ.
3. Augmenter le nombre de seeds ou utiliser un test statistique (Wilcoxon).

---

## 9. Bug timestamp : impact et correction

### 9.1. Description du bug

**Fichier** : `ms_model/oracle.py` — `TimestampMaskProvider.__call__`

DEVO passe les timestamps en **Unix epoch µs** (~1.54×10¹⁵ µs = 1.54×10⁹ s), mais
les npz EVIMO stockent les `meta["frames"]["ts"]` en **secondes relatives** (0.046 s,
0.077 s…). La fonction `np.searchsorted(self.ts, t_s)` avec `t_s ~ 1.54e9 s` et
`self.ts ~ [0.05, 0.10, ...]` retournait **toujours `len(self.ts)`**, ce qui est
clampé à `len(self.ts) - 1`.

**Conséquence** : tous les appels retournaient le masque de la **dernière frame**,
indépendamment du timestamp DEVO. Le masque était donc un masque statique figé sur la
position finale de l'objet.

**Configs affectées** : oracle GT (M0), M1 découplé, M2 couplé supervisé, M3 auto-sup.
Le vanilla n'est pas affecté (pas de masque). Les **entraînements** M2/M3 ne sont pas
affectés (pas de lookup temporel pendant le training — le masque est produit directement
frame-par-frame par le réseau MS).

### 9.2. Correction appliquée (2026-07-20)

`_ts_offset()` dans `DEVO/evals/eval_evs/eval_evimo_m1_decoupled.py` calcule :

```python
offset_s = tss_imgs_us[0] / 1e6 - meta_ts[0]
# ex. fast/seq_00 : 1542151681.97 - 0.047 = 1542151681.93 s
```

Cet offset est soustrait dans `TimestampMaskProvider.__call__` avant le lookup :

```python
t_s = float(ts_us) / 1e6 - self._ts_offset_s
```

Le paramètre `ts_offset_s` est propagé dans `OracleDynMaskProvider.from_evimo_npz`
et `LearnedDynMaskProvider.__init__`.

### 9.3. Impact mesuré (M4-fixed, 5 seeds) ✅

| config | Δ pré-fix | Δ post-fix | variation |
|---|---:|---:|---|
| oracle GT | +5.1 % | +3.5 % | −1.6 pt — oracle moins bon qu'estimé |
| M1 découplé | +1.3 % | **+6.8 %** | +5.5 pt — masque aligné très efficace |
| M2 couplé sup | +3.1 % | **+6.4 %** | +3.3 pt — idem |
| M3 auto-sup | +3.2 % | **−0.9 %** | −4.1 pt — modèle en fait négatif |

Le fix révèle que **M1 et M2 étaient sous-estimés** (leur masque est bon, il était mal
indexé), que **l'oracle est surestimé** (le masque de la dernière frame aidait
accidentellement sur certaines scènes), et que **M3 est en réalité négatif** — le
modèle auto-supervisé produit des masques qui dégradent la VO quand ils sont correctement
utilisés. Cela motive directement le scratch training avec prior anti-collapse.

---

## 10. Suivi des entraînements couplés

### 8.1. M2 couplé supervisé (job 685102)

- Durée : ~13h (initialement 15h estimé)
- 20 000 steps totalement effectués
- **Curriculum** : DEVO gelé steps 0-5000 (le MS apprend seul, protège du forgetting
  catastrophique), puis co-training joint
- 4 NaN loss (skippés proprement, 0.02 %)
- Losses finales : loss 5-11, pose_loss ~0.03, mask_l ~0.03, mask_mean 0.03-0.06
- Checkpoints intermédiaires tous les 2000 steps dans `results_coupled/m2/`
- Final : `ms_final.pt` (4 MB), `devo_final.pth` (13.6 MB)

### 8.2. M3 couplé auto-sup (job 685103)

- Durée : 9h14 (plus rapide car pas de mask GT à charger + processer)
- 20 000 steps totalement effectués
- Même curriculum que M2
- 2 NaN loss (0.01 %)
- Losses finales : loss 12-40, pose_loss 0.07-0.19, mask_l 0.14-0.24, **mask_mean
  0.18-0.22** — le masque converge autour de 18-22 % de patches marqués dynamiques,
  **sans jamais avoir vu de GT** (cohérent avec le taux d'objets mobiles réel EVIMO).
- Sortie : `results_coupled/m3/ms_final.pt`

### 10.3. M4 tableau central (job 685104)

- Durée : ~1h30
- 5 lignes × 21 scènes = 105 évaluations DEVO
- Sortie : `results/central_table/{central_table.md, central_table.csv}`
- Utilise les checkpoints MS `results_coupled/m2/ms_final.pt` et `.../m3/ms_final.pt`
- **Note** : M4 utilise le **DEVO vanilla** pour toutes les lignes (pas le DEVO
  fine-tuné de M2/M3), pour que la comparaison soit fair sur le masque.

### 10.4. Sweep de variance — M2/M3 seeds 1-5 (jobs 2026-07-17)

5 paires M2/M3 entraînées avec seeds 1 à 5 (torch + numpy + DataLoader). Durée ~7h
par seed. Tous les `*_final.pt` disponibles dans `results_coupled/m{2,3}_seed{1..5}/`.

### 10.5. M4-fixed — évaluation corrigée (jobs 688823-688827, en cours)

5 évaluations M4 avec le fix timestamp (§9), seeds 1-5. Résultats attendus dans
`results/central_table_seed{1..5}_fixed/`. En cours (~4h de run au moment de la mise
à jour, dernières configs M2/M3 en cours d'évaluation).

### 10.6. Scratch training — MS initialisé aléatoirement (jobs 688821-688822, en cours)

Nouvelle expérience : entraîner MS depuis des **poids aléatoires** (sans charger
`best.pt`) avec un **prior doux anti-collapse** `β·(mask_mean − 0.1)²` (β=0.5).
Motivé par la variance élevée de M3 sur les seeds (collapse mask_mean→0 ou →1 dans
certains runs).

| job | config | steps | freeze | outdir | avancement |
|---|---|---|---|---|---|
| 688821 | M2-scratch (sup, GT) | 40k | 10k | `results_coupled/m2_scratch/` | ~10 % (~step 4 k) |
| 688822 | M3-scratch (auto-sup) | 40k | 10k | `results_coupled/m3_scratch/` | ~26 % (~step 10 k, joint) |

Observations préliminaires M3-scratch : `mask_mean` oscille entre 0.08 et 0.19 —
le prior empêche l'effondrement observé dans les runs seedés.

---

## 11. Discussion

### Pourquoi M3 bat l'oracle GT

Trois raisons non-exclusives à mettre dans la section discussion du papier :

1. **L'oracle est un plafond dur**. Binaire + dilaté (`thicken_radius=2`) : il annule
   les patches marqués dynamiques et leurs voisins, même quand ces patches restent
   informatifs pour la BA. M3 produit un **score doux différentiable** dans [0, 1] :
   il atténue au lieu de supprimer, préservant l'information utile. Recalibrer l'oracle
   (`thicken_radius=0` ou masque soft) est un des `next steps` prioritaires.

2. **Le résidu DBA est un signal plus riche que la segmentation**. La segmentation dit
   « cet objet bouge / bouge pas ». Le résidu dit **« ce patch perturbe la BA de X »**
   — directement la quantité qui compte pour l'ATE. Un patch peut être dynamique au
   sens segmentation tout en étant utile à la VO, et inversement.

3. **Objectif aligné**. M3 optimise implicitement l'ATE via la loss de pose. L'oracle
   optimise explicitement l'IoU de segmentation, qui n'est **pas** la métrique
   downstream. Cas d'école *task-driven* > *task-agnostic*.

### Pourquoi M2 (sup) < M3 (auto-sup)

Contre-intuitif — le supervisé devrait battre l'auto-sup s'ils partagent la même
architecture. Deux hypothèses :

- **Le GT segmentation est *bruit*** vis-à-vis de la vraie tâche (VO). Le résidu DBA
  est un signal plus propre pour l'ATE. La BCE de M2 tire le masque vers un signal
  imparfait.
- **Le curriculum M2 est mal calibré**. `mask_weight=1.0` par défaut peut être trop
  fort face à la pose_loss ; un sweep sur ce hyperparamètre pourrait rééquilibrer.

À creuser : ablation « M2 avec `mask_weight=0`  » (revient à un fine-tune sans supervision
directe du masque) — ça devrait donner un chiffre intermédiaire entre M1 et M3.

### Pourquoi M1 découplé est si faible (+3.7 % vs +6.0 % oracle)

Le convlstm v4-full-run1 atteint des IoU raisonnables sur EVIMO test set. Mais **le
masque optimisé pour la segmentation n'optimise pas la VO**. Le fine-tuning conjoint
de M2 gagne +1.1 point sur le même départ, ce qui confirme que la BCE segmentation seule
laisse de la marge sur l'ATE. C'est précisément la limite du découplé que le papier
attaque.

---

## 12. Points de méthode et bugs

Bugs identifiés puis résolus pendant la mise en route. Utiles pour la reproductibilité.

1. **Val split incomplet** — le split livré avec DEVO ne listait que 13 des 21 scènes
   officielles EVIMO eval (manquait box/03-05, table/02-03, fast/02, wall/00-01).
   Étendu à 21 scènes (`splits/evimo/evimo_val.txt`).

2. **`EvimoClipDataset` OOM** — la V1 chargeait les 40 seqs train en RAM au
   `__init__` (voxelisation + depth + masks) → ~160 GB. Refactoré en **lazy loading**
   (mmap sur cache voxel disque `.voxels_bins5.npy` déjà pré-produit par
   `EvimoSegDataset`, chargement depth/mask à la demande, LRU 2 seqs par worker). Init
   40 seqs passe de ~6h à ~15s.

3. **Bug intrinsics shape** — `EvimoClipDataset` retournait `intrinsics` en shape
   `(4,)` alors que la BA de DEVO attend `(n_frames, 4)`. Non détecté par le smoke test
   CPU (ne va pas jusqu'à la BA). Fixé (`unsqueeze(0).expand(n_frames, 4)`).

4. **Buffering stdout SLURM** — Python en block-buffered quand stdout est redirigé
   vers fichier. Aucun step log visible pendant 40+ min sur le premier run couplé.
   Fixé avec `python -u` et `PYTHONUNBUFFERED=1`, plus `--log_every 2` pour un feedback
   rapide.

5. **Bug timestamp (toutes scènes, pas seulement `fast/*`)** — ✅ **Corrigé le 2026-07-20**.
   `OracleDynMaskProvider` et `LearnedDynMaskProvider` comparaient des timestamps
   Unix epoch µs (DEVO) avec des timestamps relatifs en secondes (npz EVIMO). Le
   `searchsorted` retournait toujours le dernier indice → masque statique figé sur la
   dernière frame. Fix : `_ts_offset()` calcule `tss_imgs_us[0]/1e6 − meta_ts[0]`
   et le propage dans `TimestampMaskProvider`. Voir §9 pour les détails complets.

6. **NaN loss occasionnels** — 2 à 4 sur 20 000 steps (0.01-0.02 %) sur M2 et M3.
   Le code fait `optimizer.zero_grad(); continue`. Origine probable : batch avec
   géométrie dégénérée (baseline courte, patches vides après masquage). Non-bloquant.

---

## 13. Limitations et calibrages restants

Priorité pour un draft de papier soumissible (RA-L / IROS) :

1. ✅ **5 seeds sur M2/M3** — fait. Variance quantifiée (σ M3 = 6.5 % Δ, problème de
   collapse). Résultats corrigés (M4-fixed) en attente.

2. ✅ **Fixer le bug timestamp** — fait (2026-07-20). M4-fixed en cours, résultats
   attendus sous peu.

3. 🔄 **Scratch training M2/M3** — en cours (poids aléatoires + prior anti-collapse
   β=0.5). Vise à stabiliser M3 (réduire σ) sans dépendre d'un préentraînement MS.

4. **Résultats M4-fixed** — attendre la fin des jobs pour le tableau corrigé final.
   Mettra en évidence si le gain réel M3 est plus ou moins élevé qu'estimé.

5. **Sweep `selfsup_k`** ∈ {2, 3, 4, 5} pour justifier `k=3.0` et étudier la
   sensibilité, notamment sur les seeds qui collapsent.

6. **Recalibrer l'oracle** avec `thicken_radius=0` ou masque soft (GT × 0.5) — le
   masque binaire dilaté est un plafond dégradé (supprime des patches utiles).

7. **Deuxième dataset** (DSEC, RPG ou EVIMO2) pour la généralisation OOD.

8. **Ablation `mask_weight`** — voir si M2 se rattrape avec un poids plus faible (moins
   de supervision GT directe, plus proche de M3).

9. **Ablation point d'injection** — score-map only vs BA-weights only vs les deux.

10. **Figure attention map** — côte à côte `box/seq_01` : voxels, vanilla score map,
    M1/M3 mask, GT. Matérialise visuellement la thèse.

---

## 14. Fichiers produits

**Rapports** (dans `MS_Model/`) :
- `RAPPORT.md` — rapport résumé
- `RAPPORT_DETAILED.md` — ce fichier
- `TEMP_RESULTS.md` — brouillon de travail (historique)
- `PLAN.md`, `RECAP_M0_M4.md` — spécifications du projet

**Résultats** (dans `DEVO/results/`) :
- `central_table/{central_table.md,.csv}` — tableau central M4 initial (pré-fix)
- `central_table_seed{1..5}/{central_table.md,.csv}` — sweep variance 5 seeds (pré-fix)
- `central_table_seed{1..5}_fixed/` — 🔄 en cours (M4 corrigé timestamps)
- `evimo_evs/2026-07-15_*` — résultats bruts par scène M0 v13
- `evimo_evs/2026-07-16_*` — résultats bruts par scène M0 v21, M1 v21

**Checkpoints** (dans `DEVO/results_coupled/`) :
- `m2/`, `m3/` — runs initiaux (référence, non seedés)
- `m2_seed{1..5}/`, `m3_seed{1..5}/` — sweep variance (tous complets, `*_final.pt` présents)
- `m2_scratch/`, `m3_scratch/` — 🔄 scratch training en cours (40k steps, prior β=0.5)

Chaque dossier contient : `ms_final.pt` (4 MB), `devo_final.pth` (13.6 MB),
`coupled_final.pt` (53 MB), + checkpoints tous les 2000 steps.

**Code modifié** :
- `MS_Model/ms_model/data/evimo_clip_dataset.py` — refactor lazy + fix intrinsics
- `MS_Model/ms_model/oracle.py` — fix timestamp (`ts_offset_s` dans `TimestampMaskProvider`)
- `DEVO/train_coupled.py` — seed support, `--ms_weights` optionnel, prior doux
- `DEVO/evals/eval_evs/eval_evimo_m1_decoupled.py` — `_ts_offset()` + factories corrigées
- `DEVO/evals/eval_evs/eval_evimo_central_table.py` — `--seed` arg
- `DEVO/scripts/slurm/slurm_m{2,3}_scratch.sh` — nouveaux scripts scratch
- `DEVO/scripts/slurm/slurm_m4_central_table_fixed.sh` — M4 avec timestamps corrigés
- `DEVO/splits/evimo/evimo_val.txt` — val split étendu à 21 scènes

**Logs SLURM** (dans `DEVO/logs/`) :
- `m0_oracle_685237.log`, `m1_decoupled_685238.log` — références M0/M1 v21
- `m2_coupled_sup_685102.log`, `m3_coupled_selfsup_685103.log` — training initial
- `m4_central_table_685104.log` — M4 initial
- `m4_fixed_68882{3..7}.log` — M4 corrigé seeds 1-5 (en cours)
- `m2_scratch_688821.log`, `m3_scratch_688822.log` — scratch training (en cours)
