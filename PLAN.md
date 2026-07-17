# PLAN — Task-driven event suppression pour VO événementielle apprise

Plan de recherche pour un papier, IPAL (CNRS/NUS). Cible : coupler la **motion
segmentation / suppression d'events dynamiques** à la **VO apprise (DEVO)**, à la
DynaSLAM, mais au domaine event et de façon **couplée** (pas seulement découplée).

## 0. Contexte & positionnement

Un papier vient de sortir sur exactement la brique « supprimer les events des objets
mobiles » :

> **Pellerito, Messikommer, Cioffi, Cannici, Scaramuzza — "Motion-aware Event
> Suppression for Event Cameras", RSS 2026** (arXiv:2602.23204, RPG/UZH).
> Repo téléchargé dans `../event_suppression/` (GPLv3).

Ce que fait RPG : un **U-Net récurrent (EV-FlowNet-like, convGRU)** qui prédit un
**masque dense pixel-wise `[N×1×H×W]`** d'objets dynamiques, **supervisé** (BCEDice +
flow + IoU) sur EVIMO v1 / DSEC, avec un twist temporel (prédiction à `t0` et `t1`).
**Le release public ne boucle PAS sur l'odométrie** — c'est un prédicteur de masque,
pas un système VO.

→ **RPG n'est pas le concurrent, c'est le baseline.** Leur contribution = prédiction de
masque. Notre contribution = **coupler la suppression à la VO apprise**, entraînée par
la loss de pose, au niveau patch (sparsité préservée), supervisée par la cohérence au
mouvement rigide (donc **sans GT au déploiement**).

**Découverte importante dans le code** (`DEVO/train.py:199`) : DEVO **supervise déjà sa
score map avec le résidu de la DBA** (`e_full`) pondéré par les poids `ba_weights` (ω).
Le câblage de la thèse auto-supervisée existe donc déjà en germe.

## 1. Thèse du papier (une phrase)

> **Task-driven event suppression for learned event-based VO** : au lieu de supprimer
> les events dynamiques via un module découplé et supervisé, on couple la suppression à
> la VO — l'objectif de pose façonne *ce qui* est supprimé, au niveau patch, et la
> cohérence au mouvement rigide de la DBA fournit la supervision, **sans annotation au
> déploiement**.

## 2. Contributions revendiquées

1. **Injection différentiable** d'un score « dynamique » aux deux points de DEVO —
   score map (`selector.py`) et poids ω de la DBA (`ba.py`) — **préservant la sparsité**
   (masque dense → un poids par patch par pooling à H/4×W/4).
2. **Entraînement couplé** (pré-entraînement séparé → fine-tuning conjoint avec loss de
   pose) qui **bat la suppression découplée** RPG en ATE/MPE sur séquences dynamiques.
3. Variante **auto-supervisée** par le résidu de la DBA, **déployable sans GT de
   masque**, approchant les performances de l'oracle.

## 3. Différentiabilité — le point technique central

Supprimer des events est une opération **discrète, non-différentiable** : on ne peut pas
backprop une loss de pose à travers un `delete`. → On n'injecte **jamais** par
suppression dure. On injecte un **poids doux (soft)** `p_dyn ∈ [0,1]` :

| Jalon | Fichier / point | Modif |
|---|---|---|
| 1 — découplé | `DEVO/devo/selector.py`, sortie `Scorer.forward` avant `PatchSelector` | `scores *= (1 - p_dyn)` |
| 2 — couplé | `DEVO/devo/ba.py`, param `weights` de `BA()` (déf. l.86, usage l.112) | `weights *= (1 - p_dyn)` |
| Supervision thèse auto-sup | `DEVO/train.py:199` (résidu `e_full` déjà présent) | superviser `p_dyn` par le résidu incohérent |

Les deux points sont différentiables et préservent la sparsité.

### Thèse auto-supervisée (le mécanisme, en clair)

La DBA reprojette chaque patch d'une frame à l'autre via pose + depth. Le résidu = écart
entre flux prédit et flux induit par le **mouvement rigide** de la caméra.

- Patch **statique** → une fois la pose convergée, mouvement rigide explique bien →
  **résidu faible**.
- Patch sur **objet mobile** → bouge indépendamment de la caméra → **résidu élevé /
  incohérent**.

→ Les IMO se trahissent seuls (idée du mixture model d'EV-IMO). On entraîne le prédicteur
de masque à prédire « dynamique » là où le résidu est incohérent. **Le signal de
supervision, c'est la VO elle-même** — aucun GT de segmentation requis au déploiement
(le drone n'a que ses résidus de DBA).

## 4. Plan expérimental

**Datasets** : EVIMO2 (GT depth+seg, dynamique) train + éval ; RPG pour la
généralisation. **Métriques** : ATE + MPE (pipeline `evo` / `rpg_trajectory_evaluation`
déjà en place), IoU masque en secondaire.

### Tableau central (6 lignes = tout le papier)

| Config | GT au déploiement ? | Rôle |
|---|---|---|
| DEVO vanilla | — | plancher |
| DEVO + masque **oracle** (GT EVIMO) | oui (irréaliste) | **plafond** |
| DEVO + RPG **découplé** (préprocesseur) | oui | baseline art antérieur |
| DEVO + couplé, injection **score** (jalon 1) | oui | > découplé |
| DEVO + couplé, injection **score + ω** (jalon 2) | oui | meilleur supervisé |
| DEVO + **auto-supervisé** (résidu DBA) | **non** | ≈ oracle — *la thèse* |

**Ablations** : point d'injection (score / ω / les deux) ; DEVO gelé vs fine-tune
conjoint ; supervisé vs auto-supervisé.

## 5. Roadmap avec gates go/no-go

- **M0 — expé décisive, pas chère (~1 semaine).** Jalon 1 avec **masque oracle GT** →
  score map. **Gate : l'oracle améliore-t-il l'ATE vs vanilla sur EVIMO ?** Oui → le
  plafond existe, le papier tient. Non → données pas assez dynamiques, changer de données
  AVANT d'investir. **À faire en premier, avant tout code de modèle.**
- **M1 — baseline découplé.** Masque RPG (ré-implémenté dans `MS_Model`, cf. GPL)
  entraîné sur EVIMO, branché en préprocesseur. → chiffre « art antérieur ».
- **M2 — couplé.** Fine-tuning conjoint : DEVO gelé d'abord, puis dégel progressif ;
  loss = `pose_weight·pose_loss + λ·mask_loss`. **Gate : couplé > découplé ?**
- **M3 — la nouveauté.** Injection ω + supervision auto par `e_full`.
  **Gate : auto-supervisé ≈ oracle sans GT ?** → le résultat qui vend le papier.
- **M4 —** ablations + rédaction.

## 6. Stratégie d'entraînement

Pré-entraîner séparément → fine-tuner conjointement. **Pas** de end-to-end from-scratch
(fragile) :

1. DEVO pré-entraîné sur TartanAir events (statique, synthétique) ; MS pré-entraîné sur
   EVIMO.
2. Données dynamiques GT (EVIMO2) **petites** → joint from-scratch = oubli catastrophique
   de DEVO.
3. Donc : freeze DEVO → entraîner l'injection score/ω → **dégel progressif** avec loss de
   pose + loss de masque sur données dynamiques.

## 7. Risques & parades

- **L'oracle n'aide pas (M0)** → séquences plus dynamiques, ou montrer le gain concentré
  sur les segments à fort mouvement d'objet. (Raison d'être de M0 en premier.)
- **Peu de données dynamiques GT → oubli catastrophique** → gel initial, dégel
  progressif, augmentations, appoint simu (ESIM).
- **Friction env CUDA** (extensions `cuda_corr`/`cuda_ba` de DEVO vs torch récent du
  RPG) → tout dans l'env `devo`, porter le *modèle* pas le repo.

## 8. Licence (GPLv3 du repo RPG)

- Usage **interne** (entraîner, expés, chiffres) = zéro contrainte. Obligations GPL =
  seulement à la **distribution**.
- Si on **release** du code incluant/dérivant du leur → tout doit passer **GPLv3, sources
  ouvertes** ; pas de relicence MIT/BSD/Apache. Statut des **poids** = zone grise.
- **Parade** : ré-implémenter le réseau de masque dans `MS_Model` (on a déjà un
  `convlstm`/EV-FlowNet-like), entraîner nos propres poids, citer RPG, n'utiliser leur
  repo qu'en interne comme baseline. IP propre, relicenciable.

## 9. Architecture des repos (voir §Réconciliation)

- DEVO et MS_Model restent **deux packages pip indépendants** dans le même env `devo`
  (`import devo` marche déjà ; DEVO a des extensions CUDA compilées).
- MS_Model **dépend de** `devo`.
- DEVO ne reçoit que des **hooks minimaux, rétro-compatibles** (arg optionnel `p_dyn`
  défaut `None` → comportement vanilla identique) → **DEVO vanilla préservé comme
  référence**.
- Le code de couplage (masque → score patch, boucle d'entraînement conjointe) vit dans
  **MS_Model**.
- Réconciliation exacte (submodules vs dépendance simple) : à confirmer.

## 10. Cible de publication

RA-L / IROS / ICRA, ou workshop CVPR/ICCV event-vision comme jalon intermédiaire. Le
tableau à 6 lignes + la thèse auto-supervisée suffisent pour un RA-L.

---

**Prochaine action : M0** — brancher le masque oracle EVIMO sur la score map et mesurer
le delta ATE. Une demi-journée de plomberie qui valide (ou tue) tout le reste.
