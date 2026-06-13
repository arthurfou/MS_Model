# MS_Model — Motion Segmentation Model (à développer)

Dossier actuellement vide. C'est le prochain chantier de développement après la mise
à jour de cette documentation. Voir le briefing global dans
[`../CLAUDE.md`](../CLAUDE.md).

## Objectif

Modèles PyTorch (simples au départ) de **motion segmentation sur données
événementielles**, produisant le masque/score « dynamique » consommé par DEVO :

- **Jalon 1** : masque binaire ou score continu par patch (résolution score map
  DEVO, H/4×W/4), utilisé pour annuler la score map de `DEVO/devo/selector.py` aux
  positions d'objets mobiles. À prototyper d'abord avec un masque oracle/factice
  (GT EVIMO) pour valider la plomberie d'intégration avant de coder le vrai modèle.
- **Jalon 2** : sortie compatible avec une pondération `ω` continue par patch pour
  `DEVO/devo/ba.py`, encodant statique vs dynamique selon la cohérence au mouvement
  rigide dominant.

## Contraintes

- Pas de GT au déploiement → viser non/auto-supervisé, ou entraînement en simu
  (ESIM/EVIMO2) + augmentations sim-to-real.
- Préserver la sparsité de DEVO : opérer au niveau patch/score map, pas de réseau de
  segmentation dense lourd.
- Garder ce module isolable de DEVO (interface claire en entrée/sortie) pour pouvoir
  comparer "DEVO vanilla" vs "DEVO + MS_Model" proprement.

## Données disponibles

- **EVIMO2** (`/home/arthur/IPAL/datasets/evimo_dataset/`, et copies filtrées dans
  `EVOwithMS/datasets/EV-IMO*`) : GT depth + segmentation par objet, utile pour
  supervision/évaluation du module de seg.
- ESIM/TartanAir (via DEVO) pour simulation si besoin de plus de données.

## Environnement

Conda env : `devo` (réutilisé, pas de nouvel env dédié — évite de dupliquer
torch/CUDA et simplifie l'intégration jalon 1 avec DEVO).

## État

Vide — structure du repo et architecture du premier modèle encore à définir
(en pause, à reprendre dans une prochaine session).
