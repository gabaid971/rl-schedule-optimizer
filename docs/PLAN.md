# Plan du projet

## Objectif

1. **Évaluer** le revenu d'un programme de vols (modèle volontairement simple au départ).
2. **Optimiser** les heures de départ pour maximiser ce revenu sous contraintes, via RL,
   et **comparer** honnêtement à des méthodes de recherche opérationnelle (OR).
3. **Visualiser et éditer** le programme dans une webapp : vues agrégées et par paire de
   régions ou route, décalage d'un vol avec impact immédiat sur le revenu, et diff des
   changements proposés par un optimiseur.

La question de recherche centrale : *à budget de calcul égal, et sur des programmes jamais
vus, une politique RL rivalise-t-elle avec le recuit simulé, le LNS et un solveur exact ?*

## Décisions

| Sujet | Décision |
|---|---|
| Réseau | Mono-hub : chaque vol part du hub ou y arrive |
| Taille | Jusqu'à 1000 vols par jour (on baissera si c'est trop lourd) |
| Données | Synthétiques d'abord ; source réelle envisagée : BTS (voir plus bas) |
| Contraintes | Ajoutées une à une, via une interface commune |
| Backend | Python 3.12, uv, numpy (calcul), polars (E/S), FastAPI (API) |
| Frontend | Vite + React + TypeScript, TanStack Query, Tailwind + shadcn/ui, ECharts pour les agrégats, timeline SVG maison pour le drag & drop, types générés depuis l'OpenAPI |
| OR | OR-Tools (CP-SAT) pour l'exact, heuristiques maison (recuit, LNS) |
| RL | Gymnasium, SB3 / sb3-contrib, puis PyTorch Geometric pour une politique en graphe |

## Architecture

```
src/schedule_optimizer/
  core/          modèle métier, partagé par tous (optimiseurs, API, UI)
    schedule     programme (colonnes numpy), Move, diff
    markets      marchés O&D (demande, tarif)
    revenue      RevenueModel (évaluation complète, ventilation) et RevenueState (Δ incrémental)
    constraints  Constraint : violations() pour l'UI, allowed() pour les masques et voisinages
    generator    instances synthétiques
    io           dossier de CSV (meta, stations, flights, markets)
  optimizers/    interface optimize(instance, constraints, budget) -> Result(dep, moves, history)
  rl/            environnement et politiques
  bench/         protocole de comparaison
  api/           FastAPI
frontend/        React
legacy/          prototype v0 (référence uniquement, à supprimer quand le RL v2 existera)
```

Principe : **un seul modèle de revenu et un seul jeu de contraintes**. Le chiffre affiché
quand on décale un vol dans l'UI est exactement celui que maximisent les optimiseurs.

## Modèle de revenu v1

Il s'agit d'un logit par marché et par segment horaire, avec une option « no-go »
(voir `core/revenue.py`) :

- **Marchés.** Les marchés locaux (escale ↔ hub) sont servis par les vols directs. Les
  marchés en correspondance (escale A → escale B) sont servis par toute paire
  arrivée/départ dont le temps de correspondance est dans [MCT, max].
- **Utilité d'un itinéraire.** Elle dépend de son type (direct ou correspondance), de
  l'écart à l'heure de départ préférée du segment (matin, midi, soir) et, pour une
  correspondance, de la qualité du temps au hub (trop court = risque, trop long = attente).
- **Part captée.** `A / (A + e^u_nogo)`. Elle a un **rendement décroissant**, donc
  l'optimum n'est pas trivial, contrairement au prototype v0 où chaque correspondance
  rapportait sans plafond.
- **Ventilation.** Revenu par marché, par itinéraire et par vol (une correspondance est
  répartie au prorata du temps de vol). C'est ce qui alimente l'UI.
- **Coût.** `RevenueState.delta(vol, heure)` coûte environ 66 µs à 1000 vols, contre 6 ms
  pour une évaluation complète.

Pistes v2 : capacité des vols (débordement, ou *spill*), concurrence par marché,
calibration des paramètres.

## Contraintes

- [x] Fenêtre de décalage ±Δ autour du programme initial (`TimeWindow`)
- [x] Rotations avion et temps de demi-tour minimal au hub et en escale (`Turnaround`)
- [ ] Couvre-feux et heures d'ouverture par aéroport
- [ ] Capacité de créneaux au hub (mouvements par tranche de 15 min)
- [ ] Pénalité ou plafond sur le nombre de vols déplacés (en objectif)
- [ ] Fréquences minimales, écart minimal entre deux vols d'une même route

## Phases

### Phase 0 : nettoyage ✅
L'ancien code a été déplacé dans `legacy/`. Les doublons, `setup.py`, `requirements.txt`
et le Dockerfile cassé ont été supprimés. Les dépendances passent toutes par
`pyproject.toml` et uv ; pytest et ruff sont en place.

### Phase 1 : modèle métier ✅ (v1)
Programme, marchés, revenu logit incrémental, contraintes, générateur, E/S, CLI et tests.
Le test clé vérifie que le Δ incrémental égale la différence de deux évaluations complètes.

Reste à faire :
- importer BTS (voir *Données réelles*) ;
- calibrer les paramètres. Aujourd'hui, un programme en vagues rapporte moins qu'un
  programme étalé, car la préférence horaire domine.

### Phase 2 : baselines OR et banc de test
- **Heuristiques** : recherche aléatoire, hill-climbing, recuit simulé, LNS (on réoptimise
  une banque ou une région à la fois). Toutes utilisent `RevenueState` et `allowed()`.
- **Exact** : CP-SAT avec heures discrétisées (pas de 5 min dans ±Δ) et part captée
  concave linéarisée par morceaux. Cible : optimum ou borne sur les petites instances,
  pour mesurer un **gap**.
- **Protocole** :
  - instances S (≈20 vols), M (≈200) et L (≈1000), plusieurs seeds ;
  - budgets égaux en temps et en nombre d'évaluations ;
  - métriques : gain %, gap à la meilleure solution connue, nombre de vols déplacés,
    temps de calcul ;
  - jeu de test séparé (instances jamais vues).

### Phase 3 : RL v2
1. **Environnement propre.**
   - Action factorisée : choisir un vol, puis un décalage, avec masque issu de `allowed()`.
   - Reward : Δrevenu, moins une pénalité par vol déplacé.
   - Fin d'épisode par troncature, observation compacte.
   - On garde le *meilleur* état visité, pas l'état final.
2. **Vérification de base** sur de petites instances où l'on connaît l'optimum.
3. **Politique qui généralise.** Un GNN sur le graphe vols/correspondances, entraîné sur
   une distribution d'instances, puis appliqué sans réentraînement (rollout glouton et
   échantillonnage).
4. **Hybride.** La politique propose les mouvements d'un LNS ou d'un recuit.

### Phase 4 : API et UI (MVP) ✅ (première version)
Livré : API FastAPI (`src/schedule_optimizer/api`) et UI React (`frontend/`) avec vue hub,
programme (Gantt par avion, escale ou vol, avec drag & drop), paires de régions, liste des
changements, panneau vol (courbe Δrevenu selon l'heure de départ, correspondances) et
explication du modèle. Scénarios en mémoire seulement (perdus au redémarrage).

Spécification d'origine :
- **API.** Instances et scénarios, évaluation et ventilation, simulation d'un décalage
  (what-if), violations de contraintes, heures autorisées pour un vol.
- **Vue globale.**
  - Indicateurs clés : revenu, passagers, vols déplacés.
  - Histogramme des arrivées et départs au hub par tranche de 15 min (structure en vagues).
  - Heatmap région × région du revenu ou de son Δ.
- **Vue paire de régions ou route.** Arrivées de A face aux départs vers B, matrice de
  correspondances colorée par temps de correspondance et par revenu.
- **Vue vol.**
  - Drag & drop sur la grille de 5 min, limité aux heures autorisées.
  - Δrevenu en direct, correspondances gagnées et perdues.
- **Montée en charge.** Agrégation côté serveur, jamais les N² correspondances côté
  client, rendu canvas et listes virtualisées.

### Phase 5 : résultats des optimiseurs dans l'UI
- **Scénarios versionnés** : base, RL, OR, édition manuelle. Comparaison de deux scénarios
  quelconques.
- **Vue diff** : vols déplacés (avant → après, Δrevenu marginal), filtres par région, et
  positions d'origine affichées en fantôme sur la timeline. Les contributions marginales
  ne s'additionnent pas : l'UI doit l'indiquer.
- **Optimisation lancée depuis l'UI**, en tâche asynchrone avec convergence en direct (SSE).

### Phase 6 : étude
Tableaux et courbes RL, heuristiques et exact par taille d'instance, robustesse,
conclusions.

## Données réelles (option)

Les fichiers SSIM réels sont payants (OAG, Cirium). Alternatives publiques et gratuites :

- **BTS On-Time Performance** (transtats.bts.gov) : horaires programmés de chaque vol
  domestique US, avec l'immatriculation, donc les rotations. Une journée d'une compagnie
  sur son hub (par exemple United à DEN ou IAH) représente environ 1000 vols.
  - À traiter : conversion des heures locales vers l'heure du hub, filtrage des vols qui
    ne touchent pas le hub.
- **BTS DB1B Market** : échantillon de 10 % des billets, qui donne une demande O&D et des
  tarifs réels.
