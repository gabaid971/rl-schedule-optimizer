# rl-schedule-optimizer

Évaluation et optimisation (RL contre recherche opérationnelle) des heures de départ d'un
programme de vols mono-hub. Voir [docs/PLAN.md](docs/PLAN.md).

## Installation

```bash
uv sync              # cœur + outils de dev
uv sync --extra rl   # + gymnasium / stable-baselines3 / torch
```

## Interface web

```bash
./scripts/dev.sh            # API + UI en développement -> http://localhost:5173
```

Ou sans Node en développement, une fois l'UI compilée (`cd frontend && npm install && npm run build`) :

```bash
uv run schedule-optimizer serve    # API + UI compilée -> http://localhost:8000
```

Au premier lancement, deux instances synthétiques (200 et 1000 vols) sont créées dans `data/`.

## Utilisation en Python et en ligne de commande

```bash
uv run schedule-optimizer generate --flights 1000 --seed 0   # -> data/synth_1000_spread_s0
uv run schedule-optimizer evaluate data/synth_1000_spread_s0
uv run pytest
```

```python
from schedule_optimizer.core.generator import GeneratorConfig, generate
from schedule_optimizer.core.instance import build_model, default_constraints

inst = generate(GeneratorConfig(n_flights=200))
constraints = default_constraints(inst, max_shift=60)
model = build_model(inst, constraints)

state = model.state()
f = 0
print(state.revenue, state.delta(f, int(state.dep[f]) + 15))
```

## Format d'une instance

Un dossier contient :
- `meta.json` : le code du hub ;
- `stations.csv` : code et région de chaque escale ;
- `flights.csv` : identifiant, direction `DEP`/`ARR` par rapport au hub, escale, départ et
  temps de vol en minutes, immatriculation, sièges ;
- `markets.csv` : origine, destination, demande journalière et tarif.

Le prototype initial (2023) est conservé dans `legacy/` pour référence.
