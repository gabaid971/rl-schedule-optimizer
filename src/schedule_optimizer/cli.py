"""Ligne de commande : `schedule-optimizer generate|evaluate ...`."""

from __future__ import annotations

import argparse
import time

import numpy as np

from schedule_optimizer.core.generator import GeneratorConfig, generate
from schedule_optimizer.core.instance import build_model, default_constraints
from schedule_optimizer.core.io import load_instance, save_instance


def _generate(args: argparse.Namespace) -> None:
    cfg = GeneratorConfig(n_flights=args.flights, banked=args.banked, seed=args.seed)
    instance = generate(cfg)
    out = save_instance(instance, args.out or f"data/{instance.name}")
    s = instance.schedule
    print(
        f"{instance.name} : {s.n_flights} vols, {len(s.stations)} escales, "
        f"{len(set(s.tail))} avions, {instance.markets.n_markets} marchés -> {out}"
    )


def _evaluate(args: argparse.Namespace) -> None:
    instance = load_instance(args.folder)
    constraints = default_constraints(instance, max_shift=args.max_shift)
    t0 = time.perf_counter()
    model = build_model(instance, constraints)
    t1 = time.perf_counter()
    bd = model.breakdown()
    t2 = time.perf_counter()

    s, m = instance.schedule, instance.markets
    local = m.is_local
    print(f"Instance       : {instance.name} ({s.n_flights} vols)")
    print(f"Revenu         : {bd.total:,.0f}  (potentiel {m.potential_revenue:,.0f})")
    print(f"  dont local   : {bd.market_revenue[local].sum():,.0f}")
    print(f"  dont cnx     : {bd.market_revenue[~local].sum():,.0f}")
    print(f"Passagers      : {bd.market_pax.sum():,.0f}  (sièges {s.seats.sum():,})")
    print(f"Coef. remplis. : {np.median(bd.flight_pax / s.seats):.0%} (médiane, sans capacité)")
    print(f"Itinéraires    : {model.n_itineraries:,}")
    violations = constraints.violations(s.dep)
    print(f"Violations     : {len(violations)}")
    for v in violations[:10]:
        print(f"  - [{v.constraint}] {v.message}")
    print(f"Temps          : modèle {1e3 * (t1 - t0):.0f} ms, évaluation {1e3 * (t2 - t1):.0f} ms")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "schedule_optimizer.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="schedule-optimizer")
    sub = parser.add_subparsers(required=True)

    g = sub.add_parser("generate", help="générer une instance synthétique")
    g.add_argument("--flights", type=int, default=200)
    g.add_argument("--banked", action="store_true", help="horaires calés sur des vagues")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", help="dossier de sortie (défaut : data/<nom>)")
    g.set_defaults(func=_generate)

    e = sub.add_parser("evaluate", help="évaluer le revenu d'une instance")
    e.add_argument("folder")
    e.add_argument("--max-shift", type=int, default=60)
    e.set_defaults(func=_evaluate)

    s = sub.add_parser("serve", help="lancer l'API (et l'UI compilée si présente)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
