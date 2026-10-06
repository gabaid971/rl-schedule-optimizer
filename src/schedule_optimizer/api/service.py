"""Scénarios d'édition côté serveur.

Un scénario = une instance + son programme de référence + un programme courant
modifiable (vols décalés à la main, plus tard par un optimiseur). Tout est en
mémoire ; les sorties sont des dicts prêts à sérialiser en JSON.
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from schedule_optimizer.core.constraints import ConstraintSet, TimeWindow, Turnaround
from schedule_optimizer.core.generator import GeneratorConfig, generate
from schedule_optimizer.core.instance import Instance, build_model, default_constraints
from schedule_optimizer.core.io import load_instance, save_instance
from schedule_optimizer.core.revenue import RevenueBreakdown
from schedule_optimizer.core.schedule import DEP, diff, fmt_time

GRID = 5  # pas des décalages (min)
PROFILE_BIN = 15  # largeur des tranches du profil hub (min)
NEAR_MISS = 60  # on montre aussi les correspondances ratées de moins de 60 min
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class MoveError(ValueError):
    pass


class Scenario:
    def __init__(self, instance_id: str, instance: Instance, max_shift: int) -> None:
        self.id = instance_id
        self.instance = instance
        self.max_shift = max_shift
        self.constraints: ConstraintSet = default_constraints(instance, max_shift=max_shift)
        self.model = build_model(instance, self.constraints)
        self.lock = threading.Lock()

        s = instance.schedule
        self.base_dep = s.dep.astype(np.int64)
        self.base = self.model.breakdown(self.base_dep)
        self.state = self.model.state(self.base_dep)
        self.lo, self.hi = self.constraints.bounds()
        self.turnaround = next(c for c in self.constraints.constraints if isinstance(c, Turnaround))

        self.regions = sorted({st.region for st in s.stations})
        region_idx = {r: i for i, r in enumerate(self.regions)}
        self.station_region = np.array([region_idx[st.region] for st in s.stations])
        self.flight_region = self.station_region[s.station]
        self.codes = np.array([st.code for st in s.stations] + [s.hub], dtype=object)  # HUB=-1

    # ------------------------------------------------------------ helpers
    @property
    def schedule(self):
        return self.instance.schedule

    def index(self, flight_id: str) -> int:
        try:
            return self.schedule.index_of(flight_id)
        except KeyError as e:
            raise KeyError(f"vol inconnu : {flight_id}") from e

    def _breakdown(self) -> RevenueBreakdown:
        return self.model.breakdown(self.state.dep)

    def _market_label(self, m: int) -> str:
        mk = self.instance.markets
        return f"{self.codes[mk.origin[m]]}→{self.codes[mk.dest[m]]}"

    def candidates(self, f: int) -> np.ndarray:
        c = self.base_dep[f] + np.arange(-self.max_shift, self.max_shift + 1, GRID)
        return c[(c >= self.lo[f]) & (c <= self.hi[f])]

    # ------------------------------------------------------------- vues
    def info(self) -> dict:
        s, mk = self.schedule, self.instance.markets
        return {
            "id": self.id,
            "name": self.instance.name,
            "hub": s.hub,
            "n_flights": s.n_flights,
            "n_tails": len(set(s.tail.tolist())),
            "n_markets": mk.n_markets,
            "n_itineraries": self.model.n_itineraries,
            "regions": self.regions,
            "stations": [{"code": st.code, "region": st.region} for st in s.stations],
            "max_shift": self.max_shift,
            "grid": GRID,
            "params": asdict(self.model.params),
            "constraints": [
                {"name": c.name, "description": _describe(c)} for c in self.constraints.constraints
            ],
        }

    def snapshot(self) -> dict:
        s, mk = self.schedule, self.instance.markets
        bd = self._breakdown()
        dep = self.state.dep
        arr = dep + s.block

        flights = [
            {
                "i": i,
                "id": s.flight_id[i],
                "direction": "DEP" if s.direction[i] == DEP else "ARR",
                "station": self.codes[s.station[i]],
                "region": self.regions[self.flight_region[i]],
                "dep": int(dep[i]),
                "arr": int(arr[i]),
                "base_dep": int(self.base_dep[i]),
                "block": int(s.block[i]),
                "tail": s.tail[i],
                "seats": int(s.seats[i]),
                "pax": round(float(bd.flight_pax[i]), 1),
                "revenue": round(float(bd.flight_revenue[i])),
                "base_revenue": round(float(self.base.flight_revenue[i])),
                "lo": int(self.lo[i]),
                "hi": int(self.hi[i]),
            }
            for i in range(s.n_flights)
        ]

        moves = []
        for mv in diff(s, s.with_dep(dep)):
            f = mv.flight
            revert_ok = bool(self.constraints.allowed(dep, f, [self.base_dep[f]])[0])
            moves.append(
                {
                    "i": f,
                    "id": s.flight_id[f],
                    "base_dep": mv.old_dep,
                    "dep": mv.new_dep,
                    "shift": mv.shift,
                    # ce que l'on perdrait en annulant ce seul décalage
                    "marginal": round(-self.state.delta(f, mv.old_dep)),
                    "revertible": revert_ok,
                }
            )

        violations = [
            {
                "constraint": v.constraint,
                "flights": [s.flight_id[f] for f in v.flights],
                "message": v.message,
            }
            for v in self.constraints.violations(dep)
        ]

        local = mk.is_local
        return {
            "kpis": {
                "revenue": bd.total,
                "base_revenue": self.base.total,
                "potential": mk.potential_revenue,
                "local_revenue": float(bd.market_revenue[local].sum()),
                "cnx_revenue": float(bd.market_revenue[~local].sum()),
                "base_local_revenue": float(self.base.market_revenue[local].sum()),
                "base_cnx_revenue": float(self.base.market_revenue[~local].sum()),
                "pax": float(bd.market_pax.sum()),
                "base_pax": float(self.base.market_pax.sum()),
                "cnx_pax": float(bd.market_pax[~local].sum()),
                "base_cnx_pax": float(self.base.market_pax[~local].sum()),
                "moved": len(moves),
                "violations": len(violations),
            },
            "flights": flights,
            "moves": moves,
            "violations": violations,
            "hub_profile": self._hub_profile(dep),
            "region_matrix": self._region_matrix(bd),
        }

    def _hub_profile(self, dep: np.ndarray) -> dict:
        s = self.schedule
        is_dep = s.direction == DEP
        cur = np.where(is_dep, dep, dep + s.block)
        base = np.where(is_dep, self.base_dep, self.base_dep + s.block)
        start = int(min(cur.min(), base.min()) // PROFILE_BIN * PROFILE_BIN)
        n = int((max(cur.max(), base.max()) - start) // PROFILE_BIN + 1)

        def count(t: np.ndarray, mask: np.ndarray) -> list[int]:
            return np.bincount((t[mask] - start) // PROFILE_BIN, minlength=n).tolist()

        return {
            "start": start,
            "bin": PROFILE_BIN,
            "dep": count(cur, is_dep),
            "arr": count(cur, ~is_dep),
            "base_dep": count(base, is_dep),
            "base_arr": count(base, ~is_dep),
        }

    def _region_matrix(self, bd: RevenueBreakdown) -> dict:
        """Revenu des correspondances agrégé par (région d'origine, région de destination)."""
        mk = self.instance.markets
        n_r = len(self.regions)
        cnx = ~mk.is_local
        key = self.station_region[mk.origin[cnx]] * n_r + self.station_region[mk.dest[cnx]]

        def agg(values: np.ndarray) -> list[list[float]]:
            m = np.bincount(key, weights=values[cnx], minlength=n_r * n_r).reshape(n_r, n_r)
            return np.round(m).tolist()

        return {
            "regions": self.regions,
            "revenue": agg(bd.market_revenue),
            "base_revenue": agg(self.base.market_revenue),
            "pax": agg(bd.market_pax),
            "base_pax": agg(self.base.market_pax),
        }

    def flight_detail(self, f: int) -> dict:
        s, mk, m = self.schedule, self.instance.markets, self.model
        dep = self.state.dep
        bd = self._breakdown()

        cands = self.candidates(f)
        allowed = self.constraints.allowed(dep, f, cands)
        options = [
            {"dep": int(t), "allowed": bool(ok), "delta": round(self.state.delta(f, int(t)), 1)}
            for t, ok in zip(cands, allowed)
        ]

        its = m.itineraries_of(f)
        cnx_its = its[m.it_leg2[its] >= 0]
        cnx_time = m.connection_time(dep, cnx_its)
        p = m.params
        keep = (cnx_time >= p.mct - NEAR_MISS) & (cnx_time <= p.max_cnx)
        connections = []
        for it, ct in zip(cnx_its[keep], cnx_time[keep]):
            other = int(m.it_leg2[it] if m.it_leg1[it] == f else m.it_leg1[it])
            connections.append(
                {
                    "other": s.flight_id[other],
                    "other_i": other,
                    "other_station": self.codes[s.station[other]],
                    "market": self._market_label(int(m.it_market[it])),
                    "cnx": int(ct),
                    "sellable": bool(p.mct <= ct <= p.max_cnx),
                    "pax": round(float(bd.itinerary_pax[it]), 2),
                    "revenue": round(float(bd.itinerary_revenue[it])),
                }
            )
        connections.sort(key=lambda c: (-c["pax"], c["cnx"]))

        direct = its[m.it_leg2[its] < 0]
        local = None
        if len(direct):
            it = int(direct[0])
            local = {
                "market": self._market_label(int(m.it_market[it])),
                "pax": round(float(bd.itinerary_pax[it]), 1),
                "revenue": round(float(bd.itinerary_revenue[it])),
                "demand": float(mk.demand[m.it_market[it]]),
                "fare": float(mk.fare[m.it_market[it]]),
            }

        prev, nxt = int(self.turnaround.prev[f]), int(self.turnaround.next[f])
        return {
            "i": f,
            "id": s.flight_id[f],
            "dep": int(dep[f]),
            "base_dep": int(self.base_dep[f]),
            "options": options,
            "connections": connections,
            "local": local,
            "prev": s.flight_id[prev] if prev >= 0 else None,
            "next": s.flight_id[nxt] if nxt >= 0 else None,
        }

    def region_pair(self, origin: str, dest: str) -> dict:
        if origin not in self.regions or dest not in self.regions:
            raise KeyError("région inconnue")
        s, mk, m = self.schedule, self.instance.markets, self.model
        ro, rd = self.regions.index(origin), self.regions.index(dest)
        bd = self._breakdown()

        arrivals = np.flatnonzero((s.direction != DEP) & (self.flight_region == ro))
        departures = np.flatnonzero((s.direction == DEP) & (self.flight_region == rd))

        local = mk.is_local
        o_reg = np.where(local, -1, self.station_region[np.where(local, 0, mk.origin)])
        d_reg = np.where(local, -1, self.station_region[np.where(local, 0, mk.dest)])
        mkts = np.flatnonzero((o_reg == ro) & (d_reg == rd))

        in_pair = np.isin(m.it_market, mkts)
        sellable = in_pair & (self.state.attr.sum(axis=1) > 0)
        n_sellable = np.bincount(m.it_market[sellable], minlength=mk.n_markets)
        markets = [
            {
                "market": self._market_label(int(k)),
                "demand": round(float(mk.demand[k]), 1),
                "fare": float(mk.fare[k]),
                "pax": round(float(bd.market_pax[k]), 1),
                "base_pax": round(float(self.base.market_pax[k]), 1),
                "revenue": round(float(bd.market_revenue[k])),
                "base_revenue": round(float(self.base.market_revenue[k])),
                "n_cnx": int(n_sellable[k]),
            }
            for k in mkts
        ]
        markets.sort(key=lambda x: -x["revenue"])

        sel = np.flatnonzero(sellable)
        cnx_time = m.connection_time(self.state.dep, sel)
        connections = [
            {
                "a": int(m.it_leg1[it]),
                "d": int(m.it_leg2[it]),
                "cnx": int(ct),
                "pax": round(float(bd.itinerary_pax[it]), 2),
            }
            for it, ct in zip(sel, cnx_time)
        ]
        return {
            "origin": origin,
            "dest": dest,
            "arrivals": arrivals.tolist(),
            "departures": departures.tolist(),
            "markets": markets,
            "connections": connections,
            "totals": {
                "revenue": float(bd.market_revenue[mkts].sum()),
                "base_revenue": float(self.base.market_revenue[mkts].sum()),
                "pax": float(bd.market_pax[mkts].sum()),
                "base_pax": float(self.base.market_pax[mkts].sum()),
                "demand": float(mk.demand[mkts].sum()),
            },
        }

    # ---------------------------------------------------------- édition
    def move(self, f: int, new_dep: int) -> float:
        if new_dep == self.state.dep[f]:
            return 0.0
        if not self.constraints.allowed(self.state.dep, f, [new_dep])[0]:
            raise MoveError(
                f"{self.schedule.flight_id[f]} ne peut pas partir à {fmt_time(new_dep)} "
                f"(fenêtre ou rotation)"
            )
        return self.state.apply(f, new_dep)

    def revert(self, f: int) -> float:
        return self.move(f, int(self.base_dep[f]))

    def reset(self) -> None:
        self.state = self.model.state(self.base_dep)


def _describe(c) -> str:
    if isinstance(c, TimeWindow):
        return "Départ dans une fenêtre autour de l'horaire initial"
    if isinstance(c, Turnaround):
        return "Rotations avion respectées, avec temps de demi-tour minimal"
    return c.name


class Workspace:
    """Instances disponibles (un dossier par instance) et scénarios ouverts."""

    def __init__(self, data_dir: str | Path, max_shift: int = 60) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_shift = max_shift
        self._scenarios: dict[str, Scenario] = {}
        self._lock = threading.Lock()

    def ensure_samples(self) -> None:
        if not self.list_instances():
            for n in (200, 1000):
                self.generate(GeneratorConfig(n_flights=n, seed=0))

    def list_instances(self) -> list[dict]:
        out = []
        for d in sorted(self.data_dir.iterdir()):
            if (d / "meta.json").exists() and _ID_RE.match(d.name):
                n = sum(1 for _ in (d / "flights.csv").open()) - 1
                out.append({"id": d.name, "n_flights": n})
        return out

    def generate(self, cfg: GeneratorConfig) -> str:
        instance = generate(cfg)
        save_instance(instance, self.data_dir / instance.name)
        with self._lock:
            self._scenarios.pop(instance.name, None)
        return instance.name

    def scenario(self, instance_id: str) -> Scenario:
        folder = self.data_dir / instance_id
        if not _ID_RE.match(instance_id) or not (folder / "meta.json").exists():
            raise KeyError(f"instance inconnue : {instance_id}")
        with self._lock:
            if instance_id not in self._scenarios:
                inst = load_instance(self.data_dir / instance_id)
                self._scenarios[instance_id] = Scenario(instance_id, inst, self.max_shift)
            return self._scenarios[instance_id]
