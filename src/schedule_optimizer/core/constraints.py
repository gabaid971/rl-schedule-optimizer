"""Contraintes dures sur les heures de départ.

Chaque contrainte expose deux opérations :
- `violations(dep)` : vérifie un programme complet (UI, validation de résultats) ;
- `allowed(dep, flight, candidates)` : filtre les heures candidates d'un vol, les
  autres vols restant fixes (masque d'actions RL, voisinage des heuristiques).

Pour ajouter une contrainte : sous-classer `Constraint` et l'ajouter au `ConstraintSet`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from schedule_optimizer.core.schedule import DEP, Schedule, fmt_time


@dataclass(frozen=True)
class Violation:
    constraint: str
    flights: tuple[int, ...]
    message: str


class Constraint(ABC):
    name: str

    @abstractmethod
    def violations(self, dep: np.ndarray) -> list[Violation]: ...

    @abstractmethod
    def allowed(self, dep: np.ndarray, flight: int, candidates: np.ndarray) -> np.ndarray: ...


class TimeWindow(Constraint):
    """Chaque vol doit partir dans [lo, hi]."""

    name = "fenetre"

    def __init__(self, schedule: Schedule, lo: np.ndarray, hi: np.ndarray) -> None:
        self.schedule = schedule
        self.lo = np.asarray(lo, dtype=np.int64)
        self.hi = np.asarray(hi, dtype=np.int64)
        if (self.lo > self.hi).any():
            raise ValueError("fenêtre vide pour au moins un vol")

    @classmethod
    def around(
        cls,
        schedule: Schedule,
        max_shift: int,
        earliest: int | None = None,
        latest: int | None = None,
    ) -> TimeWindow:
        """Décalage d'au plus ±max_shift minutes autour du programme d'origine,
        éventuellement borné par une heure de départ au plus tôt / au plus tard."""
        lo = schedule.dep.astype(np.int64) - max_shift
        hi = schedule.dep.astype(np.int64) + max_shift
        if earliest is not None:
            lo = np.maximum(lo, np.minimum(earliest, schedule.dep))
        if latest is not None:
            hi = np.minimum(hi, np.maximum(latest, schedule.dep))
        return cls(schedule, lo, hi)

    def violations(self, dep: np.ndarray) -> list[Violation]:
        bad = np.flatnonzero((dep < self.lo) | (dep > self.hi))
        ids = self.schedule.flight_id
        return [
            Violation(
                self.name,
                (int(f),),
                f"{ids[f]} part à {fmt_time(dep[f])}, hors fenêtre "
                f"[{fmt_time(self.lo[f])}, {fmt_time(self.hi[f])}]",
            )
            for f in bad
        ]

    def allowed(self, dep: np.ndarray, flight: int, candidates: np.ndarray) -> np.ndarray:
        return (candidates >= self.lo[flight]) & (candidates <= self.hi[flight])


class Turnaround(Constraint):
    """Un avion enchaîne ses vols dans l'ordre du programme initial, avec un temps
    de demi-tour minimal au hub et en escale.

    Les rotations sont déduites de la colonne `tail`. Deux vols consécutifs d'un même
    avion ne sont liés que s'ils sont géographiquement continus (le second part d'où
    le premier arrive) ; sinon la chaîne est coupée (données partielles).
    """

    name = "rotation"

    def __init__(
        self, schedule: Schedule, min_turn_hub: int = 45, min_turn_station: int = 35
    ) -> None:
        self.schedule = schedule
        n = schedule.n_flights
        self.prev = np.full(n, -1, dtype=np.int64)
        self.next = np.full(n, -1, dtype=np.int64)
        # temps de demi-tour après le vol f, là où il arrive
        self.turn = np.where(schedule.direction == DEP, min_turn_station, min_turn_hub)
        self.block = schedule.block.astype(np.int64)

        has_tail = schedule.tail != ""
        order = np.lexsort((schedule.dep, schedule.tail))
        order = order[has_tail[order]]
        for p, f in zip(order[:-1], order[1:]):
            if schedule.tail[p] != schedule.tail[f]:
                continue
            if schedule.direction[p] == schedule.direction[f]:
                continue  # deux départs (ou deux arrivées) de suite : vol manquant
            if schedule.direction[p] == DEP and schedule.station[p] != schedule.station[f]:
                continue  # part vers A, revient de B : vol manquant
            self.prev[f], self.next[p] = p, f

    def ready_time(self, dep: np.ndarray, flight: int) -> int:
        """Heure à partir de laquelle l'avion peut repartir après `flight`."""
        return int(dep[flight] + self.block[flight] + self.turn[flight])

    def violations(self, dep: np.ndarray) -> list[Violation]:
        f = np.flatnonzero(self.prev >= 0)
        p = self.prev[f]
        ready = dep[p] + self.block[p] + self.turn[p]
        bad = dep[f] < ready
        ids = self.schedule.flight_id
        return [
            Violation(
                self.name,
                (int(pi), int(fi)),
                f"{ids[fi]} part à {fmt_time(dep[fi])} mais l'avion n'est prêt qu'à "
                f"{fmt_time(ri)} (après {ids[pi]})",
            )
            for pi, fi, ri in zip(p[bad], f[bad], ready[bad])
        ]

    def allowed(self, dep: np.ndarray, flight: int, candidates: np.ndarray) -> np.ndarray:
        ok = np.ones(len(candidates), dtype=bool)
        p, n = self.prev[flight], self.next[flight]
        if p >= 0:
            ok &= candidates >= self.ready_time(dep, p)
        if n >= 0:
            ok &= candidates + self.block[flight] + self.turn[flight] <= dep[n]
        return ok


class ConstraintSet:
    def __init__(self, constraints: list[Constraint]) -> None:
        self.constraints = list(constraints)

    def violations(self, dep: np.ndarray) -> list[Violation]:
        dep = np.asarray(dep, dtype=np.int64)
        return [v for c in self.constraints for v in c.violations(dep)]

    def is_feasible(self, dep: np.ndarray) -> bool:
        return not self.violations(dep)

    def allowed(self, dep: np.ndarray, flight: int, candidates: np.ndarray) -> np.ndarray:
        dep = np.asarray(dep, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        ok = np.ones(len(candidates), dtype=bool)
        for c in self.constraints:
            ok &= c.allowed(dep, flight, candidates)
        return ok

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Bornes globales des heures de départ (pour élaguer le modèle de revenu)."""
        windows = [c for c in self.constraints if isinstance(c, TimeWindow)]
        if not windows:
            return None
        lo = np.max([w.lo for w in windows], axis=0)
        hi = np.min([w.hi for w in windows], axis=0)
        return lo, hi
