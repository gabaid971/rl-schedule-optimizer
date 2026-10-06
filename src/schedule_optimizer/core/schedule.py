"""Programme de vols mono-hub.

Chaque vol touche le hub : soit il en part (DEP, hub -> escale), soit il y arrive
(ARR, escale -> hub). Toutes les heures sont en minutes depuis minuit, dans une
référence horaire unique (celle du hub).

Le programme est stocké en colonnes numpy : c'est ce format que consomment le
modèle de revenu, les contraintes et les optimiseurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEP = 1  # départ du hub vers une escale
ARR = -1  # arrivée au hub depuis une escale

HUB = -1  # index « station » réservé au hub (utilisé par les marchés)


@dataclass(frozen=True)
class Station:
    code: str
    region: str


@dataclass(frozen=True, eq=False)
class Schedule:
    hub: str
    stations: tuple[Station, ...]
    flight_id: np.ndarray  # str
    direction: np.ndarray  # int8, DEP ou ARR
    station: np.ndarray  # int32, index dans `stations`
    dep: np.ndarray  # int32, heure de départ (min)
    block: np.ndarray  # int32, temps de vol (min)
    tail: np.ndarray  # str, immatriculation ("" si inconnue)
    seats: np.ndarray  # int32
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.flight_id)
        for name in ("direction", "station", "dep", "block", "tail", "seats"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"colonne {name!r} : {len(getattr(self, name))} != {n} vols")
        index = {fid: i for i, fid in enumerate(self.flight_id.tolist())}
        if len(index) != n:
            raise ValueError("flight_id doit être unique")
        if not np.isin(self.direction, (DEP, ARR)).all():
            raise ValueError("direction doit valoir DEP (1) ou ARR (-1)")
        object.__setattr__(self, "_index", index)

    @property
    def n_flights(self) -> int:
        return len(self.flight_id)

    @property
    def arr(self) -> np.ndarray:
        return self.dep + self.block

    @property
    def hub_time(self) -> np.ndarray:
        """Heure à laquelle le vol touche le hub (départ pour DEP, arrivée pour ARR)."""
        return np.where(self.direction == DEP, self.dep, self.arr)

    @property
    def region(self) -> np.ndarray:
        regions = np.array([s.region for s in self.stations], dtype=object)
        return regions[self.station]

    @property
    def station_code(self) -> np.ndarray:
        codes = np.array([s.code for s in self.stations], dtype=object)
        return codes[self.station]

    def index_of(self, flight_id: str) -> int:
        return self._index[flight_id]

    def with_dep(self, dep: np.ndarray) -> Schedule:
        """Copie du programme avec de nouvelles heures de départ."""
        dep = np.asarray(dep, dtype=np.int32)
        if dep.shape != self.dep.shape:
            raise ValueError("dimension de dep incompatible")
        return Schedule(
            hub=self.hub,
            stations=self.stations,
            flight_id=self.flight_id,
            direction=self.direction,
            station=self.station,
            dep=dep.copy(),
            block=self.block,
            tail=self.tail,
            seats=self.seats,
        )


@dataclass(frozen=True)
class Move:
    flight: int
    old_dep: int
    new_dep: int

    @property
    def shift(self) -> int:
        return self.new_dep - self.old_dep


def diff(before: Schedule, after: Schedule) -> list[Move]:
    """Liste des vols décalés entre deux versions d'un même programme."""
    if before.n_flights != after.n_flights or not np.array_equal(before.flight_id, after.flight_id):
        raise ValueError("les deux programmes ne portent pas sur les mêmes vols")
    changed = np.flatnonzero(before.dep != after.dep)
    return [Move(int(f), int(before.dep[f]), int(after.dep[f])) for f in changed]


def fmt_time(minutes: int) -> str:
    """480 -> '08:00', 1500 -> '01:00+1'."""
    day, m = divmod(int(minutes), 1440)
    suffix = f"+{day}" if day else ""
    return f"{m // 60:02d}:{m % 60:02d}{suffix}"
