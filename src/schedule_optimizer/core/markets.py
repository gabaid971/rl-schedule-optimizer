"""Marchés origine-destination (O&D).

Un marché est une paire (origine, destination) avec une demande journalière et un
tarif moyen. En mono-hub il y a deux familles :
- marchés locaux : escale <-> hub, servis par un vol direct ;
- marchés en correspondance : escale A -> escale B, servis via le hub.
Le hub est codé par l'index HUB (-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from schedule_optimizer.core.schedule import HUB


@dataclass(frozen=True, eq=False)
class Markets:
    origin: np.ndarray  # int32, index de station ou HUB
    dest: np.ndarray  # int32, index de station ou HUB
    demand: np.ndarray  # float64, passagers / jour
    fare: np.ndarray  # float64, tarif moyen
    _index: dict[tuple[int, int], int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = {(int(o), int(d)): m for m, (o, d) in enumerate(zip(self.origin, self.dest))}
        if len(index) != len(self.origin):
            raise ValueError("marché (origine, destination) en double")
        if (self.origin == self.dest).any():
            raise ValueError("origine == destination")
        object.__setattr__(self, "_index", index)

    @property
    def n_markets(self) -> int:
        return len(self.origin)

    @property
    def is_local(self) -> np.ndarray:
        return (self.origin == HUB) | (self.dest == HUB)

    def get(self, origin: int, dest: int) -> int:
        """Index du marché, ou -1 s'il n'existe pas."""
        return self._index.get((origin, dest), -1)

    @property
    def potential_revenue(self) -> float:
        """Revenu si toute la demande était captée (borne supérieure)."""
        return float(np.sum(self.demand * self.fare))
