"""Instance = programme + marchés, et contraintes par défaut associées."""

from __future__ import annotations

from dataclasses import dataclass

from schedule_optimizer.core.constraints import ConstraintSet, TimeWindow, Turnaround
from schedule_optimizer.core.markets import Markets
from schedule_optimizer.core.revenue import ChoiceParams, RevenueModel
from schedule_optimizer.core.schedule import Schedule


@dataclass(frozen=True, eq=False)
class Instance:
    schedule: Schedule
    markets: Markets
    name: str = ""


def default_constraints(
    instance: Instance,
    max_shift: int = 60,
    min_turn_hub: int = 45,
    min_turn_station: int = 35,
) -> ConstraintSet:
    s = instance.schedule
    return ConstraintSet(
        [
            TimeWindow.around(s, max_shift),
            Turnaround(s, min_turn_hub=min_turn_hub, min_turn_station=min_turn_station),
        ]
    )


def build_model(
    instance: Instance,
    constraints: ConstraintSet | None = None,
    params: ChoiceParams | None = None,
) -> RevenueModel:
    """Modèle de revenu élagué selon les fenêtres des contraintes (s'il y en a)."""
    windows = constraints.bounds() if constraints is not None else None
    return RevenueModel(instance.schedule, instance.markets, params=params, windows=windows)
