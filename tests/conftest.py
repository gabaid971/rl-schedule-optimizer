import numpy as np
import pytest

from schedule_optimizer.core.generator import GeneratorConfig, generate
from schedule_optimizer.core.instance import Instance
from schedule_optimizer.core.markets import Markets
from schedule_optimizer.core.schedule import ARR, DEP, HUB, Schedule, Station


@pytest.fixture(scope="session")
def small_instance() -> Instance:
    return generate(GeneratorConfig(n_flights=120, seed=1))


def make_schedule(flights: list[tuple[str, int, int, int, int, str]]) -> Schedule:
    """flights : (id, direction, escale, départ, temps de vol, avion)."""
    stations = (Station("AAA", "R1"), Station("BBB", "R2"), Station("CCC", "R2"))
    cols = list(zip(*flights))
    return Schedule(
        hub="HUB",
        stations=stations,
        flight_id=np.array(cols[0], dtype=object),
        direction=np.array(cols[1], dtype=np.int8),
        station=np.array(cols[2], dtype=np.int32),
        dep=np.array(cols[3], dtype=np.int32),
        block=np.array(cols[4], dtype=np.int32),
        tail=np.array(cols[5], dtype=object),
        seats=np.full(len(flights), 180, dtype=np.int32),
    )


@pytest.fixture
def tiny_instance() -> Instance:
    """AAA -> HUB (arrive 09:00) puis HUB -> BBB / HUB -> CCC."""
    schedule = make_schedule(
        [
            ("A1", ARR, 0, 480, 60, "T1"),  # AAA 08:00 -> HUB 09:00
            ("B1", DEP, 1, 615, 90, "T1"),  # HUB 10:15 -> BBB
            ("C1", DEP, 2, 615, 90, "T2"),  # HUB 10:15 -> CCC
        ]
    )
    markets = Markets(
        origin=np.array([0, HUB, HUB, 0, 0], dtype=np.int32),
        dest=np.array([HUB, 1, 2, 1, 2], dtype=np.int32),
        demand=np.array([100.0, 100.0, 100.0, 50.0, 50.0]),
        fare=np.array([100.0, 150.0, 150.0, 200.0, 200.0]),
    )
    return Instance(schedule, markets, name="tiny")
