"""Décalages successifs : chaque vol voit l'état courant (revenu et rotations)."""

import numpy as np
import pytest

from schedule_optimizer.api.service import Scenario
from schedule_optimizer.core.generator import GeneratorConfig, generate


@pytest.fixture
def scenario() -> Scenario:
    return Scenario("t", generate(GeneratorConfig(n_flights=200, seed=3)), max_shift=60)


def test_options_reflect_previous_move_on_connected_flight(scenario):
    m = scenario.model
    # une correspondance vendue arrivée a -> départ d, où a peut bouger
    for it in np.flatnonzero((m.it_leg2 >= 0) & (scenario.state.attr.sum(axis=1) > 0)):
        a, d = int(m.it_leg1[it]), int(m.it_leg2[it])
        cands = scenario.candidates(a)
        ok = cands[scenario.constraints.allowed(scenario.state.dep, a, cands)]
        ok = ok[ok != scenario.state.dep[a]]
        if len(ok):
            break
    before = {o["dep"]: o["delta"] for o in scenario.flight_detail(d)["options"]}
    scenario.move(a, int(ok[-1]))

    dep = scenario.state.dep.copy()
    current = m.evaluate(dep)
    after = scenario.flight_detail(d)["options"]
    for o in after:
        if not o["allowed"]:
            continue
        trial = dep.copy()
        trial[d] = o["dep"]
        # la courbe de d est calculée avec a déjà déplacé
        assert o["delta"] == pytest.approx(m.evaluate(trial) - current, abs=0.1)
    assert [o["delta"] for o in after] != list(before.values())


def test_rotation_bounds_follow_previous_move(scenario):
    t = scenario.turnaround
    s = scenario.schedule
    # vol f ayant un suivant n dans sa rotation, et pouvant partir plus tard
    for f in np.flatnonzero(t.next >= 0):
        cands = scenario.candidates(f)
        ok = scenario.constraints.allowed(scenario.state.dep, f, cands)
        later = cands[(cands > scenario.state.dep[f]) & ok]
        if len(later):
            break
    n = int(t.next[f])
    scenario.move(int(f), int(later[-1]))

    ready = scenario.state.dep[f] + s.block[f] + t.turn[f]  # avion prêt pour n
    for o in scenario.flight_detail(n)["options"]:
        if o["dep"] < ready:
            assert not o["allowed"]  # trop tôt : l'avion n'est pas revenu
    with pytest.raises(ValueError):
        scenario.move(n, int(ready) - 5)
    # et f ne peut plus revenir après n
    assert scenario.constraints.is_feasible(scenario.state.dep)
