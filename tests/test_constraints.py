import numpy as np

from schedule_optimizer.core.constraints import ConstraintSet, TimeWindow, Turnaround
from schedule_optimizer.core.generator import GeneratorConfig, generate
from schedule_optimizer.core.instance import default_constraints
from schedule_optimizer.core.schedule import ARR, DEP
from tests.conftest import make_schedule


def test_window_around():
    s = make_schedule([("X", DEP, 0, 600, 60, ""), ("Y", DEP, 1, 400, 60, "")])
    w = TimeWindow.around(s, 30, earliest=380)
    np.testing.assert_array_equal(w.lo, [570, 380])
    np.testing.assert_array_equal(w.hi, [630, 430])
    assert w.violations(np.array([600, 400])) == []
    assert len(w.violations(np.array([631, 379]))) == 2
    np.testing.assert_array_equal(
        w.allowed(s.dep, 0, np.array([560, 600, 640])), [False, True, False]
    )


def test_turnaround_rotation_links():
    s = make_schedule(
        [
            ("OUT", DEP, 0, 480, 60, "T1"),  # HUB 08:00 -> AAA 09:00
            ("BACK", ARR, 0, 600, 60, "T1"),  # AAA 10:00 -> HUB 11:00
            ("NEXT", DEP, 1, 720, 90, "T1"),  # HUB 12:00 -> BBB
            ("ODD", ARR, 2, 900, 90, "T1"),  # revient de CCC : chaîne coupée
        ]
    )
    t = Turnaround(s, min_turn_hub=45, min_turn_station=35)
    np.testing.assert_array_equal(t.prev, [-1, 0, 1, -1])
    assert t.violations(s.dep.astype(np.int64)) == []

    # BACK ne peut pas partir avant 09:00 + 35 min, ni après NEXT - 45 - 60
    cands = np.array([574, 575, 615, 616])
    np.testing.assert_array_equal(t.allowed(s.dep, 1, cands), [False, True, True, False])

    dep = s.dep.astype(np.int64).copy()
    dep[2] = 700  # avion arrivé à 11:00, demi-tour 45 min -> pas avant 11:45
    (v,) = t.violations(dep)
    assert v.flights == (1, 2)


def test_generated_instances_are_feasible():
    for banked in (False, True):
        for seed in range(3):
            inst = generate(GeneratorConfig(n_flights=300, banked=banked, seed=seed))
            cs = default_constraints(inst)
            assert cs.violations(inst.schedule.dep) == []
            # chaque avion forme une chaîne continue
            t = next(c for c in cs.constraints if isinstance(c, Turnaround))
            n_tails = len(set(inst.schedule.tail))
            assert (t.prev < 0).sum() == n_tails


def test_allowed_moves_keep_schedule_feasible(small_instance):
    cs = default_constraints(small_instance, max_shift=60)
    dep = small_instance.schedule.dep.astype(np.int64).copy()
    rng = np.random.default_rng(0)
    shifts = np.arange(-60, 61, 5)
    for _ in range(500):
        f = int(rng.integers(len(dep)))
        cands = dep[f] + shifts
        ok = cs.allowed(dep, f, cands)
        if ok.any():
            dep[f] = rng.choice(cands[ok])
    assert cs.violations(dep) == []


def test_constraint_set_bounds_intersects_windows(small_instance):
    s = small_instance.schedule
    cs = ConstraintSet([TimeWindow.around(s, 60), TimeWindow.around(s, 30)])
    lo, hi = cs.bounds()
    np.testing.assert_array_equal(hi - lo, np.full(s.n_flights, 60))
    assert ConstraintSet([]).bounds() is None
    assert ARR != DEP
