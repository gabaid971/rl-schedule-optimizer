import numpy as np
import pytest

from schedule_optimizer.core.instance import build_model, default_constraints
from schedule_optimizer.core.revenue import ChoiceParams, RevenueModel


def test_itineraries_tiny(tiny_instance):
    model = RevenueModel(tiny_instance.schedule, tiny_instance.markets)
    # 3 directs + 2 correspondances (AAA->BBB, AAA->CCC)
    assert model.n_itineraries == 5
    cnx = model.it_leg2 >= 0
    assert cnx.sum() == 2
    np.testing.assert_array_equal(
        model.connection_time(model.schedule.dep.astype(int), np.flatnonzero(cnx)), [75, 75]
    )


def test_connection_outside_mct_is_not_sold(tiny_instance):
    p = ChoiceParams(mct=45, max_cnx=360)
    model = RevenueModel(tiny_instance.schedule, tiny_instance.markets, params=p)
    dep = tiny_instance.schedule.dep.astype(np.int64).copy()
    cnx_mkts = [3, 4]

    dep[1] = 540 + 44  # B1 part 44 min après l'arrivée de A1 : < MCT
    bd = model.breakdown(dep)
    assert bd.market_pax[3] == 0
    dep[1] = 540 + 45
    assert model.breakdown(dep).market_pax[3] > 0
    dep[1] = 540 + 361
    assert model.breakdown(dep).market_pax[3] == 0
    assert model.breakdown(dep).market_pax[cnx_mkts[1]] > 0  # C1 non touché


def test_diminishing_returns(tiny_instance):
    """Une 2e offre sur le même marché rapporte moins que la 1re (pas de dégénérescence)."""
    model = RevenueModel(tiny_instance.schedule, tiny_instance.markets)
    dep = tiny_instance.schedule.dep.astype(np.int64)
    its = np.arange(model.n_itineraries)
    a = model.attractiveness(dep, its)[0:1]  # un itinéraire
    one = model.revenue_from_sums(a, model.value[:1])
    two = model.revenue_from_sums(2 * a, model.value[:1])
    assert 0 < two - one < one


def test_revenue_bounded_by_potential(small_instance):
    model = build_model(small_instance)
    bd = model.breakdown()
    assert 0 < bd.total < small_instance.markets.potential_revenue
    assert bd.total == pytest.approx(model.evaluate())
    assert bd.flight_revenue.sum() == pytest.approx(bd.total)
    assert bd.market_revenue.sum() == pytest.approx(bd.total)


def test_pruning_with_windows_is_exact(small_instance):
    constraints = default_constraints(small_instance, max_shift=60)
    full = build_model(small_instance)
    pruned = build_model(small_instance, constraints)
    assert pruned.n_itineraries < full.n_itineraries
    rng = np.random.default_rng(0)
    lo, hi = constraints.bounds()
    for _ in range(5):
        dep = rng.integers(lo, hi + 1)
        assert pruned.evaluate(dep) == pytest.approx(full.evaluate(dep))


def test_incremental_delta_matches_full_evaluation(small_instance):
    constraints = default_constraints(small_instance, max_shift=60)
    model = build_model(small_instance, constraints)
    lo, hi = constraints.bounds()
    state = model.state()
    rng = np.random.default_rng(42)
    for _ in range(300):
        f = int(rng.integers(small_instance.schedule.n_flights))
        t = int(rng.integers(lo[f], hi[f] + 1))
        before = model.evaluate(state.dep)
        predicted = state.delta(f, t)
        assert state.revenue == pytest.approx(before)  # delta n'applique rien
        applied = state.apply(f, t)
        after = model.evaluate(state.dep)
        assert predicted == pytest.approx(applied)
        assert applied == pytest.approx(after - before, abs=1e-6)
    assert state.revenue == pytest.approx(model.evaluate(state.dep))
