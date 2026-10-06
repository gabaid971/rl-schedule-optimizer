import numpy as np
import pytest

from schedule_optimizer.core.instance import build_model
from schedule_optimizer.core.io import load_instance, save_instance
from schedule_optimizer.core.schedule import diff, fmt_time


def test_roundtrip(small_instance, tmp_path):
    save_instance(small_instance, tmp_path / "inst")
    loaded = load_instance(tmp_path / "inst")
    a, b = small_instance.schedule, loaded.schedule
    assert loaded.name == small_instance.name
    assert a.stations == b.stations
    for col in ("flight_id", "direction", "station", "dep", "block", "tail", "seats"):
        np.testing.assert_array_equal(getattr(a, col), getattr(b, col))
    np.testing.assert_allclose(small_instance.markets.demand, loaded.markets.demand)
    assert build_model(loaded).evaluate() == pytest.approx(build_model(small_instance).evaluate())


def test_diff_and_fmt(small_instance):
    s = small_instance.schedule
    dep = s.dep.copy()
    dep[[3, 7]] += [15, -20]
    moves = diff(s, s.with_dep(dep))
    assert [(m.flight, m.shift) for m in moves] == [(3, 15), (7, -20)]
    assert fmt_time(485) == "08:05"
    assert fmt_time(1500) == "01:00+1"
