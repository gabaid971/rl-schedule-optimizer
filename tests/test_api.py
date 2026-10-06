import pytest
from fastapi.testclient import TestClient

from schedule_optimizer.api.app import create_app
from schedule_optimizer.api.service import Workspace
from schedule_optimizer.core.generator import GeneratorConfig


@pytest.fixture
def client(tmp_path):
    ws = Workspace(tmp_path)
    ws.generate(GeneratorConfig(n_flights=120, seed=1))
    with TestClient(create_app(ws)) as c:
        yield c


def test_state_and_move_roundtrip(client):
    (inst,) = client.get("/api/instances").json()
    iid = inst["id"]
    info = client.get(f"/api/instances/{iid}").json()
    assert info["n_flights"] == 120

    state = client.get(f"/api/instances/{iid}/state").json()
    k = state["kpis"]
    assert k["revenue"] == pytest.approx(k["base_revenue"])
    assert k["moved"] == 0 and k["violations"] == 0
    assert sum(state["hub_profile"]["dep"]) + sum(state["hub_profile"]["arr"]) == 120

    f = state["flights"][10]
    detail = client.get(f"/api/instances/{iid}/flights/{f['id']}").json()
    best = max((o for o in detail["options"] if o["allowed"]), key=lambda o: o["delta"])

    r = client.post(f"/api/instances/{iid}/moves", json={"flight_id": f["id"], "dep": best["dep"]})
    assert r.status_code == 200
    assert r.json()["delta"] == pytest.approx(best["delta"], abs=0.1)

    state = client.get(f"/api/instances/{iid}/state").json()
    if best["dep"] != f["dep"]:
        assert state["kpis"]["moved"] == 1
        (mv,) = state["moves"]
        assert mv["id"] == f["id"] and mv["marginal"] == pytest.approx(best["delta"], abs=1)
    assert state["kpis"]["revenue"] - state["kpis"]["base_revenue"] == pytest.approx(
        best["delta"], abs=0.1
    )

    assert client.delete(f"/api/instances/{iid}/moves/{f['id']}").status_code == 200
    assert client.get(f"/api/instances/{iid}/state").json()["kpis"]["moved"] == 0


def test_forbidden_move_is_rejected(client):
    iid = client.get("/api/instances").json()[0]["id"]
    f = client.get(f"/api/instances/{iid}/state").json()["flights"][0]
    r = client.post(f"/api/instances/{iid}/moves", json={"flight_id": f["id"], "dep": f["hi"] + 5})
    assert r.status_code == 409
    assert client.get(f"/api/instances/{iid}/flights/nope").status_code == 404
    assert client.get("/api/instances/..%2Fetc/state").status_code == 404


def test_region_pair(client):
    iid = client.get("/api/instances").json()[0]["id"]
    info = client.get(f"/api/instances/{iid}").json()
    state = client.get(f"/api/instances/{iid}/state").json()
    rm = state["region_matrix"]
    # paire de régions la plus rentable
    i, j = max(
        ((i, j) for i in range(len(rm["regions"])) for j in range(len(rm["regions"]))),
        key=lambda ij: rm["revenue"][ij[0]][ij[1]],
    )
    pair = client.get(
        f"/api/instances/{iid}/region-pair",
        params={"origin": info["regions"][i], "dest": info["regions"][j]},
    ).json()
    assert pair["totals"]["revenue"] == pytest.approx(rm["revenue"][i][j], abs=1)
    assert pair["markets"] and pair["connections"]
