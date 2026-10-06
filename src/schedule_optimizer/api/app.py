"""API HTTP (FastAPI) au-dessus des scénarios d'édition.

Lancement : `schedule-optimizer serve` (voir cli.py). Si `frontend/dist` existe, l'UI
compilée est servie sur `/` ; en développement, Vite tourne à part et proxifie `/api`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from schedule_optimizer.api.service import MoveError, Scenario, Workspace
from schedule_optimizer.core.generator import GeneratorConfig

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


class GenerateRequest(BaseModel):
    n_flights: int = Field(200, ge=10, le=2000)
    banked: bool = False
    seed: int = 0


class MoveRequest(BaseModel):
    flight_id: str
    dep: int


def create_app(workspace: Workspace | None = None) -> FastAPI:
    ws = workspace or Workspace(os.environ.get("SCHEDOPT_DATA", "data"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ws.ensure_samples()
        yield

    app = FastAPI(title="Schedule optimizer", lifespan=lifespan)

    def get(instance_id: str) -> Scenario:
        try:
            return ws.scenario(instance_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    def flight(sc: Scenario, flight_id: str) -> int:
        try:
            return sc.index(flight_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/instances")
    def list_instances():
        return ws.list_instances()

    @app.post("/api/instances")
    def generate_instance(req: GenerateRequest):
        iid = ws.generate(
            GeneratorConfig(n_flights=req.n_flights, banked=req.banked, seed=req.seed)
        )
        return {"id": iid}

    @app.get("/api/instances/{instance_id}")
    def instance_info(instance_id: str):
        return get(instance_id).info()

    @app.get("/api/instances/{instance_id}/state")
    def state(instance_id: str):
        sc = get(instance_id)
        with sc.lock:
            return sc.snapshot()

    @app.get("/api/instances/{instance_id}/flights/{flight_id}")
    def flight_detail(instance_id: str, flight_id: str):
        sc = get(instance_id)
        with sc.lock:
            return sc.flight_detail(flight(sc, flight_id))

    @app.get("/api/instances/{instance_id}/region-pair")
    def region_pair(instance_id: str, origin: str, dest: str):
        sc = get(instance_id)
        with sc.lock:
            try:
                return sc.region_pair(origin, dest)
            except KeyError as e:
                raise HTTPException(404, str(e)) from e

    @app.post("/api/instances/{instance_id}/moves")
    def move(instance_id: str, req: MoveRequest):
        sc = get(instance_id)
        with sc.lock:
            try:
                return {"delta": sc.move(flight(sc, req.flight_id), req.dep)}
            except MoveError as e:
                raise HTTPException(409, str(e)) from e

    @app.delete("/api/instances/{instance_id}/moves/{flight_id}")
    def revert(instance_id: str, flight_id: str):
        sc = get(instance_id)
        with sc.lock:
            try:
                return {"delta": sc.revert(flight(sc, flight_id))}
            except MoveError as e:
                raise HTTPException(409, str(e)) from e

    @app.post("/api/instances/{instance_id}/reset")
    def reset(instance_id: str):
        sc = get(instance_id)
        with sc.lock:
            sc.reset()
        return {"ok": True}

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")

    return app
