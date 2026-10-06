"""Lecture / écriture d'une instance dans un dossier de CSV lisibles :

meta.json      {"hub": "...", "name": "..."}
stations.csv   code, region
flights.csv    flight_id, direction (DEP|ARR), station, dep_min, block_min, tail, seats
markets.csv    origin, destination, demand, fare   (le hub est désigné par son code)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from schedule_optimizer.core.instance import Instance
from schedule_optimizer.core.markets import Markets
from schedule_optimizer.core.schedule import ARR, DEP, HUB, Schedule, Station


def flights_frame(schedule: Schedule, dep: np.ndarray | None = None) -> pl.DataFrame:
    dep = schedule.dep if dep is None else dep
    return pl.DataFrame(
        {
            "flight_id": schedule.flight_id.tolist(),
            "direction": np.where(schedule.direction == DEP, "DEP", "ARR").tolist(),
            "station": schedule.station_code.tolist(),
            "dep_min": np.asarray(dep, dtype=np.int32),
            "block_min": schedule.block,
            "tail": schedule.tail.tolist(),
            "seats": schedule.seats,
        }
    )


def save_instance(instance: Instance, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    s, m = instance.schedule, instance.markets
    (folder / "meta.json").write_text(
        json.dumps({"hub": s.hub, "name": instance.name}, ensure_ascii=False, indent=2)
    )
    pl.DataFrame(
        {"code": [st.code for st in s.stations], "region": [st.region for st in s.stations]}
    ).write_csv(folder / "stations.csv")
    flights_frame(s).write_csv(folder / "flights.csv")

    codes = np.array([st.code for st in s.stations] + [s.hub], dtype=object)  # HUB = -1
    pl.DataFrame(
        {
            "origin": codes[m.origin].tolist(),
            "destination": codes[m.dest].tolist(),
            "demand": m.demand,
            "fare": m.fare,
        }
    ).write_csv(folder / "markets.csv")
    return folder


def load_instance(folder: str | Path) -> Instance:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    hub = meta["hub"]
    st_df = pl.read_csv(folder / "stations.csv")
    stations = tuple(Station(c, r) for c, r in st_df.iter_rows())
    st_index = {st.code: i for i, st in enumerate(stations)}
    if hub in st_index:
        raise ValueError(f"le hub {hub} ne doit pas figurer dans stations.csv")

    fl = pl.read_csv(folder / "flights.csv", schema_overrides={"tail": pl.String}).with_columns(
        pl.col("tail").fill_null("")
    )
    unknown = set(fl["station"].to_list()) - set(st_index)
    if unknown:
        raise ValueError(f"escales inconnues dans flights.csv : {sorted(unknown)}")
    direction = fl["direction"].to_numpy()
    if not np.isin(direction, ("DEP", "ARR")).all():
        raise ValueError("direction doit valoir DEP ou ARR")
    schedule = Schedule(
        hub=hub,
        stations=stations,
        flight_id=np.array(fl["flight_id"].cast(pl.String).to_list(), dtype=object),
        direction=np.where(direction == "DEP", DEP, ARR).astype(np.int8),
        station=np.array([st_index[c] for c in fl["station"].to_list()], dtype=np.int32),
        dep=fl["dep_min"].to_numpy().astype(np.int32),
        block=fl["block_min"].to_numpy().astype(np.int32),
        tail=np.array(fl["tail"].to_list(), dtype=object),
        seats=fl["seats"].to_numpy().astype(np.int32),
    )

    mk = pl.read_csv(folder / "markets.csv")
    code_index = {**st_index, hub: HUB}
    markets = Markets(
        origin=np.array([code_index[c] for c in mk["origin"].to_list()], dtype=np.int32),
        dest=np.array([code_index[c] for c in mk["destination"].to_list()], dtype=np.int32),
        demand=mk["demand"].to_numpy().astype(np.float64),
        fare=mk["fare"].to_numpy().astype(np.float64),
    )
    return Instance(schedule, markets, name=meta.get("name", folder.name))
