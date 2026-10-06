"""Générateur d'instances synthétiques mono-hub.

Construction :
1. des escales réparties par région (temps de vol selon la région) et une « taille »
   (log-normale) qui pilote fréquence et demande ;
2. des rotations hub -> escale -> hub, étalées sur la journée (ou calées sur des
   vagues si `banked=True`) ;
3. une affectation gloutonne des rotations aux avions, par flotte, qui garantit des
   temps de demi-tour faisables ;
4. des marchés locaux et en correspondance (modèle gravitaire), calibrés pour que
   la demande totale soit `demand_factor` fois l'offre en sièges.
"""

from __future__ import annotations

import heapq
import string
from dataclasses import dataclass

import numpy as np

from schedule_optimizer.core.instance import Instance
from schedule_optimizer.core.markets import Markets
from schedule_optimizer.core.schedule import ARR, DEP, HUB, Schedule, Station

# région : (temps de vol min, max, poids dans le réseau)
REGIONS: dict[str, tuple[int, int, float]] = {
    "Domestique": (55, 80, 0.20),
    "Europe Ouest": (80, 120, 0.20),
    "Europe Nord": (100, 150, 0.15),
    "Europe Est": (120, 170, 0.15),
    "Méditerranée": (120, 180, 0.15),
    "Afrique du Nord": (150, 210, 0.08),
    "Moyen-Orient": (240, 320, 0.07),
}
WIDEBODY_MIN_BLOCK = 211  # au-delà : flotte gros porteur
SEATS = {"narrow": 180, "wide": 280}
BANKS = tuple(range(7 * 60, 21 * 60 + 1, 120))  # heures de départ des vagues au hub


@dataclass(frozen=True)
class GeneratorConfig:
    n_flights: int = 200
    n_stations: int | None = None  # défaut : ~1 escale pour 4 rotations
    hub: str = "HUB"
    first_dep: int = 6 * 60
    last_arr: int = 23 * 60 + 30
    min_turn_hub: int = 45
    min_turn_station: int = 35
    banked: bool = False
    night_stop_share: float = 0.6  # proba qu'une escale moyen-courrier ait un night-stop
    demand_factor: float = 1.3
    local_share: float = 0.55  # part de la demande sur les marchés locaux
    seed: int = 0


def _round5(x: float) -> int:
    return int(5 * round(x / 5))


def generate(config: GeneratorConfig | None = None) -> Instance:
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n_rot = max(1, cfg.n_flights // 2)
    n_st = min(n_rot, cfg.n_stations or max(3, round(n_rot / 4)))

    # 1. escales
    names = list(REGIONS)
    weights = np.array([REGIONS[r][2] for r in names])
    regions = rng.choice(names, size=n_st, p=weights / weights.sum())
    blocks = np.array([_round5(rng.uniform(*REGIONS[r][:2])) for r in regions])
    size = rng.lognormal(0.0, 0.8, size=n_st)
    codes = _station_codes(rng, n_st, exclude={cfg.hub})
    stations = tuple(Station(c, str(r)) for c, r in zip(codes, regions))

    # 2. fréquences (au moins une rotation par escale) et horaires. Une partie des
    # escales moyen-courrier a un « night-stop » : un avion y dort, arrive au hub le
    # matin (première vague d'arrivées) et y retourne le soir.
    freq = 1 + rng.multinomial(n_rot - n_st, size / size.sum())
    rotations = []  # (escale, départ hub, départ escale)
    night_in = []  # (escale, départ escale le matin)
    night_out = []  # (escale, départ hub le soir)
    for s in range(n_st):
        b = int(blocks[s])
        n_day = int(freq[s])
        if b < WIDEBODY_MIN_BLOCK and rng.random() < cfg.night_stop_share:
            n_day -= 1
            if cfg.banked:
                arr_hub = BANKS[1] - 60 + _round5(rng.uniform(-20, 10))
                t_eve = BANKS[-1] - 120 + _round5(rng.uniform(0, 150))
            else:
                arr_hub = cfg.first_dep + 60 + _round5(rng.uniform(0, 150))
                t_eve = 19 * 60 + _round5(rng.uniform(0, 150))
            night_in.append((s, arr_hub - b))
            night_out.append((s, t_eve))

        latest = cfg.last_arr - 2 * b - cfg.min_turn_station
        latest = max(latest, cfg.first_dep)
        for i in range(n_day):
            if cfg.banked:
                bank = BANKS[rng.integers(len(BANKS))]
                t_out = bank + _round5(rng.uniform(0, 30))
            else:
                t_out = cfg.first_dep + (i + 0.5) * (latest - cfg.first_dep) / n_day
                t_out = _round5(t_out + rng.normal(0, 30))
            t_out = int(np.clip(t_out, cfg.first_dep, latest))
            ground = cfg.min_turn_station + _round5(rng.uniform(0, 45))
            if cfg.banked:
                # retour au hub ~1h avant une vague de départs
                arr_hub = t_out + 2 * b + ground
                target = next((k - 60 for k in BANKS if k - 60 >= arr_hub), arr_hub)
                ground += min(target - arr_hub, 90)
            rotations.append((s, t_out, t_out + b + ground))

    # 3. affectation aux avions, par flotte. Les avions en night-stop deviennent
    # disponibles au hub à leur arrivée du matin ; ensuite, glouton par heure de départ :
    # on réutilise un avion disponible au hub, sinon on en ajoute un (basé au hub).
    free: dict[str, list[tuple[int, int]]] = {"narrow": [], "wide": []}  # (dispo hub, avion)
    n_tails = 0
    rows = []  # (num, direction, escale, départ, temps de vol, avion, sièges)

    def fleet_of(s: int) -> str:
        return "wide" if blocks[s] >= WIDEBODY_MIN_BLOCK else "narrow"

    def tail_code(fleet: str, tail: int) -> str:
        return f"F-{fleet[0].upper()}{tail:03d}"

    def take_tail(fleet: str, t: int) -> int:
        nonlocal n_tails
        heap = free[fleet]
        if heap and heap[0][0] + cfg.min_turn_hub <= t:
            return heapq.heappop(heap)[1]
        n_tails += 1
        return n_tails - 1

    num = iter(range(1000, 100000))
    for s, t_dep in night_in:
        b, fleet = int(blocks[s]), fleet_of(s)
        tail, n_tails = n_tails, n_tails + 1
        heapq.heappush(free[fleet], (t_dep + b, tail))
        rows.append((next(num), ARR, s, t_dep, b, tail_code(fleet, tail), SEATS[fleet]))

    # tâches qui consomment un avion au hub : rotations et départs du soir
    tasks = [(t_out, s, t_in) for s, t_out, t_in in rotations]
    tasks += [(t_eve, s, None) for s, t_eve in night_out]
    for t_out, s, t_in in sorted(tasks, key=lambda x: (x[0], x[1])):
        b, fleet = int(blocks[s]), fleet_of(s)
        tail = take_tail(fleet, t_out)
        rows.append((next(num), DEP, s, t_out, b, tail_code(fleet, tail), SEATS[fleet]))
        if t_in is not None:
            heapq.heappush(free[fleet], (t_in + b, tail))
            rows.append((next(num), ARR, s, t_in, b, tail_code(fleet, tail), SEATS[fleet]))

    rows.sort(key=lambda r: (r[3], r[0]))
    schedule = Schedule(
        hub=cfg.hub,
        stations=stations,
        flight_id=np.array([f"HB{r[0]}" for r in rows], dtype=object),
        direction=np.array([r[1] for r in rows], dtype=np.int8),
        station=np.array([r[2] for r in rows], dtype=np.int32),
        dep=np.array([r[3] for r in rows], dtype=np.int32),
        block=np.array([r[4] for r in rows], dtype=np.int32),
        tail=np.array([r[5] for r in rows], dtype=object),
        seats=np.array([r[6] for r in rows], dtype=np.int32),
    )

    markets = _markets(rng, cfg, schedule, regions, blocks, size)
    tag = "banked" if cfg.banked else "spread"
    return Instance(schedule, markets, name=f"synth_{schedule.n_flights}_{tag}_s{cfg.seed}")


def _station_codes(rng: np.random.Generator, n: int, exclude: set[str]) -> list[str]:
    codes: list[str] = []
    seen = set(exclude)
    letters = list(string.ascii_uppercase)
    while len(codes) < n:
        c = "".join(rng.choice(letters, 3))
        if c not in seen:
            seen.add(c)
            codes.append(c)
    return codes


def _markets(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    schedule: Schedule,
    regions: np.ndarray,
    blocks: np.ndarray,
    size: np.ndarray,
) -> Markets:
    n_st = len(size)
    total = cfg.demand_factor * float(schedule.seats.sum())

    # locaux : escale <-> hub, proportionnels à la taille de l'escale
    st = np.arange(n_st)
    local_d = size / size.sum() * total * cfg.local_share / 2
    local_fare = 60 + 0.8 * blocks

    # correspondances : gravitaire, atténué entre escales d'une même région
    o, d = np.meshgrid(st, st, indexing="ij")
    o, d = o.ravel(), d.ravel()
    keep = o != d
    o, d = o[keep], d[keep]
    grav = size[o] * size[d] * np.where(regions[o] == regions[d], 0.15, 1.0)
    cnx_d = grav / grav.sum() * total * (1 - cfg.local_share)
    cnx_fare = 50 + 0.6 * (blocks[o] + blocks[d])

    origin = np.concatenate([st, np.full(n_st, HUB), o])
    dest = np.concatenate([np.full(n_st, HUB), st, d])
    demand = np.concatenate([local_d, local_d, cnx_d])
    fare = np.concatenate([local_fare, local_fare, cnx_fare])
    fare = np.round(fare * rng.uniform(0.9, 1.1, size=len(fare)))
    return Markets(
        origin=origin.astype(np.int32),
        dest=dest.astype(np.int32),
        demand=demand,
        fare=fare,
    )
