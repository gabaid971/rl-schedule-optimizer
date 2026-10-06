// Types et appels de l'API FastAPI (src/schedule_optimizer/api).

export type Direction = "DEP" | "ARR";

export interface Flight {
  i: number;
  id: string;
  direction: Direction;
  station: string;
  region: string;
  dep: number;
  arr: number;
  base_dep: number;
  block: number;
  tail: string;
  seats: number;
  pax: number;
  revenue: number;
  base_revenue: number;
  lo: number;
  hi: number;
}

export interface Kpis {
  revenue: number;
  base_revenue: number;
  potential: number;
  local_revenue: number;
  cnx_revenue: number;
  base_local_revenue: number;
  base_cnx_revenue: number;
  pax: number;
  base_pax: number;
  cnx_pax: number;
  base_cnx_pax: number;
  moved: number;
  violations: number;
}

export interface MoveInfo {
  i: number;
  id: string;
  base_dep: number;
  dep: number;
  shift: number;
  marginal: number;
  revertible: boolean;
}

export interface Violation {
  constraint: string;
  flights: string[];
  message: string;
}

export interface HubProfile {
  start: number;
  bin: number;
  dep: number[];
  arr: number[];
  base_dep: number[];
  base_arr: number[];
}

export interface RegionMatrix {
  regions: string[];
  revenue: number[][];
  base_revenue: number[][];
  pax: number[][];
  base_pax: number[][];
}

export interface ScenarioState {
  kpis: Kpis;
  flights: Flight[];
  moves: MoveInfo[];
  violations: Violation[];
  hub_profile: HubProfile;
  region_matrix: RegionMatrix;
}

export interface ChoiceParams {
  mct: number;
  max_cnx: number;
  ideal_cnx: number;
  asc_direct: number;
  asc_cnx: number;
  beta_short: number;
  beta_wait: number;
  beta_delay: number;
  u_nogo: number;
  segments: [number, number][];
}

export interface InstanceInfo {
  id: string;
  name: string;
  hub: string;
  n_flights: number;
  n_tails: number;
  n_markets: number;
  n_itineraries: number;
  regions: string[];
  stations: { code: string; region: string }[];
  max_shift: number;
  grid: number;
  params: ChoiceParams;
  constraints: { name: string; description: string }[];
}

export interface FlightOption {
  dep: number;
  allowed: boolean;
  delta: number;
}

export interface Connection {
  other: string;
  other_i: number;
  other_station: string;
  market: string;
  cnx: number;
  sellable: boolean;
  pax: number;
  revenue: number;
}

export interface FlightDetail {
  i: number;
  id: string;
  dep: number;
  base_dep: number;
  options: FlightOption[];
  connections: Connection[];
  local: { market: string; pax: number; revenue: number; demand: number; fare: number } | null;
  prev: string | null;
  next: string | null;
}

export interface PairMarket {
  market: string;
  demand: number;
  fare: number;
  pax: number;
  base_pax: number;
  revenue: number;
  base_revenue: number;
  n_cnx: number;
}

export interface PairData {
  origin: string;
  dest: string;
  arrivals: number[];
  departures: number[];
  markets: PairMarket[];
  connections: { a: number; d: number; cnx: number; pax: number }[];
  totals: { revenue: number; base_revenue: number; pax: number; base_pax: number; demand: number };
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = (await r.json()).detail ?? msg;
    } catch {
      /* corps non JSON */
    }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

const base = (iid: string) => `/api/instances/${encodeURIComponent(iid)}`;

export const api = {
  instances: () => req<{ id: string; n_flights: number }[]>("/api/instances"),
  generate: (body: { n_flights: number; banked: boolean; seed: number }) =>
    req<{ id: string }>("/api/instances", { method: "POST", body: JSON.stringify(body) }),
  info: (iid: string) => req<InstanceInfo>(base(iid)),
  state: (iid: string) => req<ScenarioState>(`${base(iid)}/state`),
  flight: (iid: string, fid: string) =>
    req<FlightDetail>(`${base(iid)}/flights/${encodeURIComponent(fid)}`),
  pair: (iid: string, origin: string, dest: string) =>
    req<PairData>(
      `${base(iid)}/region-pair?origin=${encodeURIComponent(origin)}&dest=${encodeURIComponent(dest)}`,
    ),
  move: (iid: string, flight_id: string, dep: number) =>
    req<{ delta: number }>(`${base(iid)}/moves`, {
      method: "POST",
      body: JSON.stringify({ flight_id, dep }),
    }),
  revert: (iid: string, fid: string) =>
    req<{ delta: number }>(`${base(iid)}/moves/${encodeURIComponent(fid)}`, { method: "DELETE" }),
  reset: (iid: string) => req<{ ok: boolean }>(`${base(iid)}/reset`, { method: "POST" }),
};
