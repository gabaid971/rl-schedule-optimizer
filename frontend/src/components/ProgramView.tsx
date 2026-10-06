import { useMemo, useState } from "react";
import type { Flight, InstanceInfo } from "../api";
import { fmtNum } from "../format";
import { PageHeader, SearchInput, Segmented, Select, Switch } from "../ui";
import { type Row, Timeline } from "./Timeline";

type Grouping = "tail" | "station" | "flight";
const MAX_ROWS = 600;

/** Répartit des vols en lignes sans chevauchement (partitionnement d'intervalles). */
function packLanes(ids: number[], flights: Flight[]): number[][] {
  const lanes: { end: number; items: number[] }[] = [];
  for (const i of [...ids].sort((a, b) => flights[a].dep - flights[b].dep)) {
    const f = flights[i];
    const lane = lanes.find((l) => l.end + 5 <= Math.min(f.dep, f.base_dep));
    const end = Math.max(f.dep, f.base_dep) + f.block;
    if (lane) {
      lane.items.push(i);
      lane.end = end;
    } else lanes.push({ end, items: [i] });
  }
  return lanes.map((l) => l.items);
}

function timeRange(flights: Flight[]) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const f of flights) {
    lo = Math.min(lo, f.lo);
    hi = Math.max(hi, f.hi + f.block);
  }
  return [Math.floor(lo / 60) * 60, Math.ceil(hi / 60) * 60] as const;
}

const hubTime = (f: Flight) => (f.direction === "DEP" ? f.dep : f.arr);

export function ProgramView({
  iid,
  info,
  flights,
  selected,
  onSelect,
}: {
  iid: string;
  info: InstanceInfo;
  flights: Flight[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [grouping, setGrouping] = useState<Grouping>("tail");
  const [region, setRegion] = useState("all");
  const [direction, setDirection] = useState("all");
  const [search, setSearch] = useState("");
  const [movedOnly, setMovedOnly] = useState(false);
  const [t0, t1] = useMemo(() => timeRange(flights), [flights]);

  const { rows, matched, dimmed } = useMemo(() => {
    const q = search.trim().toUpperCase();
    const match = (f: Flight) =>
      (region === "all" || f.region === region) &&
      (direction === "all" || f.direction === direction) &&
      (!movedOnly || f.dep !== f.base_dep) &&
      (!q || f.id.toUpperCase().includes(q) || f.station.includes(q) || f.tail.toUpperCase().includes(q));
    const matchedIds = flights.filter(match).map((f) => f.i);
    const matchedSet = new Set(matchedIds);
    const dimmed = new Set<number>();
    const rows: Row[] = [];

    if (grouping === "flight") {
      for (const i of [...matchedIds].sort((a, b) => hubTime(flights[a]) - hubTime(flights[b]))) {
        const f = flights[i];
        rows.push({ key: f.id, label: f.id, sublabel: f.station, flights: [i] });
      }
    } else {
      // par avion : toute la rotation reste visible (vols hors filtre estompés)
      const keyOf = (f: Flight) => (grouping === "tail" ? f.tail : f.station);
      const keys = new Set(matchedIds.map((i) => keyOf(flights[i])));
      const groups = new Map<string, number[]>();
      for (const f of flights) {
        const k = keyOf(f);
        if (!keys.has(k)) continue;
        if (grouping === "station" && !matchedSet.has(f.i)) continue;
        if (!matchedSet.has(f.i)) dimmed.add(f.i);
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k)!.push(f.i);
      }
      const first = (ids: number[]) => Math.min(...ids.map((i) => flights[i].dep));
      const ordered = [...groups.entries()].sort((a, b) =>
        grouping === "tail" ? first(a[1]) - first(b[1]) : a[0].localeCompare(b[0]),
      );
      for (const [k, ids] of ordered) {
        packLanes(ids, flights).forEach((lane, n) =>
          rows.push({
            key: `${k}-${n}`,
            label: n === 0 ? k : "",
            sublabel: n === 0 && grouping === "station" ? flights[ids[0]].region : undefined,
            flights: lane,
            groupStart: n === 0,
          }),
        );
      }
    }
    return { rows, matched: matchedIds.length, dimmed };
  }, [flights, grouping, region, direction, search, movedOnly]);

  return (
    <div>
      <PageHeader title="Programme">
        <span className="mr-1 text-[12px] text-muted tabular">
          {fmtNum(matched)} vols · {fmtNum(rows.length)} lignes
        </span>
      </PageHeader>

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <SearchInput placeholder="Vol, escale ou avion" value={search} onChange={(e) => setSearch(e.target.value)} />
        <Select
          label="Région"
          value={region}
          onChange={setRegion}
          options={[{ value: "all", label: "Toutes les régions" }, ...info.regions.map((r) => ({ value: r, label: r }))]}
        />
        <Segmented
          label="Sens"
          value={direction}
          onChange={setDirection}
          options={[
            { value: "all", label: "Tous" },
            { value: "DEP", label: "Départs" },
            { value: "ARR", label: "Arrivées" },
          ]}
        />
        <div className="ml-auto flex items-center gap-4">
          <Switch checked={movedOnly} onChange={setMovedOnly} label="Modifiés uniquement" />
          <Segmented
            label="Regrouper par"
            value={grouping}
            onChange={(v) => setGrouping(v as Grouping)}
            options={[
              { value: "tail", label: "Avion" },
              { value: "station", label: "Escale" },
              { value: "flight", label: "Vol" },
            ]}
          />
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-hairline bg-surface">
        {rows.length ? (
          <Timeline
            iid={iid}
            grid={info.grid}
            flights={flights}
            rows={rows.slice(0, MAX_ROWS)}
            t0={t0}
            t1={t1}
            selected={selected}
            onSelect={onSelect}
            dimmed={dimmed}
            maxHeight={window.innerHeight - 200}
          />
        ) : (
          <p className="py-16 text-center text-[13px] text-muted">Aucun vol ne correspond à ces filtres.</p>
        )}
      </div>
      {rows.length > MAX_ROWS && (
        <p className="mt-2 text-[12px] text-muted">Seules les {MAX_ROWS} premières lignes sont affichées : affinez les filtres.</p>
      )}
    </div>
  );
}
