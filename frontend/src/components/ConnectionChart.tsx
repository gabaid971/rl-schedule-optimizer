import { useMemo, useState } from "react";
import type { Flight, PairData } from "../api";
import { fmtNum1, fmtTime } from "../format";
import { useElementWidth } from "../hooks";
import { Dot } from "../ui";
import { FlightCard } from "./Timeline";
import { DragBubble, useFlightDrag } from "./useFlightDrag";

const PAD_X = 20;
const STEP = 10; // espacement vertical des vols empilés
const R = 4;
const GAP = 150; // espace entre les deux lignes, où passent les correspondances
const TOP = 30;

interface Props {
  iid: string;
  grid: number;
  flights: Flight[];
  data: PairData;
  origin: string;
  dest: string;
  mct: number;
  maxCnx: number;
  selected: string | null;
  onSelect: (id: string) => void;
}

/**
 * Diagramme de vague : arrivées (en haut) et départs (en bas) placés à leur heure au hub,
 * reliés par les correspondances vendues (épaisseur = passagers). Survoler ou sélectionner
 * un vol isole ses correspondances et montre sa fenêtre [MCT, max] ; le glisser décale
 * son horaire en recalculant les liaisons en direct.
 */
export function ConnectionChart({ iid, grid, flights, data, origin, dest, mct, maxCnx, selected, onSelect }: Props) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);

  const hubTime = (f: Flight, dep: number) => (f.direction === "DEP" ? dep : dep + f.block);

  // domaine horaire : toutes les positions atteignables des vols de la paire
  const [t0, t1] = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const i of [...data.arrivals, ...data.departures]) {
      const f = flights[i];
      lo = Math.min(lo, hubTime(f, f.lo));
      hi = Math.max(hi, hubTime(f, f.hi));
    }
    return [Math.floor(lo / 60) * 60, Math.ceil(hi / 60) * 60];
  }, [data, flights]);

  const plotW = Math.max(300, width - 2 * PAD_X);
  const ppm = plotW / Math.max(60, t1 - t0);
  const x = (t: number) => PAD_X + (t - t0) * ppm;
  const { drag, dragAny, dragOpt, shownDep, handlers } = useFlightDrag({ iid, grid, ppm, onSelect });

  // empilement des marqueurs qui se chevauchent (état serveur, stable pendant un geste)
  const { level, nA, nD } = useMemo(() => {
    const level = new Map<number, number>();
    const stack = (ids: number[]) => {
      const lastX: number[] = [];
      for (const i of [...ids].sort((a, b) => hubTime(flights[a], flights[a].dep) - hubTime(flights[b], flights[b].dep))) {
        const px = (hubTime(flights[i], flights[i].dep) - t0) * ppm;
        let k = lastX.findIndex((lx) => px - lx >= 2 * R + 3);
        if (k < 0) k = lastX.push(px) - 1;
        else lastX[k] = px;
        level.set(i, k);
      }
      return Math.max(1, lastX.length);
    };
    return { level, nA: stack(data.arrivals), nD: stack(data.departures) };
  }, [data, flights, t0, ppm]);

  const yA = TOP + nA * STEP; // ligne de base des arrivées
  const yD = yA + GAP; // ligne de base des départs
  const height = yD + nD * STEP + 46;
  const posY = (f: Flight) => (f.direction === "DEP" ? yD + level.get(f.i)! * STEP : yA - level.get(f.i)! * STEP);

  const isArr = useMemo(() => new Set(data.arrivals), [data]);
  const maxPax = useMemo(() => Math.max(1, ...data.connections.map((c) => c.pax)), [data]);

  // vol mis en avant : glissé > survolé > sélectionné
  const sel = flights.find((f) => f.id === selected);
  const inPair = (i: number) => isArr.has(i) || data.departures.includes(i);
  const focus = dragAny?.i ?? (hover ? hover.i : sel && inPair(sel.i) ? sel.i : null);

  // correspondances du vol mis en avant (recalculées en direct pendant un glissement)
  const focusLinks = useMemo(() => {
    if (focus === null) return null;
    const f = flights[focus];
    if (!drag || drag.i !== focus) {
      return data.connections.filter((c) => c.a === focus || c.d === focus).map((c) => ({ a: c.a, d: c.d, pax: c.pax }));
    }
    const t = hubTime(f, drag.dep);
    const others = isArr.has(focus) ? data.departures : data.arrivals;
    return others
      .filter((o) => {
        const of = flights[o];
        const cnx = isArr.has(focus) ? shownDep(of) - t : t - hubTime(of, shownDep(of));
        return cnx >= mct && cnx <= maxCnx;
      })
      .map((o) => (isArr.has(focus) ? { a: focus, d: o, pax: maxPax / 3 } : { a: o, d: focus, pax: maxPax / 3 }));
  }, [focus, drag, data, flights, isArr, mct, maxCnx, maxPax]);

  const partners = useMemo(() => {
    if (!focusLinks) return null;
    return new Set(focusLinks.flatMap((l) => [l.a, l.d]));
  }, [focusLinks]);

  const ticks = useMemo(() => {
    const step = ppm * 60 < 44 ? 120 : 60;
    const out: number[] = [];
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
    return out;
  }, [t0, t1, ppm]);

  const curve = (a: number, d: number) => {
    const fa = flights[a];
    const fd = flights[d];
    const x1 = x(hubTime(fa, shownDep(fa)));
    const x2 = x(shownDep(fd));
    const y1 = yA + R + 3;
    const y2 = yD - R - 3;
    return `M${x1},${y1} C${x1},${y1 + GAP * 0.5} ${x2},${y2 - GAP * 0.5} ${x2},${y2}`;
  };
  const strokeW = (pax: number) => 0.6 + 2.6 * Math.sqrt(pax / maxPax);

  // fenêtre de correspondance du vol mis en avant
  let window_: { x1: number; x2: number; y: number; h: number } | null = null;
  if (focus !== null) {
    const f = flights[focus];
    const t = hubTime(f, shownDep(f));
    if (isArr.has(focus)) window_ = { x1: x(t + mct), x2: x(t + maxCnx), y: yD - R - 6, h: nD * STEP + 2 * R + 8 };
    else window_ = { x1: x(t - maxCnx), x2: x(t - mct), y: yA - (nA - 1) * STEP - R - 6, h: nA * STEP + R + 8 };
    window_.x1 = Math.max(PAD_X, window_.x1);
    window_.x2 = Math.min(width - PAD_X, window_.x2);
  }

  const hovered = hover ? flights[hover.i] : null;
  const hoveredLinks = hover ? data.connections.filter((c) => c.a === hover.i || c.d === hover.i) : [];
  const dragFlight = drag ? flights[drag.i] : null;

  return (
    <div ref={wrapRef} className="relative select-none">
      <svg width={Math.max(width, 1)} height={height} className="block">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={TOP - 8} y2={height - 26} stroke="var(--grid)" />
            <text x={x(t)} y={height - 8} textAnchor="middle" fontSize={11} className="fill-muted tabular">
              {fmtTime(t % 1440)}
            </text>
          </g>
        ))}

        {window_ && window_.x2 > window_.x1 && (
          <g>
            <rect x={window_.x1} y={window_.y} width={window_.x2 - window_.x1} height={window_.h} rx={6} fill="var(--accent-wash)" />
          </g>
        )}

        {/* correspondances : toutes en fond, celles du vol mis en avant par-dessus */}
        <g fill="none">
          {data.connections.map((c, k) => (
            <path
              key={k}
              d={curve(c.a, c.d)}
              stroke="var(--ink-2)"
              strokeWidth={strokeW(c.pax)}
              // l'opacité suit les passagers : les gros flux ressortent, la masse reste en fond
              strokeOpacity={focusLinks ? 0.025 : 0.02 + 0.3 * (c.pax / maxPax)}
            />
          ))}
          {focusLinks?.map((c, k) => (
            <path key={`f${k}`} d={curve(c.a, c.d)} stroke="var(--accent)" strokeWidth={strokeW(c.pax) + 0.4} strokeOpacity={0.85} />
          ))}
        </g>

        {/* vols */}
        {[...data.arrivals, ...data.departures].map((i) => {
          const f = flights[i];
          const dep = shownDep(f);
          const cx = x(hubTime(f, dep));
          const cy = posY(f);
          const color = f.direction === "DEP" ? "var(--series-dep)" : "var(--series-arr)";
          const dim = partners && !partners.has(i) && focus !== i;
          const isFocus = focus === i || selected === f.id;
          const ghost = drag?.i === i ? f.dep : f.base_dep;
          return (
            <g key={i} opacity={dim ? 0.25 : 1}>
              {ghost !== dep && (
                <>
                  <line x1={x(hubTime(f, ghost))} x2={cx} y1={cy} y2={cy} stroke={color} strokeOpacity={0.5} />
                  <circle cx={x(hubTime(f, ghost))} cy={cy} r={R - 0.5} fill="var(--surface)" stroke={color} strokeWidth={1.5} />
                </>
              )}
              <circle
                cx={cx}
                cy={cy}
                r={isFocus ? R + 1.5 : R}
                fill={color}
                stroke={isFocus ? "var(--ink)" : "var(--surface)"}
                strokeWidth={isFocus ? 2 : 1.5}
              />
              <circle
                cx={cx}
                cy={cy}
                r={R + 5}
                fill="transparent"
                className="cursor-grab active:cursor-grabbing"
                {...handlers(f)}
                onPointerEnter={(e) => !dragAny && setHover({ i, x: e.clientX, y: e.clientY })}
                onPointerLeave={() => setHover(null)}
              />
            </g>
          );
        })}

        <text x={PAD_X} y={14} fontSize={12} className="fill-ink-2">
          <tspan fontWeight={600} className="fill-ink">
            {data.arrivals.length} arrivées
          </tspan>{" "}
          depuis {origin}
        </text>
        <text x={PAD_X} y={height - 34} fontSize={12} className="fill-ink-2">
          <tspan fontWeight={600} className="fill-ink">
            {data.departures.length} départs
          </tspan>{" "}
          vers {dest}
        </text>
      </svg>

      <div className="pointer-events-none absolute right-5 top-0 flex items-center gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1">
          <Dot color="var(--series-arr)" size={7} /> Arrivée au hub
        </span>
        <span className="flex items-center gap-1">
          <Dot color="var(--series-dep)" size={7} /> Départ du hub
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-0.5 w-4 rounded bg-ink-2 opacity-60" /> Correspondance vendue
        </span>
      </div>

      {drag && dragFlight && (
        <DragBubble
          f={dragFlight}
          drag={drag}
          opt={dragOpt}
          left={Math.max(0, Math.min(x(hubTime(dragFlight, drag.dep)) - 60, width - 240))}
          top={dragFlight.direction === "DEP" ? posY(dragFlight) + 14 : posY(dragFlight) - 42}
        />
      )}
      {hovered && !dragAny && (
        <FlightCard
          f={hovered}
          x={hover!.x}
          y={hover!.y}
          extra={
            <>
              <span className="text-muted">Correspondances</span>
              <span>
                {hoveredLinks.length} · {fmtNum1(hoveredLinks.reduce((s, c) => s + c.pax, 0))} pax
              </span>
            </>
          }
        />
      )}
    </div>
  );
}
