import { useMemo, useState } from "react";
import type { Flight } from "../api";
import { fmtNum1, fmtShift, fmtTime } from "../format";
import { useElementWidth } from "../hooks";
import { Dot, dirColor } from "../ui";
import { DragBubble, useFlightDrag } from "./useFlightDrag";

export interface Row {
  key: string;
  label: string;
  sublabel?: string;
  flights: number[];
  groupStart?: boolean; // trait de séparation au-dessus
}

const LABEL_W = 132;
const ROW_H = 26;
const BAR_H = 16;
const AXIS_H = 30;

interface Props {
  iid: string;
  grid: number;
  flights: Flight[];
  rows: Row[];
  t0: number;
  t1: number;
  selected: string | null;
  onSelect: (id: string) => void;
  dimmed?: Set<number>;
  maxHeight: number;
}

/** Gantt des vols. Clic = sélection ; glisser horizontalement = décaler le départ. */
export function Timeline({ iid, grid, flights, rows, t0, t1, selected, onSelect, dimmed, maxHeight }: Props) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);

  const plotW = Math.max(240, width - LABEL_W - 16);
  const ppm = plotW / Math.max(60, t1 - t0);
  const x = (t: number) => LABEL_W + (t - t0) * ppm;
  const height = rows.length * ROW_H;
  const { drag, dragAny, dragOpt, shownDep, handlers } = useFlightDrag({ iid, grid, ppm, onSelect });

  const ticks = useMemo(() => {
    const step = ppm * 60 < 44 ? (ppm * 120 < 44 ? 180 : 120) : 60;
    const out: number[] = [];
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) if ((t - t0) * ppm > 18) out.push(t);
    return out;
  }, [t0, t1, ppm]);

  const dragRow = dragAny ? rows.findIndex((r) => r.flights.includes(dragAny.i)) : -1;
  const dragFlight = drag ? flights[drag.i] : null;

  return (
    <div ref={wrapRef} className="relative overflow-auto" style={{ maxHeight }}>
      <div className="sticky top-0 z-10 flex h-[30px] items-center border-b border-hairline bg-surface">
        <div className="flex shrink-0 items-center gap-3 pl-1 text-[11px] text-muted" style={{ width: LABEL_W }}>
          <span className="flex items-center gap-1">
            <Dot color="var(--series-dep)" size={7} /> Départ
          </span>
          <span className="flex items-center gap-1">
            <Dot color="var(--series-arr)" size={7} /> Arrivée
          </span>
        </div>
        <svg width={Math.max(width - LABEL_W, 1)} height={AXIS_H} className="block">
          {ticks.map((t) => (
            <text key={t} x={x(t) - LABEL_W} y={19} textAnchor="middle" className="fill-muted tabular" fontSize={11}>
              {fmtTime(t % 1440)}
            </text>
          ))}
        </svg>
      </div>

      <svg width={Math.max(width, 1)} height={Math.max(height, ROW_H)} className="block select-none">
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={height} stroke="var(--grid)" />
        ))}

        {rows.map((r, k) => (
          <g key={r.key}>
            {r.groupStart && k > 0 && <line x1={0} x2={width} y1={k * ROW_H + 0.5} y2={k * ROW_H + 0.5} stroke="var(--hairline)" />}
            <text x={4} y={k * ROW_H + 17} fontSize={12} className="fill-ink-2">
              {r.label}
              {r.sublabel && (
                <tspan className="fill-muted" fontSize={11} dx={6}>
                  {r.sublabel}
                </tspan>
              )}
            </text>
          </g>
        ))}

        {/* créneaux autorisés du vol glissé */}
        {dragAny?.options && dragRow >= 0 && drag && (
          <g>
            {dragAny.options.map((o) => (
              <rect
                key={o.dep}
                x={x(o.dep) - (grid * ppm) / 2}
                y={dragRow * ROW_H + 3}
                width={grid * ppm + 0.5}
                height={ROW_H - 6}
                fill={o.allowed ? "var(--good)" : "var(--critical)"}
                opacity={o.allowed ? 0.1 : 0.07}
              />
            ))}
          </g>
        )}

        {rows.map((r, k) =>
          r.flights.map((fi) => {
            const f = flights[fi];
            const dep = shownDep(f);
            const y = k * ROW_H + (ROW_H - BAR_H) / 2;
            const w = Math.max(4, f.block * ppm);
            const color = dirColor(f.direction);
            const isSel = selected === f.id;
            const isDragged = drag?.i === f.i;
            const ghostDep = isDragged ? f.dep : f.base_dep;
            const forbidden = isDragged && dragOpt && !dragOpt.allowed;
            return (
              <g key={f.i} opacity={dimmed?.has(f.i) ? 0.22 : 1}>
                {ghostDep !== dep && (
                  <rect x={x(ghostDep)} y={y} width={w} height={BAR_H} rx={4} fill="none" stroke={color} strokeOpacity={0.5} strokeDasharray="3 2" />
                )}
                <rect
                  x={x(dep)}
                  y={y}
                  width={w}
                  height={BAR_H}
                  rx={4}
                  fill={color}
                  stroke={forbidden ? "var(--critical)" : isSel ? "var(--ink)" : "none"}
                  strokeWidth={2}
                  className="cursor-grab transition-[filter] hover:brightness-110 active:cursor-grabbing"
                  {...handlers(f)}
                  onPointerEnter={(e) => !dragAny && setHover({ i: f.i, x: e.clientX, y: e.clientY })}
                  onPointerLeave={() => setHover(null)}
                />
                {w > 32 && (
                  <text x={x(dep) + 6} y={y + BAR_H - 4.5} fontSize={10.5} fontWeight={600} fill="#fff" pointerEvents="none">
                    {f.station}
                  </text>
                )}
              </g>
            );
          }),
        )}
      </svg>

      {drag && dragFlight && dragRow >= 0 && (
        <DragBubble
          f={dragFlight}
          drag={drag}
          opt={dragOpt}
          left={Math.max(LABEL_W, Math.min(x(drag.dep), width - 230))}
          top={AXIS_H + (dragRow === 0 ? ROW_H + 2 : dragRow * ROW_H - 34)}
        />
      )}
      {hover && !dragAny && <FlightCard f={flights[hover.i]} x={hover.x} y={hover.y} />}
    </div>
  );
}

export function FlightCard({ f, x, y, extra }: { f: Flight; x: number; y: number; extra?: React.ReactNode }) {
  const route = f.direction === "DEP" ? `HUB → ${f.station}` : `${f.station} → HUB`;
  return (
    <div
      className="pointer-events-none fixed z-30 w-52 rounded-lg border border-hairline-strong bg-raised p-2.5 text-[12px] shadow-pop"
      style={{ left: Math.min(x + 14, window.innerWidth - 230), top: y + 16 }}
    >
      <div className="flex items-center gap-1.5 font-semibold">
        <Dot color={dirColor(f.direction)} size={7} />
        {f.id}
        <span className="font-normal text-ink-2">{route}</span>
      </div>
      <div className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-ink-2 tabular">
        <span className="text-muted">Départ</span>
        <span>
          {fmtTime(f.dep)}
          {f.dep !== f.base_dep && <span className="text-muted"> ({fmtShift(f.dep - f.base_dep)})</span>}
        </span>
        <span className="text-muted">Arrivée</span>
        <span>{fmtTime(f.arr)}</span>
        <span className="text-muted">Passagers</span>
        <span>{fmtNum1(f.pax)}</span>
        {extra}
      </div>
    </div>
  );
}
