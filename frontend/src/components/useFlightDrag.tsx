import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, type Flight, type FlightDetail, type FlightOption } from "../api";
import { fmtShift, fmtSignedMoney, fmtTime } from "../format";
import { useEdit, useToast } from "../hooks";

export interface Drag {
  i: number;
  startX: number;
  dep: number;
  moved: boolean;
  options?: FlightOption[];
}

/**
 * Glisser horizontalement un vol pour décaler son départ (pas de `grid` minutes, borné à
 * sa fenêtre). Les options (autorisé, Δ revenu) sont chargées au début du geste.
 * Un simple clic sélectionne le vol.
 */
export function useFlightDrag({
  iid,
  grid,
  ppm,
  onSelect,
}: {
  iid: string;
  grid: number;
  ppm: number; // pixels par minute
  onSelect: (id: string) => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const { move } = useEdit(iid);
  const [drag, setDrag] = useState<Drag | null>(null);
  const ref = useRef<Drag | null>(null);
  const set = (d: Drag | null) => {
    ref.current = d;
    setDrag(d);
  };

  const pending = move.isPending ? move.variables : undefined;
  /** Heure de départ à afficher : geste en cours > requête en vol > état serveur. */
  const shownDep = (f: Flight) => (drag?.i === f.i ? drag.dep : pending?.fid === f.id ? pending.dep : f.dep);
  const dragOpt = drag?.options?.find((o) => o.dep === drag.dep);

  const handlers = (f: Flight) => ({
    onPointerDown: (e: React.PointerEvent) => {
      if (e.button !== 0) return;
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      set({ i: f.i, startX: e.clientX, dep: f.dep, moved: false });
      qc.fetchQuery<FlightDetail>({ queryKey: ["flight", iid, f.id], queryFn: () => api.flight(iid, f.id) })
        .then((det) => {
          const cur = ref.current;
          if (cur && cur.i === f.i) set({ ...cur, options: det.options });
        })
        .catch(() => {});
    },
    onPointerMove: (e: React.PointerEvent) => {
      const d = ref.current;
      if (!d || d.i !== f.i) return;
      const dx = e.clientX - d.startX;
      if (!d.moved && Math.abs(dx) < 4) return;
      const dep = Math.min(f.hi, Math.max(f.lo, f.dep + Math.round(dx / ppm / grid) * grid));
      if (dep !== d.dep || !d.moved) set({ ...d, dep, moved: true });
    },
    onPointerUp: () => {
      const d = ref.current;
      set(null);
      if (!d || d.i !== f.i) return;
      onSelect(f.id);
      if (!d.moved || d.dep === f.dep) return;
      const opt = d.options?.find((o) => o.dep === d.dep);
      if (opt && !opt.allowed) {
        toast(`${f.id} ne peut pas partir à ${fmtTime(d.dep)} : rotation avion incompatible`, "error");
        return;
      }
      move.mutate({ fid: f.id, dep: d.dep });
    },
  });

  return { drag: drag?.moved ? drag : null, dragAny: drag, dragOpt, shownDep, handlers };
}

/** Bulle affichée pendant un glissement. */
export function DragBubble({ f, drag, opt, left, top }: { f: Flight; drag: Drag; opt?: FlightOption; left: number; top: number }) {
  return (
    <div
      className="pointer-events-none absolute z-20 flex items-center gap-2 whitespace-nowrap rounded-lg border border-hairline-strong bg-raised px-2.5 py-1.5 text-[12px] shadow-pop tabular"
      style={{ left, top }}
    >
      <span className="font-semibold">{fmtTime(drag.dep)}</span>
      <span className="text-muted">{fmtShift(drag.dep - f.base_dep)}</span>
      {opt ? (
        opt.allowed ? (
          <span className={opt.delta >= 0 ? "font-medium text-good" : "font-medium text-critical"}>
            {fmtSignedMoney(opt.delta)}
          </span>
        ) : (
          <span className="font-medium text-critical">Rotation incompatible</span>
        )
      ) : (
        <span className="text-muted">…</span>
      )}
    </div>
  );
}
