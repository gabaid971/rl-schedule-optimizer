import { AlertTriangle, ArrowRight, MousePointerClick, RotateCcw, Undo2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { Flight, MoveInfo, Violation } from "../api";
import { fmtShift, fmtTime } from "../format";
import { useEdit } from "../hooks";
import { Button, Delta, Dot, dirColor, Info, PageHeader, Segmented, Tip } from "../ui";

export function ChangesView({
  iid,
  moves,
  violations,
  flights,
  totalDelta,
  onSelect,
}: {
  iid: string;
  moves: MoveInfo[];
  violations: Violation[];
  flights: Flight[];
  totalDelta: number;
  onSelect: (id: string) => void;
}) {
  const { revert, reset } = useEdit(iid);
  const [sort, setSort] = useState("impact");
  const sorted = useMemo(
    () => [...moves].sort((a, b) => (sort === "impact" ? Math.abs(b.marginal) - Math.abs(a.marginal) : a.base_dep - b.base_dep)),
    [moves, sort],
  );

  return (
    <div>
      <PageHeader
        title={
          <>
            Modifications
            {moves.length > 0 && (
              <span className="text-[14px] font-normal text-ink-2">
                {moves.length} vol{moves.length > 1 ? "s" : ""} · <Delta value={totalDelta} />
              </span>
            )}
          </>
        }
      >
        {moves.length > 0 && (
          <>
            <Segmented
              label="Tri"
              value={sort}
              onChange={setSort}
              options={[
                { value: "impact", label: "Par impact" },
                { value: "time", label: "Par horaire" },
              ]}
            />
            <Button disabled={reset.isPending} onClick={() => reset.mutate()}>
              <RotateCcw size={14} />
              Tout annuler
            </Button>
          </>
        )}
      </PageHeader>

      {violations.length > 0 && (
        <div className="mb-4 rounded-xl border border-critical/30 bg-critical-wash px-4 py-3 text-[13px]">
          <div className="mb-1 flex items-center gap-2 font-semibold text-critical">
            <AlertTriangle size={15} /> {violations.length} contrainte(s) non respectée(s)
          </div>
          <ul className="space-y-0.5 text-ink-2">
            {violations.map((v, k) => (
              <li key={k}>{v.message}</li>
            ))}
          </ul>
        </div>
      )}

      {moves.length === 0 ? (
        <div className="flex flex-col items-center rounded-xl border border-dashed border-hairline-strong py-16 text-center">
          <MousePointerClick size={22} className="mb-3 text-muted" />
          <p className="text-[14px] font-medium">Aucune modification</p>
          <p className="mt-1 max-w-sm text-[13px] text-muted">
            Glissez un vol dans le Programme ou les Correspondances, ou choisissez un horaire dans le panneau d'un vol.
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-hairline bg-surface">
          <table className="w-full text-[13px] tabular">
            <thead>
              <tr className="border-b border-hairline text-left text-[12px] text-muted">
                <th className="px-4 py-2.5 font-normal">Vol</th>
                <th className="py-2.5 font-normal">Route</th>
                <th className="py-2.5 font-normal">Horaire</th>
                <th className="py-2.5 font-normal">Décalage</th>
                <th className="py-2.5 text-right font-normal">
                  <span className="inline-flex items-center gap-1.5">
                    Impact
                    <Info>
                      Revenu perdu si l'on annulait ce seul décalage. Ces impacts ne s'additionnent pas exactement :
                      deux décalages peuvent créer ensemble une même correspondance.
                    </Info>
                  </span>
                </th>
                <th className="w-14 px-4 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((m) => {
                const f = flights[m.i];
                return (
                  <tr key={m.id} className="group border-t border-hairline first:border-t-0 hover:bg-sunken/60">
                    <td className="px-4 py-2.5">
                      <button className="flex items-center gap-2 font-medium hover:text-accent" onClick={() => onSelect(m.id)}>
                        <Dot color={dirColor(f.direction)} size={7} />
                        {m.id}
                      </button>
                    </td>
                    <td className="py-2.5 text-ink-2">
                      {f.direction === "DEP" ? `HUB → ${f.station}` : `${f.station} → HUB`}
                      <span className="ml-2 text-muted">{f.region}</span>
                    </td>
                    <td className="py-2.5">
                      <span className="inline-flex items-center gap-1.5">
                        <span className="text-muted line-through decoration-[var(--hairline-strong)]">{fmtTime(m.base_dep)}</span>
                        <ArrowRight size={12} className="text-muted" />
                        <span className="font-medium">{fmtTime(m.dep)}</span>
                      </span>
                    </td>
                    <td className="py-2.5 text-ink-2">{fmtShift(m.shift)}</td>
                    <td className="py-2.5 text-right">
                      <Delta value={m.marginal} />
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <Tip content={m.revertible ? "Remettre l'horaire initial" : "Impossible : la rotation avion a changé"}>
                        <span>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            disabled={!m.revertible || revert.isPending}
                            onClick={() => revert.mutate(m.id)}
                            aria-label="Annuler ce décalage"
                          >
                            <Undo2 size={15} />
                          </Button>
                        </span>
                      </Tip>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
