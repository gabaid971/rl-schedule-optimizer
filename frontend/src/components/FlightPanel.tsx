import { ChevronRight, RotateCcw, Sparkles, X } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import type { Connection, Flight, FlightDetail, InstanceInfo } from "../api";
import { fmtMoney, fmtNum1, fmtShift, fmtSignedMoney, fmtTime } from "../format";
import { useColorScheme, useEdit, useFlightDetail } from "../hooks";
import { Button, Delta, Dot, dirColor, Info, Segmented } from "../ui";
import { type ClickParams, EChart, type EChartsOption, tokens, tooltipStyle } from "./EChart";

export function FlightPanel({
  iid,
  info,
  flight,
  onSelect,
  onClose,
}: {
  iid: string;
  info: InstanceInfo;
  flight: Flight;
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  const { data } = useFlightDetail(iid, flight.id);
  const { move, revert } = useEdit(iid);
  const detail = data && data.id === flight.id ? data : undefined;
  const moved = flight.dep !== flight.base_dep;
  const route = flight.direction === "DEP" ? [info.hub, flight.station] : [flight.station, info.hub];

  const best = detail?.options
    .filter((o) => o.allowed)
    .reduce((a, b) => (b.delta > a.delta ? b : a), { dep: flight.dep, delta: 0, allowed: true });

  return (
    <aside className="w-full shrink-0 lg:sticky lg:top-[72px] lg:max-h-[calc(100vh-88px)] lg:w-[360px] lg:overflow-y-auto">
      <div className="rounded-xl border border-hairline bg-surface">
        {/* en-tête */}
        <div className="flex items-start justify-between gap-2 border-b border-hairline px-4 py-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Dot color={dirColor(flight.direction)} />
              <span className="text-[15px] font-semibold">{flight.id}</span>
              <span className="text-[13px] text-ink-2">
                {route[0]} → {route[1]}
              </span>
            </div>
            <div className="mt-0.5 text-[12px] text-muted">
              {flight.region} · {flight.tail} · {flight.seats} sièges
            </div>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Fermer">
            <X size={16} />
          </Button>
        </div>

        {/* faits */}
        <dl className="grid grid-cols-3 gap-x-3 gap-y-3 px-4 py-3 text-[13px] tabular">
          <Fact label="Départ" value={fmtTime(flight.dep)} sub={moved ? `au lieu de ${fmtTime(flight.base_dep)}` : undefined} />
          <Fact label="Arrivée" value={fmtTime(flight.arr)} />
          <Fact label="Fenêtre" value={`${fmtTime(flight.lo)}–${fmtTime(flight.hi)}`} />
          <Fact label="Passagers" value={fmtNum1(flight.pax)} />
          <div className="col-span-2">
            <dt className="text-[12px] text-muted">Revenu attribué</dt>
            <dd className="flex items-baseline gap-2 font-medium">
              {fmtMoney(flight.revenue)}
              <Delta value={flight.revenue - flight.base_revenue} className="text-[12px]" />
            </dd>
          </div>
        </dl>

        {/* effet d'un décalage */}
        <div className="border-t border-hairline px-4 py-3">
          <h3 className="flex items-center gap-1.5 text-[13px] font-semibold">
            Effet d'un décalage
            <Info>
              Variation du revenu total si ce vol partait à une autre heure, tous les autres vols restant fixes.
              Cliquez un point pour appliquer l'horaire. Les points gris sont interdits par la rotation avion.
            </Info>
          </h3>
          {detail && !detail.options.some((o) => o.allowed && o.dep !== flight.dep) ? (
            <p className="mt-2 rounded-lg bg-sunken px-3 py-2.5 text-[12.5px] text-ink-2">
              Aucun autre horaire possible : la rotation de {flight.tail} ne laisse pas de marge autour de ce vol.
            </p>
          ) : detail ? (
            <OptionsChart detail={detail} onPick={(dep) => move.mutate({ fid: flight.id, dep })} />
          ) : (
            <div className="h-[170px]" />
          )}
          <div className="mt-2 flex flex-wrap gap-2">
            {best && best.delta > 0.5 && (
              <Button variant="soft" size="sm" onClick={() => move.mutate({ fid: flight.id, dep: best.dep })}>
                <Sparkles size={13} />
                Meilleur horaire : {fmtTime(best.dep)} ({fmtSignedMoney(best.delta)})
              </Button>
            )}
            {moved && (
              <Button variant="ghost" size="sm" disabled={revert.isPending} onClick={() => revert.mutate(flight.id)}>
                <RotateCcw size={13} />
                Revenir à {fmtTime(flight.base_dep)}
              </Button>
            )}
          </div>
        </div>

        {detail && (
          <>
            <Connections detail={detail} flight={flight} mct={info.params.mct} onSelect={onSelect} />
            <div className="border-t border-hairline px-4 py-3">
              <h3 className="mb-2 text-[13px] font-semibold">Rotation {flight.tail}</h3>
              <div className="flex items-center gap-1 text-[12.5px]">
                <RotationChip id={detail.prev} onSelect={onSelect} />
                <ChevronRight size={14} className="text-muted" />
                <span className="rounded-md bg-sunken px-2 py-1 font-medium">{flight.id}</span>
                <ChevronRight size={14} className="text-muted" />
                <RotationChip id={detail.next} onSelect={onSelect} />
              </div>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

function Fact({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <dt className="text-[12px] text-muted">{label}</dt>
      <dd className="font-medium">{value}</dd>
      {sub && <dd className="text-[11.5px] text-muted">{sub}</dd>}
    </div>
  );
}

function RotationChip({ id, onSelect }: { id: string | null; onSelect: (id: string) => void }) {
  if (!id) return <span className="px-2 py-1 text-muted">—</span>;
  return (
    <button className="rounded-md px-2 py-1 text-accent hover:bg-sunken" onClick={() => onSelect(id)}>
      {id}
    </button>
  );
}

function Connections({
  detail,
  flight,
  mct,
  onSelect,
}: {
  detail: FlightDetail;
  flight: Flight;
  mct: number;
  onSelect: (id: string) => void;
}) {
  const [tab, setTab] = useState("sold");
  const sold = detail.connections.filter((c) => c.sellable);
  const missed = detail.connections.filter((c) => !c.sellable);
  const items = tab === "sold" ? sold : missed;
  const title = flight.direction === "DEP" ? "Vols d'apport" : "Correspondances";

  return (
    <div className="border-t border-hairline px-4 py-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-[13px] font-semibold">{title}</h3>
        <Segmented
          label="Type de correspondance"
          value={tab}
          onChange={setTab}
          options={[
            { value: "sold", label: `Vendues ${sold.length}` },
            { value: "missed", label: `Ratées ${missed.length}` },
          ]}
        />
      </div>
      {tab === "missed" && missed.length > 0 && (
        <p className="mb-1 text-[12px] text-muted">Moins de {mct} min de correspondance : non vendues.</p>
      )}
      {items.length ? (
        <div className="max-h-64 overflow-y-auto">
          <table className="w-full text-[12.5px] tabular">
            <tbody>
              {items.map((c: Connection) => (
                <tr key={c.other + c.market} className="border-t border-hairline first:border-t-0">
                  <td className="py-1.5">
                    <button className="font-medium text-accent hover:underline" onClick={() => onSelect(c.other)}>
                      {c.other}
                    </button>
                  </td>
                  <td className="py-1.5 text-ink-2">{c.market.replace("→", " → ")}</td>
                  <td className={c.cnx < mct ? "py-1.5 text-right text-critical" : "py-1.5 text-right text-ink-2"}>
                    {c.cnx} min
                  </td>
                  <td className="w-12 py-1.5 text-right">{c.sellable ? fmtNum1(c.pax) : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="py-3 text-[12.5px] text-muted">Aucune.</p>
      )}
    </div>
  );
}

function OptionsChart({ detail, onPick }: { detail: FlightDetail; onPick: (dep: number) => void }) {
  const scheme = useColorScheme();
  const option = useMemo<EChartsOption>(() => {
    const t = tokens();
    const idx = (dep: number) => detail.options.findIndex((o) => o.dep === dep);
    const marks = [
      { xAxis: idx(detail.dep), label: "actuel" },
      ...(detail.base_dep !== detail.dep ? [{ xAxis: idx(detail.base_dep), label: "initial" }] : []),
    ].filter((m) => m.xAxis >= 0);
    return {
      animation: false,
      grid: { left: 44, right: 8, top: 18, bottom: 22 },
      tooltip: {
        trigger: "item",
        ...tooltipStyle(t),
        formatter: (p: ClickParams) => {
          const o = detail.options[p.dataIndex];
          return o.allowed
            ? `<b>${fmtTime(o.dep)}</b> <span style="color:${t.muted}">${fmtShift(o.dep - detail.base_dep)}</span><br/>${fmtSignedMoney(o.delta)}`
            : `<b>${fmtTime(o.dep)}</b><br/>Rotation incompatible`;
        },
      },
      xAxis: {
        type: "category",
        data: detail.options.map((o) => fmtTime(o.dep)),
        boundaryGap: false,
        axisLine: { lineStyle: { color: t.axis } },
        axisTick: { show: false },
        axisLabel: { color: t.muted, fontSize: 10.5, interval: 5 },
      },
      yAxis: {
        type: "value",
        splitNumber: 3,
        minInterval: 1,
        splitLine: { lineStyle: { color: t.grid } },
        axisLabel: { color: t.muted, fontSize: 10.5, formatter: (v: number) => fmtSignedMoney(v).replace(" €", "") },
      },
      series: [
        {
          type: "line",
          data: detail.options.map((o) => (o.allowed ? o.delta : null)),
          connectNulls: false,
          symbol: "circle",
          symbolSize: 7,
          lineStyle: { color: t.dep, width: 2 },
          itemStyle: { color: t.dep, borderColor: t.surface, borderWidth: 1.5 },
          areaStyle: { color: t.dep, opacity: 0.08, origin: 0 },
          markLine: {
            symbol: "none",
            silent: true,
            lineStyle: { color: t.ink2, width: 1, type: "solid" },
            label: { color: t.ink2, fontSize: 10, formatter: (p: ClickParams) => p.data.label },
            data: marks,
          },
        },
        {
          type: "line",
          data: detail.options.map((o) => (o.allowed ? null : 0)),
          symbol: "circle",
          symbolSize: 5,
          lineStyle: { opacity: 0 },
          itemStyle: { color: t.axis },
        },
      ],
    };
  }, [detail, scheme]);

  const onClick = useCallback(
    (p: ClickParams) => {
      const o = detail.options[p.dataIndex];
      if (p.seriesIndex === 0 && o?.allowed && o.dep !== detail.dep) onPick(o.dep);
    },
    [detail, onPick],
  );

  return <EChart option={option} height={170} onClick={onClick} />;
}
