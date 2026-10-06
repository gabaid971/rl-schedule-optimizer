import { ArrowLeftRight } from "lucide-react";
import { useState } from "react";
import type { Flight, InstanceInfo, PairMarket } from "../api";
import { fmtMoney, fmtNum, fmtNum1 } from "../format";
import { usePair } from "../hooks";
import { Button, Delta, Metric, PageHeader, Section, Select, Tip } from "../ui";
import { ConnectionChart } from "./ConnectionChart";

const TOP_MARKETS = 8;

export function ConnectionsView({
  iid,
  info,
  flights,
  origin,
  dest,
  onChange,
  selected,
  onSelect,
}: {
  iid: string;
  info: InstanceInfo;
  flights: Flight[];
  origin: string;
  dest: string;
  onChange: (origin: string, dest: string) => void;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const { data } = usePair(iid, origin, dest);
  const [all, setAll] = useState(false);
  const regions = info.regions.map((r) => ({ value: r, label: r }));
  const tot = data?.totals;

  return (
    <div>
      <PageHeader
        title={
          <>
            <span className="text-ink-2 font-normal">De</span>
            <Select variant="inline" label="Région d'origine" value={origin} onChange={(v) => onChange(v, dest)} options={regions} />
            <Tip content="Inverser le sens">
              <Button variant="ghost" size="icon" onClick={() => onChange(dest, origin)} aria-label="Inverser le sens">
                <ArrowLeftRight size={16} />
              </Button>
            </Tip>
            <span className="text-ink-2 font-normal">vers</span>
            <Select variant="inline" label="Région de destination" value={dest} onChange={(v) => onChange(origin, v)} options={regions} />
          </>
        }
      />

      {tot && data && (
        <div className="mb-6 grid grid-cols-2 gap-x-10 gap-y-4 sm:flex">
          <Metric label="Revenu des correspondances" value={fmtMoney(tot.revenue)} delta={tot.revenue - tot.base_revenue} money />
          <Metric label="Passagers / jour" value={fmtNum(tot.pax)} delta={tot.pax - tot.base_pax} />
          <Metric
            label="Part de la demande captée"
            value={`${Math.round((100 * tot.pax) / Math.max(tot.demand, 1))} %`}
            hint={`sur ${fmtNum(tot.demand)} pax`}
          />
          <Metric label="Correspondances vendues" value={fmtNum(data.connections.length)} />
        </div>
      )}

      {data && (
        <div className="space-y-6">
          <Section
            title="Vague de correspondances au hub"
            info={
              <>
                Chaque point est un vol, placé à son heure au hub. Les courbes relient les correspondances vendues
                (épaisseur = passagers). Survolez ou cliquez un vol pour isoler ses correspondances et voir sa fenêtre
                de {info.params.mct} à {info.params.max_cnx} min ; glissez-le pour le décaler.
              </>
            }
          >
            {data.arrivals.length && data.departures.length ? (
              <ConnectionChart
                iid={iid}
                grid={info.grid}
                flights={flights}
                data={data}
                origin={origin}
                dest={dest}
                mct={info.params.mct}
                maxCnx={info.params.max_cnx}
                selected={selected}
                onSelect={onSelect}
              />
            ) : (
              <p className="py-10 text-center text-[13px] text-muted">Aucun vol sur cette paire de régions.</p>
            )}
          </Section>

          <Section
            title="Principaux marchés"
            info="Demande et passagers captés par paire d'escales, revenu et écart au programme initial."
            right={
              data.markets.length > TOP_MARKETS && (
                <Button variant="ghost" size="sm" onClick={() => setAll(!all)}>
                  {all ? "Réduire" : `Voir les ${data.markets.length} marchés`}
                </Button>
              )
            }
          >
            <MarketTable markets={all ? data.markets : data.markets.slice(0, TOP_MARKETS)} />
          </Section>
        </div>
      )}
    </div>
  );
}

function MarketTable({ markets }: { markets: PairMarket[] }) {
  return (
    <table className="w-full text-[13px] tabular">
      <thead>
        <tr className="text-left text-[12px] text-muted">
          <th className="pb-2 font-normal">Marché</th>
          <th className="pb-2 font-normal">Passagers captés / demande</th>
          <th className="pb-2 text-right font-normal">Revenu</th>
          <th className="w-28 pb-2 text-right font-normal">Écart</th>
        </tr>
      </thead>
      <tbody>
        {markets.map((m) => {
          const share = m.pax / Math.max(m.demand, 1e-9);
          return (
            <tr key={m.market} className="border-t border-hairline">
              <td className="py-2 font-medium">{m.market.replace("→", " → ")}</td>
              <td className="py-2">
                <div className="flex items-center gap-3">
                  <div className="h-1.5 w-40 overflow-hidden rounded-full bg-sunken">
                    <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, share * 100)}%` }} />
                  </div>
                  <span className="text-ink-2">
                    {fmtNum1(m.pax)} <span className="text-muted">/ {fmtNum1(m.demand)}</span>
                  </span>
                </div>
              </td>
              <td className="py-2 text-right">{fmtMoney(m.revenue)}</td>
              <td className="py-2 text-right">
                <Delta value={m.revenue - m.base_revenue} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
