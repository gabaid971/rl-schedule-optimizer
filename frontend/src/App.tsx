import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Info as InfoIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { ChangesView } from "./components/ChangesView";
import { ConnectionsView } from "./components/ConnectionsView";
import { FlightPanel } from "./components/FlightPanel";
import { ModelDialog } from "./components/ModelDialog";
import { NetworkView } from "./components/NetworkView";
import { ProgramView } from "./components/ProgramView";
import { type Tab, TopBar } from "./components/TopBar";
import { readStorage, ToastContext, useInstanceInfo, useScenario, writeStorage } from "./hooks";
import { TooltipProvider } from "./ui";

export default function App() {
  const { data: instances } = useQuery({ queryKey: ["instances"], queryFn: api.instances });
  const [iid, setIid] = useState<string | null>(() => readStorage("iid"));
  const [toasts, setToasts] = useState<{ id: number; msg: string; kind: string }[]>([]);

  useEffect(() => {
    if (instances?.length && (!iid || !instances.some((i) => i.id === iid))) setIid(instances[0].id);
  }, [instances, iid]);

  const toast = useCallback((msg: string, kind: "error" | "info" = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  const selectInstance = (id: string) => {
    setIid(id);
    writeStorage("iid", id);
  };

  return (
    <ToastContext.Provider value={toast}>
      <TooltipProvider>
        {iid ? <Workspace key={iid} iid={iid} onInstance={selectInstance} /> : <Loading />}
        <div className="fixed bottom-5 left-1/2 z-50 flex -translate-x-1/2 flex-col items-center gap-2">
          {toasts.map((t) => (
            <div
              key={t.id}
              role="status"
              className="flex items-center gap-2 rounded-lg border border-hairline-strong bg-raised px-3.5 py-2.5 text-[13px] shadow-pop"
            >
              {t.kind === "error" ? (
                <AlertCircle size={15} className="text-critical" />
              ) : (
                <InfoIcon size={15} className="text-accent" />
              )}
              {t.msg}
            </div>
          ))}
        </div>
      </TooltipProvider>
    </ToastContext.Provider>
  );
}

function Loading({ label = "Chargement…" }: { label?: string }) {
  return <p className="p-10 text-center text-[13px] text-muted">{label}</p>;
}

function Workspace({ iid, onInstance }: { iid: string; onInstance: (id: string) => void }) {
  const { data: info } = useInstanceInfo(iid);
  const { data: state, error } = useScenario(iid);
  const [tab, setTab] = useState<Tab>(() => {
    const t = readStorage("tab");
    return t === "network" || t === "program" || t === "connections" || t === "changes" ? t : "network";
  });
  const [selected, setSelected] = useState<string | null>(null);
  const [pair, setPair] = useState<[string, string] | null>(null);
  const [showModel, setShowModel] = useState(false);

  const changeTab = (t: Tab) => {
    setTab(t);
    writeStorage("tab", t);
  };

  // paire par défaut : la plus rentable
  const defaultPair = useMemo<[string, string] | null>(() => {
    if (!state) return null;
    const m = state.region_matrix;
    let best: [string, string] = [m.regions[0], m.regions[0]];
    let v = -1;
    m.revenue.forEach((row, o) =>
      row.forEach((x, d) => {
        if (x > v) {
          v = x;
          best = [m.regions[o], m.regions[d]];
        }
      }),
    );
    return best;
  }, [state]);
  const [origin, dest] = pair ?? defaultPair ?? ["", ""];

  const selFlight = useMemo(
    () => (selected && state ? state.flights.find((f) => f.id === selected) : undefined),
    [selected, state],
  );
  const showPanel = selFlight && tab !== "network";

  return (
    <div className="min-h-screen">
      <TopBar iid={iid} kpis={state?.kpis} tab={tab} onTab={changeTab} onInstance={onInstance} onModel={() => setShowModel(true)} />
      {error && (
        <p className="m-6 flex items-center gap-2 text-[13px] text-critical">
          <AlertCircle size={15} /> {(error as Error).message}
        </p>
      )}

      {state && info ? (
        <main className="mx-auto flex max-w-[1680px] flex-col gap-6 px-6 py-6 lg:flex-row lg:items-start">
          <section className="min-w-0 flex-1">
            {tab === "network" && (
              <NetworkView
                kpis={state.kpis}
                profile={state.hub_profile}
                matrix={state.region_matrix}
                onPickPair={(o, d) => {
                  setPair([o, d]);
                  changeTab("connections");
                }}
              />
            )}
            {tab === "program" && (
              <ProgramView iid={iid} info={info} flights={state.flights} selected={selected} onSelect={setSelected} />
            )}
            {tab === "connections" && origin && (
              <ConnectionsView
                iid={iid}
                info={info}
                flights={state.flights}
                origin={origin}
                dest={dest}
                onChange={(o, d) => setPair([o, d])}
                selected={selected}
                onSelect={setSelected}
              />
            )}
            {tab === "changes" && (
              <ChangesView
                iid={iid}
                moves={state.moves}
                violations={state.violations}
                flights={state.flights}
                totalDelta={state.kpis.revenue - state.kpis.base_revenue}
                onSelect={(id) => {
                  setSelected(id);
                  changeTab("program");
                }}
              />
            )}
          </section>
          {showPanel && (
            <FlightPanel iid={iid} info={info} flight={selFlight} onSelect={setSelected} onClose={() => setSelected(null)} />
          )}
        </main>
      ) : (
        !error && <Loading label="Calcul du revenu…" />
      )}

      {info && <ModelDialog info={info} open={showModel} onOpenChange={setShowModel} />}
    </div>
  );
}
