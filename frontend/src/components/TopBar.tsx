import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronsUpDown, CircleHelp, Plane, Plus } from "lucide-react";
import { Popover } from "radix-ui";
import { useState } from "react";
import { api, type Kpis } from "../api";
import { fmtMoney, fmtNum } from "../format";
import { useToast } from "../hooks";
import { Button, cx, Delta, Dialog, Field, NumberInput, Switch, Tip } from "../ui";

export type Tab = "network" | "program" | "connections" | "changes";

/** "synth_1000_spread_s0" -> titre lisible */
export function instanceLabel(id: string) {
  const m = /^synth_(\d+)_(spread|banked)_s(\d+)$/.exec(id);
  if (!m) return { title: id, meta: "" };
  return { title: `${fmtNum(+m[1])} vols`, meta: `${m[2] === "banked" ? "en vagues" : "étalé"} · graine ${m[3]}` };
}

export function TopBar({
  iid,
  kpis,
  tab,
  onTab,
  onInstance,
  onModel,
}: {
  iid: string;
  kpis?: Kpis;
  tab: Tab;
  onTab: (t: Tab) => void;
  onInstance: (id: string) => void;
  onModel: () => void;
}) {
  const tabs: [Tab, string][] = [
    ["network", "Réseau"],
    ["program", "Programme"],
    ["connections", "Correspondances"],
    ["changes", "Modifications"],
  ];
  const d = kpis ? kpis.revenue - kpis.base_revenue : 0;

  return (
    <header className="sticky top-0 z-30 border-b border-hairline bg-surface/85 backdrop-blur-md">
      <div className="flex h-14 items-center gap-3 px-5">
        <div className="flex items-center gap-2 pr-1">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-ink text-page">
            <Plane size={15} strokeWidth={2.25} />
          </span>
          <span className="hidden text-[14px] font-semibold tracking-tight md:inline">Schedule Optimizer</span>
        </div>
        <span className="h-5 w-px bg-[var(--hairline-strong)]" />
        <InstancePicker iid={iid} onInstance={onInstance} />

        <nav className="ml-3 flex items-center gap-0.5">
          {tabs.map(([t, label]) => (
            <button
              key={t}
              onClick={() => onTab(t)}
              className={cx(
                "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-[13px] transition-colors",
                tab === t ? "bg-sunken font-medium text-ink" : "text-ink-2 hover:text-ink",
              )}
            >
              {label}
              {t === "changes" && !!kpis?.moved && (
                <span className="rounded-full bg-accent px-1.5 text-[11px] font-semibold leading-[18px] text-white">{kpis.moved}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-4">
          {kpis && (
            <div className="text-right leading-tight">
              <div className="text-[11px] text-muted">Revenu estimé / jour</div>
              <div className="flex items-baseline justify-end gap-2">
                <Delta value={d} className="text-[12px]" />
                <span className="text-[15px] font-semibold tabular">{fmtMoney(kpis.revenue)}</span>
              </div>
            </div>
          )}
          <Tip content="Comment le revenu est calculé" side="bottom">
            <Button variant="ghost" size="icon" onClick={onModel} aria-label="Modèle de revenu">
              <CircleHelp size={17} />
            </Button>
          </Tip>
        </div>
      </div>
    </header>
  );
}

function InstancePicker({ iid, onInstance }: { iid: string; onInstance: (id: string) => void }) {
  const { data: instances } = useQuery({ queryKey: ["instances"], queryFn: api.instances });
  const [open, setOpen] = useState(false);
  const [genOpen, setGenOpen] = useState(false);
  const cur = instanceLabel(iid);

  return (
    <>
      <Popover.Root open={open} onOpenChange={setOpen}>
        <Popover.Trigger asChild>
          <button className="inline-flex h-8 items-center gap-2 rounded-md px-2 text-[13px] hover:bg-sunken">
            <span className="font-medium">{cur.title}</span>
            {cur.meta && <span className="text-muted">{cur.meta}</span>}
            <ChevronsUpDown size={14} className="text-muted" />
          </button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            align="start"
            sideOffset={6}
            className="z-50 w-72 rounded-lg border border-hairline-strong bg-raised p-1 shadow-pop outline-none"
          >
            <div className="px-2 pb-1 pt-1.5 text-[11px] font-medium uppercase tracking-wide text-muted">Instances</div>
            {instances?.map((i) => {
              const l = instanceLabel(i.id);
              return (
                <button
                  key={i.id}
                  className="flex h-9 w-full items-center gap-2 rounded-md px-2 text-left text-[13px] hover:bg-sunken"
                  onClick={() => {
                    onInstance(i.id);
                    setOpen(false);
                  }}
                >
                  <span className="w-4">{i.id === iid && <Check size={14} className="text-accent" />}</span>
                  <span className="font-medium">{l.title}</span>
                  <span className="text-muted">{l.meta}</span>
                </button>
              );
            })}
            <div className="my-1 h-px bg-[var(--hairline)]" />
            <button
              className="flex h-9 w-full items-center gap-2 rounded-md px-2 text-[13px] text-ink-2 hover:bg-sunken hover:text-ink"
              onClick={() => {
                setOpen(false);
                setGenOpen(true);
              }}
            >
              <Plus size={14} className="ml-0.5" /> Générer une instance…
            </button>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
      <GenerateDialog open={genOpen} onOpenChange={setGenOpen} onDone={onInstance} />
    </>
  );
}

function GenerateDialog({ open, onOpenChange, onDone }: { open: boolean; onOpenChange: (v: boolean) => void; onDone: (id: string) => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [n, setN] = useState(200);
  const [banked, setBanked] = useState(false);
  const [seed, setSeed] = useState(1);
  const gen = useMutation({
    mutationFn: () => api.generate({ n_flights: n, banked, seed }),
    onSuccess: async ({ id }) => {
      await qc.invalidateQueries({ queryKey: ["instances"] });
      qc.removeQueries({ queryKey: ["state", id] });
      qc.removeQueries({ queryKey: ["info", id] });
      onOpenChange(false);
      onDone(id);
    },
    onError: (e: Error) => toast(e.message, "error"),
  });
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Générer une instance"
      description="Programme mono-hub synthétique : escales par région, rotations avion faisables, demande de type gravitaire."
      width="max-w-md"
    >
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          gen.mutate();
        }}
      >
        <Field label="Nombre de vols">
          <NumberInput min={10} max={2000} step={10} value={n} onChange={(e) => setN(+e.target.value)} />
        </Field>
        <Field label="Graine aléatoire">
          <NumberInput value={seed} onChange={(e) => setSeed(+e.target.value)} />
        </Field>
        <div className="pt-1">
          <Switch checked={banked} onChange={setBanked} label="Horaires organisés en vagues" />
        </div>
        <div className="flex justify-end gap-2 pt-3">
          <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
            Annuler
          </Button>
          <Button type="submit" variant="primary" disabled={gen.isPending}>
            {gen.isPending ? "Génération…" : "Générer"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
