import { useCallback, useMemo, useState } from "react";
import type { HubProfile, Kpis, RegionMatrix } from "../api";
import { fmtCompact, fmtMoney, fmtNum, fmtSignedMoney, fmtTime } from "../format";
import { useColorScheme } from "../hooks";
import { Dot, Metric, PageHeader, Section, Segmented, Switch } from "../ui";
import { BLUE_POLE, BLUE_RAMP, type ClickParams, EChart, type EChartsOption, RED_POLE, tokens, tooltipStyle } from "./EChart";

export function NetworkView({
  kpis,
  profile,
  matrix,
  onPickPair,
}: {
  kpis: Kpis;
  profile: HubProfile;
  matrix: RegionMatrix;
  onPickPair: (origin: string, dest: string) => void;
}) {
  const hasMoves = kpis.moved > 0;
  const [compare, setCompare] = useState(false);
  const [mode, setMode] = useState("revenue");

  return (
    <div>
      <PageHeader title="Réseau" />
      <div className="mb-8 grid grid-cols-2 gap-x-10 gap-y-4 sm:flex">
        <Metric label="Revenu local" value={fmtMoney(kpis.local_revenue)} delta={kpis.local_revenue - kpis.base_local_revenue} money />
        <Metric label="Revenu des correspondances" value={fmtMoney(kpis.cnx_revenue)} delta={kpis.cnx_revenue - kpis.base_cnx_revenue} money />
        <Metric label="Passagers / jour" value={fmtNum(kpis.pax)} hint={`dont ${fmtNum(kpis.cnx_pax)} en correspondance`} />
        <Metric
          label="Part du potentiel captée"
          value={`${((100 * kpis.revenue) / kpis.potential).toFixed(1).replace(".", ",")} %`}
          hint={`potentiel ${fmtCompact(kpis.potential)} €`}
        />
      </div>

      <div className="space-y-8">
        <Section
          title="Vagues au hub"
          info="Nombre d'arrivées (au-dessus) et de départs (en dessous) au hub par tranche de 15 minutes. Une bonne vague = des arrivées groupées suivies de départs groupés."
          right={
            <div className="flex items-center gap-4">
              <Legend items={[["var(--series-arr)", "Arrivées"], ["var(--series-dep)", "Départs"], ...(compare && hasMoves ? [["var(--ink-2)", "Initial"] as [string, string]] : [])]} />
              {hasMoves && <Switch checked={compare} onChange={setCompare} label="Comparer à l'initial" />}
            </div>
          }
        >
          <ProfileChart profile={profile} showBase={compare && hasMoves} />
        </Section>

        <Section
          title="Correspondances entre régions"
          info="Revenu des correspondances de la région en ligne vers la région en colonne. Cliquez une case pour ouvrir le détail de la paire."
          right={
            hasMoves && (
              <Segmented
                label="Mesure"
                value={mode}
                onChange={setMode}
                options={[
                  { value: "revenue", label: "Revenu" },
                  { value: "delta", label: "Écart à l'initial" },
                ]}
              />
            )
          }
        >
          <MatrixChart matrix={matrix} isDelta={hasMoves && mode === "delta"} onPickPair={onPickPair} />
        </Section>
      </div>
    </div>
  );
}

function Legend({ items }: { items: [string, string][] }) {
  return (
    <span className="flex items-center gap-3 text-[12px] text-ink-2">
      {items.map(([c, l]) => (
        <span key={l} className="flex items-center gap-1.5">
          <Dot color={c} size={7} />
          {l}
        </span>
      ))}
    </span>
  );
}

function ProfileChart({ profile, showBase }: { profile: HubProfile; showBase: boolean }) {
  const scheme = useColorScheme();
  const option = useMemo<EChartsOption>(() => {
    const t = tokens();
    const labels = profile.arr.map((_, k) => fmtTime(profile.start + k * profile.bin));
    const baseLine = (data: number[]) => ({
      type: "line",
      step: "middle",
      data,
      symbol: "none",
      lineStyle: { color: t.ink2, width: 1.5 },
      z: 3,
      silent: true,
    });
    return {
      animation: false,
      grid: { left: 28, right: 4, top: 8, bottom: 24 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,127,127,0.08)" } },
        ...tooltipStyle(t),
        formatter: (ps: ClickParams[]) => {
          const k = ps[0].dataIndex;
          const end = fmtTime(profile.start + (k + 1) * profile.bin);
          const row = (label: string, cur: number, b: number) =>
            `<div style="display:flex;justify-content:space-between;gap:16px"><span>${label}</span><b>${cur}${
              showBase && cur !== b ? ` <span style="color:${t.muted};font-weight:400">(${b})</span>` : ""
            }</b></div>`;
          return `<div style="color:${t.muted};margin-bottom:2px">${labels[k]} – ${end}</div>${row("Arrivées", profile.arr[k], profile.base_arr[k])}${row(
            "Départs",
            profile.dep[k],
            profile.base_dep[k],
          )}`;
        },
      },
      xAxis: {
        type: "category",
        data: labels,
        axisLine: { lineStyle: { color: t.axis } },
        axisTick: { show: false },
        axisLabel: { color: t.muted, interval: 7, fontSize: 11 },
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: t.grid } },
        axisLabel: { color: t.muted, fontSize: 11, formatter: (v: number) => String(Math.abs(v)) },
      },
      series: [
        { type: "bar", stack: "m", data: profile.arr, barMaxWidth: 24, itemStyle: { color: t.arr, borderRadius: [3, 3, 0, 0] } },
        { type: "bar", stack: "m", data: profile.dep.map((v) => -v), barMaxWidth: 24, itemStyle: { color: t.dep, borderRadius: [0, 0, 3, 3] } },
        ...(showBase ? [baseLine(profile.base_arr), baseLine(profile.base_dep.map((v) => -v))] : []),
      ],
    };
  }, [profile, showBase, scheme]);
  return <EChart option={option} height={240} />;
}

function MatrixChart({
  matrix,
  isDelta,
  onPickPair,
}: {
  matrix: RegionMatrix;
  isDelta: boolean;
  onPickPair: (origin: string, dest: string) => void;
}) {
  const scheme = useColorScheme();
  const R = matrix.regions;

  const option = useMemo<EChartsOption>(() => {
    const t = tokens();
    const cells: [number, number, number][] = [];
    let maxAbs = 1;
    let max = 1;
    R.forEach((_, o) =>
      R.forEach((_, d) => {
        const cur = matrix.revenue[o][d];
        const v = isDelta ? cur - matrix.base_revenue[o][d] : cur;
        cells.push([d, o, v]);
        maxAbs = Math.max(maxAbs, Math.abs(v));
        max = Math.max(max, v);
      }),
    );
    // texte blanc sur les cases foncées, encre sur les claires (le milieu divergent suit le thème)
    const items = cells.map((value) => {
      const frac = isDelta ? Math.abs(value[2]) / maxAbs : value[2] / max;
      return { value, label: { color: frac < 0.45 ? (isDelta ? t.ink : "#141413") : "#ffffff" } };
    });
    return {
      animation: false,
      grid: { left: 112, right: 4, top: 26, bottom: 4 },
      tooltip: {
        ...tooltipStyle(t),
        formatter: (p: ClickParams) => {
          const [d, o] = p.value as [number, number, number];
          const cur = matrix.revenue[o][d];
          const diff = cur - matrix.base_revenue[o][d];
          return `<b>${R[o]} → ${R[d]}</b><br/>${fmtMoney(cur)} · ${fmtNum(matrix.pax[o][d])} pax${
            Math.abs(diff) >= 0.5 ? `<br/>${fmtSignedMoney(diff)} vs initial` : ""
          }`;
        },
      },
      xAxis: {
        type: "category",
        data: R,
        position: "top",
        axisLabel: { color: t.ink2, fontSize: 11, interval: 0 },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: "category",
        data: R,
        inverse: true,
        axisLabel: { color: t.ink2, fontSize: 11 },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      visualMap: {
        show: false,
        min: isDelta ? -maxAbs : 0,
        max: isDelta ? maxAbs : max,
        inRange: { color: isDelta ? [RED_POLE, t.divMid, BLUE_POLE] : BLUE_RAMP },
      },
      series: [
        {
          type: "heatmap",
          data: items,
          itemStyle: { borderColor: t.surface, borderWidth: 3, borderRadius: 6 },
          label: {
            show: true,
            fontSize: 11,
            formatter: (p: ClickParams) => {
              const v = (p.value as number[])[2];
              if (isDelta) return Math.abs(v) < 1 ? "" : `${v > 0 ? "+" : "−"}${fmtCompact(Math.abs(v))}`;
              return v ? fmtCompact(v) : "";
            },
          },
          emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 2 } },
          cursor: "pointer",
        },
      ],
    };
  }, [matrix, isDelta, R, scheme]);

  const onClick = useCallback(
    (p: ClickParams) => {
      const [d, o] = p.value as [number, number, number];
      onPickPair(R[o], R[d]);
    },
    [R, onPickPair],
  );

  return <EChart option={option} height={R.length * 46 + 34} onClick={onClick} />;
}
