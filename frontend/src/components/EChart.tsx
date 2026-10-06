import { BarChart, HeatmapChart, LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export type EChartsOption = echarts.EChartsCoreOption;
// paramètres d'événement ECharts (forme variable selon la série)
export type ClickParams = any;

/** Lit les jetons CSS courants, pour que les graphiques suivent le thème. */
export function tokens() {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    surface: v("--surface"),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    grid: v("--grid"),
    axis: v("--axis"),
    dep: v("--series-dep"),
    arr: v("--series-arr"),
    good: v("--good"),
    critical: v("--critical"),
    divMid: v("--div-mid"),
  };
}

// rampe séquentielle bleue (100 -> 700) et pôles divergents bleu / rouge
export const BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
export const RED_POLE = "#d03b3b";
export const BLUE_POLE = "#256abf";

export const tooltipStyle = (t: ReturnType<typeof tokens>) => ({
  backgroundColor: t.surface,
  borderColor: t.axis,
  borderWidth: 1,
  padding: [8, 10],
  textStyle: { color: t.ink, fontSize: 12, fontFamily: "Inter Variable, system-ui, sans-serif" },
  extraCssText: "box-shadow: 0 8px 28px rgba(0,0,0,.14); border-radius: 8px;",
});

export function EChart({
  option,
  height,
  onClick,
}: {
  option: EChartsOption;
  height: number;
  onClick?: (p: ClickParams) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    const c = echarts.init(ref.current!, undefined, { renderer: "canvas" });
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(ref.current!);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(
      { textStyle: { fontFamily: "Inter Variable, system-ui, sans-serif" }, ...option },
      { notMerge: true },
    );
  }, [option]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !onClick) return;
    c.on("click", onClick);
    return () => {
      c.off("click", onClick);
    };
  }, [onClick]);

  return <div ref={ref} style={{ height }} />;
}
