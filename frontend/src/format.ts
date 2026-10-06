const nf0 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
const compact = new Intl.NumberFormat("fr-FR", { notation: "compact", maximumFractionDigits: 1 });

/** 480 -> "08:00", 1500 -> "01:00+1" */
export function fmtTime(m: number): string {
  const day = Math.floor(m / 1440);
  const r = ((m % 1440) + 1440) % 1440;
  const s = `${String(Math.floor(r / 60)).padStart(2, "0")}:${String(r % 60).padStart(2, "0")}`;
  return day > 0 ? `${s}+${day}` : s;
}

export const fmtNum = (x: number) => nf0.format(x);
export const fmtNum1 = (x: number) => nf1.format(x);
export const fmtMoney = (x: number) => `${nf0.format(x)} €`;
export const fmtCompact = (x: number) => compact.format(x);

const sign = (x: number) => (x > 0 ? "+" : x < 0 ? "−" : "±");

export const fmtSignedMoney = (x: number) => `${sign(x)}${nf0.format(Math.abs(x))} €`;
export const fmtSigned = (x: number) => `${sign(x)}${nf0.format(Math.abs(x))}`;
export const fmtPct = (x: number) =>
  x !== 0 && Math.abs(x) < 5e-5
    ? `${sign(x)}<0,01 %`
    : `${sign(x)}${(Math.abs(x) * 100).toFixed(2).replace(".", ",")} %`;
export const fmtShift = (m: number) => (m === 0 ? "0 min" : `${sign(m)}${Math.abs(m)} min`);

/** Classe de couleur d'un écart (un revenu qui monte est une bonne nouvelle). */
export const deltaClass = (x: number, eps = 0.5) =>
  x > eps ? "text-good" : x < -eps ? "text-critical" : "text-muted";
