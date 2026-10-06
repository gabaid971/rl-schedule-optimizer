// Composants d'interface de base (Radix + Tailwind), utilisés partout.
import { ArrowDownRight, ArrowUpRight, Check, ChevronDown, Search, X } from "lucide-react";
import { Dialog as D, Select as S, Switch as Sw, ToggleGroup as TG, Tooltip as T } from "radix-ui";
import type { ComponentProps, ReactNode } from "react";
import { fmtSigned, fmtSignedMoney } from "../format";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

// ------------------------------------------------------------------ Button
type ButtonProps = ComponentProps<"button"> & {
  variant?: "default" | "primary" | "ghost" | "soft";
  size?: "sm" | "md" | "icon" | "icon-sm";
};

export function Button({ variant = "default", size = "md", className, ...props }: ButtonProps) {
  return (
    <button
      className={cx(
        "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-colors disabled:pointer-events-none disabled:opacity-40",
        size === "sm" && "h-7 px-2 text-[12.5px]",
        size === "md" && "h-8 px-3 text-[13px]",
        size === "icon" && "h-8 w-8",
        size === "icon-sm" && "h-7 w-7",
        variant === "default" && "border border-hairline-strong bg-surface text-ink shadow-xs hover:bg-sunken",
        variant === "primary" && "bg-accent text-white hover:brightness-110",
        variant === "soft" && "bg-accent-wash text-accent hover:brightness-95",
        variant === "ghost" && "text-ink-2 hover:bg-sunken hover:text-ink",
        className,
      )}
      {...props}
    />
  );
}

// ------------------------------------------------------------------ Select
export interface Option {
  value: string;
  label: ReactNode;
}

export function Select({
  value,
  onChange,
  options,
  label,
  variant = "default",
  className,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Option[];
  label: string;
  variant?: "default" | "inline";
  className?: string;
}) {
  return (
    <S.Root value={value} onValueChange={onChange}>
      <S.Trigger
        aria-label={label}
        className={cx(
          "inline-flex items-center justify-between gap-2 outline-none",
          variant === "default" &&
            "h-8 rounded-md border border-hairline-strong bg-surface px-2.5 text-[13px] text-ink shadow-xs hover:bg-sunken",
          variant === "inline" &&
            "h-9 rounded-lg bg-sunken px-3 text-[17px] font-semibold text-ink hover:bg-[var(--hairline-strong)]",
          className,
        )}
      >
        <S.Value />
        <S.Icon>
          <ChevronDown size={variant === "inline" ? 16 : 14} className="text-muted" />
        </S.Icon>
      </S.Trigger>
      <S.Portal>
        <S.Content
          position="popper"
          sideOffset={4}
          className="z-50 max-h-[min(380px,var(--radix-select-content-available-height))] min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border border-hairline-strong bg-raised p-1 shadow-pop"
        >
          <S.Viewport>
            {options.map((o) => (
              <S.Item
                key={o.value}
                value={o.value}
                className="relative flex h-8 cursor-default select-none items-center rounded-md pl-7 pr-3 text-[13px] text-ink outline-none data-[highlighted]:bg-sunken"
              >
                <S.ItemIndicator className="absolute left-2 inline-flex">
                  <Check size={14} className="text-accent" />
                </S.ItemIndicator>
                <S.ItemText>{o.label}</S.ItemText>
              </S.Item>
            ))}
          </S.Viewport>
        </S.Content>
      </S.Portal>
    </S.Root>
  );
}

// --------------------------------------------------------------- Segmented
export function Segmented({
  value,
  onChange,
  options,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Option[];
  label: string;
}) {
  return (
    <TG.Root
      type="single"
      aria-label={label}
      value={value}
      onValueChange={(v) => v && onChange(v)}
      className="inline-flex h-8 items-center rounded-md bg-sunken p-0.5"
    >
      {options.map((o) => (
        <TG.Item
          key={o.value}
          value={o.value}
          className="inline-flex h-7 items-center gap-1.5 rounded-[5px] px-2.5 text-[13px] text-ink-2 hover:text-ink data-[state=on]:bg-surface data-[state=on]:font-medium data-[state=on]:text-ink data-[state=on]:shadow-xs"
        >
          {o.label}
        </TG.Item>
      ))}
    </TG.Root>
  );
}

// ------------------------------------------------------------------ Switch
export function Switch({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <label className={cx("inline-flex items-center gap-2 text-[13px]", disabled ? "text-muted" : "text-ink-2")}>
      <Sw.Root
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
        className="relative h-[18px] w-8 rounded-full bg-[var(--hairline-strong)] transition-colors data-[state=checked]:bg-accent disabled:opacity-50"
      >
        <Sw.Thumb className="block h-3.5 w-3.5 translate-x-0.5 rounded-full bg-white shadow-sm transition-transform data-[state=checked]:translate-x-[16px]" />
      </Sw.Root>
      {label}
    </label>
  );
}

// ----------------------------------------------------------------- Tooltip
export function Tip({ content, children, side = "top" }: { content: ReactNode; children: ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <T.Root delayDuration={250}>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-72 rounded-md bg-ink px-2.5 py-1.5 text-[12px] leading-snug text-page shadow-pop"
        >
          {content}
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}
export const TooltipProvider = T.Provider;

// ------------------------------------------------------------------ Dialog
export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  width = "max-w-lg",
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  title: string;
  description?: string;
  children: ReactNode;
  width?: string;
}) {
  return (
    <D.Root open={open} onOpenChange={onOpenChange}>
      <D.Portal>
        <D.Overlay className="fixed inset-0 z-40 bg-black/35 backdrop-blur-[2px]" />
        <D.Content
          className={cx(
            "fixed left-1/2 top-[8vh] z-50 max-h-[84vh] w-[calc(100vw-32px)] -translate-x-1/2 overflow-y-auto rounded-xl border border-hairline-strong bg-raised p-6 shadow-pop outline-none",
            width,
          )}
        >
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <D.Title className="text-[16px] font-semibold">{title}</D.Title>
              {description ? (
                <D.Description className="mt-1 text-[13px] text-ink-2">{description}</D.Description>
              ) : (
                <D.Description className="sr-only">{title}</D.Description>
              )}
            </div>
            <D.Close asChild>
              <Button variant="ghost" size="icon-sm" aria-label="Fermer">
                <X size={16} />
              </Button>
            </D.Close>
          </div>
          {children}
        </D.Content>
      </D.Portal>
    </D.Root>
  );
}

// ------------------------------------------------------------------- Input
export function SearchInput(props: ComponentProps<"input">) {
  return (
    <div className="relative">
      <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted" />
      <input
        {...props}
        className={cx(
          "h-8 w-56 rounded-md border border-hairline-strong bg-surface pl-8 pr-2.5 text-[13px] text-ink shadow-xs outline-none placeholder:text-muted focus:border-accent",
          props.className,
        )}
      />
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-4 text-[13px] text-ink-2">
      {label}
      {children}
    </label>
  );
}

export function NumberInput(props: ComponentProps<"input">) {
  return (
    <input
      type="number"
      {...props}
      className="h-8 w-28 rounded-md border border-hairline-strong bg-surface px-2.5 text-right text-[13px] text-ink tabular outline-none focus:border-accent"
    />
  );
}

// ------------------------------------------------------------------- Delta
/** Écart signé, coloré (une hausse de revenu est une bonne nouvelle). */
export function Delta({ value, money = true, className }: { value: number; money?: boolean; className?: string }) {
  if (Math.abs(value) < 0.5) return null;
  const up = value > 0;
  const Icon = up ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={cx("inline-flex items-center gap-0.5 font-medium tabular", up ? "text-good" : "text-critical", className)}>
      <Icon size={13} strokeWidth={2.25} />
      {money ? fmtSignedMoney(value) : fmtSigned(value)}
    </span>
  );
}

// ----------------------------------------------------------------- Metric
export function Metric({ label, value, delta, money, hint }: { label: string; value: string; delta?: number; money?: boolean; hint?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[12px] text-muted">{label}</div>
      <div className="mt-0.5 text-[20px] font-semibold tracking-tight tabular">{value}</div>
      <div className="h-4 text-[12px]">
        {delta !== undefined && <Delta value={delta} money={money} />}
        {hint && <span className="text-muted">{hint}</span>}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- Section
export function Section({ title, info, right, children }: { title: string; info?: ReactNode; right?: ReactNode; children: ReactNode }) {
  return (
    <section>
      <div className="mb-2 flex min-h-8 items-center justify-between gap-4">
        <h2 className="flex items-center gap-1.5 text-[13px] font-semibold text-ink">
          {title}
          {info && <Info>{info}</Info>}
        </h2>
        {right}
      </div>
      <div className="rounded-xl border border-hairline bg-surface p-4">{children}</div>
    </section>
  );
}

export function Info({ children }: { children: ReactNode }) {
  return (
    <Tip content={children}>
      <button className="inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-semibold text-muted ring-1 ring-[var(--hairline-strong)] hover:text-ink" aria-label="Aide">
        ?
      </button>
    </Tip>
  );
}

export function PageHeader({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <h1 className="flex flex-wrap items-center gap-2 text-[18px] font-semibold tracking-tight">{title}</h1>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

export function Dot({ color, size = 8 }: { color: string; size?: number }) {
  return <span className="inline-block shrink-0 rounded-full" style={{ background: color, width: size, height: size }} />;
}

export const dirColor = (d: "DEP" | "ARR") => (d === "DEP" ? "var(--series-dep)" : "var(--series-arr)");
