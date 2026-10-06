import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { api } from "./api";

export const useInstanceInfo = (iid: string) =>
  useQuery({ queryKey: ["info", iid], queryFn: () => api.info(iid) });

export const useScenario = (iid: string) =>
  useQuery({ queryKey: ["state", iid], queryFn: () => api.state(iid), placeholderData: (p) => p });

export const useFlightDetail = (iid: string, fid: string | null) =>
  useQuery({
    queryKey: ["flight", iid, fid],
    queryFn: () => api.flight(iid, fid!),
    enabled: fid !== null,
    placeholderData: (p) => p,
  });

export const usePair = (iid: string, origin: string | null, dest: string | null) =>
  useQuery({
    queryKey: ["pair", iid, origin, dest],
    queryFn: () => api.pair(iid, origin!, dest!),
    enabled: origin !== null && dest !== null,
    placeholderData: (p) => p,
  });

/** Notifications éphémères (erreurs d'édition, confirmations). */
export const ToastContext = createContext<(msg: string, kind?: "error" | "info") => void>(() => {});
export const useToast = () => useContext(ToastContext);

/** Mutations d'édition : chaque succès rafraîchit l'état du scénario. */
export function useEdit(iid: string) {
  const qc = useQueryClient();
  const toast = useToast();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["state", iid] });
    qc.invalidateQueries({ queryKey: ["flight", iid] });
    qc.invalidateQueries({ queryKey: ["pair", iid] });
  };
  const onError = (e: Error) => toast(e.message, "error");
  const move = useMutation({
    mutationFn: ({ fid, dep }: { fid: string; dep: number }) => api.move(iid, fid, dep),
    onSuccess: refresh,
    onError,
  });
  const revert = useMutation({
    mutationFn: (fid: string) => api.revert(iid, fid),
    onSuccess: refresh,
    onError,
  });
  const reset = useMutation({ mutationFn: () => api.reset(iid), onSuccess: refresh, onError });
  return { move, revert, reset };
}

export function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Thème système courant (pour recolorer les graphiques). */
export function useColorScheme() {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const h = () => setDark(mq.matches);
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }, []);
  return dark ? "dark" : "light";
}

export function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* stockage indisponible */
  }
}
