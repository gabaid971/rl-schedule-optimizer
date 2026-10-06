import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";
import type { InstanceInfo } from "../api";
import { fmtTime } from "../format";
import { Dialog } from "../ui";

function TeX({ children, block }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: block, throwOnError: false }), [children, block]);
  return block ? (
    <div className="my-3 overflow-x-auto rounded-lg bg-sunken px-4 py-1" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}

/** Explication du modèle de revenu et paramètres de l'instance. */
export function ModelDialog({ info, open, onOpenChange }: { info: InstanceInfo; open: boolean; onOpenChange: (v: boolean) => void }) {
  const p = info.params;
  const params: [string, string, string][] = [
    ["\\text{MCT}", `${p.mct} min`, "temps de correspondance minimum"],
    ["c^{\\max}", `${p.max_cnx} min`, "au-delà, la correspondance n'est pas vendue"],
    ["c^{*}", `${p.ideal_cnx} min`, "temps de correspondance confortable"],
    ["\\mathrm{ASC}_{\\text{direct}}", String(p.asc_direct), "attractivité de base d'un vol direct"],
    ["\\mathrm{ASC}_{\\text{cnx}}", String(p.asc_cnx), "attractivité de base d'une correspondance"],
    ["\\beta_{\\text{court}}", `${p.beta_short} / h`, "pénalité sous c* (risque de rater la correspondance)"],
    ["\\beta_{\\text{att}}", `${p.beta_wait} / h`, "pénalité d'attente au-delà de c*"],
    ["\\beta_{\\text{h}}", `${p.beta_delay} / h`, "pénalité d'écart à l'heure préférée"],
    ["u_0", String(p.u_nogo), "attrait de l'option « ne pas voler avec nous »"],
  ];
  const segments = p.segments.map(([t, w]) => `${fmtTime(t)} (${Math.round(w * 100)} %)`).join(", ");

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Comment le revenu est calculé"
      description="Un modèle de choix simple : pour chaque trajet, les passagers choisissent entre nos vols et ne pas voler avec nous."
      width="max-w-2xl"
    >
      <div className="space-y-5 text-[13.5px] leading-relaxed text-ink-2">
        <section>
          <h3 className="mb-1 font-semibold text-ink">1. Ce qui est vendu</h3>
          <p>
            Chaque marché <TeX>m</TeX> (paire origine–destination) a une demande journalière <TeX>D_m</TeX> et un tarif{" "}
            <TeX>p_m</TeX>. Un marché local (escale ↔ {info.hub}) est servi par un vol direct ; un marché en correspondance
            (escale A → escale B) par toute paire arrivée/départ au hub dont le temps de correspondance <TeX>c</TeX> est
            compris entre <TeX>{`\\text{MCT}`}</TeX> et <TeX>{`c^{\\max}`}</TeX>. La demande est répartie en segments{" "}
            <TeX>k</TeX> de poids <TeX>w_k</TeX>, chacun avec une heure préférée <TeX>t_k</TeX> : {segments}.
          </p>
        </section>

        <section>
          <h3 className="mb-1 font-semibold text-ink">2. Attractivité d'un itinéraire</h3>
          <p>
            Pour un itinéraire <TeX>i</TeX> partant à l'heure <TeX>t_i</TeX> :
          </p>
          <TeX block>
            {`u_{ik} = \\mathrm{ASC}_i \\;-\\; \\beta_{\\text{h}}\\,|t_i - t_k| \\;-\\; \\beta_{\\text{court}}\\,(c^{*} - c_i)^{+} \\;-\\; \\beta_{\\text{att}}\\,(c_i - c^{*})^{+}`}
          </TeX>
          <p>Les écarts sont en heures ; les deux derniers termes ne concernent que les correspondances.</p>
        </section>

        <section>
          <h3 className="mb-1 font-semibold text-ink">3. Part captée et revenu</h3>
          <TeX block>
            {`S_{mk} = \\frac{A_{mk}}{A_{mk} + e^{u_0}}, \\qquad A_{mk} = \\sum_{i \\in m} e^{u_{ik}}, \\qquad R = \\sum_m \\sum_k p_m\\, D_m\\, w_k\\, S_{mk}`}
          </TeX>
          <p>
            La part captée sature : une deuxième offre sur un marché déjà bien servi rapporte moins que la première. Pour
            l'affichage par vol, le revenu d'une correspondance est réparti entre ses deux vols au prorata du temps de vol.
            La capacité des avions n'est pas encore prise en compte.
          </p>
        </section>

        <section>
          <h3 className="mb-2 font-semibold text-ink">Paramètres</h3>
          <table className="w-full tabular">
            <tbody>
              {params.map(([k, v, d]) => (
                <tr key={k} className="border-t border-hairline">
                  <td className="w-28 py-1.5">
                    <TeX>{k}</TeX>
                  </td>
                  <td className="w-20 py-1.5 font-medium text-ink">{v}</td>
                  <td className="py-1.5">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section>
          <h3 className="mb-1 font-semibold text-ink">Contraintes</h3>
          <ul className="list-disc space-y-0.5 pl-5">
            {info.constraints.map((c) => (
              <li key={c.name}>
                {c.description}
                {c.name === "fenetre" && ` : ±${info.max_shift} min, par pas de ${info.grid} min`}
              </li>
            ))}
          </ul>
        </section>

        <p className="border-t border-hairline pt-4 text-[12.5px] text-muted">
          Calcul vectorisé en numpy sur {info.n_itineraries.toLocaleString("fr-FR")} itinéraires. Les sommes{" "}
          <TeX>{"A_{mk}"}</TeX> sont gardées en cache : décaler un vol ne recalcule que les itinéraires qui l'utilisent.
        </p>
      </div>
    </Dialog>
  );
}
