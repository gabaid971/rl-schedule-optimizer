"""Estimation du revenu d'un programme (v1 : logit, sans capacité).

Pour chaque marché O&D, la demande est découpée en segments horaires (matin,
midi, soir) ayant chacun une heure de départ préférée. Un passager du segment k
choisit entre les itinéraires proposés (vols directs ou correspondances au hub)
et une option « no-go » (concurrence, autre mode, ne pas voyager) selon un logit :

    part_mk = A_mk / (A_mk + exp(u_nogo)),   A_mk = somme_i exp(u_ik)
    revenu  = somme_m somme_k fare_m * demand_m * w_k * part_mk

L'utilité d'un itinéraire dépend de :
- son type (direct / correspondance) ;
- l'écart entre l'heure de départ à l'origine et l'heure préférée du segment ;
- pour une correspondance, le temps au hub (trop court = risque, trop long = attente).
  En dehors de [mct, max_cnx] la correspondance n'est pas vendue.

Ajouter une correspondance sur un marché déjà bien servi rapporte donc de moins en
moins : l'optimum n'est plus « tout mettre dans une seule vague ».

`RevenueState` maintient les sommes A_mk pour évaluer en O(degré du vol) l'effet
du décalage d'un seul vol, ce dont ont besoin les optimiseurs (RL, recuit…).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schedule_optimizer.core.markets import Markets
from schedule_optimizer.core.schedule import ARR, DEP, HUB, Schedule


@dataclass(frozen=True)
class ChoiceParams:
    mct: int = 45  # temps de correspondance minimum (min)
    max_cnx: int = 360  # au-delà, la correspondance n'est pas vendue (min)
    ideal_cnx: int = 75  # temps de correspondance « confortable » (min)
    asc_direct: float = 1.0  # attractivité de base d'un vol direct
    asc_cnx: float = 0.0  # attractivité de base d'une correspondance
    beta_short: float = 1.0  # pénalité par heure en dessous de ideal_cnx
    beta_wait: float = 0.6  # pénalité par heure d'attente au-delà de ideal_cnx
    beta_delay: float = 0.3  # pénalité par heure d'écart à l'heure préférée
    u_nogo: float = 1.0  # utilité de l'option « no-go »
    # (heure de départ préférée en min, poids) ; les poids somment à 1
    segments: tuple[tuple[int, float], ...] = ((450, 0.35), (750, 0.25), (1080, 0.40))


class RevenueModel:
    def __init__(
        self,
        schedule: Schedule,
        markets: Markets,
        params: ChoiceParams | None = None,
        windows: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """
        windows : bornes (lo, hi) des heures de départ atteignables par chaque vol.
            Si fourni, les paires (arrivée, départ) qui ne peuvent jamais former une
            correspondance valide sont écartées (gros gain mémoire/temps à 1000 vols).
            Les heures évaluées ensuite doivent rester dans ces bornes.
        """
        self.schedule = schedule
        self.markets = markets
        self.params = params or ChoiceParams()
        p = self.params

        self.pref = np.array([t for t, _ in p.segments], dtype=np.float64)
        weights = np.array([w for _, w in p.segments], dtype=np.float64)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("les poids des segments doivent sommer à 1")
        # value[m, k] = revenu si tout le segment k du marché m est capté
        self.value = (markets.fare * markets.demand)[:, None] * weights[None, :]
        self.demand_mk = markets.demand[:, None] * weights[None, :]
        self.e_nogo = float(np.exp(p.u_nogo))
        self.block = schedule.block.astype(np.int64)

        self._build_itineraries(windows)

    # ------------------------------------------------------------------ build
    def _build_itineraries(self, windows: tuple[np.ndarray, np.ndarray] | None) -> None:
        s, mk = self.schedule, self.markets
        n_st = len(s.stations)
        flights = np.arange(s.n_flights)

        # vols directs : marché local hub -> escale ou escale -> hub
        local_out = np.array([mk.get(HUB, i) for i in range(n_st)], dtype=np.int64)
        local_in = np.array([mk.get(i, HUB) for i in range(n_st)], dtype=np.int64)
        direct_mkt = np.where(s.direction == DEP, local_out[s.station], local_in[s.station])
        keep = direct_mkt >= 0
        d_leg1, d_mkt = flights[keep], direct_mkt[keep]

        # correspondances : arrivée depuis A puis départ vers B (A != B)
        cnx_mkt = np.full((n_st, n_st), -1, dtype=np.int64)
        for m in range(mk.n_markets):
            o, d = int(mk.origin[m]), int(mk.dest[m])
            if o != HUB and d != HUB:
                cnx_mkt[o, d] = m
        arr_f = flights[s.direction == ARR]
        dep_f = flights[s.direction == DEP]
        a, d = np.meshgrid(arr_f, dep_f, indexing="ij")
        a, d = a.ravel(), d.ravel()
        c_mkt = cnx_mkt[s.station[a], s.station[d]]
        keep = c_mkt >= 0
        if windows is not None:
            lo, hi = (np.asarray(w, dtype=np.int64) for w in windows)
            cnx_min = lo[d] - (hi[a] + self.block[a])
            cnx_max = hi[d] - (lo[a] + self.block[a])
            keep &= (cnx_max >= self.params.mct) & (cnx_min <= self.params.max_cnx)
        a, d, c_mkt = a[keep], d[keep], c_mkt[keep]

        self.it_leg1 = np.concatenate([d_leg1, a])  # vol au départ de l'origine
        self.it_leg2 = np.concatenate([np.full(len(d_leg1), -1), d])  # -1 si direct
        self.it_market = np.concatenate([d_mkt, c_mkt])
        self.n_itineraries = len(self.it_market)

        # index vol -> itinéraires qui l'utilisent (format CSR)
        its = np.arange(self.n_itineraries)
        is_cnx = self.it_leg2 >= 0
        f_all = np.concatenate([self.it_leg1, self.it_leg2[is_cnx]])
        i_all = np.concatenate([its, its[is_cnx]])
        order = np.argsort(f_all, kind="stable")
        self._flight_its = i_all[order]
        counts = np.bincount(f_all, minlength=s.n_flights)
        self._indptr = np.concatenate([[0], np.cumsum(counts)])

    def itineraries_of(self, flight: int) -> np.ndarray:
        return self._flight_its[self._indptr[flight] : self._indptr[flight + 1]]

    # -------------------------------------------------------------- évaluation
    def connection_time(self, dep: np.ndarray, its: np.ndarray) -> np.ndarray:
        """Temps de correspondance au hub (sans objet pour un direct : renvoie 0)."""
        l1, l2 = self.it_leg1[its], self.it_leg2[its]
        l2 = np.where(l2 >= 0, l2, l1)
        cnx = dep[l2] - (dep[l1] + self.block[l1])
        return np.where(self.it_leg2[its] >= 0, cnx, 0)

    def attractiveness(self, dep: np.ndarray, its: np.ndarray) -> np.ndarray:
        """exp(utilité) par itinéraire et par segment, 0 si non vendable. Shape (len(its), K)."""
        p = self.params
        is_cnx = self.it_leg2[its] >= 0
        cnx = self.connection_time(dep, its)
        u = np.where(is_cnx, p.asc_cnx, p.asc_direct).astype(np.float64)
        penalty = (
            p.beta_short * np.maximum(p.ideal_cnx - cnx, 0)
            + p.beta_wait * np.maximum(cnx - p.ideal_cnx, 0)
        ) / 60.0
        u -= np.where(is_cnx, penalty, 0.0)
        sellable = ~is_cnx | ((cnx >= p.mct) & (cnx <= p.max_cnx))

        t0 = dep[self.it_leg1[its]].astype(np.float64)
        u_k = u[:, None] - p.beta_delay * np.abs(t0[:, None] - self.pref[None, :]) / 60.0
        return np.exp(u_k) * sellable[:, None]

    def market_sums(self, attr: np.ndarray) -> np.ndarray:
        """A[m, k] = somme des attractivités des itinéraires du marché m."""
        n_m = self.markets.n_markets
        return np.stack(
            [
                np.bincount(self.it_market, weights=attr[:, k], minlength=n_m)
                for k in range(attr.shape[1])
            ],
            axis=1,
        )

    def revenue_from_sums(self, sums: np.ndarray, value: np.ndarray | None = None) -> float:
        value = self.value if value is None else value
        return float(np.sum(value * sums / (sums + self.e_nogo)))

    def _dep(self, dep: np.ndarray | None) -> np.ndarray:
        return (self.schedule.dep if dep is None else np.asarray(dep)).astype(np.int64)

    def evaluate(self, dep: np.ndarray | None = None) -> float:
        dep = self._dep(dep)
        attr = self.attractiveness(dep, np.arange(self.n_itineraries))
        return self.revenue_from_sums(self.market_sums(attr))

    def state(self, dep: np.ndarray | None = None) -> RevenueState:
        return RevenueState(self, self._dep(dep))

    def breakdown(self, dep: np.ndarray | None = None) -> RevenueBreakdown:
        dep = self._dep(dep)
        attr = self.attractiveness(dep, np.arange(self.n_itineraries))
        sums = self.market_sums(attr)
        # le logit répartit la demande captée au prorata des attractivités
        per_unit = self.demand_mk / (sums + self.e_nogo)  # (M, K)
        it_pax = np.sum(attr * per_unit[self.it_market], axis=1)
        it_rev = it_pax * self.markets.fare[self.it_market]

        n_f = self.schedule.n_flights
        is_cnx = self.it_leg2 >= 0
        l1, l2 = self.it_leg1, self.it_leg2
        # revenu d'une correspondance réparti entre les deux vols au prorata du temps de vol
        b1 = self.block[l1].astype(np.float64)
        b2 = np.where(is_cnx, self.block[np.where(is_cnx, l2, l1)], 0).astype(np.float64)
        share1 = b1 / (b1 + b2)
        flight_pax = np.bincount(l1, weights=it_pax, minlength=n_f) + np.bincount(
            l2[is_cnx], weights=it_pax[is_cnx], minlength=n_f
        )
        flight_rev = np.bincount(l1, weights=it_rev * share1, minlength=n_f) + np.bincount(
            l2[is_cnx], weights=(it_rev * (1 - share1))[is_cnx], minlength=n_f
        )
        market_pax = np.bincount(self.it_market, weights=it_pax, minlength=self.markets.n_markets)
        return RevenueBreakdown(
            total=float(it_rev.sum()),
            market_pax=market_pax,
            market_revenue=market_pax * self.markets.fare,
            itinerary_pax=it_pax,
            itinerary_revenue=it_rev,
            flight_pax=flight_pax,
            flight_revenue=flight_rev,
        )


@dataclass(frozen=True, eq=False)
class RevenueBreakdown:
    total: float
    market_pax: np.ndarray
    market_revenue: np.ndarray
    itinerary_pax: np.ndarray
    itinerary_revenue: np.ndarray
    flight_pax: np.ndarray  # passagers à bord (directs + correspondances)
    flight_revenue: np.ndarray  # revenu attribué au vol (prorata temps de vol)


class RevenueState:
    """Programme courant + sommes logit en cache, pour évaluer des décalages un à un."""

    def __init__(self, model: RevenueModel, dep: np.ndarray) -> None:
        self.model = model
        self.dep = dep.astype(np.int64).copy()
        self.attr = model.attractiveness(self.dep, np.arange(model.n_itineraries))
        self.sums = model.market_sums(self.attr)
        self.revenue = model.revenue_from_sums(self.sums)

    def delta(self, flight: int, new_dep: int) -> float:
        """Variation de revenu si `flight` partait à `new_dep` (sans l'appliquer)."""
        return self._delta(flight, new_dep)[0]

    def apply(self, flight: int, new_dep: int) -> float:
        """Applique le décalage et renvoie la variation de revenu."""
        d, its, new_attr, mkts, new_sums = self._delta(flight, new_dep)
        self.dep[flight] = new_dep
        self.attr[its] = new_attr
        self.sums[mkts] = new_sums
        self.revenue += d
        return d

    def _delta(self, flight: int, new_dep: int):
        m = self.model
        its = m.itineraries_of(flight)
        old = self.dep[flight]
        self.dep[flight] = new_dep
        try:
            new_attr = m.attractiveness(self.dep, its)
        finally:
            self.dep[flight] = old
        mkts, inv = np.unique(m.it_market[its], return_inverse=True)
        d_sums = np.zeros((len(mkts), new_attr.shape[1]))
        np.add.at(d_sums, inv, new_attr - self.attr[its])
        old_sums = self.sums[mkts]
        new_sums = old_sums + d_sums
        value = m.value[mkts]
        d = m.revenue_from_sums(new_sums, value) - m.revenue_from_sums(old_sums, value)
        return d, its, new_attr, mkts, new_sums
