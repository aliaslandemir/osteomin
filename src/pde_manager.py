from __future__ import annotations
import numpy as np

__all__ = ["PDEManager"]

class PDEManager:
    """
    Fields (dimensionless):
      Ca, PO4, β-GP (bgp), Collagen, ACP, HA, ALP.
    No diffusion (D=0). Media refresh resets free ions and β-GP to reservoir.
    Supplements are scalar "levels" applied by the model each day:
      - mk4_level scales ACP→HA maturation rate
    Tracks per-day HA increment (last_dHA_sum) to consume MK4.
    """
    def __init__(
        self, *, width:int, height:int, dx:float, pde_substeps:int,
        ca_bc_value:float, po4_bc_value:float, bgp_init:float,
        D_ca:float, D_po4:float, D_bgp:float,
        k_acp_nucl:float, Km_supersat:float, n_supersat:float,
        k_mature:float, Km_mature:float, n_mature:float,
        D_collagen:float, collagen_decay:float,
        collagen_prod_cap:float, alp_decay:float,
        aa_consume_per_collagen: float = 0.20,
        dexa_consume_per_alp: float = 0.10,
        mk4_consume_per_dHA: float = 0.05,
    ) -> None:
        self.width, self.height = width, height
        self.dx = dx
        self.sub = max(1, pde_substeps)
        self.dt = 1.0 / float(self.sub)

        self.ca_bc_value = ca_bc_value
        self.po4_bc_value = po4_bc_value
        self.bgp_bc_value = bgp_init

        shape = (height, width)
        self.ca   = np.full(shape, ca_bc_value, dtype=np.float32)
        self.po4  = np.full(shape, po4_bc_value, dtype=np.float32)
        self.bgp  = np.full(shape, bgp_init,   dtype=np.float32)
        self.collagen = np.zeros(shape, dtype=np.float32)
        self.acp  = np.zeros(shape, dtype=np.float32)
        self.ha   = np.zeros(shape, dtype=np.float32)
        self.alp  = np.zeros(shape, dtype=np.float32)

        # kinetics
        self.k_acp_nucl, self.Km_supersat, self.n_supersat = k_acp_nucl, Km_supersat, n_supersat
        self.k_mature, self.Km_mature, self.n_mature = k_mature, Km_mature, n_mature
        self.collagen_decay = collagen_decay
        self.collagen_cap = collagen_prod_cap
        self.alp_decay = alp_decay

        # supplement levels (set daily by model; all start at 1.0)
        self.aa_level = 1.0
        self.dexa_level = 1.0
        self.mk4_level = 1.0

        # consumption couplings (used by model; stored here for reference)
        self.aa_consume_per_collagen = aa_consume_per_collagen
        self.dexa_consume_per_alp = dexa_consume_per_alp
        self.mk4_consume_per_dHA = mk4_consume_per_dHA

        # bookkeeping
        self.last_dHA_sum: float = 0.0

    def supersaturation(self) -> np.ndarray:
        return (self.ca * self.po4) * (1.0 + 1.5 * np.tanh(self.collagen))

    def step(self) -> None:
        # reset daily HA increment
        self.last_dHA_sum = 0.0

        # collagen local dynamics
        self.collagen -= self.dt * self.collagen_decay * self.collagen
        self.collagen = np.minimum(self.collagen, self.collagen_cap)
        self.collagen = np.clip(self.collagen, 0.0, None)

        # ALP turnover
        self.alp -= self.dt * self.alp_decay * self.alp
        self.alp = np.clip(self.alp, 0.0, None)

        # β-GP → PO4 via ALP (consumes bgp)
        r_release = 0.05 * self.alp * self.bgp
        self.po4 += self.dt * r_release
        self.bgp -= self.dt * r_release
        self.bgp = np.clip(self.bgp, 0.0, None)

        # nucleation (ACP) with stoichiometric cap 3:2 Ca:PO4
        S = self.supersaturation()
        Sn = S**self.n_supersat
        nucl = self.k_acp_nucl * Sn / (self.Km_supersat + Sn + 1e-12)
        cap = np.minimum.reduce([nucl, self.ca / 3.0, self.po4 / 2.0])
        self.acp += self.dt * cap
        self.ca  -= self.dt * 3.0 * cap
        self.po4 -= self.dt * 2.0 * cap

        # ACP → HA maturation (scaled by MK4 level)
        a_pow = self.acp**self.n_mature
        k_eff = self.k_mature * (0.2 + 0.8 * self.mk4_level)  # mk4 in [0..1] → 0.2..1.0 scaling
        rate = k_eff * a_pow / (self.Km_mature + a_pow + 1e-12)
        dHA = self.dt * np.minimum(rate, self.acp)
        self.acp -= dHA
        self.ha  += dHA
        self.last_dHA_sum = float(dHA.sum())

    def media_refresh(self) -> None:
        """Reset freely available ions and β-GP reservoir; supplements are replenished by the model."""
        self.ca[:]  = self.ca_bc_value
        self.po4[:] = self.po4_bc_value
        self.bgp[:] = self.bgp_bc_value
