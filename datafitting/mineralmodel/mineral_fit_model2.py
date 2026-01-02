#!/usr/bin/env python3
"""
unifiedmineralmodel_v3_salib.py

Endpoint-derived mineral proxy model (ARS area %, 0..Hmax) with post-day-14 activation,
allowing Dex/MK4 to INCREASE or DECREASE mineralization (eta_j unconstrained),
PLUS SALib Sobol sensitivity.

Core model:
  dH/dt = (k(t)/Hmax) * (Hmax - H)
  S(t) = 1/(1+exp(-Gamma*(t-t_switch)))
  k_C1(t) = k_b
  k_C2(t) = k_b + k_AA * S(t)
  k_Cj(t) = k_b + (eta_j * k_AA) * S(t)   (j=3..8; eta can be <1 or >1)

Closed-form solution:
  H(t) = Hmax - (Hmax-H0) * exp( -K(t)/Hmax )
  K(t) = k_b*t + eta*k_AA*I(t)
  I(t) = ∫_0^t S(u) du  (closed-form)

Endpoint anchoring at T_end:
  K(T_end) = -Hmax ln((Hmax-H(T_end))/(Hmax-H0))
  k_b = K(C1)/T
  k_AA = (K(C2) - k_b*T)/I(T)
  eta_j = (K(Cj) - k_b*T)/(k_AA*I(T))   (if k_AA>0)

Safety:
  If k_AA>0, we clip eta so k_b + eta*k_AA >= 0 to avoid negative k(t).
  If you really want to allow negative k(t), pass --allow_negative_rates.

Sensitivity (SALib Sobol):
  Metric options:
    - auc     : AUC of H(t) from 0..T_end
    - t50     : time to reach 50% of endpoint of selected condition
    - hend    : H(T_end) (usually uninformative if endpoints are included because it's anchored)

  Problems:
    - main params only (Gamma, t_switch, Hmax [,H0])
    - optionally include endpoint means as uncertain inputs (C1..C8)

Install SALib if needed:
  pip install SALib
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Math helpers
# ----------------------------
def sigmoid(t: np.ndarray, t0: float, Gamma: float) -> np.ndarray:
    x = Gamma * (t - t0)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def integral_sigmoid_0_t(t: np.ndarray, t0: float, Gamma: float) -> np.ndarray:
    """
    I(t)=∫_0^t 1/(1+exp(-Gamma*(u-t0))) du
       = (1/Gamma)[ ln(1+exp(Gamma*(t-t0))) - ln(1+exp(-Gamma*t0)) ]
    """
    if Gamma <= 0:
        raise ValueError("Gamma must be > 0")
    a = float(Gamma)
    zt = np.clip(a * (t - t0), -60.0, 60.0)
    z0 = np.clip(-a * t0, -60.0, 60.0)
    return (np.log1p(np.exp(zt)) - np.log1p(np.exp(z0))) / a


def required_integral_k(H_end: float, Hmax: float, H0: float) -> float:
    """
    K(T)=∫k dt = -Hmax * ln((Hmax-H_end)/(Hmax-H0))
    """
    if not (0 <= H0 < Hmax):
        raise ValueError("Require 0 <= H0 < Hmax.")
    if not (0 <= H_end < Hmax):
        raise ValueError("Require 0 <= H_end < Hmax. If endpoint near plateau, increase Hmax.")
    Y0 = Hmax - H0
    YT = Hmax - H_end
    if YT <= 0:
        raise ValueError("Endpoint too close to Hmax; increase Hmax.")
    return float(-Hmax * np.log(YT / Y0))


def H_closed_form(
    t: np.ndarray,
    H0: float,
    Hmax: float,
    k_b: float,
    k_AA: float,
    eta: float,
    t_switch: float,
    Gamma: float,
) -> np.ndarray:
    I_t = integral_sigmoid_0_t(t, t_switch, Gamma)
    K_t = float(k_b) * t + float(eta) * float(k_AA) * I_t
    Y0 = float(Hmax) - float(H0)
    Yt = Y0 * np.exp(-K_t / float(Hmax))
    Ht = float(Hmax) - Yt
    return np.clip(Ht, 0.0, float(Hmax))


def auc(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, t))


def t50_of_endpoint(t: np.ndarray, y: np.ndarray, y_end: float) -> float:
    """
    Time to reach 50% of endpoint value (linear interpolation in time).
    Returns np.nan if never crosses.
    """
    target = 0.5 * float(y_end)
    idx = np.where(y >= target)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    # linear interp between i-1 and i
    x0, x1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return float(x1)
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


# ----------------------------
# Plot helpers
# ----------------------------
def set_pub_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def save_fig(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", pad_inches=0.24)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.24)


def nice_xticks(T_end: float) -> list[float]:
    ticks = [0, 5, 10, 14, 20, 25, T_end]
    return sorted(set(float(x) for x in ticks))


def plot_activation(t, t_switch, Gamma, T_end, outdir):
    set_pub_style()
    S = sigmoid(t, t_switch, Gamma)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(t, S, lw=3.0, label="S(t)")
    ax.axvline(t_switch, ls="--", lw=1.4, alpha=0.6)
    ax.set_xlim(0, T_end + 0.6)
    ax.set_xticks(nice_xticks(T_end))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Day (post-seeding)", labelpad=10)
    ax.set_ylabel("Activation")
    ax.set_title("Smooth post-day-14 activation", pad=10)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, outdir, "mineral_activation")
    plt.close(fig)


def plot_timecourses(t, curves, endpoints, t_switch, T_end, outdir):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    for cid in order:
        if cid in curves:
            ax.plot(t, curves[cid], lw=2.6, label=cid)

    for cid in order:
        if endpoints.get(cid, None) is None or cid not in curves:
            continue
        ax.scatter([t[-1]], [float(endpoints[cid])], s=80, edgecolors="black", linewidths=0.9, zorder=6)

    ax.axvline(t_switch, ls="--", lw=1.4, alpha=0.6)
    ax.set_xlim(0, T_end + 0.6)
    ax.set_xticks(nice_xticks(T_end))
    ax.set_xlabel("Day (post-seeding)", labelpad=10)
    ax.set_ylabel("ARS-positive area (% of field)")
    ax.set_title("Endpoint-derived mineral proxy trajectories (minimal saturating model)", pad=10)
    ax.grid(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.subplots_adjust(bottom=0.16)
    save_fig(fig, outdir, "mineral_timecourses_all")
    plt.close(fig)


def plot_sobol_bar(names, S1, ST, outdir: Path, keep_prefixes=("Gamma", "t_switch", "Hmax", "H0")):
    # keep only main parameters in the plot
    keep_idx = [i for i, n in enumerate(names) if n in keep_prefixes]
    names_k = [names[i] for i in keep_idx]
    S1_k = np.array([S1[i] for i in keep_idx], dtype=float)
    ST_k = np.array([ST[i] for i in keep_idx], dtype=float)

    set_pub_style()
    x = np.arange(len(names_k))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - w/2, S1_k, w, label=r"$S_1$")
    ax.bar(x + w/2, ST_k, w, label=r"$S_T$")

    ax.set_xticks(x)
    ax.set_xticklabels(names_k, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Sobol index")
    ax.set_title("Sobol sensitivity (model parameters)", pad=10)
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.22)

    save_fig(fig, outdir, "sobol_bar_params")
    plt.close(fig)



# ----------------------------
# Model fit from endpoints
# ----------------------------
def derive_rates_and_etas(endpoints: dict, Hmax: float, H0: float, T: float, t_switch: float, Gamma: float, allow_negative_rates: bool):
    tT = np.array([T], dtype=float)
    I_T = float(integral_sigmoid_0_t(tT, t_switch, Gamma)[0])

    # k_b from C1
    K1 = required_integral_k(float(endpoints["C1"]), Hmax, H0)
    k_b = K1 / T

    # k_AA from C2
    K2 = required_integral_k(float(endpoints["C2"]), Hmax, H0)
    k_AA = (K2 - k_b * T) / I_T
    if k_AA < 0:
        k_AA = 0.0

    # eta for C3..C8
    eta = {"C1": 0.0, "C2": 1.0}
    for cid in ["C3", "C4", "C5", "C6", "C7", "C8"]:
        if endpoints.get(cid, None) is None:
            continue
        Kj = required_integral_k(float(endpoints[cid]), Hmax, H0)
        if k_AA <= 0:
            eta[cid] = 1.0
        else:
            eta[cid] = (Kj - k_b * T) / (k_AA * I_T)

    # keep k(t) >= 0 if requested
    if (not allow_negative_rates) and (k_AA > 0):
        eta_min_rate = -k_b / k_AA  # ensures k_b + eta*k_AA >= 0 when S=1
        for cid in list(eta.keys()):
            if cid == "C1":
                continue
            if eta[cid] < eta_min_rate:
                eta[cid] = eta_min_rate

    return k_b, k_AA, eta, I_T


def simulate_all(endpoints: dict, Hmax: float, H0: float, T_end: float, dt_days: float, t_switch: float, Gamma: float, allow_negative_rates: bool):
    t = np.arange(0.0, T_end + dt_days * 0.5, dt_days)
    k_b, k_AA, eta, I_T = derive_rates_and_etas(endpoints, Hmax, H0, T_end, t_switch, Gamma, allow_negative_rates)

    curves = {}
    curves["C1"] = H_closed_form(t, H0, Hmax, k_b, 0.0, 0.0, t_switch, Gamma)
    curves["C2"] = H_closed_form(t, H0, Hmax, k_b, k_AA, 1.0, t_switch, Gamma)
    for cid in ["C3", "C4", "C5", "C6", "C7", "C8"]:
        if endpoints.get(cid, None) is None:
            continue
        curves[cid] = H_closed_form(t, H0, Hmax, k_b, k_AA, float(eta[cid]), t_switch, Gamma)

    return t, curves, (k_b, k_AA, eta, I_T)


# ----------------------------
# SALib Sobol
# ----------------------------
def run_salib_sobol(
    endpoints_base: dict,
    args,
    outdir: Path,
):
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except Exception as e:
        raise RuntimeError(
            "SALib is not available. Install with: pip install SALib\n"
            f"Original import error: {e}"
        )

    # define uncertain parameters
    names = []
    bounds = []

    # main model parameters
    names += ["Gamma", "t_switch", "Hmax"]
    bounds += [
        [args.sens_Gamma_min, args.sens_Gamma_max],
        [args.sens_tswitch_min, args.sens_tswitch_max],
        [args.sens_Hmax_min, args.sens_Hmax_max],
    ]
    if args.sens_include_H0:
        names += ["H0"]
        bounds += [[args.sens_H0_min, args.sens_H0_max]]

    # optionally include endpoints as uncertain inputs (recommended if you want "data uncertainty")
    endpoint_keys = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    if args.sens_include_endpoints:
        for k in endpoint_keys:
            if endpoints_base.get(k, None) is None:
                continue
            # +/- percent bounds around the provided mean
            mu = float(endpoints_base[k])
            lo = mu * (1.0 - args.sens_endpoint_frac)
            hi = mu * (1.0 + args.sens_endpoint_frac)
            # keep within [0, Hmax] later; for sampling bounds keep nonnegative
            names += [k]
            bounds += [[max(0.0, lo), max(0.0, hi)]]

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    # Saltelli sampling
    X = saltelli.sample(problem, args.sens_N, calc_second_order=args.sens_second_order)
    rng = np.random.default_rng(args.sens_seed)

    # evaluate metric
    Y = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        # unpack
        p = {names[j]: float(X[i, j]) for j in range(len(names))}
        Gamma = p["Gamma"]
        t_switch = p["t_switch"]
        Hmax = p["Hmax"]
        H0 = p.get("H0", float(args.H0))

        # endpoints: either fixed baseline or sampled
        endpoints = dict(endpoints_base)
        if args.sens_include_endpoints:
            for k in endpoint_keys:
                if k in p:
                    endpoints[k] = float(p[k])

        # guard: endpoints must be < Hmax
        # if any endpoint >= Hmax, rescale Hmax up slightly for this evaluation
        max_end = max(float(v) for v in endpoints.values() if v is not None)
        if max_end >= Hmax:
            Hmax = max_end + max(1.0, 0.05 * max_end)

        # simulate
        t, curves, _ = simulate_all(
            endpoints=endpoints,
            Hmax=Hmax,
            H0=H0,
            T_end=float(args.T_end),
            dt_days=float(args.dt_days),
            t_switch=t_switch,
            Gamma=Gamma,
            allow_negative_rates=bool(args.allow_negative_rates),
        )

        cid = args.sens_condition
        if cid not in curves:
            cid = "C2"

        if args.sens_metric == "auc":
            Y[i] = auc(t, curves[cid])
        elif args.sens_metric == "t50":
            Y[i] = t50_of_endpoint(t, curves[cid], endpoints.get(cid, curves[cid][-1]))
        elif args.sens_metric == "hend":
            Y[i] = float(curves[cid][-1])
        else:
            raise ValueError("Unknown sens_metric")

        # small random jitter to break perfect ties if metric degenerates (rare but possible)
        if args.sens_metric == "hend" and args.sens_include_endpoints:
            Y[i] += 1e-9 * rng.standard_normal()

    # analyze
    Si = sobol.analyze(problem, Y, calc_second_order=args.sens_second_order, print_to_console=False)

    S1 = np.clip(Si["S1"], 0, 1)
    ST = np.clip(Si["ST"], 0, 1)
    out = pd.DataFrame(
        {
            "param": names,
            "S1": S1,
            "S1_conf": Si.get("S1_conf", np.full_like(S1, np.nan)),
            "ST": ST,
            "ST_conf": Si.get("ST_conf", np.full_like(ST, np.nan)),
        }
    )
    out.to_csv(outdir / "sobol_indices.csv", index=False)
    plot_sobol_bar(names, S1, ST, outdir)


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Endpoint-derived mineral model + SALib Sobol sensitivity.")

    # endpoints (means)
    p.add_argument("--C1", type=float, required=True)
    p.add_argument("--C2", type=float, required=True)
    p.add_argument("--C3", type=float, default=None)
    p.add_argument("--C4", type=float, default=None)
    p.add_argument("--C5", type=float, default=None)
    p.add_argument("--C6", type=float, default=None)
    p.add_argument("--C7", type=float, default=None)
    p.add_argument("--C8", type=float, default=None)

    # fixed controls
    p.add_argument("--Hmax", type=float, default=100.0)
    p.add_argument("--H0", type=float, default=0.0)
    p.add_argument("--t_switch", type=float, default=14.0)
    p.add_argument("--Gamma", type=float, default=2.5)
    p.add_argument("--T_end", type=float, default=28.0)
    p.add_argument("--dt_days", type=float, default=1.0 / 24.0)

    # allow negative k(t) if you really want (not recommended)
    p.add_argument("--allow_negative_rates", action="store_true")

    p.add_argument("--outdir", type=str, default="unifiedmineral_out_v3")

    # SALib Sobol
    p.add_argument("--do_sobol", action="store_true")
    p.add_argument("--sens_N", type=int, default=2048, help="Base sample size for Saltelli (power of 2 recommended).")
    p.add_argument("--sens_seed", type=int, default=7)
    p.add_argument("--sens_second_order", action="store_true", help="Compute second-order indices (more expensive).")
    p.add_argument("--sens_condition", type=str, default="C2", choices=["C1","C2","C3","C4","C5","C6","C7","C8"])
    p.add_argument("--sens_metric", type=str, default="auc", choices=["auc", "t50", "hend"])

    # Sobol problem toggles
    p.add_argument("--sens_include_endpoints", action="store_true",
                   help="Treat C1..C8 endpoints as uncertain inputs (+/- sens_endpoint_frac).")
    p.add_argument("--sens_endpoint_frac", type=float, default=0.15,
                   help="Fractional +/- uncertainty for endpoints if included (e.g. 0.15 = ±15%).")
    p.add_argument("--sens_include_H0", action="store_true")

    # Bounds for main parameters
    p.add_argument("--sens_Gamma_min", type=float, default=0.6)
    p.add_argument("--sens_Gamma_max", type=float, default=8.0)
    p.add_argument("--sens_tswitch_min", type=float, default=12.0)
    p.add_argument("--sens_tswitch_max", type=float, default=16.0)
    p.add_argument("--sens_Hmax_min", type=float, default=80.0)
    p.add_argument("--sens_Hmax_max", type=float, default=140.0)
    p.add_argument("--sens_H0_min", type=float, default=0.0)
    p.add_argument("--sens_H0_max", type=float, default=2.0)

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    endpoints = {
        "C1": float(args.C1),
        "C2": float(args.C2),
        "C3": args.C3,
        "C4": args.C4,
        "C5": args.C5,
        "C6": args.C6,
        "C7": args.C7,
        "C8": args.C8,
    }

    # simulate
    t, curves, (k_b, k_AA, eta, I_T) = simulate_all(
        endpoints=endpoints,
        Hmax=float(args.Hmax),
        H0=float(args.H0),
        T_end=float(args.T_end),
        dt_days=float(args.dt_days),
        t_switch=float(args.t_switch),
        Gamma=float(args.Gamma),
        allow_negative_rates=bool(args.allow_negative_rates),
    )

    # save CSV
    df = pd.DataFrame({"day": t, "S_activation": sigmoid(t, float(args.t_switch), float(args.Gamma))})
    for cid, H in curves.items():
        df[f"H_{cid}"] = H
    df.to_csv(outdir / "mineral_timeseries.csv", index=False)

    # summary
    lines = [
        "Endpoint-derived mineral model v3 (+ SALib Sobol)",
        f"Hmax={args.Hmax}, H0={args.H0}, T_end={args.T_end}, t_switch={args.t_switch}, Gamma={args.Gamma}",
        f"I(T_end)=∫S dt over [0,T] = {I_T:.6g} days",
        "",
        f"Derived k_basal = {k_b:.6g}  (1/day)",
        f"Derived k_AA    = {k_AA:.6g}  (1/day)",
        "",
        "Inferred eta (C2 fixed=1.0; may be <1 or >1):",
    ]
    for cid in ["C2","C3","C4","C5","C6","C7","C8"]:
        if cid == "C2":
            lines.append("C2: eta=1.0 (fixed)")
        elif cid in eta:
            lines.append(f"{cid}: eta={eta[cid]:.6g}")
    lines += ["", "Sanity check: endpoints:"]
    for cid, H in curves.items():
        lines.append(f"{cid}: H(T_end)={H[-1]:.6g}  vs endpoint={endpoints.get(cid)}")
    (outdir / "mineral_fit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # plots
    plot_activation(t, float(args.t_switch), float(args.Gamma), float(args.T_end), outdir)
    plot_timecourses(t, curves, endpoints, float(args.t_switch), float(args.T_end), outdir)

    # SALib Sobol
    if args.do_sobol:
        run_salib_sobol(endpoints, args, outdir)

    print(f"[✓] outdir: {outdir.resolve()}")
    print(f"[✓] k_basal={k_b:.3e}  k_AA={k_AA:.3e}")
    for cid in ["C3","C4","C5","C6","C7","C8"]:
        if cid in eta:
            print(f"[✓] {cid} eta={eta[cid]:.3g}")


if __name__ == "__main__":
    main()





"""

python mineral_fit_model2.py \
  --C1 0.6471 --C2 50.8723 --C3 62.1497 --C4 52.5035 \
  --C5 33.9445 --C6 26.2764 --C7 28.6130 --C8 26.6941 \
  --outdir mineral_out_v3 \
  --do_sobol \
  --sens_include_endpoints \
  --sens_endpoint_frac 0.15 \
  --sens_N 2048 \
  --sens_metric t50 \
  --sens_condition C2


"""