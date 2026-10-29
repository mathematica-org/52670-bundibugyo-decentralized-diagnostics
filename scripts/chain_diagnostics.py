"""MCMC mixing diagnostics from a thinned calibration checkpoint.

The full `emcee_chain.h5` and the run's `report.json` are not distributed, so the
acceptance fraction, log-probability trace and full-resolution autocorrelation time
cannot be read back. What survives is the thinned checkpoint the scenarios were run
from (`data/inputs/source_checkpoint_iter44000.npz`): theta for every walker at every
`thin`-th step after burn-in. This script recovers what that file supports:

- trace plots per calibrated parameter, on the true iteration axis;
- the fraction of walkers that moved between consecutive thinned steps. A walker
  that has not moved in `thin` iterations accepted nothing in that span, so the mean
  move fraction divided by `thin` is a LOWER BOUND on the per-step acceptance rate
  (union bound; the true rate is higher only if acceptances cluster);
- the integrated autocorrelation time per parameter, estimated on the thinned chain
  and rescaled by `thin`;
- split-R-hat treating each walker as a chain. Walkers of an ensemble sampler are not
  independent, so this is a heuristic for "has each walker covered the posterior".

ROW LAYOUT. The checkpoint is emcee's flat layout, row = step * n_walkers + walker,
so the arrays reshape to (n_steps, n_walkers). Reshaping to (n_walkers, n_steps)
interleaves walkers and yields traces that look like white noise.

The thinning is not stored; it is reconstructed from `_iter` exactly as
`bdbv/engine/samplers.py` computed it when writing the checkpoint, and checked
against the row count.

    uv run python scripts/chain_diagnostics.py
    uv run python scripts/chain_diagnostics.py --checkpoint calibration/outputs/checkpoint_ppc.npz

Writes to `figures/`: `chain_traces.png`, `chain_mixing.png` and
`chain_diagnostics.json`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import yaml
from make_figures import BLUE, FIG, GRAY, LIGHT_GRAY, RED, _save

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data" / "inputs" / "source_checkpoint_iter44000.npz"
CAL_CFG = ROOT / "calibration" / "config.yaml"
NON_CAL = ("seed_day", "sim_seed")  # derived from theta / pseudo-marginal blobs
CONV_MULT = 50  # emcee early-stop rule used by the sampler: iter > 50 * tau
N_EXAMPLE_WALKERS = 3
CAVEATS = (
    (
        "Acceptance is a lower bound from thinned positions, not emcee's "
        "acceptance_fraction, which needs the full chain."
    ),
    (
        "tau is estimated from ~100 thinned points spanning ~10 tau, so it is noisy and "
        "likely biased low; walkers that never moved are excluded from it, which biases "
        "it lower still."
    ),
    ("split-R-hat treats ensemble walkers as independent chains; read it as a heuristic."),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _thinning(it: int, n_walkers: int, n_burn: int) -> tuple[int, int]:
    """(discard, thin) as `main_emcee` chose them for the checkpoint at `it`."""
    discard = min(n_burn, max(it // 3, 1))
    kept = max(it - discard, 1)
    return discard, max(1, (kept * n_walkers) // 50000)


def load_chain(path: Path, n_walkers: int, n_burn: int) -> dict:
    """Checkpoint as a (n_steps, n_walkers, n_dim) chain plus its iteration axis."""
    with np.load(path) as z:
        keys = [k for k in z.files if not k.startswith("_") and k not in NON_CAL]
        it = int(np.ravel(z["_iter"])[0])
        tau_stored = float(np.ravel(z["_tau_max"])[0]) if "_tau_max" in z else float("nan")
        flat = np.column_stack([np.asarray(z[k], float) for k in keys])
    n_rows = flat.shape[0]
    if n_rows % n_walkers:
        raise SystemExit(f"{n_rows} rows is not a multiple of {n_walkers} walkers")
    n_steps = n_rows // n_walkers
    discard, thin = _thinning(it, n_walkers, n_burn)
    expected = (it - discard) // thin
    if n_steps != expected:
        raise SystemExit(
            f"{n_steps} thinned steps, but discard={discard}, thin={thin} at iter {it} "
            f"implies {expected}; pass --walkers/--burn matching the run"
        )
    # emcee's get_chain(discard, thin) keeps raw indices discard + thin - 1 + j * thin.
    iters = discard + thin * (np.arange(n_steps) + 1)
    return {
        "keys": keys,
        "chain": flat.reshape(n_steps, n_walkers, -1),
        "iters": iters,
        "iter": it,
        "discard": discard,
        "thin": thin,
        "tau_stored": tau_stored,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _moving(x: np.ndarray) -> np.ndarray:
    """Per-parameter mask of walkers that are not constant over the whole window.
    A never-moving walker has zero variance, which turns emcee's ACF into 0/0."""
    return np.ptp(x, axis=0) > 0


def mean_acf(x: np.ndarray) -> np.ndarray:
    """Walker-averaged normalised ACF of one parameter, (n_steps, n_walkers)."""
    ok = _moving(x)
    return np.mean([emcee.autocorr.function_1d(x[:, w]) for w in np.flatnonzero(ok)], axis=0)


def split_rhat(x: np.ndarray) -> float:
    """Gelman-Rubin R-hat over split halves of each walker."""
    h = x.shape[0] // 2
    c = np.concatenate([x[:h], x[h : 2 * h]], axis=1)
    within = c.var(axis=0, ddof=1).mean()
    between = h * c.mean(axis=0).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * within + between / h) / within))


def diagnose(ch: dict) -> dict:
    X, thin, keys = ch["chain"], ch["thin"], ch["keys"]
    n_steps, n_walkers, _ = X.shape
    moved = np.any(X[1:] != X[:-1], axis=-1)  # (n_steps - 1, n_walkers)
    per_walker = moved.mean(axis=0)
    quarters = np.array_split(np.arange(moved.shape[0]), 4)

    params = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, k in enumerate(keys):
            x = X[:, :, j]
            ok = _moving(x)
            tau = float(emcee.autocorr.integrated_time(x[:, ok], tol=0)[0]) * thin
            h = n_steps // 2
            params[k] = {
                "tau_iterations": tau,
                "chain_length_in_tau": n_steps * thin / tau,
                "split_rhat_across_walkers": split_rhat(x[:, ok]),
                "median_first_half": float(np.median(x[:h])),
                "median_second_half": float(np.median(x[h:])),
                "walkers_used": int(ok.sum()),
            }
    taus = np.array([p["tau_iterations"] for p in params.values()])
    return {
        "moved": moved,
        "per_walker": per_walker,
        "summary": {
            "layout": "emcee flat: row = step * n_walkers + walker",
            "n_walkers": n_walkers,
            "n_thinned_steps": n_steps,
            "checkpoint_iteration": ch["iter"],
            "discard": ch["discard"],
            "thin": thin,
            "iteration_span": [int(ch["iters"][0]), int(ch["iters"][-1])],
            "move_fraction": float(moved.mean()),
            "move_fraction_by_quarter": [float(moved[q].mean()) for q in quarters],
            "walkers_never_moving": int((per_walker == 0).sum()),
            "per_walker_move_fraction_quantiles": {
                f"p{p}": float(np.percentile(per_walker, p)) for p in (10, 25, 50, 75, 90)
            },
            "acceptance_per_step_lower_bound": float(moved.mean() / thin),
            "tau_max_stored": ch["tau_stored"],
            "tau_max_thinned_estimate": float(taus.max()),
            "tau_needed_for_convergence": ch["iter"] / CONV_MULT,
            "params": params,
            "caveats": list(CAVEATS),
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, pad=1)


def fig_traces(ch: dict, d: dict) -> None:
    """One panel per parameter: every walker in light gray, the ensemble 5/50/95%
    across walkers in blue, and a few example walkers in red."""
    X, iters, keys = ch["chain"], ch["iters"] / 1000, ch["keys"]
    n_cols = 2
    n_rows = -(-len(keys) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.1, 1.25 * n_rows), sharex=True)
    # Example walkers: the median mover, plus the least and most mobile walkers that
    # still moved, so the panel shows the range of holding behaviour.
    order = np.argsort(d["per_walker"])
    movers = order[d["per_walker"][order] > 0]
    examples = movers[[0, len(movers) // 2, -1]][:N_EXAMPLE_WALKERS]
    for j, (ax, k) in enumerate(zip(axes.flat, keys)):
        x = X[:, :, j]
        ax.plot(iters, x, color=LIGHT_GRAY, lw=0.3, alpha=0.25, rasterized=True)
        q5, q50, q95 = np.percentile(x, [5, 50, 95], axis=1)
        ax.plot(iters, q50, color=BLUE, lw=1.4, label="Ensemble median")
        ax.plot(iters, q5, color=BLUE, lw=0.8, ls="--", label="Ensemble 5-95%")
        ax.plot(iters, q95, color=BLUE, lw=0.8, ls="--")
        for n, w in enumerate(examples):
            ax.step(
                iters,
                x[:, w],
                where="post",
                color=RED,
                lw=0.8,
                alpha=0.9,
                label="Example walkers" if n == 0 else None,
            )
        ax.set_title(k, fontsize=7, loc="left", pad=2)
        ax.tick_params(labelsize=6)
        _style(ax)
    for ax in axes.flat[len(keys) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Iteration (thousands)", fontsize=7)
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=5.5, ncol=3)
    fig.suptitle(
        f"Traces: {X.shape[1]} walkers, every {ch['thin']}th step after burn-in "
        f"{ch['discard']:,} (checkpoint iteration {ch['iter']:,})",
        fontsize=8,
    )
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    _save(fig, "chain_traces.png")


def fig_mixing(ch: dict, d: dict) -> None:
    s, thin, X = d["summary"], ch["thin"], ch["chain"]
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))

    ax = axes[0, 0]
    ax.plot(ch["iters"][1:] / 1000, d["moved"].mean(axis=1), color=BLUE, lw=1.4)
    ax.axhline(s["move_fraction"], color=GRAY, lw=0.8, ls="--")
    ax.text(
        ch["iters"][1] / 1000,
        s["move_fraction"],
        f"mean {s['move_fraction']:.2f}",
        ha="left",
        va="top",
        fontsize=6,
        color="0.3",
    )
    ax.set_ylim(0, None)
    ax.set_xlabel("Iteration (thousands)")
    ax.set_ylabel(f"Walkers moved in last {thin} steps")
    ax.set_title("Fraction of walkers moving", loc="left")

    ax = axes[0, 1]
    ax.hist(d["per_walker"], bins=30, color=BLUE, edgecolor="white", lw=0.5)
    ax.set_xlabel(f"Fraction of {thin}-step intervals with a move")
    ax.set_ylabel("Walkers")
    ax.set_title(f"Per-walker mobility ({s['walkers_never_moving']} never moved)", loc="left")

    ax = axes[1, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        acfs = np.array([mean_acf(X[:, :, j]) for j in range(X.shape[2])])
    max_lag = min(acfs.shape[1], 40)
    lags = np.arange(max_lag) * thin / 1000
    ax.plot(lags, acfs[:, :max_lag].T, color=LIGHT_GRAY, lw=0.8)
    ax.plot(lags, acfs[:, :max_lag].mean(axis=0), color=BLUE, lw=1.6, label="Mean of parameters")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel("Lag (thousand iterations)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Walker-averaged ACF (14 parameters, gray, overlap)", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    keys = ch["keys"]
    taus = np.array([s["params"][k]["tau_iterations"] for k in keys])
    y = np.arange(len(keys))
    ax.scatter(taus, y, s=16, color=BLUE, zorder=3)
    ax.axvline(
        s["tau_needed_for_convergence"],
        color=GRAY,
        lw=0.8,
        ls="--",
        label=f"Converged: iter / {CONV_MULT} = {s['tau_needed_for_convergence']:,.0f}",
    )
    if np.isfinite(s["tau_max_stored"]):
        ax.axvline(
            s["tau_max_stored"],
            color=RED,
            lw=0.8,
            ls=":",
            label=f"Stored tau_max {s['tau_max_stored']:,.0f}",
        )
    ax.legend(loc="center left", bbox_to_anchor=(0.2, 0.5), frameon=False)
    ax.set_yticks(y, keys)
    ax.set_xlim(0, None)
    ax.set_xlabel("Integrated autocorrelation time (iterations)")
    ax.set_title("tau per parameter (thinned estimate)", loc="left")

    for ax in axes.flat:
        _style(ax)
    fig.tight_layout()
    _save(fig, "chain_mixing.png")


def main() -> None:
    em = yaml.safe_load(CAL_CFG.read_text()).get("emcee", {})
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--walkers", type=int, default=int(em.get("n_walkers", 240)))
    ap.add_argument("--burn", type=int, default=int(em.get("n_burn", 1500)))
    args = ap.parse_args()

    ch = load_chain(args.checkpoint, args.walkers, args.burn)
    d = diagnose(ch)
    s = d["summary"]
    s["checkpoint"] = str(args.checkpoint.resolve().relative_to(ROOT, walk_up=True))
    fig_traces(ch, d)
    fig_mixing(ch, d)
    (FIG / "chain_diagnostics.json").write_text(json.dumps(s, indent=1))

    print(
        f"{s['n_walkers']} walkers x {s['n_thinned_steps']} steps, thin {s['thin']}, "
        f"iterations {s['iteration_span'][0]:,}-{s['iteration_span'][1]:,}"
    )
    print(
        f"move fraction {s['move_fraction']:.3f} (by quarter "
        f"{', '.join(f'{v:.3f}' for v in s['move_fraction_by_quarter'])}); "
        f"{s['walkers_never_moving']} walkers never moved"
    )
    print(f"per-step acceptance >= {s['acceptance_per_step_lower_bound']:.1e}")
    print(
        f"tau_max {s['tau_max_thinned_estimate']:,.0f} thinned vs {s['tau_max_stored']:,.0f} "
        f"stored; convergence needs < {s['tau_needed_for_convergence']:,.0f}"
    )
    print(f"{'parameter':<22} {'tau':>7} {'N/tau':>6} {'R-hat':>6}")
    for k, p in s["params"].items():
        print(
            f"{k:<22} {p['tau_iterations']:7,.0f} {p['chain_length_in_tau']:6.1f} "
            f"{p['split_rhat_across_walkers']:6.2f}"
        )
    print(f"wrote {FIG / 'chain_traces.png'}, chain_mixing.png, chain_diagnostics.json")


if __name__ == "__main__":
    main()
