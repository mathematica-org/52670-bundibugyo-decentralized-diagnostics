"""Posterior predictive check of the status-quo forecast against post-horizon data.

`make_figures.fig_appendix_ppc` draws figA02 from frozen arrays that end at the
2026-08-02 horizon. This script extends that check past the horizon: it replays the
scored posterior draws (`data/figure_inputs/sq_draws.npz`) along the scenario sweep's
status-quo path and compares them with the SitRep data now staged in
`data/BDBV2026-Data`.

The replay is the forecast the manuscript quotes, not a fresh one. Each draw runs
`scenarios/lib.build_tree` under the primary national-response ramp to the sweep's
full horizon (cutoff + 540 d), exactly as `scenarios/run.py` does, so continuation 0 of
draw i is bit-identical to the `base` row (i, k=0) in `results.jsonl`. Simulating to a
shorter horizon would give statistically equivalent but different realisations.

Stage newer data first, then run from the repository root:

    uv run python scripts/stage_data.py --skip-hdx --bdbv-commit ""
    uv run python scripts/forecast_ppc.py [--draws N] [--conts K] [--workers W]
    uv run python scripts/forecast_ppc.py --replot   # redraw from the cached arrays

Writes to `figures/`: `figA02_posterior_predictive_forecast.png`,
`figA02b_forecast_by_zone.png` (per-zone confirmed before, after and across the
horizon), the simulated and observed arrays (`forecast_ppc.npz`), and post-horizon coverage scores
(`forecast_ppc_scores.json`).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scenarios"), str(ROOT / "scripts")]
# The engine resolves its config and output paths from the entry point that imports
# it; import it as the scenario sweep does (see tests/test_prune.py).
_argv0, sys.argv[0] = sys.argv[0], str(ROOT / "scenarios" / "run.py")
import lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from make_figures import FIG, _band, _save, _style_time_axis
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bdbv import engine as E
from bdbv.engine.simulation import _new_in_window, cumulative_by_day
from bdbv.model import data as D
from bdbv.model.model import from_day, to_day
from bdbv.model.tree import tree_to_frame

sys.argv[0] = _argv0

POST = ROOT / "data" / "figure_inputs" / "sq_draws.npz"
FROZEN = ROOT / "data" / "figure_inputs" / "posterior_predictive.npz"
CACHE = FIG / "forecast_ppc.npz"
SCORES = FIG / "forecast_ppc_scores.json"
RAMP_KEYS = ("nat_scale_T", "nat_scale_min", "nat_scale_n")
HELD_OUT = "#c9252d"
FORECAST_SHADE = "0.55"

_S: dict = {}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _engine(cfg: dict) -> tuple[dict, dict]:
    """Shared engine context with the primary ramp arm merged in, as in run.py."""
    iv = cfg["intervention"]
    with contextlib.redirect_stdout(io.StringIO()):
        shared = E.build_context(cfg)["shared"]
    primary = next(a for a in iv["ramp_arms"] if a["role"] == "primary")
    shared["fx"] = {**shared["fx"], **{k: float(primary[k]) for k in RAMP_KEYS}}
    shared["_max_inf"] = int(iv["max_infections"])
    E.init_worker(shared)
    return shared, lib.build_ctx(cfg, shared)


def _init(state: dict) -> None:
    _S.clear()
    _S.update(state)
    E.init_worker(state["shared"])


def _work(task: tuple[int, int]) -> dict:
    """One (draw, continuation): the forecast tree summarised on the extended grid.

    Each summary mirrors `bdbv.engine.scoring.ppc_draw`, with its window extended past
    the cutoff: confirmation is the capacity-queued `t_detect`, a death is confirmed at
    max(t_detect, t_death), and per-zone cumulative counts use `care_zone`.
    """
    i, k = task
    shared, ctx, grid = _S["shared"], _S["ctx"], _S["grid"]
    T, prm, p = lib.build_tree(
        _S["post"], i, shared, ctx["horizon_day"], cutoff_day=ctx["cutoff_day"], k=k
    )
    df = tree_to_frame(T, prm, confirm_capacity=lib._sim_kw(p, shared)["confirm_capacity"])
    td = df["t_detect"].to_numpy(dtype=float)
    m = df["detected"].to_numpy(dtype=bool) & np.isfinite(td)
    cd = m & df["died"].to_numpy(dtype=bool)
    death_day = np.maximum(td[cd], df["t_death"].to_numpy(dtype=float)[cd])
    to_cut = m & (td <= ctx["cutoff_day"])
    to_end = m & (td <= grid["day_ints"][-1])
    care = df["care_zone"].to_numpy(dtype=int)
    return {
        "nat": cumulative_by_day(td[m], grid["day_ints"]),
        "lat": cumulative_by_day(df["t_infect"].to_numpy(dtype=float), grid["day_ints"]),
        "conf_new": _new_in_window(td[m], grid["conf_edges"]),
        "deaths_binned": _new_in_window(death_day, grid["death_edges"]),
        "deaths_cum": cumulative_by_day(death_day, grid["day_ints"]),
        "zone_pre": np.bincount(care[to_cut], minlength=ctx["Z"]).astype(float),
        "zone_cum": np.bincount(care[to_end], minlength=ctx["Z"]).astype(float),
    }


def _extend(edges: np.ndarray, cut: int, end: int, step: int) -> np.ndarray:
    """In-window edges unchanged, then whole `step`-day cells from the cutoff to `end`."""
    return np.concatenate([edges, np.arange(cut + step, end + 1, step)]).astype(int)


def _observed(shared: dict, zones: list[str], grid: dict, cut: int) -> dict:
    """Observed comparators on the extended grid, built as `build_context` builds them."""
    dates = pd.DatetimeIndex([from_day(int(d)) for d in grid["day_ints"]])

    def daily(df: pd.DataFrame, col: str) -> np.ndarray:
        s = df.drop_duplicates("date", keep="last").set_index("date")[col]
        return s.reindex(dates).ffill().fillna(0).to_numpy(dtype=float)

    nat = daily(D.load_national_confirmed_daily(), "national_cumulative_confirmed_cases")
    deaths = daily(D.load_national_deaths_daily(), "national_cumulative_confirmed_deaths")
    cc = D.load_confirmed_cases_daily()
    by_zone = (
        cc[cc["nom"] != "NA"]
        .pivot_table(
            index="date", columns="nom", values="cumulative_confirmed_cases", aggfunc="sum"
        )
        .reindex(dates)
        .ffill()
        .fillna(0)
    )
    dropped = sorted(set(by_zone.columns) - set(zones))
    if dropped:
        print(f"[obs] WARNING zones outside the model footprint, not plotted: {dropped}")
    by_zone = by_zone.reindex(columns=zones, fill_value=0.0)
    i_cut = int(np.searchsorted(grid["day_ints"], cut))
    obs_conf_new = np.diff(np.interp(grid["conf_edges"], grid["day_ints"], nat))
    # The fit window must be the data the model was fitted to; flag any revision.
    fit = np.asarray(shared["obs_conf_new"], float)
    if not np.array_equal(obs_conf_new[: fit.size], fit):
        print("[obs] WARNING staged data revise the in-window confirmed series used in the fit")
    return {
        "obs_nat_cum": nat,
        "obs_conf_new": obs_conf_new,
        "obs_deaths_binned": np.diff(np.interp(grid["death_edges"], grid["day_ints"], deaths)),
        "obs_deaths_cum": deaths,
        "obs_zone_pre": by_zone.iloc[i_cut].to_numpy(dtype=float),
        "obs_zone_cum": by_zone.iloc[-1].to_numpy(dtype=float),
    }


def _data_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=D.data_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def simulate(n_draws: int | None, n_conts: int, workers: int, end: str | None) -> None:
    cfg = yaml.safe_load((ROOT / "scenarios" / "config.yaml").read_text())
    shared, ctx = _engine(cfg)
    cut = ctx["cutoff_day"]

    last_obs = min(
        D.load_national_confirmed_daily()["date"].max(),
        D.load_national_deaths_daily()["date"].max(),
    )
    end_day = to_day(end) if end else to_day(last_obs)
    if end_day <= cut:
        raise SystemExit(
            f"ERROR: staged data end {last_obs.date()}, at or before the horizon "
            f"{from_day(cut).date()}. Stage newer data first:\n"
            '    uv run python scripts/stage_data.py --skip-hdx --bdbv-commit ""'
        )
    end_day = min(end_day, to_day(last_obs), ctx["horizon_day"])
    day_ints = np.arange(int(shared["day_ints"][0]), end_day + 1)
    grid = {
        "day_ints": day_ints,
        "conf_edges": _extend(np.asarray(shared["conf_edges"]), cut, end_day, 3),
        "death_edges": _extend(np.asarray(shared["death_edges"]), cut, end_day, 14),
    }

    post = dict(np.load(POST))
    n_rows = len(post["R0"])
    n_draws = min(n_draws or n_rows, n_rows)
    # The scored rows are already a random permutation (draw_split.py), so a prefix is
    # an unbiased subset, and row i here is row i of the sweep's `base` stream.
    tasks = [(i, k) for i in range(n_draws) for k in range(n_conts)]
    print(
        f"[sim] {n_draws} draws x {n_conts} continuation(s) to "
        f"{from_day(ctx['horizon_day']).date()}; scored "
        f"{from_day(int(day_ints[0])).date()} -> {from_day(end_day).date()} "
        f"(horizon {from_day(cut).date()}) on {workers} worker(s)",
        flush=True,
    )
    state = {"shared": shared, "ctx": ctx, "post": post, "grid": grid}
    with mp.Pool(workers, initializer=_init, initargs=(state,)) as pool:
        out = pool.map(_work, tasks, chunksize=max(1, len(tasks) // (workers * 8)))

    obs = _observed(shared, ctx["zones"], grid, cut)
    frozen = np.load(FROZEN)
    FIG.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        **{f"{k}_pp": np.array([o[k] for o in out]) for k in out[0]},
        **obs,
        **grid,
        zones=np.array(ctx["zones"]),
        cutoff_day=cut,
        epi_date=frozen["epi_date"],
        epi_mean=frozen["epi_mean"],
        epi_lo=frozen["epi_lo"],
        epi_hi=frozen["epi_hi"],
        n_draws=n_draws,
        n_conts=n_conts,
        data_commit=_data_commit(),
    )
    print(f"[sim] wrote {CACHE}")


# ---------------------------------------------------------------------------
# Scores and figure
# ---------------------------------------------------------------------------


def _coverage(sim: np.ndarray, obs: np.ndarray) -> dict:
    """Share of held-out observations inside the central 50% / 90% model intervals,
    and the observation's predictive percentile at the last point."""
    q5, q25, q75, q95 = np.percentile(sim, [5, 25, 75, 95], axis=0)
    return {
        "n_points": int(obs.size),
        "in_50": float(np.mean((obs >= q25) & (obs <= q75))),
        "in_90": float(np.mean((obs >= q5) & (obs <= q95))),
        "last_observed": float(obs[-1]),
        "last_median": float(np.median(sim[:, -1])),
        "last_percentile": float(100 * np.mean(sim[:, -1] < obs[-1])),
    }


def scores(z) -> dict:
    cut = int(z["cutoff_day"])
    days = z["day_ints"]
    post_day = days > cut
    post_conf = z["conf_edges"][1:] > cut
    post_death = z["death_edges"][1:] > cut
    return {
        "horizon": str(from_day(cut).date()),
        "evaluated_through": str(from_day(int(days[-1])).date()),
        "n_draws": int(z["n_draws"]),
        "n_continuations": int(z["n_conts"]),
        "data_commit": str(z["data_commit"]),
        "cumulative_confirmed": _coverage(z["nat_pp"][:, post_day], z["obs_nat_cum"][post_day]),
        "confirmed_3d": _coverage(z["conf_new_pp"][:, post_conf], z["obs_conf_new"][post_conf]),
        "deaths_14d": _coverage(
            z["deaths_binned_pp"][:, post_death], z["obs_deaths_binned"][post_death]
        ),
        "cumulative_deaths": _coverage(
            z["deaths_cum_pp"][:, post_day], z["obs_deaths_cum"][post_day]
        ),
    }


def _mid(edges: np.ndarray) -> np.ndarray:
    return np.array([from_day(int(d)) for d in (edges[:-1] + edges[1:]) / 2], "datetime64[D]")


def _observed_line(ax, x, y):
    ax.plot(x, y, "k.-", lw=1.0, ms=2.6, zorder=3)


def _forecast_period(ax, cut_date, end_date):
    ax.axvspan(cut_date, end_date, color=FORECAST_SHADE, alpha=0.12, lw=0, zorder=0)
    ax.axvline(cut_date, color="0.35", ls="--", lw=0.9, zorder=1)
    # Label the horizon on the x-axis as a minor tick, leaving the month ticks alone.
    ts = pd.Timestamp(cut_date)
    ax.set_xticks([cut_date], [f"{ts:%b} {ts.day}"], minor=True)
    ax.tick_params(axis="x", which="minor", length=3.5, rotation=35, labelsize=6)


def plot(z) -> None:
    cut = int(z["cutoff_day"])
    days = z["day_ints"]
    dates = np.array([from_day(int(d)) for d in days], "datetime64[D]")
    cut_date = np.datetime64(from_day(cut).date(), "D")
    end_date = dates[-1]
    conf_mid, death_mid = _mid(z["conf_edges"]), _mid(z["death_edges"])

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.3))

    series = [
        (axes[0, 0], dates, "nat_pp", "obs_nat_cum", "#2b6cb0",
         "Cumulative confirmed", "Cases"),
        (axes[0, 1], conf_mid, "conf_new_pp", "obs_conf_new", "#dd6b20",
         "3-day confirmed incidence", "Cases / 3 d"),
        (axes[1, 0], death_mid, "deaths_binned_pp", "obs_deaths_binned", "#805ad5",
         "Confirmed deaths", "Deaths / 14-d cell"),
        (axes[1, 1], dates, "deaths_cum_pp", "obs_deaths_cum", "#805ad5",
         "Cumulative confirmed deaths", "Deaths"),
    ]  # fmt: skip
    for ax, x, sim, obs, color, title, ylabel in series:
        _forecast_period(ax, cut_date, end_date)
        _band(ax, x, z[sim], color, "Model")
        _observed_line(ax, x, z[obs])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        _style_time_axis(ax)

    # Named zones, plus every remaining zone summed in its own panel and y-scale.
    spec = axes[2, 0].get_subplotspec()
    axes[2, 0].remove()
    sub = spec.subgridspec(1, 2, width_ratios=[N_ZONES, 2], wspace=0.3)
    ax, ax_o = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    sel, labels = _ranked_zones(z)
    obs_z, sim_z = _with_other(z["obs_zone_cum"], z["zone_cum_pp"], sel)
    q5, q50, q95 = np.percentile(sim_z, [5, 50, 95], axis=0)
    for a, g in ((ax, np.arange(sel.size)), (ax_o, np.array([sel.size]))):
        x = np.arange(g.size)
        a.bar(x - 0.2, obs_z[g], width=0.38, color=HELD_OUT, label="Observed")
        a.bar(
            x + 0.2,
            q50[g],
            width=0.38,
            yerr=[q50[g] - q5[g], q95[g] - q50[g]],
            color="#3182bd",
            ecolor="#08519c",
            capsize=1.5,
            alpha=0.9,
            label="Model median (5-95%)",
        )
        a.set_xticks(x)
        a.set_xticklabels([labels[i] for i in g], rotation=65, ha="right")
    ax_o.set_xlim(-0.6, 0.6)
    ax.set_title(f"Cumulative confirmed by health zone, to {end_date}")
    ax.set_ylabel("Cases")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2, 1]
    _forecast_period(ax, cut_date, end_date)
    _band(ax, dates, z["lat_pp"], "#2f9e44", "Model latent")
    ax.errorbar(
        [np.datetime64(str(z["epi_date"]))],
        [float(z["epi_mean"])],
        yerr=[
            [float(z["epi_mean"]) - float(z["epi_lo"])],
            [float(z["epi_hi"]) - float(z["epi_mean"])],
        ],
        fmt="o",
        color="red",
        capsize=3,
        ms=4,
        label="Latent anchor",
    )
    ax.set_title("Latent burden (no post-horizon observable)")
    ax.set_ylabel("Infections")
    ax.legend(loc="upper left", frameon=False)
    _style_time_axis(ax)

    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22, lw=0.5)

    fig.legend(
        handles=[
            Patch(facecolor="0.4", alpha=0.3, lw=0, label="Model 5-95%"),
            Line2D([], [], color="0.2", lw=1.8, label="Model median"),
            Line2D([], [], color="k", marker=".", ms=5, lw=1.0, label="Observed"),
            Patch(facecolor=FORECAST_SHADE, alpha=0.25, lw=0,
                  label=f"Forecast period, not fitted (horizon {cut_date})"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6,
        bbox_to_anchor=(0.5, 0.0),
    )  # fmt: skip
    fig.text(
        0.5,
        -0.012,
        f"{int(z['n_draws'])} posterior draws x {int(z['n_conts'])} continuation(s), "
        f"status-quo forecast; BDBV2026-Data {str(z['data_commit'])[:7]}",
        ha="center",
        fontsize=5.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, "figA02_posterior_predictive_forecast.png")


# Named health zones in both zone figures; the rest are summed as one "Other" group.
N_ZONES = 20


def _ranked_zones(z, n: int = N_ZONES) -> tuple[np.ndarray, list[str]]:
    """The `n` zones with the most observed confirmed cases over the whole period, and
    tick labels for them plus the summed remainder (with how many of it saw cases)."""
    obs = z["obs_zone_cum"]
    sel = np.argsort(-obs, kind="stable")[:n]
    n_rest = obs.size - sel.size
    n_cases = int((np.delete(obs, sel) > 0).sum())
    other = f"Other {n_rest} zones\n({n_cases} with observed cases)"
    return sel, [str(z["zones"][i]) for i in sel] + [other]


def _with_other(obs: np.ndarray, sim: np.ndarray, sel: np.ndarray):
    """Counts for the zones in `sel`, then one appended column summing all the others
    (per draw on the model side, so its interval is the interval of the sum)."""
    rest = np.setdiff1d(np.arange(obs.size), sel)
    return (
        np.append(obs[sel], obs[rest].sum()),
        np.column_stack([sim[:, sel], sim[:, rest].sum(axis=1)]),
    )


# Period colours (validated for CVD separation); observed is solid, model is a tint.
PERIOD_COLORS = ("#2a78d6", "#eb6834", "0.3")


def plot_zones(z, per_row: int = 10) -> None:
    """Per-zone confirmed cases, observed vs model, for three periods: the fit window
    (start to horizon), the forecast window (horizon to end) and the whole range.
    Zones are ranked by observed whole-range count, split into rows of `per_row` so
    smaller zones get their own y-scale; the zones outside the top `N_ZONES` are summed
    in a separate panel."""
    cut_date = from_day(int(z["cutoff_day"])).date()
    start_date = from_day(int(z["day_ints"][0])).date()
    end_date = from_day(int(z["day_ints"][-1])).date()
    pre, cum = z["zone_pre_pp"], z["zone_cum_pp"]
    obs_pre, obs_cum = z["obs_zone_pre"], z["obs_zone_cum"]
    periods = [
        (f"Fit window, {start_date} to {cut_date}", obs_pre, pre),
        (f"Forecast, {cut_date} to {end_date}", obs_cum - obs_pre, cum - pre),
        (f"Whole period, {start_date} to {end_date}", obs_cum, cum),
    ]
    sel, labels = _ranked_zones(z)
    # Columns 0..N_ZONES-1 are the ranked zones; the last column sums the rest.
    groups = [_with_other(obs, sim, sel) for _, obs, sim in periods]
    rows = [np.arange(r, min(r + per_row, sel.size)) for r in range(0, sel.size, per_row)]

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6.5})
    fig = plt.figure(figsize=(11.5, 3.4 * len(rows)))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[per_row, 1.5])
    panels = [
        (fig.add_subplot(gs[r, 0]), idx, f"Health zones ranked {idx[0] + 1}-{idx[-1] + 1} "
         "by observed whole-period count")
        for r, idx in enumerate(rows)
    ]  # fmt: skip
    panels.append((fig.add_subplot(gs[:, 1]), np.array([sel.size]), "All other zones"))
    w = 0.13
    for ax, idx, title in panels:
        x = np.arange(idx.size)
        for j, ((obs, sim), color) in enumerate(zip(groups, PERIOD_COLORS)):
            centre = x + (j - 1) * 0.31
            q5, q50, q95 = np.percentile(sim[:, idx], [5, 50, 95], axis=0)
            ax.bar(centre - w / 2, obs[idx], width=w, color=color, lw=0)
            ax.bar(centre + w / 2, q50, width=w, color=color, alpha=0.35, ec=color, lw=0.6)
            ax.errorbar(
                centre + w / 2, q50, yerr=[q50 - q5, q95 - q50],
                fmt="none", ecolor="0.15", elinewidth=0.7, capsize=1.5,
            )  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels([labels[i] for i in idx])
        ax.set_xlim(-0.6, idx.size - 0.4)
        for b in x[:-1] + 0.5:
            ax.axvline(b, color="0.88", lw=0.6, zorder=0)
        ax.set_ylabel("Confirmed cases")
        ax.set_title(title, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22, lw=0.5)

    handles = [Patch(facecolor=c, lw=0, label=t) for (t, _, _), c in zip(periods, PERIOD_COLORS)]
    handles += [
        Patch(facecolor="0.4", lw=0, label="Observed (left bar of each pair)"),
        Patch(facecolor="0.4", alpha=0.35, ec="0.4", lw=0.6,
              label="Model median, 5-95% (right bar)"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    fig.suptitle(
        "Confirmed cases by health zone: fit window, forecast and whole period", fontsize=9
    )
    fig.text(
        0.5,
        -0.008,
        f"{int(z['n_draws'])} posterior draws x {int(z['n_conts'])} continuation(s), "
        f"status-quo forecast; horizon {cut_date}; BDBV2026-Data {str(z['data_commit'])[:7]}",
        ha="center",
        fontsize=5.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "figA02b_forecast_by_zone.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=None, help="posterior draws (default: all)")
    ap.add_argument(
        "--conts", type=int, default=1, help="continuations per draw (default: 1, the stored seed)"
    )
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--end", default=None, help="last date to score (default: last observed)")
    ap.add_argument("--replot", action="store_true", help=f"redraw from {CACHE.name}")
    a = ap.parse_args()

    if not a.replot:
        simulate(a.draws, a.conts, a.workers, a.end)
    elif not CACHE.exists():
        raise SystemExit(f"ERROR: {CACHE} not found; run without --replot first")
    z = np.load(CACHE)
    s = scores(z)
    SCORES.write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    plot(z)
    plot_zones(z)
    print(f"wrote {FIG / 'figA02_posterior_predictive_forecast.png'}")
    print(f"wrote {FIG / 'figA02b_forecast_by_zone.png'}")


if __name__ == "__main__":
    main()
