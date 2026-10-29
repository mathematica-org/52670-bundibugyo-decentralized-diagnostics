# Pipeline: inputs and outputs by step

Draft working notes, traced from the code on 2026-10-28. Each step below gives the
command, the directory to run it from, the configuration it reads, and the files it
reads and writes. For data provenance and licensing see `data/DATA.md`; for install and
the short command list see `README.md`.

```
 0 stage data ──► data/BDBV2026-Data/, data/flowminder_hdx/
                        │
 1 calibration ─────────┴──► calibration/outputs/checkpoint_ppc.npz, posterior.npz, ...
                        │
 2 prepare checkpoint ──┴──► scenarios/outputs/source_checkpoint_iter<N>.npz
                        │      (or copy data/inputs/source_checkpoint_iter44000.npz
                        │       to reproduce the preprint)
 3 extract subsample ───┴──► scenarios/outputs/sq_draws.npz   (2,500 draws)
                        │
 4 scenario sweeps ─────┴──► scenarios/outputs/results[_tag].jsonl, placement, report
                        │
 5 post-processing ─────┴──► scenarios/outputs/analysis.json, *avertable*.json
                        │
 6 freeze figure inputs ┴──► data/figure_inputs/   (MANUAL: no script, see below)
                        │
 7 figures ─────────────┴──► figures/*.png
```

## Conventions that affect where files go

- **Outputs land next to the `run.py` that was invoked.** The engine derives its
  `outputs/` and `figures/` directories from `sys.argv[0]` (`bdbv/engine/state.py:24`),
  so `calibration/run.py` writes to `calibration/outputs/` and `calibration/figures/`,
  and `scenarios/run.py` writes to `scenarios/outputs/`. Run each from its own directory.
- **Two configs, one shared body.** `scenarios/config.yaml` is `calibration/config.yaml`
  plus an `intervention:` block. At the last commit every shared key is identical; keep
  them in sync, because the scenario sweep rebuilds the calibration context from its own
  copy.
- **`--tag X` suffixes every scenario artefact** (`results_X.jsonl`, `placement_X.json`,
  `report_X.json`, `analysis_X.json`, `avertable_X.json`). Without a tag, a run writes
  the default names and appends to (or, with `--fresh`, deletes) the main results.
- **Resume is keyed on `base` rows.** `scenarios/run.py` treats a (draw, continuation)
  as done when `results[_tag].jsonl` has a `base` row for it. A run whose `--streams`
  omit `base` cannot resume and restarts from scratch.
- **Staged data location.** The model reads `data/BDBV2026-Data/` unless
  `BDBV_DATA_DIR` is set (`bdbv/model/data.py:24-46`).

---

## Step 0: stage the public data

| | |
|---|---|
| Command | `uv run python scripts/stage_data.py [--bdbv-commit SHA] [--skip-hdx] [--allow-hdx-drift] [--force]` |
| Run from | repository root |
| Config | pinned in the script: BDBV2026-Data commit `60f6701…`, HDX package IDs and SHA256s |
| Reads | GitHub `INRB-UMIE/BDBV2026-Data`; HDX Flowminder relocation CSVs |
| Writes | `data/BDBV2026-Data/` (git clone at the pinned commit); `data/flowminder_hdx/raw/`, `data/flowminder_hdx/processed/flowminder_hdx__outflow_2026_03.matrix.csv` |

`--bdbv-commit ""` checks out the newest QA-verified release instead. That is what
`scripts/forecast_ppc.py` needs for post-horizon data, and it no longer reproduces
the preprint's in-window inputs.

### What the model reads from the staged and tracked data

Loaded through `bdbv/model/data.py` and `bdbv/engine/context.py` whenever a context is
built (calibration, scenarios, `analyze.py`, `forecast_ppc.py`):

| Source | Files |
|---|---|
| SitRep series (`BDBV2026-Data/data/insp_sitrep/processed/`) | per-zone and national cumulative confirmed cases; per-zone and national cumulative confirmed deaths; suspects in isolation; recoveries; new admissions; contacts traced |
| Lab capacity | `testing_capacity__pcr_machines__static.csv` |
| Geography and access | `osrm__travel_time__static.matrix.csv`; `grid3_healthsites__healthsite_count__static.csv`; `worldpop__pop_count__static.csv`; `build/drc_health_zones.geojson`; `data/aliases.csv` |
| Mobility | `flowminder_hdx/processed/flowminder_hdx__outflow_2026_03.matrix.csv`; Flowminder short-trips matrices for Ituri and Nord-Kivu |
| Tracked curated inputs | `data/inputs/lab_rollout.yaml` (lab go-live dates → the 8 served zones); `data/inputs/MVE_isolation_stockflow_validation.xlsx`; `bdbv/model/care_facilities.yaml`; `bdbv/province_covariates.csv` |

Admissions, recoveries and suspect-isolation series are drawn as overlays and are not
scored by the likelihood (`calibration/config.yaml`, "Other locked decisions").

---

## Step 1: calibration

| | |
|---|---|
| Command | `uv run python run.py --sampler emcee [--steps N] [--walkers W] [--workers K] [--burn B] [--init-npz PATH]` |
| Run from | `calibration/` |
| Config | `calibration/config.yaml`: `priors`, `fixed`, `brake_reforms`, `emcee` (walkers, steps, burn), likelihood settings, `seed` |
| Reads | the staged data from step 0 |
| Writes (`calibration/outputs/`) | `emcee_chain.h5` (full chain, about a GB) plus rotating backups `.bak0`–`.bak2`; `checkpoint_ppc.npz` (thinned theta + selected `sim_seed`, rewritten at each chunk boundary, carries `_iter`); `posterior.npz` and `report.json` at the end of the run |
| Control | a `STOP` file in `calibration/outputs/` is checked between chunks to end the chain early |

**Mixing diagnostics a calibration run gives you.** The log prints mean acceptance and
`tau_max` after each chunk. `report.json` keeps the final `acceptance_mean`,
`autocorr_mean`, `autocorr_max` and `ess_approx`, but no per-parameter τ. No trace plots
are drawn. `emcee_chain.h5` holds everything needed for exact per-walker acceptance,
log-probability traces and full-resolution τ, but no script reads it yet. For the
preprint chain only the thinned checkpoint survives, so use `chain_diagnostics.py`
(below).

**Calibration diagnostic figures (optional, not in README).** `run.py` with its default
SIR sampler and `--posterior-only` reloads `outputs/posterior.npz` and `report.json`,
runs prior and posterior predictive checks, and writes to `calibration/figures/`:
`00_all_in_one.png`, `01_ppc_national.png`, `02_capacity_iso.png`,
`03_posterior.png`, `04_spatial.png`, `05_scored_only.png`,
`06_likelihood_diagnosis.png`. This path should accept the emcee `posterior.npz`; I
haven't run it here.

**Chain diagnostics from a checkpoint** (these need no staged data):

| Script | Reads | Writes |
|---|---|---|
| `scripts/chain_diagnostics.py [--checkpoint PATH]` | checkpoint (default `data/inputs/source_checkpoint_iter44000.npz`); `calibration/config.yaml` (`emcee` block) | `figures/chain_traces.png`, `figures/chain_mixing.png`, `figures/chain_diagnostics.json` |
| `scripts/prior_posterior.py [--checkpoint PATH]` | same checkpoint; `calibration/config.yaml` (`priors`) | `figures/prior_posterior.json` |

The preprint used the thinned checkpoint at **iteration 44,000** of a 60,000-step chain
(49,920 draws = 480 walkers × 104 thinned steps), not the finished chain. It is
bundled as `data/inputs/source_checkpoint_iter44000.npz`. It is not stationary and
carries no stored likelihood (`data/DATA.md`).

---

## Step 2: prepare the scenario checkpoint

| | |
|---|---|
| Command | `uv run python scripts/prepare_scenario_checkpoint.py [--source PATH] [--dest PATH]` |
| Run from | repository root |
| Reads | `--source`, default `calibration/outputs/checkpoint_ppc.npz`; reads its `_iter` |
| Writes | `scenarios/outputs/source_checkpoint_iter<_iter>.npz` |

To reproduce the preprint, use `--source data/inputs/source_checkpoint_iter44000.npz`.
If the written filename isn't `source_checkpoint_iter44000.npz`, update
`intervention.posterior` in `scenarios/config.yaml` to match.

---

## Step 3: extract the scenario subsample

| | |
|---|---|
| Command | `uv run python extract_subsample.py [--n N] [--seed S] [--out PATH]` |
| Run from | `scenarios/` |
| Config | `scenarios/config.yaml` → `intervention.posterior` (source), `intervention.subsample` (destination), `n_draws` + `greedy_n_pre` (size), `seed` |
| Reads | `scenarios/outputs/source_checkpoint_iter44000.npz`; the source's sibling `report.json` if present (median cross-check) |
| Writes | `scenarios/outputs/sq_draws.npz` (2,500 draws: parameters, `seed_day`, `sim_seed`); `scenarios/outputs/subsample_meta.json` |

Each draw is a parameter set plus a seed, **not** a simulated tree. Trees are
re-simulated from draws in step 4 and are bit-reproducible from the stored seeds.

---

## Step 4: scenario sweeps

`scenarios/run.py` splits `sq_draws.npz` into **500 greedy-training draws** and
**2,000 scoring draws** (`scenarios/draw_split.py`). Next, the greedy pre-pass builds
the lab placement order from the training draws, which is the slow part (about 46 min
on 2 CPUs). Then each scoring draw runs as one task per (draw, continuation).

| | |
|---|---|
| Command | `uv run python run.py [--streams S1,S2,…] [--draws N] [--conts K] [--workers W] [--tag X] [--rep-delay D] [--subsample PATH] [--fresh]` |
| Run from | `scenarios/` |
| Config | `scenarios/config.yaml` → whole calibration body + `intervention:` (`n_draws` 2000, `n_continuations` 5, `greedy_n_pre` 500, `horizon_days` 540, `report_marks`, `budgets`, `strategies`, `n_random_seeds`, `representative_arm`, `sdb_targets`/`sdb_scopes`, `ramp_arms`, `max_infections`) |
| Reads | `scenarios/outputs/sq_draws.npz` (or `--subsample`); staged data |
| Writes (`scenarios/outputs/`) | `placement[_X].json` (greedy order + random placements); `results[_X].jsonl` (one scalar row per scenario evaluation, appended); `report[_X].json` (run metadata: draws, continuations, split seed, streams, greedy top 15) |

**Streams.** `base` holds the status-quo trees. Every other stream evaluates
interventions against them. Trees aren't stored, only scalar summaries per row.

| Stream | What | Trees |
|---|---|---|
| `base` | status-quo forecast, accrual at +30/90/180/540 d | all K continuations |
| `time` | deployment delay × mode × budget | continuation 0 |
| `timerdt` | `time` × RDT/burial ladder (only runs when named) | continuation 0 |
| `place` | greedy vs random × budget × mode | continuation 0 |
| `pm` | safe-and-dignified-burial coverage × scope | continuation 0 |
| `casc` | mechanism decomposition at the representative arm | continuation 0 |
| `maps` | per-zone averted and residual burden | all K |
| `ramp` | each `ramp_arms` entry, with the representative arm priced against that arm's own status quo | all K |

Every stream except `ramp` runs under the single `role: primary` ramp arm only.

**Runs behind the preprint:**

| Run | Command | Output | Contents |
|---|---|---|---|
| Main sweep | `run.py --streams base,time,place,timerdt` | `results.jsonl` (~800 MB) | 10,000 `base` rows (2,000 draws × 5 continuations); `place` 1,148,000, `time` 196,000, `timerdt` 784,000 rows |
| 30-day placement | `run.py --streams place --tag d30 --rep-delay 30` | `results_d30.jsonl` | `place` rows only; `--rep-delay` also re-picks the greedy order for the later deployment |

---

## Step 5: post-processing

All of these read only the scalar rows. None of them simulate.

| Script (run from `scenarios/`) | Reads | Writes |
|---|---|---|
| `analyze.py [--tag X]` | `outputs/results[_X].jsonl`, `placement[_X].json`, `report[_X].json`; `config.yaml` + staged data (for observed totals at cutoff) | `outputs/analysis[_X].json` |
| `avertable.py [--tag X]` | `outputs/results[_X].jsonl` (streams `base`, `place`, `pm`, `time`, `casc`) | `outputs/avertable[_X].json` |
| `time_avertable.py IN OUT time` | `outputs/results.jsonl` | `outputs/time_avertable.json` |
| `time_avertable.py IN OUT timerdt` | `outputs/results.jsonl` | `outputs/timerdt_avertable.json` |
| `place_avertable.py IN OUT pct_deaths` | `outputs/results_d30.jsonl` | `outputs/place_avertable_d30_pct_deaths.json` |

`analyze.py` holds every row in memory. `avertable.py` streams the file and runs
in constant memory.

---

## Step 6: freeze figure inputs (manual)

`scripts/make_figures.py` reads only `data/figure_inputs/`. **No script in the
repository writes to that directory.** Each file was copied or exported by hand:

| `data/figure_inputs/` file | Produced from | Export code in repo? |
|---|---|---|
| `sq_draws.npz` | the 2,000 scoring rows of `scenarios/outputs/sq_draws.npz` | no (the split is `draw_split.py`) |
| `avertable.json` | `scenarios/outputs/avertable.json` | copy |
| `time_avertable.json` | `scenarios/outputs/time_avertable.json` | copy |
| `timerdt_avertable.json` | `scenarios/outputs/timerdt_avertable.json` | copy |
| `place_avertable_d30_pct_deaths.json` | `scenarios/outputs/place_avertable_d30_pct_deaths.json` | copy |
| `status_quo_540.csv` | `base` rows of `results.jsonl`, one row per draw (values look like means over the 5 continuations) | **no** |
| `new_zone_replay_d30_local.csv` | per-draw replay for the 30-day local-PCR placement panel | **no** |
| `posterior_predictive.npz` | calibration posterior predictive trajectories + observed arrays | **no** |
| `observed_at_cutoff.json` | national observed totals at 2026-08-02 | **no** |
| `placement_map.geojson`, `placement_map_metadata.json` | one illustrative 20-lab, 30-day draw joined to health-zone geometry | **no** |

---

## Step 7: figures

### Preprint figures

| | |
|---|---|
| Command | `uv run python scripts/make_figures.py` |
| Run from | repository root |
| Reads | `data/figure_inputs/` only (no staged data, no config) |
| Writes | `figures/*.png` |

| Figure | Inputs |
|---|---|
| `fig02_response_functions.png` | `sq_draws.npz` |
| `fig03_status_quo.png` | `status_quo_540.csv`, `observed_at_cutoff.json` |
| `fig04_speed.png` | `time_avertable.json`, `avertable.json` (denominator) |
| `fig05_placement_zones.png` | `place_avertable_d30_pct_deaths.json`, `new_zone_replay_d30_local.csv` |
| `fig06_placement_map.png` | `placement_map.geojson`, `placement_map_metadata.json` |
| `fig07_rdt.png` | `timerdt_avertable.json`, `avertable.json` (denominator) |
| `figA01_posterior_pairplot.png` | `sq_draws.npz` |
| `figA02_posterior_predictive.png` | `posterior_predictive.npz` (fitted window only) |

### Post-horizon forecast check (new, not in README)

| | |
|---|---|
| Command | `uv run python scripts/forecast_ppc.py [--draws N] [--conts K] [--workers W] [--end DATE] [--replot]` |
| Run from | repository root, after `stage_data.py --skip-hdx --bdbv-commit ""` |
| Config | `scenarios/config.yaml` (primary `ramp_arms` entry; `horizon_days`; `max_infections`) |
| Reads | `data/figure_inputs/sq_draws.npz` (2,000 draws, replayed as continuation 0 by default, so bit-identical to the `base` rows); staged SitRep national cases, national deaths and per-zone cases through the latest date; `data/figure_inputs/posterior_predictive.npz` (latent anchor point only) |
| Writes (`figures/`) | `figA02_posterior_predictive_forecast.png`, `figA02b_forecast_by_zone.png`, `forecast_ppc.npz` (cache; `--replot` redraws from it), `forecast_ppc_scores.json` |

---

## Smoke test: do not run in a checkout holding real results

`uv run python scripts/run_smoke_tests.py` runs a 1-step emcee calibration, copies the
bundled checkpoint into `scenarios/outputs/`, then runs
`extract_subsample.py --n 3` and `run.py --draws 1 --conts 1 --fresh`, all untagged. That
**overwrites `scenarios/outputs/sq_draws.npz`, deletes and rewrites `results.jsonl`, and
overwrites `placement.json` and `report.json`**. It also overwrites
`calibration/outputs/`. Then it rebuilds the preprint figures from `data/figure_inputs/`,
which it leaves alone.

---

## State of this checkout (2026-10-28)

- `calibration/outputs/` is empty. Calibration hasn't been run here; scenario inputs
  come from the bundled iteration-44,000 checkpoint.
- `scenarios/outputs/results.jsonl` is the complete main sweep (counts in step 4).
- `scenarios/outputs/results_d30.jsonl` has `place` rows for **886 of the 2,000
  draws** only, so it looks like an interrupted run. It can't resume, because it has
  no `base` rows. Whether the frozen `place_avertable_d30_pct_deaths.json` came from a
  complete run can't be told from this checkout.
- `scenarios/config.yaml` has an uncommitted second ramp arm, `noclock`
  (`nat_scale_min: 1.0`, `role: contrast`), used by the `--tag noclock_smoke` run
  (`*_noclock_smoke.*` in `scenarios/outputs/`). It only affects the `ramp` stream.
- `data/inputs/20261024_national_cumulative_confirmed_daily.csv` is untracked and isn't
  read by any Python file.
