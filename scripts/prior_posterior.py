"""Prior-to-posterior shrinkage for each calibrated parameter, from a thinned checkpoint.

A parameter whose posterior looks like its prior is not informed by the data and adds a
dimension to the sampler without adding information. Per parameter this reports:

- width ratio: posterior 5-95% width over prior 5-95% width (1 = no learning);
- sd ratio: posterior sd over prior sd;
- median shift in prior sds;
- the largest absolute correlation with any other calibrated parameter, since two
  parameters can each look unidentified yet be jointly constrained (or redundant);
- the first-half and second-half posterior medians, because the checkpoint is not
  stationary and a parameter still drifting has not settled.

The draws are pooled over walkers and steps. The checkpoint's effective sample size is
small (the chain is ~10 autocorrelation times long), so treat the ratios as indicative.
Priors are read from `calibration/config.yaml`.

    uv run python scripts/prior_posterior.py [--checkpoint PATH]

Writes `figures/prior_posterior.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bdbv.model.priors import sample_prior

CHECKPOINT = ROOT / "data" / "inputs" / "source_checkpoint_iter44000.npz"
CAL_CFG = ROOT / "calibration" / "config.yaml"
NON_CAL = ("seed_day", "sim_seed")
N_PRIOR = 200_000
LO, HI = 5, 95


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--walkers", type=int, default=480)
    args = ap.parse_args()

    priors = yaml.safe_load(CAL_CFG.read_text())["priors"]
    with np.load(args.checkpoint) as z:
        keys = [k for k in z.files if not k.startswith("_") and k not in NON_CAL]
        post = {k: np.asarray(z[k], float) for k in keys}
    n_steps = len(post[keys[0]]) // args.walkers
    corr = np.corrcoef(np.column_stack([post[k] for k in keys]), rowvar=False)
    np.fill_diagonal(corr, 0.0)

    rng = np.random.default_rng(0)
    out = {}
    print(
        f"{'parameter':<22} {'width':>6} {'sd':>6} {'shift/sd':>9} {'max|r|':>7}  drift(1st->2nd half)"
    )
    for j, k in enumerate(keys):
        pr = np.array([sample_prior(priors[k], rng) for _ in range(N_PRIOR)])
        x = post[k]
        p_lo, p_med, p_hi = np.percentile(x, [LO, 50, HI])
        q_lo, q_med, q_hi = np.percentile(pr, [LO, 50, HI])
        half = n_steps // 2 * args.walkers
        m1, m2 = np.median(x[:half]), np.median(x[half:])
        other = keys[int(np.argmax(np.abs(corr[j])))]
        out[k] = {
            "width_ratio": float((p_hi - p_lo) / (q_hi - q_lo)),
            "sd_ratio": float(x.std() / pr.std()),
            "median_shift_in_prior_sd": float((p_med - q_med) / pr.std()),
            "max_abs_corr": float(np.abs(corr[j]).max()),
            "max_corr_with": other,
            "prior_5_50_95": [float(q_lo), float(q_med), float(q_hi)],
            "posterior_5_50_95": [float(p_lo), float(p_med), float(p_hi)],
            "median_first_half": float(m1),
            "median_second_half": float(m2),
            "drift_in_posterior_sd": float((m2 - m1) / x.std()),
        }
        o = out[k]
        print(
            f"{k:<22} {o['width_ratio']:6.2f} {o['sd_ratio']:6.2f} "
            f"{o['median_shift_in_prior_sd']:+9.2f} {o['max_abs_corr']:7.2f}  "
            f"{o['drift_in_posterior_sd']:+.2f} sd  (r with {other})"
        )
    dest = ROOT / "figures" / "prior_posterior.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
