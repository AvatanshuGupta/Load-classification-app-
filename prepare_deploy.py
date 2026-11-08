"""One-off helper: builds the placeholder defaults and example rows from your
ORIGINAL (unscaled) dataset.

    python prepare_deploy.py --csv path/to/data.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

FEATURES = (
    ["Vrms", "Irms", "P", "Q1", "S", "PF", "phi_deg", "Zmag", "THD_V", "THD_I"]
    + [f"{q}_L{i}" for i in range(4) for q in "VIP"]
)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out-dir", default="artifacts")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = pd.read_csv(args.csv)
out = Path(args.out_dir)
out.mkdir(exist_ok=True)

stats = {
    f: {
        "median": float(df[f].median()),
        "min": float(df[f].min()),
        "max": float(df[f].max()),
    }
    for f in FEATURES
}
(out / "feature_stats.json").write_text(json.dumps(stats, indent=2))

# one example row for every (behaviour, circuit) pair
ex = df.groupby(["behaviour", "circuit"], group_keys=False).sample(1, random_state=args.seed)
samples = [
    {
        "behaviour": r["behaviour"],
        "circuit": r["circuit"],
        "values": {f: float(r[f]) for f in FEATURES},
    }
    for _, r in ex.iterrows()
]
(out / "samples.json").write_text(json.dumps(samples, indent=2))
print(f"Wrote {len(FEATURES)} feature stats and {len(samples)} samples to {out}/")
