"""Self-verification: files present, quick reproduction runs, Appendix-C keys produced."""
from pathlib import Path
import json, subprocess, sys

REQUIRED_FILES = ["README.md","requirements.txt","reproduce.py","make_figures.py",
    "skeleton.py","leverage.py","experiments.py","robustness.py","learned_memory.py",
    "results_reference/results_full_reference.json",
    "results_reference/e9_reference.json","results_reference/learned_reference.json"]
REQUIRED_KEYS = ["E1_decay_laws.slope_pointwise","E1_decay_laws.slope_mean",
    "E1_decay_laws.slope_full","E2_frontier.square_ratio",
    "E4_correction_cost.slope_pointwise","E4_correction_cost.slope_field",
    "E4_correction_cost.slope_scalar","E5_drift_recovery.slope",
    "E6_separability.slope_histogram_fuel","E7_sequential.efficiency"]

def get_nested(d,key):
    for p in key.split("."): d=d[p]
    return d

missing=[p for p in REQUIRED_FILES if not Path(p).exists()]
if missing: raise SystemExit("Missing files: "+", ".join(missing))
print("[1/3] file manifest OK")
subprocess.run([sys.executable,"reproduce.py","--quick"],check=True,
               stdout=subprocess.DEVNULL)
print("[2/3] quick reproduction OK")
res=json.loads(Path("results/results_quick.json").read_text())
bad=[k for k in REQUIRED_KEYS if not _ok(res,k)] if False else []
for k in REQUIRED_KEYS:
    try: get_nested(res,k)
    except KeyError: bad.append(k)
if bad: raise SystemExit("Missing JSON keys: "+", ".join(bad))
print("[3/3] all Appendix-C keys produced")
print("Package verification passed.")
