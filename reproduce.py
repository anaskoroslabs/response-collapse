#!/usr/bin/env python3
"""One-command reproduction driver.

  python reproduce.py --quick    # qualitative reproduction, ~2 min, smaller grids
  python reproduce.py --full     # every scalar reported in the paper (~15-25 min)

Outputs:
  results/results.json          (--full)  or  results/results_quick.json (--quick)
  figs/fig_decay.pdf, figs/fig_correction.pdf   (from --full results)

Every scalar in the paper maps to a JSON key; the mapping table is in README.md.
All seeds are fixed; runs are deterministic.
"""
import argparse, json, os, sys
import numpy as np
import experiments as X

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if args.check:
        ref = json.load(open("results_reference/results_full_reference.json"))
        cur = json.load(open("results/results.json"))
        TOL = {("E1_decay_laws","slope_pointwise"):0.02,("E1_decay_laws","slope_mean"):0.02,
               ("E1_decay_laws","slope_full"):0.02,("E4_correction_cost","slope_pointwise"):0.04,
               ("E4_correction_cost","slope_field"):0.04,("E4_correction_cost","slope_scalar"):0.04,
               ("E2_frontier","square_ratio"):0.01,("E5_drift_recovery","slope"):0.02}
        bad=[]
        for (blk,key),tol in TOL.items():
            a,b=ref[blk][key],cur[blk][key]
            if abs(a-b)>tol: bad.append(f"{blk}.{key}: ref {a} vs run {b} (tol {tol})")
        if bad: raise SystemExit("CHECK FAILED:\n  "+"\n  ".join(bad))
        print("CHECK PASSED: all reported scalars within tolerance."); return
    OUT = {}
    if args.quick:
        OUT["E1_decay_laws"] = X.E1(Ts=(10**3,10**4,10**5), reps=3, n=3000)
        OUT["E2_frontier"]   = X.E2(T=10**4, n=20000)
        OUT["E3_decoupling_ablation"] = X.E3(T=10**4, n=20000)
        OUT["E4_correction_cost"] = X.E4(Ts=(10**4,10**5), reps=3)
        OUT["E5_drift_recovery"]  = X.E5(T0s=(100,1000), reps=60)
        OUT["E6_separability"]    = X.E6(Ts=(10**3,10**4))
        OUT["E7_sequential"]      = X.E7(Ts=(10**4,), reps=3)
        path = "results/results_quick.json"
    else:
        OUT["E1_decay_laws"] = X.E1()
        OUT["E2_frontier"]   = X.E2()
        OUT["E3_decoupling_ablation"] = X.E3()
        OUT["E4_correction_cost"] = X.E4()
        OUT["E5_drift_recovery"]  = X.E5()
        OUT["E6_separability"]    = X.E6()
        OUT["E7_sequential"]      = X.E7()
        path = "results/results.json"
    with open(path, "w") as f: json.dump(OUT, f, indent=2)
    print(json.dumps(OUT, indent=2))
    print(f"\nWrote {path}", file=sys.stderr)

if __name__ == "__main__":
    main()
