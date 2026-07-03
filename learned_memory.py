import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import json
d=json.load(open("results_reference/results_full_reference.json"))
for k in ["E1_decay_laws","E2_frontier","E3_decoupling_ablation","E4_correction_cost",
          "E5_drift_recovery","E6_separability","E7_sequential"]:
    assert k in d, k
print("reference JSON keys OK")
