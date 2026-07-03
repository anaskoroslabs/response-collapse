import json; d=json.load(open("results_reference/results_full_reference.json"))
assert d["E2_frontier"]["square_ratio"]<=1.02 and d["E2_frontier"]["adversary_ratio"]<=1.0
print("frontier bound respected in reference results")
