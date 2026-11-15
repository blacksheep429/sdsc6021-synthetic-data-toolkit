
from datetime import datetime
from math import isclose
from statistics import mean, stdev
import platform
import sys

SEEDS = (0, 1, 2)
CONDITIONS = (
    "real_only", "random_50", "clip_50",
    "random_150", "clip_150"
)

# Simulated data only — NOT actual experiment results
toy = {
    "real_only":  [0.60, 0.62, 0.61],
    "random_50":  [0.61, 0.63, 0.62],
    "clip_50":    [0.65, 0.64, 0.67],
    "random_150": [0.63, 0.64, 0.64],
    "clip_150":   [0.66, 0.65, 0.68],
}

print("Time:", datetime.now().astimezone().isoformat())
print("Python:", sys.version.split()[0])
print("OS:", platform.platform())
print("MODE: TOY DATA ONLY; NO OFFICIAL TEST RESULTS")

# 1. Check top-1 accuracy
true_labels = [0, 1, 2, 3, 4]
pred_labels = [0, 1, 2, 0, 4]

acc = sum(
    t == p for t, p in zip(true_labels, pred_labels)
) / len(true_labels)

assert isclose(acc, 0.8)
print("PASS: top-1 accuracy")

# 2. Check complete 5 x 3 experiment matrix
records = [
    {"condition_id": c, "seed": s, "accuracy": toy[c][s]}
    for c in CONDITIONS for s in SEEDS
][::-1]

by_key = {
    (r["seed"], r["condition_id"]): r["accuracy"]
    for r in records
}

expected_keys = {
    (s, c) for s in SEEDS for c in CONDITIONS
}

assert len(records) == len(by_key) == 15
assert set(by_key) == expected_keys

print("PASS: complete unique 15-run matrix")

# 3. Independently calculate summary statistics
summaries = {}

for c in CONDITIONS:
    values = [by_key[s, c] for s in SEEDS]

    assert all(0 <= v <= 1 for v in values)

    paired_deltas = [
        100 * (by_key[s, c] - by_key[s, "real_only"])
        for s in SEEDS
    ]

    summaries[c] = {
        "mean_acc": mean(values),
        "sd_acc": stdev(values),
        "mean_delta_pp": mean(paired_deltas),
        "sd_delta_pp": stdev(paired_deltas),
    }

# 4. Compare with independently known expected values
result = summaries["clip_50"]

assert isclose(
    result["mean_acc"], 0.6533333333333333
)
assert isclose(
    result["sd_acc"], 0.01527525231651948
)
assert isclose(
    result["mean_delta_pp"], 4.333333333333333
)
assert isclose(
    result["sd_delta_pp"], 2.0816659994661344
)
assert isclose(
    summaries["real_only"]["mean_delta_pp"], 0.0
)

print("PASS: mean and sample SD (ddof=1)")
print("PASS: same-seed paired differences in percentage points")
print("RESULT: ALL TOY-DATA PROTOCOL CHECKS PASS")
