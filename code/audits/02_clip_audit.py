
import json
import platform
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

bundle = Path("training_bundle.zip")

def check(ok, message):
    if not ok:
        raise AssertionError(message)

with zipfile.ZipFile(bundle) as z:
    def read(name):
        return json.loads(z.read(name).decode("utf-8"))

    candidates = read("manifests/candidates.json")
    scores = read("manifests/clip_scores.json")["records"]
    rankings = read("manifests/rankings.json")
    selections = read("manifests/selections.json")["selections"]

print("Time:", datetime.now().astimezone().isoformat())
print("Python:", sys.version.split()[0])
print("OS:", platform.platform())

# 1. Check candidate and CLIP score records
check(len(candidates) == 2000, "Wrong candidate count")
check(len(scores) == 2000, "Wrong score count")
check(len(rankings) == 4000, "Wrong ranking count")

counts = Counter(x["class_id"] for x in candidates)
check(counts == {i: 200 for i in range(10)}, "Wrong class counts")

by_id = {x["image_id"]: x for x in candidates}
scored = {x["image_id"]: x for x in scores}

check(len(by_id) == 2000, "Duplicate candidate IDs")
check(len(scored) == 2000, "Duplicate score IDs")
check(by_id.keys() == scored.keys(), "ID mismatch")

for image_id, item in by_id.items():
    s = scored[image_id]
    p = s["probabilities"]

    check(item["class_id"] == s["class_id"],
          "Class mismatch: " + image_id)

    check(item["sha256"] == s["image_sha256"],
          "Image hash mismatch: " + image_id)

    check(len(p) == 10 and
          all(0 <= x <= 1 for x in p) and
          abs(sum(p) - 1) < 1e-4,
          "Invalid probabilities: " + image_id)

    check(abs(s["target_confidence"] -
              p[item["class_id"]]) < 1e-6,
          "Confidence mismatch: " + image_id)

print("PASS: candidates and CLIP score consistency")

# 2. Check rankings and selections
check(selections["real_only"] == [], "Invalid baseline")

for method in ("random", "clip"):
    expected_50 = []
    expected_150 = []

    for class_id in range(10):
        rows = sorted(
            (r for r in rankings
             if r["method"] == method
             and r["class_id"] == class_id),
            key=lambda r: r["rank"]
        )

        pool = {
            x["image_id"] for x in candidates
            if x["class_id"] == class_id
        }
        ids = [r["image_id"] for r in rows]

        check(
            [r["rank"] for r in rows] == list(range(1, 201)),
            f"{method}, class {class_id}: invalid ranks"
        )

        check(set(ids) == pool and len(ids) == 200,
              f"{method}, class {class_id}: invalid pool")

        if method == "clip":
            expected = sorted(
                pool,
                key=lambda image_id: (
                    -scored[image_id]["target_confidence"],
                    image_id
                )
            )
            check(ids == expected,
                  f"Class {class_id}: CLIP ranking mismatch")

        expected_50.extend(ids[:50])
        expected_150.extend(ids[:150])

    check(selections[f"{method}_50"] == expected_50,
          f"{method}_50 selection mismatch")

    check(selections[f"{method}_150"] == expected_150,
          f"{method}_150 selection mismatch")

    check(len(set(expected_50)) == 500,
          f"{method}_50 duplicate IDs")

    check(len(set(expected_150)) == 1500,
          f"{method}_150 duplicate IDs")

    check(set(expected_50).issubset(set(expected_150)),
          f"{method}: nesting failed")

    print(f"PASS: {method} ranking, selection and nesting")

print("RESULT: ALL CHECKS PASS")
print("Scope: stored records only; no CLIP re-inference.")
