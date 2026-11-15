"""Formal-stage recovery, batch membership and paired statistics contracts."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from a_protocol import ROOT,CONDITIONS,file_hash,generation_batches,load_config
from classifier import train_heads
from evaluation import paired_summary


class FormalChecks(unittest.TestCase):
    def test_original_batches_are_stable_for_partial_resume(self):
        items = list(range(10))
        batches = generation_batches(items,4)
        self.assertEqual(batches,[[0,1,2,3],[4,5,6,7],[8,9]])
        complete = {0,1,2,4,5,6,7}
        pending = [b for b in batches if not all(i in complete for i in b)]
        self.assertEqual(pending,[[0,1,2,3],[8,9]])

    def results(self):
        return [{"seed":seed,"condition_id":name,"accuracy":0.5+0.1*seed+(0 if name=="real_only" else 0.05)}
                for seed in (0,1,2) for name,_,_ in CONDITIONS]

    def test_statistics_pair_by_seed_and_use_sample_standard_deviation(self):
        rows = paired_summary(list(reversed(self.results())),[0,1,2])
        self.assertAlmostEqual(rows[0]["mean_accuracy"],0.6)
        self.assertAlmostEqual(rows[0]["std_accuracy_ddof1"],0.1)
        for row in rows[1:]:
            self.assertAlmostEqual(row["paired_delta_pp_mean"],5.0)
            self.assertAlmostEqual(row["paired_delta_pp_std_ddof1"],0.0)

    def test_statistics_reject_missing_or_duplicate_conditions(self):
        rows = self.results()
        with self.assertRaises(ValueError):
            paired_summary(rows[:-1],[0,1,2])
        with self.assertRaises(ValueError):
            paired_summary(rows[:-1]+[rows[0]],[0,1,2])

    def test_formal_head_resume_checks_inputs_outputs_and_checkpoint_integrity(self):
        config = copy.deepcopy(load_config("configs/study.json"))
        config["classifier"]["hyperparameters"]["updates"] = 2  # Fabricated unit fixture only.
        real = [{"image_id":f"r{i}","sha256":f"real{i}","class_id":i//20,
                 "original_train_index":i,"split":"train"} for i in range(200)]
        synthetic = [{"image_id":f"s{i}","sha256":f"synth{i}","class_id":i//200} for i in range(2000)]
        selections = {name:[] if not count else [f"s{c*200+i}" for c in range(10) for i in range(count)]
                      for name,count,_ in CONDITIONS}
        rng = np.random.default_rng(123)
        arrays = rng.normal(size=(2200,4,512)).astype(np.float32)
        features = {r["image_id"]:x for r,x in zip(real+synthetic,arrays)}
        directory = ROOT/"runs/_cache/tmp/tests"
        directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            root = Path(temporary)
            rows,audit = train_heads(config,root,real,synthetic,selections,features,seed=0,mode="main")
            self.assertEqual(len(rows),5)
            self.assertTrue(all(r["accuracy"] is None and r["dev_count"]==0 for r in rows))
            before = [file_hash(root/r["relative_path"]) for r in audit["models"]]
            resumed,second = train_heads(config,root,real,synthetic,selections,features,seed=0,mode="main")
            self.assertEqual(second["reused_classifiers"],5)
            self.assertEqual(before,[file_hash(root/r["relative_path"]) for r in second["models"]])
            path = root/audit["models"][0]["relative_path"]
            with path.open("ab") as handle:
                handle.write(b"deliberate corruption in a temporary unit fixture")
            with self.assertRaises(ValueError):
                train_heads(config,root,real,synthetic,selections,features,seed=0,mode="main")


if __name__ == "__main__":
    unittest.main()
