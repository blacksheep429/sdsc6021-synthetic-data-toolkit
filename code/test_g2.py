"""Checks that protect experiment interpretation, before the actual G2 GPU run."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from a_protocol import (ROOT, atomic_json, candidate_plan, critical_sources, file_hash,
                        load_config, proposal_hash, require_g2)
from classifier import objective, sample_schedule
from data_features import feature_key, paired_split
from selection import make_selections


class ScientificChecks(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/study.json")

    def test_development_labels_are_within_the_same_twenty_shot_sample(self):
        labels = np.repeat(np.arange(10),60)
        train,dev = paired_split(labels,0,20,4)
        main,_ = paired_split(labels,0,20,0)
        self.assertEqual(set(main),set(train)|set(dev))
        self.assertFalse(set(train)&set(dev))
        self.assertEqual(len(train),160)
        self.assertEqual(len(dev),40)
        self.assertEqual([sum(labels[i] == c for i in train) for c in range(10)],[16]*10)

    def pool(self):
        records,scores = [],[]
        for label in range(10):
            for j in range(6):
                image_id = f"{label:02d}_{j:02d}"
                records.append({"image_id":image_id,"class_id":label,"sha256":image_id})
                scores.append({"image_id":image_id,"class_id":label,"image_sha256":image_id,"target_confidence":0.5})
        return records,scores

    def test_selection_is_equal_sized_nested_and_tie_order_independent(self):
        records,scores = self.pool()
        selected,ranks = make_selections(records,scores,[2,5],123,"test")
        repeated,repeated_ranks = make_selections(list(reversed(records)),list(reversed(scores)),[2,5],123,"test")
        self.assertEqual(selected,repeated)
        self.assertEqual(ranks,repeated_ranks)
        for method in ("random","clip"):
            self.assertEqual(len(selected[method+"_2"]),20)
            self.assertEqual(len(selected[method+"_5"]),50)
            self.assertTrue(set(selected[method+"_2"]).issubset(selected[method+"_5"]))
        self.assertEqual(selected["clip_2"][:2],["00_00","00_01"])

    def test_selection_rejects_scores_from_another_image(self):
        records,scores = self.pool()
        scores[0]["image_sha256"] = "changed"
        with self.assertRaises(ValueError):
            make_selections(records,scores,[2,5],123,"test")

    def test_mixed_weight_does_not_change_when_synthetic_rows_or_views_repeat(self):
        real = torch.arange(80,dtype=torch.float32).reshape(2,4,10)/10
        real.requires_grad_(True)
        syn = torch.flip(real.detach(),dims=[-1])
        labels = torch.tensor([2,7])
        value = objective(real,labels,syn,labels)
        expected = 0.5*F.cross_entropy(real.reshape(-1,10),labels.repeat_interleave(4)) + 0.5*F.cross_entropy(syn.reshape(-1,10),labels.repeat_interleave(4))
        self.assertTrue(torch.allclose(value,expected))
        repeated = objective(real,labels,syn.repeat(13,1,1),labels.repeat(13))
        self.assertTrue(torch.allclose(value,repeated))
        self.assertTrue(torch.allclose(torch.autograd.grad(value,real)[0],torch.autograd.grad(repeated,real)[0]))
        more_views = objective(real.repeat_interleave(2,dim=1),labels,syn,labels)
        self.assertTrue(torch.allclose(value,more_views))
        baseline = objective(real,labels)
        self.assertTrue(torch.allclose(baseline,F.cross_entropy(real.reshape(-1,10),labels.repeat_interleave(4))))

    def test_paired_real_sampling_keeps_all_classes_in_every_update(self):
        labels = np.repeat(np.arange(10),16)
        first = sample_schedule(labels,17,2,11)
        self.assertTrue(torch.equal(first,sample_schedule(labels,17,2,11)))
        self.assertEqual(tuple(first.shape),(17,20))
        for row in first.numpy():
            self.assertEqual(np.bincount(labels[row],minlength=10).tolist(),[2]*10)

    def test_feature_key_invalidates_image_views_but_not_head_learning_rate(self):
        record = {"image_id":"x","sha256":"original"}
        key = feature_key(self.config,record,4)
        self.assertNotEqual(key,feature_key(self.config,dict(record,sha256="changed"),4))
        self.assertNotEqual(key,feature_key(self.config,record,1))
        changed = copy.deepcopy(self.config)
        changed["classifier"]["view_seed"] += 1
        self.assertNotEqual(key,feature_key(changed,record,4))
        changed = copy.deepcopy(self.config)
        changed["classifier"]["hyperparameters"]["lr"] *= 2
        self.assertEqual(key,feature_key(changed,record,4))

    def test_frozen_flag_cannot_replace_matching_g2_evidence(self):
        config = copy.deepcopy(self.config)
        config["protocol_status"] = "frozen"
        config["gates"] = {"g2_passed":True}
        with self.assertRaises(ValueError):
            require_g2(config)
        directory = ROOT / "runs/_cache/tmp/tests"
        directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary)/"audit.json"
            atomic_json(path,{"status":"success","proposal_hash":proposal_hash(config),"code_sha256":critical_sources()})
            config["gates"]["g2_evidence"] = {"path":str(path.relative_to(ROOT)),"sha256":file_hash(path)}
            require_g2(config)
            config["generation"]["guidance_scale"] += 1
            with self.assertRaises(ValueError):
                require_g2(config)

    def test_balanced_smoke_never_uses_the_formal_generation_seeds(self):
        formal = candidate_plan(self.config)
        smoke = candidate_plan(self.config,smoke_per_class=3)
        self.assertEqual(len(smoke),30)
        self.assertFalse(set(r["generation_seed"] for r in formal)&set(r["generation_seed"] for r in smoke))
        self.assertFalse(set(r["image_id"] for r in formal)&set(r["image_id"] for r in smoke))


if __name__ == "__main__":
    unittest.main()
