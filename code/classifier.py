"""Paired linear heads on frozen features, with equal update and source budgets."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from a_protocol import CLASSES, atomic_json, config_hash, file_hash


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """First average views of an original, then average original images."""
    n,views,classes = logits.shape
    return F.cross_entropy(logits.reshape(n*views,classes),labels.repeat_interleave(views),reduction="none").reshape(n,views).mean(1).mean()


def objective(real_logits: torch.Tensor, real_labels: torch.Tensor,
              synthetic_logits: torch.Tensor | None = None, synthetic_labels: torch.Tensor | None = None) -> torch.Tensor:
    real = source_loss(real_logits,real_labels)
    if synthetic_logits is None:
        return real
    return 0.5*real + 0.5*source_loss(synthetic_logits,synthetic_labels)


def sample_schedule(labels: np.ndarray, steps: int, per_class: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    columns = []
    for class_id in range(10):
        group = np.flatnonzero(labels == class_id)
        if not len(group):
            raise ValueError("Each training source must contain every class")
        columns.append(rng.choice(group,size=(steps,per_class),replace=True))
    return torch.from_numpy(np.concatenate(columns,axis=1)).long()


def train_heads(config: dict, run_dir: Path, real: list[dict], synthetic: list[dict],
                selections: dict, features: dict, *, seed: int, mode: str) -> tuple[list[dict],dict]:
    if mode != "smoke":
        raise ValueError("Formal classifier execution is reserved for G3 integration")
    hp = config["classifier"]["hyperparameters"]
    train = [r for r in real if r["split"] == "train"]
    dev = [r for r in real if r["split"] == "dev"]
    if not dev or set(r["original_train_index"] for r in train) & set(r["original_train_index"] for r in dev):
        raise ValueError("Missing development data or split leakage")
    arrays = lambda rows: torch.from_numpy(np.stack([features[r["image_id"]] for r in rows])).float()
    rx,dx = arrays(train),arrays(dev)
    ry,dy = torch.tensor([r["class_id"] for r in train]),torch.tensor([r["class_id"] for r in dev])
    schedule = sample_schedule(ry.numpy(),hp["updates"],hp["images_per_class_per_batch"],seed+11000)
    schedule_hash = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = nn.Linear(512,10).state_dict()
    init_hash = state_hash(initial)
    by_id = {r["image_id"]:r for r in synthetic}
    metrics,all_predictions,model_records = [],[],[]
    start = time.perf_counter()
    for condition,selected in selections.items():
        print(f"[train] smoke {condition}, seed={seed}, updates={hp['updates']}",flush=True)
        condition_start = time.perf_counter()
        head = nn.Linear(512,10)
        head.load_state_dict(initial)
        assert state_hash(head.state_dict()) == init_hash
        optimizer = torch.optim.SGD(head.parameters(),lr=hp["lr"],momentum=hp["momentum"],weight_decay=hp["weight_decay"])
        sx,sy,synthetic_schedule = None,None,None
        if selected:
            rows = [by_id[i] for i in selected]
            sx = arrays(rows)
            sy = torch.tensor([r["class_id"] for r in rows])
            synthetic_schedule = sample_schedule(sy.numpy(),hp["updates"],hp["images_per_class_per_batch"],seed+22000)
        losses = []
        head.train()
        for step,indices in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            real_logits = head(rx[indices])
            syn_indices = synthetic_schedule[step] if selected else None
            loss = objective(real_logits,ry[indices],head(sx[syn_indices]) if selected else None,
                             sy[syn_indices] if selected else None)
            if not torch.isfinite(loss):
                raise ValueError("Classifier loss is not finite")
            loss.backward()
            optimizer.step()
            if step in {0,hp["updates"]-1}:
                losses.append({"update":step+1,"loss":float(loss.detach())})
        head.eval()
        with torch.inference_mode():
            before = head(dx[:,0])
        path = run_dir / f"models/{condition}_seed{seed}.pt"
        path.parent.mkdir(parents=True,exist_ok=True)
        checkpoint = {"schema_version":1,"mode":mode,"condition_id":condition,"seed":seed,
            "config_hash":config_hash(config),"head_state":head.state_dict(),"feature_dimension":512,
            "classes":list(CLASSES),"classifier_config":config["classifier"],
            "initial_state_sha256":init_hash,"real_schedule_sha256":schedule_hash,"updates":hp["updates"]}
        temporary = path.with_name(path.name+".tmp")
        torch.save(checkpoint,temporary)
        temporary.replace(path)
        restored = torch.load(path,map_location="cpu",weights_only=True)
        reloaded = nn.Linear(512,10)
        reloaded.load_state_dict(restored["head_state"])
        reloaded.eval()
        with torch.inference_mode():
            after = reloaded(dx[:,0])
        if not torch.equal(before,after):
            raise ValueError("Reloaded classifier predictions changed")
        predictions = after.argmax(1)
        accuracy = float((predictions == dy).float().mean())
        item = {"schema_version":1,"mode":mode,"evaluation_split":"official_train_internal_dev",
            "condition_id":condition,"seed":seed,"synthetic_per_class":len(selected)//10,
            "accuracy":accuracy,"dev_count":len(dev),"updates":hp["updates"],"initial_state_sha256":init_hash,
            "real_schedule_sha256":schedule_hash,"reload_identical":True,"loss_checks":losses,
            "training_and_reload_seconds":time.perf_counter()-condition_start}
        metrics.append(item)
        model_records.append({"relative_path":str(path.relative_to(run_dir)).replace("\\","/"),"sha256":file_hash(path)})
        for row,prediction in zip(dev,predictions.tolist()):
            all_predictions.append({"schema_version":1,"mode":mode,"sample_id":row["image_id"],
                "condition_id":condition,"seed":seed,"true_label":row["class_id"],"predicted_label":prediction,
                "config_hash":config_hash(config)})
    # Accuracy is always reconstructed from saved predictions, not copied from a display.
    for metric in metrics:
        rows = [r for r in all_predictions if r["condition_id"] == metric["condition_id"]]
        recomputed = sum(r["true_label"] == r["predicted_label"] for r in rows)/len(rows)
        if abs(recomputed-metric["accuracy"]) > 1e-6:
            raise ValueError("Saved predictions and accuracy differ")
        metric["accuracy"] = recomputed
    audit = {"status":"success","mode":mode,"classifiers":len(metrics),"paired_initialization":True,
             "paired_real_batches":True,"equal_updates":True,"source_weights":{"real":0.5,"synthetic":0.5},
             "baseline_real_weight":1.0,"view_loss_reduction":"mean views per original, then mean originals",
             "reload_identical":True,"seconds":time.perf_counter()-start,"models":model_records,
             "classifier_device":"cpu","hyperparameters_predeclared":hp}
    atomic_json(run_dir / "metrics/dev_predictions.json",all_predictions)
    atomic_json(run_dir / "metrics/dev_results.json",metrics)
    atomic_json(run_dir / "models/audit.json",audit)
    return metrics,audit
