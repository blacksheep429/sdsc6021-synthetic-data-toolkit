"""G3 stages around the same generation/selection/features/head implementations."""
from __future__ import annotations
import json
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from a_protocol import (ROOT, CONDITIONS, atomic_json, candidate_plan, config_hash,
                        critical_sources, file_hash, inside, require_g2, verified_item)
from a_runtime import configure_caches, environment
from classifier import train_heads
from data_features import extract_features, materialize_real
from selection import make_selections, score_images


def main_root(config: dict) -> Path:
    configure_caches()
    require_g2(config)
    root = inside(ROOT/"runs",config["run_id"])
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    if protocol["mode"] != "main" or protocol["config_hash"] != config_hash(config):
        raise ValueError("Formal run belongs to another protocol")
    prior = json.loads((root/"environment.json").read_text(encoding="utf-8"))
    current = environment()
    if prior["packages"] != current["packages"] or any(prior["source_sha256"].get(k) != v for k,v in critical_sources().items()):
        raise ValueError("Formal source or dependency versions changed")
    return root


def verified_candidates(config: dict,root: Path) -> list[dict]:
    plan = candidate_plan(config)
    rows = [verified_item(root,item,config_hash(config)) for item in plan]
    if any(r is None for r in rows):
        raise ValueError("Formal candidate pool is incomplete")
    saved = json.loads((root/"manifests/candidates.json").read_text(encoding="utf-8"))
    if config_hash(saved) != config_hash(rows):
        raise ValueError("Aggregate candidate manifest differs from verified per-image records")
    return rows


def receipt(root: Path,stage: str,started: float,result: dict) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = dict(result,schema_version=1,stage=stage,invocation_seconds=time.perf_counter()-started)
    atomic_json(root/f"logs/{stage}_{stamp}.json",output)
    return output


def select_main(config: dict) -> dict:
    start = time.perf_counter()
    root = main_root(config)
    candidates = verified_candidates(config,root)
    scores,timing = score_images(config,root,candidates)
    selections,ranks = make_selections(candidates,scores,config["selection"]["sizes"],config["selection"]["random_seed"],config["run_id"])
    atomic_json(root/"manifests/selections.json",{"schema_version":1,"mode":"main","config_hash":config_hash(config),"selections":selections})
    atomic_json(root/"manifests/rankings.json",ranks)
    report = {"status":"success","mode":"main","config_hash":config_hash(config),"timing":timing,
        "candidates":len(candidates),"actual_counts":{k:len(v)//10 for k,v in selections.items()},
        "nested":True,"outputs_sha256":{p:file_hash(root/p) for p in
            ("manifests/clip_scores.json","manifests/selections.json","manifests/rankings.json")},"official_test_labels_read":False}
    atomic_json(root/"selection_report.json",report)
    return receipt(root,"select",start,report)


def selections_main(config: dict,root: Path) -> dict:
    report = json.loads((root/"selection_report.json").read_text(encoding="utf-8"))
    if report["status"] != "success" or report["config_hash"] != config_hash(config):
        raise ValueError("Selection stage is incomplete")
    for path,digest in report["outputs_sha256"].items():
        if file_hash(inside(root,path)) != digest:
            raise ValueError("Selection artifact changed")
    selected = json.loads((root/"manifests/selections.json").read_text(encoding="utf-8"))["selections"]
    if set(selected) != {name for name,_,_ in CONDITIONS}:
        raise ValueError("Formal conditions differ")
    by_id = {r["image_id"]:r for r in verified_candidates(config,root)}
    for name,count,_ in CONDITIONS:
        ids = selected[name]
        if len(ids) != 10*count or len(set(ids)) != len(ids) or any(sum(by_id[i]["class_id"]==c for i in ids)!=count for c in range(10)):
            raise ValueError("Formal selection class counts differ")
    return {name:selected[name] for name,_,_ in CONDITIONS}


def features_main(config: dict) -> dict:
    import torch
    torch.set_num_threads(4)
    started = time.perf_counter()
    root = main_root(config)
    selections_main(config,root)
    synthetic = verified_candidates(config,root)
    features,synthetic_audit = extract_features(config,root,synthetic)
    records = {r["image_id"]:r for r in synthetic}
    split_audits = {}
    for seed in config["dataset"]["split_seeds"]:
        directory = root/f"splits/seed{seed}"
        real,provenance = materialize_real(config,directory,smoke=False,seed=seed)
        real_features,audit = extract_features(config,directory,real)
        for row in real:
            name = row["image_id"]
            if name in features and not np.array_equal(features[name],real_features[name]):
                raise ValueError("Same real image acquired different feature views across paired seeds")
            features[name] = real_features[name]
            records[name] = row
        split_audits[str(seed)] = {"data":provenance,"features":audit,
            "real_manifest_sha256":file_hash(directory/"manifests/real.json")}
    names = sorted(features)
    array = np.stack([features[name] for name in names])
    if array.shape != (len(names),4,512) or not np.isfinite(array).all():
        raise ValueError("Formal feature bundle is invalid")
    bundle = root/"features/training_features.npz"
    temporary = bundle.with_name(bundle.name+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,x=array,image_ids=np.asarray(names),labels=np.asarray([records[n]["class_id"] for n in names],dtype=np.int64))
    temporary.replace(bundle)
    metadata = [{"row":i,"image_id":name,"class_id":records[name]["class_id"],"source":records[name]["source"],
                 "image_sha256":records[name]["sha256"],"views":4} for i,name in enumerate(names)]
    atomic_json(root/"features/bundle_index.json",metadata)
    report = {"status":"success","mode":"main","config_hash":config_hash(config),
        "synthetic":synthetic_audit,"paired_splits":split_audits,"images":len(names),"views":len(names)*4,
        "bundle":{"relative_path":"features/training_features.npz","sha256":file_hash(bundle),"shape":list(array.shape)},
        "index_sha256":file_hash(root/"features/bundle_index.json"),"official_test_labels_read":False,
        "seconds":time.perf_counter()-started}
    atomic_json(root/"features_report.json",report)
    return receipt(root,"features",started,report)


def load_training_features(config: dict,root: Path) -> tuple[dict,dict]:
    report = json.loads((root/"features_report.json").read_text(encoding="utf-8"))
    if report["status"] != "success" or report["config_hash"] != config_hash(config):
        raise ValueError("Formal features are incomplete")
    bundle = inside(root,report["bundle"]["relative_path"])
    if file_hash(bundle) != report["bundle"]["sha256"] or file_hash(root/"features/bundle_index.json") != report["index_sha256"]:
        raise ValueError("Formal feature bundle/index changed")
    metadata = json.loads((root/"features/bundle_index.json").read_text(encoding="utf-8"))
    with np.load(bundle,allow_pickle=False) as data:
        names,arrays,labels = data["image_ids"].tolist(),data["x"].copy(),data["labels"].tolist()
    if names != [r["image_id"] for r in metadata] or labels != [r["class_id"] for r in metadata] or len(set(names)) != len(names):
        raise ValueError("Feature bundle row/ID/class mapping differs")
    if list(arrays.shape) != report["bundle"]["shape"] or not np.isfinite(arrays).all():
        raise ValueError("Feature bundle shape/values differ")
    return dict(zip(names,arrays)),report


def verify_model_seal(config: dict,root: Path) -> dict:
    seal = json.loads((root/"models/frozen.json").read_text(encoding="utf-8"))
    expected = {(s,name) for s in config["dataset"]["split_seeds"] for name,_,_ in CONDITIONS}
    if (seal["status"] != "success" or seal["config_hash"] != config_hash(config)
            or seal["code_sha256"] != critical_sources() or len(seal["models"]) != 15
            or {(r["seed"],r["condition_id"]) for r in seal["models"]} != expected):
        raise ValueError("Formal model seal does not match the full paired matrix")
    for row in seal["models"]:
        if file_hash(inside(root,row["relative_path"])) != row["sha256"]:
            raise ValueError("A sealed classifier changed")
    if (file_hash(root/"manifests/selections.json") != seal["selection_manifest_sha256"]
            or file_hash(root/"features/training_features.npz") != seal["feature_bundle_sha256"]):
        raise ValueError("Sealed training inputs changed")
    return seal


def train_main(config: dict) -> dict:
    started = time.perf_counter()
    root = main_root(config)
    if (root/"models/frozen.json").exists():
        return receipt(root,"train",started,dict(verify_model_seal(config,root),reused_verified=True))
    selected = selections_main(config,root)
    features,feature_report = load_training_features(config,root)
    synthetic = verified_candidates(config,root)
    jobs,models,audits = [],[],{}
    for seed in config["dataset"]["split_seeds"]:
        directory = root/f"splits/seed{seed}"
        path = directory/"manifests/real.json"
        if file_hash(path) != feature_report["paired_splits"][str(seed)]["real_manifest_sha256"]:
            raise ValueError("Paired real split changed after feature extraction")
        real = json.loads(path.read_text(encoding="utf-8"))
        results,audit = train_heads(config,directory,real,synthetic,selected,features,seed=seed,mode="main")
        if len({r["initial_state_sha256"] for r in results}) != 1 or len({r["real_schedule_sha256"] for r in results}) != 1 or not all(r["reload_identical"] for r in results):
            raise ValueError("Formal paired head initialization/batches/reload differ")
        jobs.extend(results)
        audits[str(seed)] = audit
        for result,artifact in zip(results,audit["models"]):
            models.append({"condition_id":result["condition_id"],"seed":seed,
                "relative_path":f"splits/seed{seed}/"+artifact["relative_path"],"sha256":artifact["sha256"],
                "updates":result["updates"],"initial_state_sha256":result["initial_state_sha256"],
                "real_schedule_sha256":result["real_schedule_sha256"]})
    if len(jobs) != 15 or any(r["accuracy"] is not None for r in jobs):
        raise ValueError("G3 must complete 15 models without estimating test accuracy")
    seal = {"schema_version":1,"status":"success","mode":"main","config_hash":config_hash(config),
        "code_sha256":critical_sources(),"models":models,"paired_seed_audits":audits,
        "final_test_evaluated":False,"official_test_labels_read":False,
        "feature_bundle_sha256":feature_report["bundle"]["sha256"],
        "selection_manifest_sha256":file_hash(root/"manifests/selections.json"),
        "source_commit":environment()["code_commit"],"training_seconds":time.perf_counter()-started}
    atomic_json(root/"models/training_jobs.json",jobs)
    atomic_json(root/"models/frozen.json",seal)
    verify_model_seal(config,root)
    return receipt(root,"train",started,seal)


def export_formal(config: dict) -> dict:
    root = main_root(config)
    seal = verify_model_seal(config,root)
    candidates = verified_candidates(config,root)
    selections_main(config,root)
    _,feature_report = load_training_features(config,root)
    shared = ROOT/"shared"/config["run_id"]
    shared.mkdir(parents=True,exist_ok=True)
    files = ("manifests/candidates.json","manifests/clip_scores.json","manifests/rankings.json",
             "manifests/selections.json","selection_report.json","features/bundle_index.json",
             "features_report.json","models/frozen.json","models/training_jobs.json")
    for relative in files:
        atomic_json(shared/relative,json.loads(inside(root,relative).read_text(encoding="utf-8")))
    for seed in config["dataset"]["split_seeds"]:
        atomic_json(shared/f"splits/seed{seed}.json",json.loads((root/f"splits/seed{seed}/manifests/real.json").read_text(encoding="utf-8")))
    atomic_json(shared/"protocol.json",config)
    artifacts = [{"artifact_id":f"{r['condition_id']}_seed{r['seed']}","kind":"linear_classifier","run_id":config["run_id"],
        "relative_path":r["relative_path"],"sha256":r["sha256"],"config_hash":config_hash(config),
        "provided_by":"A / Codex under user direction","access":"local_only_not_delivered"} for r in seal["models"]]
    artifacts.append({"artifact_id":"training_features","kind":"portable_npz",**feature_report["bundle"],
                      "run_id":config["run_id"],"access":"local_only_not_delivered"})
    atomic_json(shared/"artifacts.json",artifacts)
    generation_history = []
    for path in sorted((root/"logs").glob("generation_*.json")):
        generation_history.append({"source":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":file_hash(path),
                                   "record":json.loads(path.read_text(encoding="utf-8"))})
    report = {"schema_version":1,"status":"G3_complete_G4_not_started","run_id":config["run_id"],
        "config_hash":config_hash(config),"code_sha256":critical_sources(),"candidates":len(candidates),
        "candidates_per_class":config["generation"]["candidates_per_class"],"classifiers":len(seal["models"]),
        "generation_history":generation_history,"features":feature_report,"training_seconds":seal["training_seconds"],
        "official_test_labels_read":False,"formal_accuracy":None,"model_seal_sha256":file_hash(root/"models/frozen.json"),
        "limitation":"Training reload-check predictions are not an estimate of generalization. G4 test evaluation, label audit and user trial remain outstanding."}
    if (root/"evaluation/latest.json").exists():
        pointer = json.loads((root/"evaluation/latest.json").read_text(encoding="utf-8"))
        path = inside(root,pointer["report_path"])
        if pointer["model_seal_sha256"] != report["model_seal_sha256"] or file_hash(path) != pointer["report_sha256"]:
            raise ValueError("Final evaluation source/result differs")
        final = json.loads(path.read_text(encoding="utf-8"))
        if final["status"] != "success":
            raise ValueError("Final evaluation has not succeeded")
        atomic_json(shared/"final_evaluation.json",final)
        report.update(status="G4_evaluation_complete_audits_pending",official_test_labels_read=True,
                      formal_accuracy=final["summary"],clip_zero_shot=final["clip_zero_shot"])
    atomic_json(shared/"check.json",report)
    return {"status":"exported","shared_dir":str(shared.relative_to(ROOT)).replace("\\","/"),"classifiers":15,"test_evaluated":False}
