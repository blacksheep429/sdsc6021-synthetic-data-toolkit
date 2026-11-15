"""G4-only evaluator. This module never reads test labels on import or during G3."""
from __future__ import annotations
import json
import pickle
import shutil
import statistics
import tarfile
import time
from datetime import datetime,timezone

import numpy as np
from PIL import Image

from a_protocol import ROOT, CLASSES, CONDITIONS, atomic_json, config_hash, file_hash, inside
from data_features import md5,extract_features
from formal import main_root,verify_model_seal
from selection import score_images


def paired_summary(results: list[dict],seeds: list[int]) -> list[dict]:
    expected = {(s,name) for s in seeds for name,_,_ in CONDITIONS}
    by_key = {(r["seed"],r["condition_id"]):r["accuracy"] for r in results}
    if len(results) != len(expected) or set(by_key) != expected:
        raise ValueError("Incomplete/duplicate formal evaluation matrix")
    output = []
    for name,_,_ in CONDITIONS:
        values = [float(by_key[s,name]) for s in seeds]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Invalid accuracy")
        delta = [100*(by_key[s,name]-by_key[s,"real_only"]) for s in seeds]
        output.append({"condition_id":name,"mean_accuracy":statistics.mean(values),
            "std_accuracy_ddof1":statistics.stdev(values),"paired_delta_pp_mean":statistics.mean(delta),
            "paired_delta_pp_std_ddof1":statistics.stdev(delta),"seeds":seeds,"accuracies":values})
    return output


def final_test_data():
    """The only function that extracts/unpickles the official test batch."""
    from torchvision.datasets import CIFAR10
    directory = ROOT/"runs/_cache/cifar10/final_test_only"
    directory.mkdir(parents=True,exist_ok=True)
    name,digest = CIFAR10.test_list[0]
    path = directory/name
    if not path.exists():
        archive = ROOT/"runs/_cache/downloads"/CIFAR10.filename
        if md5(archive) != CIFAR10.tgz_md5:
            raise ValueError("Official test archive checksum differs")
        with tarfile.open(archive,"r:gz") as tar:
            member = tar.getmember("cifar-10-batches-py/"+name)
            if not member.isfile():
                raise ValueError("Unexpected test archive member")
            temporary = path.with_name(path.name+".tmp")
            with tar.extractfile(member) as source,temporary.open("wb") as target:
                shutil.copyfileobj(source,target)
            if md5(temporary) != digest:
                raise ValueError("Official test batch checksum differs")
            temporary.replace(path)
    if md5(path) != digest:
        raise ValueError("Cached official test batch checksum differs")
    with path.open("rb") as handle:
        batch = pickle.load(handle,encoding="latin1")
    return batch["data"].reshape(-1,3,32,32).transpose(0,2,3,1),np.asarray(batch["labels"]),digest


def evaluate_final(config: dict) -> dict:
    import torch
    torch.set_num_threads(4)
    started = time.perf_counter()
    root = main_root(config)
    seal = verify_model_seal(config,root)
    seal_sha = file_hash(root/"models/frozen.json")
    pointer = root/"evaluation/latest.json"
    if pointer.exists():
        prior = json.loads(pointer.read_text(encoding="utf-8"))
        if prior["model_seal_sha256"] != seal_sha or file_hash(inside(root,prior["report_path"])) != prior["report_sha256"]:
            raise ValueError("Existing final evaluation source/result changed")
        report = json.loads(inside(root,prior["report_path"]).read_text(encoding="utf-8"))
        for relative,digest in report["prediction_sha256"].items():
            if file_hash(inside(root,relative)) != digest:
                raise ValueError("Saved final predictions changed")
        return dict(report,reused_verified=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = root/"evaluation"/stamp
    run.mkdir(parents=True,exist_ok=False)
    report = {"schema_version":1,"status":"running","mode":"final_test","config_hash":config_hash(config),
              "model_seal_sha256":seal_sha,"stage":"test_data","official_test_labels_read":False}
    report_path = run/"report.json"
    atomic_json(report_path,report)
    try:
        images,labels,test_md5 = final_test_data()
        report.update(official_test_labels_read=True,test_batch_md5=test_md5)
        if len(labels) != 10000 or np.bincount(labels,minlength=10).tolist() != [1000]*10:
            raise ValueError("Unexpected official test size/classes")
        records = []
        for i,image in enumerate(images):
            relative = f"images/test_{i:05d}.png"
            path = inside(run,relative)
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(image).save(path)
            # Feature and zero-shot helpers receive no test labels.
            records.append({"schema_version":1,"image_id":f"real_test_{i:05d}","relative_path":relative,
                            "sha256":file_hash(path),"split":"test","source":"real"})
        report["stage"] = "test_features"
        atomic_json(report_path,report)
        features,feature_audit = extract_features(config,run,records)
        x = torch.from_numpy(np.stack([features[r["image_id"]][0] for r in records])).float()
        results,prediction_files = [],{}
        inference_start = time.perf_counter()
        for row in seal["models"]:
            checkpoint = torch.load(inside(root,row["relative_path"]),map_location="cpu",weights_only=True)
            if checkpoint["mode"] != "main" or checkpoint["config_hash"] != config_hash(config) or checkpoint["classifier_config"] != config["classifier"] or checkpoint["classes"] != list(CLASSES):
                raise ValueError("Final classifier preprocessing/config differs")
            head = torch.nn.Linear(512,10)
            head.load_state_dict(checkpoint["head_state"])
            head.eval()
            with torch.inference_mode():
                predictions = head(x).argmax(1).numpy()
            rows = [{"sample_id":r["image_id"],"seed":row["seed"],"condition_id":row["condition_id"],
                     "true_label":int(y),"predicted_label":int(p),"config_hash":config_hash(config)}
                    for r,y,p in zip(records,labels,predictions)]
            path = run/f"predictions/{row['condition_id']}_seed{row['seed']}.json"
            atomic_json(path,rows)
            prediction_files[str(path.relative_to(root)).replace("\\","/")] = file_hash(path)
            results.append({"seed":row["seed"],"condition_id":row["condition_id"],
                "accuracy":float(np.mean(predictions==labels)),"per_class_accuracy":
                {name:float(np.mean(predictions[labels==c]==c)) for c,name in enumerate(CLASSES)}})
        report["head_inference_and_save_seconds"] = time.perf_counter()-inference_start
        report["stage"] = "clip_zero_shot"
        atomic_json(report_path,report)
        scores,clip_timing = score_images(config,run,records)
        clip_predictions = np.asarray([r["predicted_class_id"] for r in scores])
        zero_rows = [{"sample_id":r["image_id"],"true_label":int(y),"predicted_label":int(p)}
                     for r,y,p in zip(records,labels,clip_predictions)]
        path = run/"predictions/clip_zero_shot.json"
        atomic_json(path,zero_rows)
        prediction_files[str(path.relative_to(root)).replace("\\","/")] = file_hash(path)
        report.update(status="success",stage="complete",test_count=len(labels),results=results,
            summary=paired_summary(results,config["dataset"]["split_seeds"]),features=feature_audit,
            clip_zero_shot={"accuracy":float(np.mean(clip_predictions==labels)),"evaluations":1,"timing":clip_timing},
            prediction_sha256=prediction_files)
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report["seconds"] = time.perf_counter()-started
        atomic_json(report_path,report)
    atomic_json(pointer,{"status":"success","model_seal_sha256":seal_sha,
        "report_path":str(report_path.relative_to(root)).replace("\\","/"),"report_sha256":file_hash(report_path)})
    return report
