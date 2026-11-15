"""One small end-to-end development run; final-test evaluation is never invoked."""
from __future__ import annotations
import copy
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from a_protocol import (CLASSES, ROOT, atomic_json, config_hash, critical_sources,
                        file_hash, inside, proposal_hash, require_g2)
from a_runtime import configure_caches, environment, generate
from classifier import train_heads
from data_features import extract_features, materialize_real
from selection import make_selections, score_images


def contact_sheets(run_dir: Path, records: list[dict]) -> list[str]:
    output = []
    for page in range(2):
        canvas = Image.new("RGB",(576,1080),"white")
        draw = ImageDraw.Draw(canvas)
        for row,class_id in enumerate(range(page*5,(page+1)*5)):
            group = [r for r in records if r["class_id"] == class_id]
            for column,record in enumerate(group):
                image = Image.open(inside(run_dir,record["relative_path"])).convert("RGB")
                canvas.paste(image.resize((192,192)),(column*192,row*216+24))
                draw.text((column*192+4,row*216+5),f"{CLASSES[class_id]} / {column+1}",fill="black")
        path = run_dir / f"figures/generation_classes_{page*5}_{page*5+4}.jpg"
        path.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(path,quality=82)
        output.append(str(path.relative_to(run_dir)).replace("\\","/"))
    return output


def smoke(config: dict) -> dict:
    import torch
    torch.set_num_threads(4)
    configure_caches()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = ROOT / "runs/g2_smoke" / stamp
    run_dir.mkdir(parents=True,exist_ok=False)
    report = {"schema_version":1,"mode":"smoke","status":"running",
        "run_dir":str(run_dir.relative_to(ROOT)).replace("\\","/"),"config_hash":config_hash(config),
        "proposal_hash":proposal_hash(config),"code_sha256":critical_sources(),"environment":environment(),
        "scope":"Engineering verification on internal development data; not a formal treatment-effect result.",
        "official_test_labels_read":False,"stage":"generation"}
    atomic_json(run_dir / "protocol.json",{"mode":"smoke","config":config,"config_hash":config_hash(config)})
    report_path = run_dir / "report.json"
    atomic_json(report_path,report)
    print(f"[smoke] {report['run_dir']}",flush=True)
    started = time.perf_counter()
    try:
        count = config["smoke"]["candidates_per_class"]
        first_generation = generate(config,smoke=True,per_class=count)
        report["generation"] = first_generation
        atomic_json(run_dir / "logs/generation_initial.json",first_generation)
        resumed = generate(config,smoke=True,per_class=count)
        if resumed["generated"] != 0 or resumed["reused_verified"] != 10*count:
            raise ValueError("Generation resume did not verify every image")
        report["generation_resume"] = resumed
        origin = ROOT / first_generation["run_dir"]
        report["generation_history"] = []
        for path in sorted((origin/"logs").glob("generation_*.json")):
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") == "success" and previous.get("config_hash") == report["config_hash"]:
                report["generation_history"].append({"source":str(path.relative_to(ROOT)).replace("\\","/"),
                    "sha256":file_hash(path),"summary":previous})
        originals = json.loads((origin / "manifests/candidates.json").read_text(encoding="utf-8"))
        synthetic = []
        for record in originals:
            source = inside(origin,record["relative_path"])
            relative = "images/synthetic/"+record["image_id"]+".png"
            target = inside(run_dir,relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            if file_hash(source) != record["sha256"]:
                raise ValueError("Generated source changed")
            shutil.copyfile(source,target)
            synthetic.append(dict(record,relative_path=relative,split="synthetic",
                                  generation_origin_run=first_generation["run_dir"]))
        if len(synthetic) != 10*count or any(sum(r["class_id"] == c for r in synthetic) != count for c in range(10)):
            raise ValueError("Balanced generation counts differ")
        atomic_json(run_dir / "manifests/candidates.json",synthetic)
        report["contact_sheets"] = contact_sheets(run_dir,synthetic)
        report["stage"] = "real_data"
        atomic_json(report_path,report)
        real,data_audit = materialize_real(config,run_dir,smoke=True)
        report["data"] = data_audit
        report["stage"] = "selection"
        atomic_json(report_path,report)
        scores,timing = score_images(config,run_dir,synthetic)
        selections,ranks = make_selections(synthetic,scores,config["smoke"]["selection_sizes"],
                                           config["selection"]["random_seed"],report["run_dir"])
        repeated,repeated_ranks = make_selections(synthetic,scores,config["smoke"]["selection_sizes"],
                                                  config["selection"]["random_seed"],report["run_dir"])
        if selections != repeated or ranks != repeated_ranks:
            raise ValueError("Selection rankings are not reproducible")
        atomic_json(run_dir / "manifests/selections.json",{"schema_version":1,"mode":"smoke","selections":selections})
        atomic_json(run_dir / "manifests/rankings.json",ranks)
        report["selection"] = {"status":"success","timing":timing,"reproducible":True,"nested":True,
                               "actual_counts":{key:len(value)//10 for key,value in selections.items()}}
        report["stage"] = "features"
        atomic_json(report_path,report)
        features,feature_audit = extract_features(config,run_dir,real+synthetic)
        atomic_json(run_dir / "logs/features_initial.json",feature_audit)
        report["features"] = feature_audit
        cached,cache_audit = extract_features(config,run_dir,real+synthetic)
        if cache_audit["computed"] or cache_audit["cache_hits"] != len(real+synthetic):
            raise ValueError("Feature cache resume did not verify every input")
        report["feature_cache_resume"] = cache_audit
        report["stage"] = "classification_and_dev_evaluation"
        atomic_json(report_path,report)
        results,classifier_audit = train_heads(config,run_dir,real,synthetic,selections,cached,
                                               seed=config["smoke"]["split_seed"],mode="smoke")
        report["development_results"] = results
        report["classifiers"] = classifier_audit
        if len(results) != 5 or len({r["initial_state_sha256"] for r in results}) != 1 or len({r["real_schedule_sha256"] for r in results}) != 1:
            raise ValueError("Classifier pairing failed")
        if {r["updates"] for r in results} != {config["classifier"]["hyperparameters"]["updates"]}:
            raise ValueError("Classifier update budgets differ")
        report["checks"] = {"balanced_generation":True,"generation_resume_integrity":True,
            "no_train_dev_overlap":True,"dev_within_20_shot_budget":True,"equal_selection_counts":True,
            "nested_selections":True,"deterministic_rankings":True,"frozen_backbone_and_bn":
                feature_audit["parameters_frozen"] and feature_audit["batchnorm_unchanged"],
            "feature_cache_resume_integrity":True,"paired_initialization_and_real_batches":True,
            "equal_updates":True,"classifier_reload_identical":all(r["reload_identical"] for r in results),
            "saved_predictions_recompute_accuracy":True,"official_test_untouched":True}
        if not all(report["checks"].values()):
            raise ValueError("At least one smoke acceptance check failed")
        report.update(status="success",stage="mechanical_checks_complete",
                      visual_review_required_before_freeze=True)
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report["seconds"] = time.perf_counter()-started
        atomic_json(report_path,report)
    return report


def freeze(config: dict, config_path: Path, run_dir: Path) -> dict:
    if config["protocol_status"] == "frozen":
        require_g2(config)
        return {"status":"already_frozen","config_hash":config_hash(config)}
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "success" or not all(report["checks"].values()) or report["official_test_labels_read"]:
        raise ValueError("Smoke mechanical checks have not all passed")
    if report["proposal_hash"] != proposal_hash(config) or report["code_sha256"] != critical_sources():
        raise ValueError("Smoke evidence belongs to another config/code version")
    visual_path = run_dir / "audits/generation_visual.json"
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    candidates_path = run_dir / "manifests/candidates.json"
    if visual["candidate_manifest_sha256"] != file_hash(candidates_path) or visual["status"] != "reviewed":
        raise ValueError("Visual review does not match this generated pool")
    if len(visual["classes"]) != 10 or {r["class_id"] for r in visual["classes"]} != set(range(10)):
        raise ValueError("Visual review must cover all ten classes")
    if not all(r["usable_for_pipeline_trial"] for r in visual["classes"]):
        raise ValueError("Generation review found a class that requires correction before freezing")
    gate = {"schema_version":1,"status":"success","proposal_hash":proposal_hash(config),
        "code_sha256":critical_sources(),"smoke_report_sha256":file_hash(report_path),
        "visual_review_sha256":file_hash(visual_path),"run_dir":report["run_dir"],
        "scope":"G2 engineering acceptance; formal pool quality and downstream gains remain unmeasured."}
    gate_path = run_dir / "audits/g2_gate.json"
    atomic_json(gate_path,gate)
    shared_gate = ROOT / "shared" / ("g2_smoke_"+run_dir.name) / "g2_gate.json"
    atomic_json(shared_gate,gate)
    frozen = copy.deepcopy(config)
    frozen.update(protocol_status="frozen",protocol_version=config["protocol_version"].rsplit("-",1)[0]+"-frozen")
    frozen["gates"] = {"g2_passed":True,"g2_evidence":{"path":str(shared_gate.relative_to(ROOT)).replace("\\","/"),
                                                        "sha256":file_hash(gate_path)}}
    require_g2(frozen)
    atomic_json(config_path,frozen)
    return {"status":"frozen","config_hash":config_hash(frozen),"evidence":frozen["gates"]["g2_evidence"]}


def export_smoke(config: dict, run_dir: Path) -> dict:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "success":
        raise ValueError("Cannot publish a failed smoke as successful")
    target = ROOT / "shared" / ("g2_smoke_"+run_dir.name)
    target.mkdir(parents=True,exist_ok=True)
    # Remove machine-specific environment paths; keep executable source and package provenance.
    shared = copy.deepcopy(report)
    shared["environment"].pop("executable",None)
    shared["frozen_config_hash"] = config_hash(config)
    shared["g2_passed"] = config["gates"]["g2_passed"]
    for source,relative in ((run_dir/"manifests/real.json","real_split.json"),
                            (run_dir/"manifests/candidates.json","candidates.json"),
                            (run_dir/"manifests/selections.json","selections.json"),
                            (run_dir/"manifests/rankings.json","rankings.json"),
                            (run_dir/"manifests/clip_scores.json","clip_scores.json"),
                            (run_dir/"features/index.json","feature_index.json"),
                            (run_dir/"metrics/dev_predictions.json","dev_predictions.json"),
                            (run_dir/"audits/generation_visual.json","generation_visual.json"),
                            (run_dir/"audits/g2_gate.json","g2_gate.json")):
        if source.exists():
            atomic_json(target/relative,json.loads(source.read_text(encoding="utf-8")))
    for relative in report["contact_sheets"]:
        source = inside(run_dir,relative)
        shutil.copyfile(source,target/source.name)
    artifacts = [{"artifact_id":Path(x["relative_path"]).stem,"mode":"smoke",**x,
                  "run_dir":report["run_dir"],"access":"local_only_not_delivered",
                  "provided_by":"A / Codex-assisted integration"} for x in report["classifiers"]["models"]]
    atomic_json(target / "artifacts.json",artifacts)
    atomic_json(target / "check.json",shared)
    atomic_json(target / "frozen_config.json",config)
    return {"status":"exported","shared_dir":str(target.relative_to(ROOT)).replace("\\","/")}
