"""Configuration, run planning and integrity helpers; standard library only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CONDITIONS = (("real_only", 0, None), ("random_50", 50, "random"),
              ("clip_50", 50, "clip"), ("random_150", 150, "random"),
              ("clip_150", 150, "clip"))


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def inside(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path must name an item inside its run directory: {relative}")
    return path


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    data, gen, classifier = config["dataset"], config["generation"], config["classifier"]
    checks = (
        (config["schema_version"] == 1, "schema_version must be 1"),
        (data["name"] == "CIFAR10" and data["classes"] == list(CLASSES), "Official CIFAR-10 mapping required"),
        (data["real_per_class"] == 20 and data["split_seeds"] == [0, 1, 2], "Main protocol is 20-shot with seeds 0/1/2"),
        (gen["candidates_per_class"] == 200 and config["selection"]["sizes"] == [50, 150], "Main candidate and retained counts differ"),
        (gen["model_id"] == "stable-diffusion-v1-5/stable-diffusion-v1-5", "Main generator must be SD1.5"),
        (classifier["architecture"] == "resnet18" and classifier["freeze_backbone"] is True
         and classifier["freeze_batchnorm"] is True, "Backbone and BN must remain frozen"),
        (classifier["synthetic_loss_weight"] == 0.5 and classifier["train_views"] == 4, "Mixed weight/view protocol differs"),
        (gen["width"] == 512 and gen["height"] == 512 and gen["steps"] > 0, "Invalid generation size or steps"),
        (gen["dtype"] == "float16" and gen["variant"] == "fp16" and gen["generator_device"] == "cpu", "Unsupported precision/variant/RNG"),
        (gen["scheduler"] == "DPMSolverMultistepScheduler", "Unsupported scheduler"),
        (config["protocol_status"] in {"draft", "frozen"}, "Unknown protocol status"),
        (1 <= gen.get("batch_size",1) <= 4, "Generation batch must be between 1 and 4"),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)
    inside(ROOT / "runs", config["run_id"])
    for component in (gen, config["clip"]):
        revision = component["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Model revisions must be pinned commit hashes")
    gen["prompt_template"].format(class_name="airplane")
    if "class_phrases" in data and len(data["class_phrases"]) != len(CLASSES):
        raise ValueError("One prompt phrase is required for each official class")
    if classifier.get("hyperparameters"):
        hp = classifier["hyperparameters"]
        if hp["optimizer"] != "SGD" or hp["lr"] <= 0 or hp["updates"] <= 0 or hp["images_per_class_per_batch"] <= 0:
            raise ValueError("Invalid predeclared classifier settings")
    return config


def require_g2(config: dict) -> None:
    if config["protocol_status"] != "frozen" or config["gates"].get("g2_passed") is not True:
        raise ValueError("Full generation requires a frozen protocol and G2. Use generate --smoke for G1.")
    if not config["classifier"].get("hyperparameters"):
        raise ValueError("Classifier parameters have not been frozen")
    evidence = config["gates"].get("g2_evidence")
    if not evidence:
        raise ValueError("G2 requires an actual successful smoke audit")
    path = inside(ROOT, evidence["path"])
    if file_hash(path) != evidence["sha256"]:
        raise ValueError("G2 audit changed")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "success" or audit.get("proposal_hash") != proposal_hash(config):
        raise ValueError("G2 did not validate this protocol")
    if audit["code_sha256"] != critical_sources():
        raise ValueError("Execution code changed since G2; rerun smoke before freezing")


def critical_sources() -> dict:
    return {p.name: file_hash(p) for p in sorted((ROOT / "code").glob("*.py")) if not p.name.startswith("test_")}


def proposal_hash(config: dict) -> str:
    return config_hash({key: value for key, value in config.items()
                        if key not in {"gates", "protocol_status", "protocol_version"}})


def class_texts(config: dict, template: str) -> list[str]:
    return [template.format(class_name=name) for name in config["dataset"].get("class_phrases", CLASSES)]


def generation_core_hash(config: dict) -> str:
    """Sampling inputs, excluding only explicit invalid-output handling policies."""
    return config_hash({"generation":{k:v for k,v in config["generation"].items()
                        if k not in {"invalid_output_policy","safety_flag_policy"}},
                        "classes":config["dataset"]["classes"],
                        "class_phrases":config["dataset"].get("class_phrases",list(CLASSES))})


def generation_batches(items: list, size: int) -> list[list]:
    """Keep original batch membership fixed when resuming a partial run."""
    if not 1 <= size <= 4:
        raise ValueError("Invalid generation batch size")
    return [items[i:i+size] for i in range(0,len(items),size)]


def candidate_plan(config: dict, *, smoke_per_class: int | None = None) -> list[dict]:
    gen = config["generation"]
    count = gen["candidates_per_class"] if smoke_per_class is None else smoke_per_class
    prefix = "synth" if smoke_per_class is None else "smoke_synth"
    offset = 0 if smoke_per_class is None else config["smoke"]["generation_seed_offset"]
    texts = class_texts(config, gen["prompt_template"])
    return [{"schema_version": 1, "image_id": f"{prefix}_{class_id:02d}_{index:04d}",
             "class_id": class_id, "class_name": name, "source": "synthetic",
             "relative_path": f"images/{class_id:02d}/{index:04d}.png",
             "generation_seed": gen["seed_base"] + offset + class_id * 100_000 + index,
             "prompt": texts[class_id], "status": "planned"}
            for class_id, name in enumerate(CLASSES)
            for index in range(count)]


def training_plan(config: dict) -> list[dict]:
    return [{"condition_id": name, "seed": seed, "synthetic_per_class": count,
             "selection": method, "real_per_class": 20, "status": "planned"}
            for seed in config["dataset"]["split_seeds"] for name, count, method in CONDITIONS]


def verified_item(run_dir: Path, item: dict, fingerprint: str) -> dict | None:
    record_path = inside(run_dir, f"manifests/items/{item['image_id']}.json")
    image_path = inside(run_dir, item["relative_path"])
    if not record_path.exists() and not image_path.exists():
        return None
    if not record_path.exists() or not image_path.exists():
        raise ValueError(f"Incomplete existing output: {item['image_id']}; inspect it before resuming")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {"config_hash": fingerprint, "status": "success", "image_id": item["image_id"],
                "generation_seed": item["generation_seed"], "prompt": item["prompt"],
                "relative_path": item["relative_path"], "class_id": item["class_id"]}
    if any(record.get(key) != value for key, value in expected.items()) or file_hash(image_path) != record.get("sha256"):
        raise ValueError(f"Existing output failed integrity/config check: {item['image_id']}")
    if record.get("safety_flag"):
        review_path = inside(run_dir,record["safety_review_relative_path"])
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if (file_hash(review_path) != record["safety_review_sha256"] or review.get("sha256") != record["sha256"]
                or review.get("decision") != "safe_to_retain_same_image" or review.get("config_hash") != fingerprint
                or review.get("generation_seed") != item["generation_seed"]):
            raise ValueError("Cached image no longer has its exact safety review proof")
    return record
