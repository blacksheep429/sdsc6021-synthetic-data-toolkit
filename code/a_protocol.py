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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    return config


def require_g2(config: dict) -> None:
    if config["protocol_status"] != "frozen" or config["gates"].get("g2_passed") is not True:
        raise ValueError("Full generation requires a frozen protocol and G2. Use generate --smoke for G1.")
    if not config["classifier"].get("hyperparameters"):
        raise ValueError("Classifier parameters have not been frozen")


def candidate_plan(config: dict) -> list[dict]:
    gen = config["generation"]
    return [{"schema_version": 1, "image_id": f"synth_{class_id:02d}_{index:04d}",
             "class_id": class_id, "class_name": name, "source": "synthetic",
             "relative_path": f"images/{class_id:02d}/{index:04d}.png",
             "generation_seed": gen["seed_base"] + class_id * 100_000 + index,
             "prompt": gen["prompt_template"].format(class_name=name), "status": "planned"}
            for class_id, name in enumerate(CLASSES)
            for index in range(gen["candidates_per_class"])]


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
    return record
