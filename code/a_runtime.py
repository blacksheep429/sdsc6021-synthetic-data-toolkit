"""A-owned CUDA/model diagnostics and resumable standard SD1.5 generation."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from a_protocol import (ROOT, CLASSES, atomic_json, candidate_plan, config_hash,
                        file_hash, inside, require_g2, verified_item)


def configure_caches() -> None:
    temporary = ROOT / "runs/_cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "runs/_cache/torch"))


def environment() -> dict:
    wanted = ("torch", "torchvision", "diffusers", "transformers", "accelerate",
              "numpy", "pillow", "safetensors", "huggingface-hub", "hf-xet", "ftfy", "wcwidth")
    packages = {}
    for name in wanted:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    changes = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    sources = {path.name: file_hash(path) for path in sorted((ROOT / "code").glob("*.py"))}
    return {"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "code_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "worktree_dirty": bool(changes.stdout.strip()),
            "source_sha256": sources}


def cuda_check() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; G1 requires actual GPU inference")
    start = time.perf_counter()
    x = torch.ones((32, 32), device="cuda")
    product = x @ x
    torch.cuda.synchronize()
    if not torch.equal(product, torch.full_like(product, 32)):
        raise RuntimeError("CUDA matrix multiplication produced an incorrect result")
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "compiled_architectures": torch.cuda.get_arch_list(), "matrix_check": "passed",
            "seconds": time.perf_counter() - start}


def release_cuda() -> None:
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def load_pipeline(config: dict):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    gen = config["generation"]
    pipe = StableDiffusionPipeline.from_pretrained(
        gen["model_id"], revision=gen["revision"], variant=gen["variant"],
        torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, solver_order=gen["solver_order"],
        algorithm_type=gen["algorithm_type"], use_karras_sigmas=gen["use_karras_sigmas"])
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(config: dict, *, smoke: bool, limit: int = 1, run_dir: Path | None = None) -> dict:
    configure_caches()
    if not smoke:
        require_g2(config)
    elif not 1 <= limit <= 10:
        raise ValueError("G1 smoke must request between 1 and 10 images")
    fingerprint = config_hash(config)
    if run_dir is None:
        relative = f"generation_smoke/{fingerprint[:12]}" if smoke else config["run_id"]
        run_dir = inside(ROOT / "runs", relative)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding="utf-8"))
        if prior["config_hash"] != fingerprint or prior["mode"] != ("smoke" if smoke else "main"):
            raise ValueError("Run directory belongs to a different protocol or mode")
    atomic_json(protocol_path, {"config_hash": fingerprint, "mode": "smoke" if smoke else "main", "config": config})
    current_environment = environment()
    environment_path = run_dir / "environment.json"
    if environment_path.exists():
        prior_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        critical_packages = ("torch", "torchvision", "diffusers", "transformers", "pillow")
        critical_sources = ("a_protocol.py", "a_runtime.py")
        if (any(prior_environment["packages"].get(name) != current_environment["packages"].get(name)
                for name in critical_packages)
                or any(prior_environment["source_sha256"].get(name) != current_environment["source_sha256"].get(name)
                       for name in critical_sources)):
            raise ValueError("Generation code/package versions changed; preserve this run and use a new run directory")
    else:
        atomic_json(environment_path, current_environment)
    items = candidate_plan(config)[:limit] if smoke else candidate_plan(config)
    records, todo = [], []
    for item in items:
        record = verified_item(run_dir, item, fingerprint)
        (records if record else todo).append(record if record else item)
    reused = len(records)
    load_seconds = 0.0
    stage_start = time.perf_counter()
    if todo:
        import numpy as np
        import torch
        cuda_check()
        start = time.perf_counter()
        pipe = load_pipeline(config)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        gen = config["generation"]
        try:
            for item in todo:
                print(f"[generate] {item['image_id']} seed={item['generation_seed']}", flush=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    output = pipe(item["prompt"], negative_prompt=gen["negative_prompt"],
                                  num_inference_steps=gen["steps"], guidance_scale=gen["guidance_scale"],
                                  width=gen["width"], height=gen["height"],
                                  generator=torch.Generator(device="cpu").manual_seed(item["generation_seed"]))
                image = output.images[0].convert("RGB")
                pixels = np.asarray(image)
                if image.size != (gen["width"], gen["height"]) or int(pixels.max()) <= 1:
                    raise RuntimeError(f"Invalid/black image: {item['image_id']}")
                if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                    raise RuntimeError(f"Flagged output: {item['image_id']}; retain failure record")
                path = inside(run_dir, item["relative_path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".tmp")
                image.save(temporary, format="PNG")
                temporary.replace(path)
                torch.cuda.synchronize()
                record = dict(item, status="success", config_hash=fingerprint, sha256=file_hash(path),
                              seconds=time.perf_counter() - start, mode="smoke" if smoke else "main")
                atomic_json(inside(run_dir, f"manifests/items/{item['image_id']}.json"), record)
                records.append(record)
        except Exception as exc:
            atomic_json(run_dir / "generation_failure.json", {
                "status": "failed", "image_id": item["image_id"], "config_hash": fingerprint,
                "error_type": type(exc).__name__, "error": str(exc)})
            raise
        finally:
            del pipe
            release_cuda()
    records.sort(key=lambda record: record["image_id"])
    atomic_json(run_dir / "manifests/candidates.json", records)
    summary = {"status": "success", "mode": "smoke" if smoke else "main", "config_hash": fingerprint,
               "run_dir": str(run_dir.relative_to(ROOT)), "count": len(records), "generated": len(todo),
               "reused_verified": reused, "pipeline_load_seconds": load_seconds,
               "stage_seconds": time.perf_counter() - stage_start}
    atomic_json(run_dir / "generation_summary.json", summary)
    return summary


def model_checks(config: dict, image_path: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from torchvision import models
    image = Image.open(image_path).convert("RGB")
    clip_config = config["clip"]
    print("[doctor] CLIP inference", flush=True)
    start = time.perf_counter()
    clip = CLIPModel.from_pretrained(clip_config["model_id"], revision=clip_config["revision"]).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(clip_config["model_id"], revision=clip_config["revision"], use_fast=False)
    texts = [clip_config["text_template"].format(class_name=name) for name in CLASSES]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        probabilities = clip(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1):
        raise RuntimeError("Invalid CLIP probabilities")
    clip_result = {"model_id": clip_config["model_id"], "revision": clip_config["revision"],
                   "probabilities": dict(zip(CLASSES, map(float, probabilities))),
                   "top_class": CLASSES[int(probabilities.argmax())], "seconds": time.perf_counter() - start}
    del clip, processor, inputs
    release_cuda()
    print("[doctor] frozen ResNet-18 inference", flush=True)
    start = time.perf_counter()
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.requires_grad_(False)
    backbone.to("cuda").eval()
    batchnorm = [(module, module.running_mean.clone(), module.running_var.clone())
                 for module in backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    with torch.inference_mode():
        feature = backbone(weights.transforms()(image).unsqueeze(0).to("cuda"))
    torch.cuda.synchronize()
    unchanged = all(torch.equal(module.running_mean, mean) and torch.equal(module.running_var, variance)
                    for module, mean, variance in batchnorm)
    if feature.shape != (1, 512) or not torch.isfinite(feature).all() or not unchanged:
        raise RuntimeError("ResNet shape/finite/BatchNorm check failed")
    result = {"architecture": "resnet18", "weights": str(weights), "shape": list(feature.shape),
              "parameters_frozen": not any(parameter.requires_grad for parameter in backbone.parameters()),
              "batchnorm_unchanged": unchanged, "seconds": time.perf_counter() - start}
    del backbone, batchnorm, feature
    release_cuda()
    return {"clip": clip_result, "resnet": result}


def doctor(config: dict, *, models: bool) -> dict:
    configure_caches()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = ROOT / "runs/doctor" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "mode": "doctor", "config_hash": config_hash(config),
              "run_dir": str(run_dir.relative_to(ROOT)), "environment": environment()}
    stage = "cuda"
    report["stage"] = stage
    atomic_json(run_dir / "report.json", report)
    print(f"[doctor] {report['run_dir']}", flush=True)
    try:
        report["cuda"] = cuda_check()
        if models:
            stage = "sd_generation"
            report["stage"] = stage
            atomic_json(run_dir / "report.json", report)
            report["generation"] = generate(config, smoke=True, limit=1, run_dir=run_dir / "generation")
            stage = "clip_and_resnet"
            report["stage"] = stage
            atomic_json(run_dir / "report.json", report)
            report["models"] = model_checks(config, run_dir / "generation/images/00/0000.png")
        report["models_checked"] = models
        report["status"] = "success"
    except Exception as exc:
        report.update(status="failed", failed_stage=stage, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(run_dir / "report.json", report)
    return report
