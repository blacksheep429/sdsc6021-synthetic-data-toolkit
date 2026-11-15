"""A-owned CUDA/model diagnostics and resumable standard SD1.5 generation."""
from __future__ import annotations

import gc
import io
import hashlib
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

from a_protocol import (ROOT, CLASSES, atomic_json, candidate_plan, class_texts, config_hash, generation_batches,
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


def generate(config: dict, *, smoke: bool, limit: int = 1, per_class: int | None = None,
             run_dir: Path | None = None) -> dict:
    configure_caches()
    if not smoke:
        require_g2(config)
    elif per_class is not None and not 1 <= per_class <= 3:
        raise ValueError("Balanced smoke supports 1-3 images per class")
    elif per_class is None and not 1 <= limit <= 10:
        raise ValueError("G1 smoke must request between 1 and 10 images")
    fingerprint = config_hash(config)
    if run_dir is None:
        relative = f"generation_smoke/{fingerprint[:12]}_balanced{per_class}" if smoke and per_class else (f"generation_smoke/{fingerprint[:12]}" if smoke else config["run_id"])
        run_dir = inside(ROOT / "runs", relative)
    items = candidate_plan(config, smoke_per_class=per_class) if smoke and per_class else (candidate_plan(config)[:limit] if smoke else candidate_plan(config))
    requested = [item["image_id"] for item in items]
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding="utf-8"))
        if prior["config_hash"] != fingerprint or prior["mode"] != ("smoke" if smoke else "main"):
            raise ValueError("Run directory belongs to a different protocol or mode")
        if prior.get("requested_image_ids", requested) != requested:
            raise ValueError("Requested generation subset changed; use a different smoke directory")
    atomic_json(protocol_path, {"config_hash": fingerprint, "mode": "smoke" if smoke else "main", "config": config,
                                "requested_image_ids": requested})
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
    records, todo = [], []
    for item in items:
        record = verified_item(run_dir, item, fingerprint)
        (records if record else todo).append(record if record else item)
    reused = len(records)
    known = {r["image_id"]:r for r in records}
    batches = generation_batches(items,config["generation"].get("batch_size",1))
    load_seconds = 0.0
    warmup_seconds = 0.0
    peak_memory = 0
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
        active_id = "warmup"
        try:
            torch.cuda.reset_peak_memory_stats()
            if gen.get("warmup_steps", 0):
                start = time.perf_counter()
                with torch.inference_mode():
                    pipe([i["prompt"] for i in batches[0]], negative_prompt=[gen["negative_prompt"]]*len(batches[0]),
                         num_inference_steps=gen["warmup_steps"], guidance_scale=gen["guidance_scale"],
                         width=gen["width"], height=gen["height"],
                         generator=[torch.Generator(device="cpu").manual_seed(gen["seed_base"]-1-i) for i in range(len(batches[0]))])
                torch.cuda.synchronize()
                warmup_seconds = time.perf_counter() - start
            for batch_index,batch in enumerate(batches):
                if all(i["image_id"] in known for i in batch):
                    continue
                active_id = ",".join(i["image_id"] for i in batch)
                print(f"[generate] batch={batch_index+1}/{len(batches)} {active_id}", flush=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    output = pipe([i["prompt"] for i in batch], negative_prompt=[gen["negative_prompt"]]*len(batch),
                                  num_inference_steps=gen["steps"], guidance_scale=gen["guidance_scale"],
                                  width=gen["width"], height=gen["height"],
                                  generator=[torch.Generator(device="cpu").manual_seed(i["generation_seed"]) for i in batch])
                torch.cuda.synchronize()
                batch_seconds = time.perf_counter()-start
                images = [im.convert("RGB") for im in output.images]
                if len(images) != len(batch):
                    raise RuntimeError("Pipeline returned an incorrect batch size")
                # Validate the entire batch before publishing any successful image.
                for item,image,flag in zip(batch,images,output.nsfw_content_detected or [False]*len(batch)):
                    if image.size != (gen["width"],gen["height"]) or int(np.asarray(image).max()) <= 1 or flag:
                        raise RuntimeError(f"Invalid/black/flagged output: {item['image_id']}; no seed substitution")
                for item,image in zip(batch,images):
                    save_start = time.perf_counter()
                    data = io.BytesIO()
                    image.save(data,format="PNG")
                    payload = data.getvalue()
                    digest = hashlib.sha256(payload).hexdigest()
                    if item["image_id"] in known:
                        if known[item["image_id"]]["sha256"] != digest:
                            raise RuntimeError("Partial batch regeneration changed a previously verified image")
                        continue
                    path = inside(run_dir,item["relative_path"])
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temporary = path.with_name(path.name+".tmp")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    serialization_seconds = time.perf_counter()-save_start
                    record = dict(item,status="success",config_hash=fingerprint,sha256=digest,
                        seconds=batch_seconds/len(batch)+serialization_seconds,
                        timing_definition="amortized batch inference plus own PNG serialization",
                        batch_index=batch_index,batch_members=[i["image_id"] for i in batch],
                        batch_inference_seconds=batch_seconds,serialization_seconds=serialization_seconds,
                        mode="smoke" if smoke else "main")
                    atomic_json(inside(run_dir,f"manifests/items/{item['image_id']}.json"),record)
                    known[item["image_id"]] = record
                    records.append(record)
                atomic_json(run_dir/"generation_progress.json",{"status":"running","completed":len(known),
                    "total":len(items),"config_hash":fingerprint,"batch_index":batch_index})
            peak_memory = torch.cuda.max_memory_allocated()
        except Exception as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            atomic_json(run_dir / "logs" / f"generation_failure_{stamp}.json", {
                "status": "failed", "image_id_or_batch": active_id, "config_hash": fingerprint,
                "elapsed_seconds":time.perf_counter()-stage_start,"pipeline_load_seconds":load_seconds,
                "warmup_seconds":warmup_seconds,"completed":len(known),
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
               "batch_size":config["generation"].get("batch_size",1),
               "warmup_seconds": warmup_seconds, "peak_cuda_allocated_bytes": peak_memory,
               "stage_seconds": time.perf_counter() - stage_start}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    atomic_json(run_dir / "logs" / f"generation_{stamp}.json", summary)
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
    texts = class_texts(config, clip_config["text_template"])
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
