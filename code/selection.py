"""Full CLIP probabilities, deterministic rankings and equal-size selections."""
from __future__ import annotations
import json
import importlib.metadata
import time
from pathlib import Path

import numpy as np
from PIL import Image

from a_protocol import CLASSES, atomic_json, class_texts, config_hash, file_hash, inside


def score_images(config: dict, run_dir: Path, records: list[dict]) -> tuple[list[dict],dict]:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from a_runtime import release_cuda
    start = time.perf_counter()
    clip_config = config["clip"]
    texts = class_texts(config,clip_config["text_template"])
    from huggingface_hub import hf_hub_download
    weights_path = Path(hf_hub_download(clip_config["model_id"],"pytorch_model.bin",
                                      revision=clip_config["revision"],local_files_only=True))
    weights_sha = file_hash(weights_path)
    if weights_sha != "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f":
        raise ValueError("CLIP checkpoint differs from the pinned official file")
    for record in records:
        if file_hash(inside(run_dir,record["relative_path"])) != record["sha256"]:
            raise ValueError("CLIP input image changed")
    fingerprint = config_hash({"model":clip_config,"texts":texts,"weights_sha256":weights_sha,
        "packages":{name:importlib.metadata.version(name) for name in ("torch","transformers","pillow")},
        "inputs":[(r["image_id"],r["sha256"],r.get("class_id")) for r in records],
        "source_sha256":file_hash(Path(__file__))})
    path = run_dir / "manifests/clip_scores.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["input_hash"] != fingerprint or prior["status"] != "success":
            raise ValueError("CLIP cached input/protocol differs")
        if prior["scores_sha256"] != config_hash(prior["records"]):
            raise ValueError("CLIP score cache changed")
        return prior["records"],dict(prior["timing"],reused_verified=True,current_seconds=time.perf_counter()-start)
    print(f"[select] CLIP scoring {len(records)} images",flush=True)
    model = CLIPModel.from_pretrained(clip_config["model_id"],revision=clip_config["revision"]).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(clip_config["model_id"],revision=clip_config["revision"],use_fast=False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-start
    scores = []
    inference_start = time.perf_counter()
    try:
        for first in range(0,len(records),16):
            group = records[first:first+16]
            images = [Image.open(inside(run_dir,r["relative_path"])).convert("RGB") for r in group]
            inputs = processor(text=texts,images=images,return_tensors="pt",padding=True).to("cuda")
            with torch.inference_mode():
                probabilities = model(**inputs).logits_per_image.softmax(dim=-1).float().cpu().numpy()
            if probabilities.shape != (len(group),10) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1):
                raise ValueError("CLIP probabilities are invalid")
            for record,p in zip(group,probabilities):
                scores.append({"schema_version":1,"image_id":record["image_id"],"class_id":record.get("class_id"),
                    "image_sha256":record["sha256"],"probabilities":list(map(float,p)),
                    "target_confidence":float(p[record["class_id"]]) if "class_id" in record else None,"predicted_class_id":int(p.argmax())})
        torch.cuda.synchronize()
        timing = {"load_seconds":load_seconds,"inference_seconds":time.perf_counter()-inference_start,
                  "seconds":time.perf_counter()-start,"reused_verified":False}
    finally:
        del model,processor
        release_cuda()
    atomic_json(path,{"schema_version":1,"status":"success","input_hash":fingerprint,
                     "records":scores,"scores_sha256":config_hash(scores),"texts":texts,"timing":timing})
    return scores,timing


def make_selections(records: list[dict], scores: list[dict], sizes: list[int], random_seed: int,
                    pool_id: str) -> tuple[dict,list[dict]]:
    by_id = {r["image_id"]:r for r in records}
    scored = {r["image_id"]:r for r in scores}
    if len(by_id) != len(records) or len(scored) != len(scores) or set(by_id) != set(scored):
        raise ValueError("Duplicate, missing or unexpected candidate/score IDs")
    if len(sizes) != 2 or not 0 < sizes[0] < sizes[1]:
        raise ValueError("Two increasing retained counts are required")
    selections = {"real_only":[]}
    ranks = []
    for method in ("random","clip"):
        for count in sizes:
            selections[f"{method}_{count}"] = []
        for class_id in range(len(CLASSES)):
            group = sorted([r for r in records if r["class_id"] == class_id],key=lambda r:r["image_id"])
            if len(group) < sizes[-1]:
                raise ValueError("Insufficient candidates for requested class count")
            for record in group:
                score = scored[record["image_id"]]
                if score["class_id"] != class_id or score["image_sha256"] != record["sha256"] or not 0 <= score["target_confidence"] <= 1:
                    raise ValueError("CLIP score does not match candidate class/image")
            if method == "random":
                order = np.random.default_rng(random_seed+class_id).permutation(len(group))
                group = [group[i] for i in order]
            else:
                group.sort(key=lambda r:(-scored[r["image_id"]]["target_confidence"],r["image_id"]))
            for rank,record in enumerate(group,1):
                ranks.append({"schema_version":1,"pool_id":pool_id,"method":method,"class_id":class_id,
                    "image_id":record["image_id"],"rank":rank,
                    "target_confidence":scored[record["image_id"]]["target_confidence"],
                    "random_seed":random_seed if method == "random" else None})
            for count in sizes:
                selections[f"{method}_{count}"].extend(r["image_id"] for r in group[:count])
    for method in ("random","clip"):
        if not set(selections[f"{method}_{sizes[0]}"]).issubset(selections[f"{method}_{sizes[1]}"]):
            raise ValueError("Selections are not nested")
        for count in sizes:
            ids = selections[f"{method}_{count}"]
            if len(ids) != 10*count or any(sum(by_id[i]["class_id"] == c for i in ids) != count for c in range(10)):
                raise ValueError("Selection class counts differ")
    return selections,ranks
