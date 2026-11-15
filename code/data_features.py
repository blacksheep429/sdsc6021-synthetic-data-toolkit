"""Verified CIFAR training data and frozen, reproducible ResNet feature caches."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from a_protocol import CLASSES, ROOT, atomic_json, config_hash, file_hash, inside


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cifar_train() -> tuple[np.ndarray, np.ndarray, dict]:
    """Never extract or unpickle test_batch. The official archive includes it."""
    from torchvision.datasets import CIFAR10
    directory = ROOT / "runs/_cache/cifar10/train_only"
    directory.mkdir(parents=True, exist_ok=True)
    expected = CIFAR10.train_list + [(CIFAR10.meta["filename"], CIFAR10.meta["md5"])]
    missing = [(name, digest) for name, digest in expected
               if not (directory / name).exists() or md5(directory / name) != digest]
    if missing:
        archive = ROOT / "runs/_cache/downloads" / CIFAR10.filename
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            partial = archive.with_name(archive.name + ".part")
            if partial.exists() and md5(partial) == CIFAR10.tgz_md5:
                partial.replace(archive)
            else:
                print("[data] downloading official CIFAR-10 archive", flush=True)
                with urllib.request.urlopen(CIFAR10.url, timeout=30) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if md5(partial) != CIFAR10.tgz_md5:
                    raise ValueError("CIFAR archive MD5 mismatch; retain the partial for inspection")
                partial.replace(archive)
        if md5(archive) != CIFAR10.tgz_md5:
            raise ValueError("CIFAR archive MD5 mismatch")
        with tarfile.open(archive, "r:gz") as tar:
            for name, digest in missing:
                # Read only exact whitelisted regular files, no extractall/path traversal.
                member = tar.getmember("cifar-10-batches-py/" + name)
                if not member.isfile():
                    raise ValueError("Unexpected CIFAR archive member")
                temporary = directory / (name + ".tmp")
                with tar.extractfile(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if md5(temporary) != digest:
                    raise ValueError("CIFAR training batch MD5 mismatch")
                temporary.replace(directory / name)
    with (directory / "batches.meta").open("rb") as handle:
        metadata = pickle.load(handle, encoding="latin1")
    if metadata["label_names"] != list(CLASSES):
        raise ValueError("CIFAR official label mapping differs")
    arrays, labels = [], []
    for name, _ in CIFAR10.train_list:
        with (directory / name).open("rb") as handle:
            batch = pickle.load(handle, encoding="latin1")
        arrays.append(batch["data"])
        labels.extend(batch["labels"])
    images = np.concatenate(arrays).reshape(-1,3,32,32).transpose(0,2,3,1)
    return images, np.asarray(labels, dtype=np.int64), {
        "source":CIFAR10.url, "official_train_file_md5":dict(expected),
        "official_test_batch_extracted":False, "official_test_labels_read":False,
        "archive_contains_test_bytes":True}


def paired_split(labels: np.ndarray, seed: int, total: int = 20, dev: int = 0) -> tuple[list[int], list[int]]:
    if not 0 <= dev < total:
        raise ValueError("Development images must be part of the 20-shot budget")
    rng = np.random.default_rng(seed)
    train, development = [], []
    for class_id in range(10):
        chosen = rng.permutation(np.flatnonzero(labels == class_id))[:total]
        if len(chosen) != total:
            raise ValueError("Insufficient real examples")
        train.extend(map(int, chosen[dev:]))
        development.extend(map(int, chosen[:dev]))
    if set(train) & set(development):
        raise ValueError("Train/development leakage")
    return train, development


def materialize_real(config: dict, run_dir: Path, *, smoke: bool) -> tuple[list[dict], dict]:
    images, labels, provenance = cifar_train()
    seed = config["smoke"]["split_seed"] if smoke else config["dataset"]["split_seeds"][0]
    dev_count = config["smoke"]["dev_per_class"] if smoke else 0
    train, development = paired_split(labels, seed, config["dataset"]["real_per_class"], dev_count)
    records = []
    for split, indices in (("train",train), ("dev",development)):
        for index in indices:
            image_id = f"real_train_{index:05d}"
            relative = f"images/real/{image_id}.png"
            path = inside(run_dir, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if not np.array_equal(np.asarray(Image.open(path).convert("RGB")), images[index]):
                    raise ValueError("Existing real image differs from the verified dataset")
            else:
                temporary = path.with_name(path.name + ".tmp")
                Image.fromarray(images[index]).save(temporary, format="PNG")
                temporary.replace(path)
            label = int(labels[index])
            records.append({"schema_version":1,"image_id":image_id,"class_id":label,
                "class_name":CLASSES[label],"source":"real","original_train_index":index,
                "relative_path":relative,"sha256":file_hash(path),"split":split,"split_seed":seed})
    provenance.update(mode="smoke" if smoke else "main", train_count=len(train),dev_count=len(development),
                      selected_label_budget=len(train)+len(development), dev_is_subset_of_20_shot=True)
    atomic_json(run_dir / "manifests/real.json", records)
    atomic_json(run_dir / "manifests/data_provenance.json", provenance)
    return records, provenance


def feature_key(config: dict, record: dict, views: int) -> str:
    return config_hash({"image_sha256":record["sha256"],"image_id":record["image_id"],
        "views":views,"classifier":{k:v for k,v in config["classifier"].items()
                                    if k not in {"hyperparameters","synthetic_loss_weight"}},
        "extractor_source_sha256":file_hash(Path(__file__)),
        "torch":importlib.metadata.version("torch"), "torchvision":importlib.metadata.version("torchvision"),
        "weights_sha256":"f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"})


def extract_features(config: dict, run_dir: Path, records: list[dict]) -> tuple[dict, dict]:
    import torch
    from torchvision import models, transforms
    from a_runtime import release_cuda
    start = time.perf_counter()
    cache = ROOT / "runs/_cache/features"
    cache.mkdir(parents=True, exist_ok=True)
    output, index, todo = {}, [], []
    for record in records:
        if file_hash(inside(run_dir,record["relative_path"])) != record["sha256"]:
            raise ValueError("Feature input image changed")
        views = 1 if record.get("split") == "dev" else config["classifier"]["train_views"]
        key = feature_key(config,record,views)
        path, meta = cache / (key+".npz"), cache / (key+".json")
        if path.exists() or meta.exists():
            if not path.exists() or not meta.exists():
                raise ValueError("Incomplete feature cache entry")
            prior = json.loads(meta.read_text(encoding="utf-8"))
            if prior["key"] != key or file_hash(path) != prior["sha256"]:
                raise ValueError("Feature cache integrity failure")
            with np.load(path,allow_pickle=False) as saved:
                array = saved["x"].copy()
            if array.shape != (views,512) or not np.isfinite(array).all():
                raise ValueError("Invalid feature cache shape/values")
            output[record["image_id"]] = array
            index.append(dict(prior,image_id=record["image_id"],reused=True))
        else:
            todo.append((record,views,key,path,meta))
    audit = {"status":"success","parameters_frozen":True,"batchnorm_unchanged":True,
             "cache_hits":len(output),"computed":len(todo),"feature_dimension":512}
    if todo:
        print(f"[features] {len(todo)} images; {len(output)} cache hits",flush=True)
        load_start = time.perf_counter()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        backbone.requires_grad_(False)
        backbone.to("cuda").eval()
        snapshots = {k:v.clone() for k,v in backbone.state_dict().items()
                     if "running_" in k or "num_batches_tracked" in k}
        audit["load_seconds"] = time.perf_counter()-load_start
        cfg = config["classifier"]
        aug = cfg["augmentation"]
        augment = transforms.Compose([
            transforms.RandomResizedCrop(aug["size"],scale=tuple(aug["crop_scale"]),ratio=tuple(aug["crop_ratio"]),
                                         interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(aug["horizontal_flip"]),transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        base = weights.transforms()
        work_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            # Batch whole images, keeping all views of an image together.
            per_batch = max(1,cfg["feature_batch_size"]//cfg["train_views"])
            for first in range(0,len(todo),per_batch):
                group = todo[first:first+per_batch]
                tensors, counts = [], []
                for record,views,_,_,_ in group:
                    image = Image.open(inside(run_dir,record["relative_path"])).convert("RGB")
                    tensors.append(base(image))
                    for view in range(1,views):
                        seed = int(config_hash([cfg["view_seed"],record["image_id"],view])[:15],16)
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(seed)
                            tensors.append(augment(image))
                    counts.append(views)
                with torch.inference_mode():
                    features = backbone(torch.stack(tensors).to("cuda"))
                    if cfg["feature_normalization"] == "l2":
                        features = torch.nn.functional.normalize(features,dim=-1)
                    arrays = features.float().cpu().numpy()
                if arrays.shape != (len(tensors),512) or not np.isfinite(arrays).all():
                    raise ValueError("ResNet features are invalid")
                offset = 0
                for (record,views,key,path,meta),count in zip(group,counts):
                    array = arrays[offset:offset+count].copy()
                    offset += count
                    temporary = path.with_name(path.name+".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle,x=array)
                    temporary.replace(path)
                    entry = {"schema_version":1,"key":key,"sha256":file_hash(path),
                             "relative_cache_path":str(path.relative_to(ROOT)).replace("\\","/"),"shape":list(array.shape)}
                    atomic_json(meta,entry)
                    index.append(dict(entry,image_id=record["image_id"],reused=False))
                    output[record["image_id"]] = array
            torch.cuda.synchronize()
            audit["batchnorm_unchanged"] = all(torch.equal(backbone.state_dict()[k],v) for k,v in snapshots.items())
            audit["parameters_frozen"] = not any(p.requires_grad for p in backbone.parameters())
            audit["inference_seconds"] = time.perf_counter()-work_start
            audit["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if not audit["batchnorm_unchanged"] or not audit["parameters_frozen"]:
                raise ValueError("Backbone/BatchNorm freezing failed")
        finally:
            del backbone, snapshots
            release_cuda()
    audit["seconds"] = time.perf_counter()-start
    atomic_json(run_dir / "features/index.json", sorted(index,key=lambda x:x["image_id"]))
    atomic_json(run_dir / "features/audit.json", audit)
    return output,audit
