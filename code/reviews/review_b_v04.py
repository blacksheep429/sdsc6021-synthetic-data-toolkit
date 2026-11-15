"""Read-only B review of A's frozen 0.4 training bundle; never evaluates test data.

This review lives below code/reviews so the sealed execution-source list is unchanged.
It independently reconstructs splits, features and optimizer trajectories, without
calling the formal CLI, classifier.train_heads, or data_features.extract_features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
CONDITIONS = {'real_only':0,'random_50':50,'clip_50':50,'random_150':150,'clip_150':150}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def state_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode()); h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def check(condition,message):
    if not condition: raise ValueError(message)


def source_ce(head,features,labels):
    losses = torch.nn.functional.cross_entropy(head(features).reshape(-1,10),
                labels[:,None].expand(-1,features.shape[1]).reshape(-1),reduction='none')
    return losses.reshape(len(features),features.shape[1]).mean(dim=1).mean()


def schedule(labels,seed,hp):
    rng=np.random.default_rng(seed)
    return torch.as_tensor(np.concatenate([
        rng.choice(np.where(labels==c)[0],size=(hp['updates'],hp['images_per_class_per_batch']),replace=True)
        for c in range(10)],axis=1),dtype=torch.long)


def review(input_root,output_root):
    torch.set_num_threads(4)
    started=time.perf_counter()
    cfg=js(ROOT/'configs/study.json')
    config_digest=fingerprint(cfg)
    seal=js(input_root/'models/frozen.json')
    check(cfg['protocol_version']=='0.4-frozen' and cfg['protocol_status']=='frozen','Wrong frozen protocol')
    check(js(input_root/'protocol.json')['config_hash']==config_digest==seal['config_hash'],'Config mismatch')
    check(seal['official_test_labels_read'] is False and seal['final_test_evaluated'] is False,'Expected untested G3 models')
    for name,digest in seal['code_sha256'].items():
        check(sha(ROOT/'code'/name)==digest,f'Frozen source changed: {name}')
    print('[review] Frozen configuration and execution source match',flush=True)
    file_index=js(input_root/'handoff/file_index.json')
    before={}
    for name,expected in file_index.items():
        path=(input_root/name).resolve()
        check(path.is_relative_to(input_root.resolve()),'Unsafe bundle path')
        digest=sha(path)
        check(digest==expected['sha256'] and path.stat().st_size==expected['bytes'],f'Corrupt bundle file: {name}')
        before[name]=digest
    before['handoff/file_index.json']=sha(input_root/'handoff/file_index.json')
    report={'schema_version':1,'status':'running','reviewer':'B-side Codex-assisted independent implementation',
            'utc':datetime.now(timezone.utc).isoformat(),'run_id':'synthetic_v2','config_hash':config_digest,
            'baseline_commit':subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip(),
            'sealed_source_commit':seal['source_commit'],'review_source_sha256':sha(__file__),
            'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'cpu',
              'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pillow')}},
            'formal_environment_equivalence_claimed':False,'official_test_labels_read':False,
            'checks':{'indexed_files_verified':len(file_index),'frozen_execution_sources_match':True}}
    candidates=js(input_root/'manifests/candidates.json')
    check(len(candidates)==2000,'Candidate count')
    synthetic={r['image_id']:r for r in candidates}
    check(len(synthetic)==2000 and Counter(r['class_id'] for r in candidates)==Counter({c:200 for c in range(10)}),'Candidate mapping/counts')
    selections=js(input_root/'manifests/selections.json')['selections']
    check(set(selections)==set(CONDITIONS),'Condition matrix differs')
    for condition,k in CONDITIONS.items():
        ids=selections[condition]
        check(len(ids)==len(set(ids))==k*10,f'Selection count/duplicates: {condition}')
        check(Counter(synthetic[i]['class_id'] for i in ids)==Counter({c:k for c in range(10)}) if k else not ids,'Selection balance')
    for method in ('random','clip'):
        check(set(selections[method+'_50']) <= set(selections[method+'_150']),'Nested sets differ')
    check(sha(input_root/'manifests/selections.json')==seal['selection_manifest_sha256'],'Selection seal differs')
    with np.load(input_root/'features/training_features.npz',allow_pickle=False) as data:
        x=data['x'].copy(); ids=data['image_ids'].tolist(); labels=data['labels'].copy()
    index=js(input_root/'features/bundle_index.json')
    check(sha(input_root/'features/training_features.npz')==seal['feature_bundle_sha256'],'Feature seal differs')
    check(x.shape==(2598,4,512) and x.dtype==np.float32 and np.isfinite(x).all(),'Feature dimensions/type/values')
    check(len(ids)==len(set(ids))==len(index)==len(labels),'Feature ID counts')
    check(ids==[r['image_id'] for r in index] and np.array_equal(labels,[r['class_id'] for r in index]),'Feature label/ID order')
    check([r['row'] for r in index]==list(range(2598)) and all(r['views']==4 for r in index),'Feature row/view mapping')
    norm_error=float(np.max(np.abs(np.linalg.norm(x,axis=-1)-1)))
    check(norm_error<1e-5,'Feature L2 normalization differs')
    by_id={name:i for i,name in enumerate(ids)}
    real_by_id={}; splits={}
    for seed in (0,1,2):
        rows=js(input_root/f'splits/seed{seed}/manifests/real.json'); splits[seed]=rows
        check(len(rows)==200 and len({r['image_id'] for r in rows})==200,'Split count/duplicates')
        check(Counter(r['class_id'] for r in rows)==Counter({c:20 for c in range(10)}),'Class balance')
        check(all(r['split']=='train' and r['split_seed']==seed and r['class_name']==CLASSES[r['class_id']] for r in rows),'Label/source mapping')
        for row in rows:
            prior=real_by_id.setdefault(row['image_id'],row)
            check(prior['sha256']==row['sha256'] and prior['class_id']==row['class_id'],'Shared real ID mismatch')
    check(len(real_by_id)==598 and set(real_by_id)|set(synthetic)==set(ids),'Feature coverage')
    for row in index:
        source=(real_by_id if row['source']=='real' else synthetic)[row['image_id']]
        check(source['class_id']==row['class_id'] and source['sha256']==row['image_sha256'],'Feature provenance mismatch')
    # Only whitelisted official training files are read; no test_batch access.
    data_root=ROOT/'runs/_cache/cifar10/cifar-10-batches-py'
    arrays=[]; official_labels=[]
    for name,md5 in CIFAR10.train_list+[(CIFAR10.meta['filename'],CIFAR10.meta['md5'])]:
        path=data_root/name
        check(hashlib.md5(path.read_bytes()).hexdigest()==md5,f'Official train checksum: {name}')
        with path.open('rb') as f: data=pickle.load(f,encoding='latin1')
        if name==CIFAR10.meta['filename']: check(data['label_names']==CLASSES,'Official mapping')
        else: arrays.append(data['data']); official_labels.extend(data['labels'])
    official_images=np.concatenate(arrays).reshape(-1,3,32,32).transpose(0,2,3,1)
    official_labels=np.array(official_labels)
    for seed,rows in splits.items():
        rng=np.random.default_rng(seed)
        expected=np.concatenate([rng.permutation(np.flatnonzero(official_labels==c))[:20] for c in range(10)]).tolist()
        check(expected==[r['original_train_index'] for r in rows],'Real split RNG/index sequence differs')
    real_paths={}
    for image_id,row in real_by_id.items():
        path=input_root/f"splits/seed{row['split_seed']}"/row['relative_path']
        real_paths[image_id]=path
        check(sha(path)==row['sha256'],'Real PNG hash mismatch')
        with Image.open(path) as image: pixels=np.asarray(image.convert('RGB'))
        idx=row['original_train_index']
        check(row['class_id']==int(official_labels[idx]) and np.array_equal(pixels,official_images[idx]),'Official label/pixel mismatch')
    report['checks'].update(candidate_manifest_count=2000,real_unique=598,real_split_count=3,
        real_images_per_split=200,official_real_labels_and_pixels_match=True,feature_shape=list(x.shape),
        feature_rows_ids_classes_match=True,l2_max_norm_error=norm_error,selections_equal_and_nested=True)
    print('[review] All split indices, real pixels, feature IDs/classes and nested counts verified',flush=True)
    # Independently exercise the 50/50 objective's invariance to duplicated synthetic data.
    h=torch.nn.Linear(512,10); rx=torch.from_numpy(x[:20]); ry=torch.from_numpy(labels[:20]).long()
    sx=torch.from_numpy(x[-20:]); sy=torch.from_numpy(labels[-20:]).long()
    loss=0.5*source_ce(h,rx,ry)+0.5*source_ce(h,sx,sy)
    duplicate=0.5*source_ce(h,rx,ry)+0.5*source_ce(h,sx.repeat(3,1,1),sy.repeat(3))
    check(torch.allclose(loss,duplicate,atol=1e-7),'Source weighting changed with duplicated synthetic data')
    report['checks']['equal_source_weight_invariance']=True
    records=[]
    hp=cfg['classifier']['hyperparameters']
    check(hp=={'optimizer':'SGD','lr':0.1,'momentum':0.9,'weight_decay':0.0001,'updates':500,'images_per_class_per_batch':2},'Unexpected optimizer')
    check(len(seal['models'])==15,'Sealed matrix size')
    for seed in (0,1,2):
        rows=splits[seed]; tr_ids=[r['image_id'] for r in rows]
        rx=torch.from_numpy(x[[by_id[i] for i in tr_ids]]); ry=torch.tensor([r['class_id'] for r in rows])
        real_batches=schedule(ry.numpy(),seed+11000,hp)
        real_batch_hash=hashlib.sha256(real_batches.numpy().tobytes()).hexdigest()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); initial=torch.nn.Linear(512,10).state_dict()
        expected_init=state_sha(initial)
        saved_predictions=js(input_root/f'splits/seed{seed}/metrics/train_reload_predictions.json')
        for condition,k in CONDITIONS.items():
            model_spec=next(r for r in seal['models'] if r['seed']==seed and r['condition_id']==condition)
            path=input_root/model_spec['relative_path']
            done=js(path.with_suffix('.done.json'))
            check(sha(path)==model_spec['sha256']==done['model_sha256'],'Model file seal mismatch')
            ckpt=torch.load(path,map_location='cpu',weights_only=True)
            check(ckpt['classes']==CLASSES and ckpt['feature_dimension']==512,'Checkpoint label/dimension')
            check(ckpt['config_hash']==config_digest and ckpt['classifier_config']==cfg['classifier'],'Checkpoint configuration')
            check(ckpt['updates']==model_spec['updates']==500,'Update count mismatch')
            check(ckpt['initial_state_sha256']==model_spec['initial_state_sha256']==expected_init,'Paired initialization mismatch')
            check(ckpt['real_schedule_sha256']==model_spec['real_schedule_sha256']==real_batch_hash,'Paired real batches mismatch')
            selected=selections[condition]
            check(ckpt['real_train_ids']==tr_ids and ckpt['selected_synthetic_ids']==selected,'Training IDs differ')
            original=torch.nn.Linear(512,10); original.load_state_dict(ckpt['head_state']);original.eval()
            with torch.inference_mode(): logits=original(rx[:,0])
            pred_by_id={r['sample_id']:r for r in saved_predictions if r['condition_id']==condition}
            check(len(pred_by_id)==200,'Reload predictions incomplete')
            check(all(pred_by_id[i]['prediction_split']=='train_reload_check_only' and
                      pred_by_id[i]['predicted_label']==int(p) and pred_by_id[i]['true_label']==int(y)
                      for i,p,y in zip(tr_ids,logits.argmax(1),ry)),'Recorded reload predictions differ')
            model=torch.nn.Linear(512,10);model.load_state_dict(initial)
            opt=torch.optim.SGD(model.parameters(),lr=hp['lr'],momentum=hp['momentum'],weight_decay=hp['weight_decay'])
            if selected:
                srows=[synthetic[i] for i in selected]
                sx=torch.from_numpy(x[[by_id[i] for i in selected]])
                sy=torch.tensor([r['class_id'] for r in srows]); sbatches=schedule(sy.numpy(),seed+22000,hp)
            start=time.perf_counter()
            for step,indices in enumerate(real_batches):
                opt.zero_grad(set_to_none=True)
                loss=source_ce(model,rx[indices],ry[indices])
                if selected:
                    si=sbatches[step];loss=0.5*loss+0.5*source_ce(model,sx[si],sy[si])
                loss.backward();opt.step()
            delta=max(float((model.state_dict()[n]-ckpt['head_state'][n]).abs().max()) for n in ckpt['head_state'])
            with torch.inference_mode(): replay_logits=model(rx[:,0])
            logits_delta=float((replay_logits-logits).abs().max())
            check(delta<=1e-6 and logits_delta<=1e-5,'Independent optimizer replay mismatch')
            check(torch.equal(logits.argmax(1),replay_logits.argmax(1)),'Replayed predictions changed')
            record={'seed':seed,'condition_id':condition,'updates_replayed':500,'model_sha256':sha(path),
                    'initialization_matches':True,'real_batch_schedule_matches':True,'reload_predictions_match':True,
                    'weight_max_abs_error':delta,'replay_logit_max_abs_error':logits_delta,
                    'state_bit_identical':state_sha(model.state_dict())==state_sha(ckpt['head_state']),
                    'original_reload_logit_hash_matches':hashlib.sha256(logits.numpy().tobytes()).hexdigest()==done['reload_logits_sha256'],
                    'review_seconds':time.perf_counter()-start}
            records.append(record);print('[review] head',seed,condition,'max weight error',delta,flush=True)
    report['model_reviews']=records
    # Recompute all available REAL image views on this CPU, retaining official files untouched.
    weights_path=ROOT/'runs/_cache/torch/checkpoints/resnet18-f37072fd.pth'
    check(sha(weights_path)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec','ResNet weight checksum')
    backbone=models.resnet18(weights=None)
    backbone.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
    backbone.fc=torch.nn.Identity();backbone.requires_grad_(False);backbone.eval()
    before_state=state_sha(backbone.state_dict())
    aug=cfg['classifier']['augmentation']
    base=models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    augmentation=transforms.Compose([transforms.RandomResizedCrop(aug['size'],scale=tuple(aug['crop_scale']),ratio=tuple(aug['crop_ratio']),interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(aug['horizontal_flip']),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    names=sorted(real_by_id);feature_error=0.0;feature_records=[];cpu_blocks=[];start=time.perf_counter()
    for offset in range(0,len(names),8):
        batch=names[offset:offset+8];views=[]
        for image_id in batch:
            with Image.open(real_paths[image_id]) as f:image=f.convert('RGB')
            views.append(base(image))
            for v in (1,2,3):
                view_seed=int(fingerprint([cfg['classifier']['view_seed'],image_id,v])[:15],16)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(view_seed);views.append(augmentation(image))
        with torch.inference_mode():
            calculated=torch.nn.functional.normalize(backbone(torch.stack(views)),dim=-1).reshape(len(batch),4,512).numpy()
        cpu_blocks.append(calculated)
        per_view=np.max(np.abs(calculated-x[[by_id[i] for i in batch]]),axis=2)
        errors=per_view.max(axis=1)
        for image_id,error,view_errors in zip(batch,errors,per_view):
            feature_records.append({'image_id':image_id,'max_abs_error':float(error),'per_view_max_abs_error':view_errors.tolist()})
        feature_error=max(feature_error,float(errors.max()))
        print(f'[review] real feature replay {offset+len(batch)}/{len(names)}',flush=True)
    write(output_root/'real_feature_comparisons.json',feature_records)
    check(state_sha(backbone.state_dict())==before_state and not any(p.requires_grad for p in backbone.parameters()) and not any(m.training for m in backbone.modules()),'Backbone weights or BN changed')
    cpu_array=np.concatenate(cpu_blocks)
    np.savez_compressed(output_root/'cpu_real_features.npz',x=cpu_array,image_ids=np.asarray(names))
    a_array=x[[by_id[i] for i in names]]
    minimum_cosine=float(np.min(np.sum(cpu_array*a_array,axis=-1)))
    cpu_by_id={image_id:i for i,image_id in enumerate(names)}
    compatibility=[]
    for spec in seal['models']:
        checkpoint=torch.load(input_root/spec['relative_path'],map_location='cpu',weights_only=True)
        head=torch.nn.Linear(512,10);head.load_state_dict(checkpoint['head_state']);head.eval()
        real_ids=[r['image_id'] for r in splits[spec['seed']]]
        with torch.inference_mode():
            original_logits=head(torch.from_numpy(x[[by_id[i] for i in real_ids],0]))
            cpu_logits=head(torch.from_numpy(cpu_array[[cpu_by_id[i] for i in real_ids],0]))
        compatibility.append({'seed':spec['seed'],'condition_id':spec['condition_id'],
                              'reference_split':'train_reload_check_only','images':200,
                              'prediction_changes':int((original_logits.argmax(1)!=cpu_logits.argmax(1)).sum()),
                              'logit_max_abs_difference':float((original_logits-cpu_logits).abs().max())})
    report['feature_prediction_compatibility']=compatibility
    report['feature_review']={'real_images_recomputed':598,'views_recomputed':2392,'max_abs_error':feature_error,
        'predeclared_abs_tolerance':1e-5,'all_backbone_state_and_bn_unchanged':True,'seconds':time.perf_counter()-start,
        'strict_numeric_match_passed':feature_error<=1e-5,'minimum_cosine_similarity':minimum_cosine,
        'synthetic_features_recomputed':0,'synthetic_feature_provenance_and_id_class_mapping_checked':True}
    write(output_root/'real_feature_comparisons.json',feature_records)
    check(all(sha(input_root/name)==digest for name,digest in before.items()),'Original bundle files were modified')
    report.update(status='passed' if feature_error<=1e-5 else 'completed_with_feature_numerical_finding',
                  input_unchanged=True,official_test_labels_read=False,
                  head_weight_abs_tolerance=1e-6,head_logit_abs_tolerance=1e-5,
                  total_review_seconds=time.perf_counter()-started,
                  limitations=['CPU review differs from A frozen CUDA environment.',
                    'Synthetic PNG archives were not downloaded or re-extracted; source hashes, IDs, labels and feature rows were cross-checked.',
                    'No official test accuracy, CLIP zero-shot evaluation, human annotation, or non-author trial was performed.'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'runs/b_review_v04/input')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/b_review_v04/results')
    args=parser.parse_args();output=args.output.resolve()
    check(not output.is_relative_to(args.input.resolve()),'Review outputs cannot be written inside sealed input')
    try:
        report=review(args.input.resolve(),output)
    except Exception as exc:
        write(output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),'official_test_labels_read':False})
        raise
    write(output/'review.json',report)
    print(json.dumps({'status':report['status'],'models_replayed':len(report['model_reviews']),
                      'real_features_recomputed':report['feature_review']['real_images_recomputed']},indent=2))


if __name__=='__main__': main()
