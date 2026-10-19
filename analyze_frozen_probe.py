#!/usr/bin/env python3
"""Independent locked-test analysis for the frozen 2x2 readout experiment.

Requires NumPy >=1.22, not torch/ROS. Reads no test NPZ before verifying the
global 24-model lock and precommitted analysis specification. --self-test uses
small mathematical examples and the already published old-bank scores only.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/loss_ordering_v1"
CELLS=("independent_balanced","ordered_balanced","independent_proper","ordered_proper")
HORIZONS=(5,10,20)
POPULATIONS=("common_m20","per_horizon_sensitivity")
CONTRASTS={"ordering_balanced":(-1,1,0,0),"ordering_proper":(0,0,-1,1),
           "loss_independent":(-1,0,1,0),"loss_ordered":(0,-1,0,1),"interaction":(1,-1,-1,1)}
CLIP=1e-12


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def inside(root,relative):
    path=(root/relative).resolve()
    require(root.resolve() in path.parents,"Path escapes locked artifact directory")
    return path


def verify_lock(args,spec):
    """This runs before evaluation metadata or score files are opened."""
    path=args.probe_root/"model_lock.json"
    require(path.is_file(),"Test access forbidden: global model lock is missing")
    lock=read_json(path)
    require(lock.get("status")=="locked","Test access forbidden: model lock is not locked")
    records=lock.get("models",[])
    expected=set(itertools.product(spec["encoder_seeds"],CELLS))
    require(len(records)==24 and {(r["encoder_seed"],r["cell"]) for r in records}==expected,"Global lock must contain all 24 unique heads")
    identities={"analysis_spec_sha256":sha256(args.analysis_spec),"spec_sha256":sha256(args.probe_spec),
                "registry_sha256":sha256(args.registry),"protocol_sha256":sha256(args.protocol),
                "script_sha256":sha256(args.producer_script)}
    for k,v in identities.items():
        require(lock.get(k)==v,"Locked identity mismatch before test access: "+k)
    for r in records:
        require(sha256(inside(args.probe_root,r["path"]))==r["sha256"],"Locked selected head changed")
    verify_source_roles(args,spec,lock)
    return lock,identities,sha256(path)


def verify_source_roles(args,spec,lock):
    """Verify shared encoder/actor training identities before any test access."""
    protocol=read_json(args.protocol)
    require(protocol.get("same_checkpoint_for_encoder_and_behavior") is True,"Joint-source bootstrap requires shared checkpoint roles")
    require(spec["encoder_seeds"]==spec["behavior_seeds"],"Encoder/behavior source identities differ")
    expected={str(s) for s in spec["encoder_seeds"]}
    mapping=protocol["checkpoint_role_mapping"]
    require(set(mapping)==expected and set(lock["encoders"])==expected,"Need six shared source identities")
    path=args.checkpoints or args.protocol.parent/"checkpoints.json"
    require(sha256(path)==protocol["checkpoint_list_sha256"],"Checkpoint list differs from protocol")
    provenance_path=args.protocol.parent/"preflight/source_provenance.json"
    require(sha256(provenance_path)==protocol["checkpoint_provenance_sha256"],"Checkpoint collection provenance changed")
    provenance=read_json(provenance_path)["checkpoints"]
    require(set(provenance)==expected,"Collection provenance must contain the same six checkpoint identities")
    entries=read_json(path)
    listed={str(int(e["seed"])):e for e in entries}
    require(len(entries)==len(listed)==6 and set(listed)==expected,"Checkpoint list must contain six unique seeds")
    require(len({mapping[s]["sha256"] for s in expected})==6,"Training source checkpoints are not distinct")
    for s in expected:
        fingerprint=mapping[s]["sha256"]
        require(len(fingerprint)==64 and all(c in "0123456789abcdef" for c in fingerprint),"Invalid source checkpoint fingerprint")
        require(listed[s]["sha256"]==fingerprint==lock["encoders"][s]["checkpoint_file_sha256"]==provenance[s]["sha256"],"Encoder/source-role checkpoint mismatch")
        require(lock["encoders"][s]["checkpoint_list_sha256"]==protocol["checkpoint_list_sha256"],"Encoder used a different checkpoint list")


def verify_behavior_sources(args,spec,lock_sha,episodes):
    """Bind every analyzed episode to the actor source in the locked protocol.

    The collector verified whole checkpoint content before rollout. Here we
    independently traverse aggregate -> worker manifest -> raw episode hashes,
    without importing torch or loading the multi-GB checkpoint files again.
    """
    protocol=read_json(args.protocol);mapping=protocol["checkpoint_role_mapping"]
    bank=(args.bank_root or args.probe_root.parent/"bank")/"test"
    aggregate_path=bank/"manifest.json";aggregate=read_json(aggregate_path)
    require(aggregate.get("complete") is True and aggregate.get("split")=="test","Incomplete test bank aggregate")
    require(aggregate.get("protocol_sha256")==sha256(args.protocol) and
            aggregate.get("assignment_registry_sha256")==sha256(args.registry),"Behavior bank protocol/registry mismatch")
    require(aggregate.get("model_lock_sha256")==lock_sha,"Test collection is not bound to the global model lock")
    bank_lock=read_json(args.probe_root/"test_bank_lock.json")
    require(bank_lock["model_lock_sha256"]==lock_sha and
            bank_lock["split_bank_manifest_sha256"]==sha256(aggregate_path),"Locked test bank changed")
    evaluated={e["episode_id"]:e for e in episodes}
    locked={e["episode_id"]:e["source_sha256"] for e in bank_lock["episodes"]}
    entries=aggregate["episodes"]
    require(len(entries)==len(evaluated)==len(locked)==spec["test_episodes"],"Behavior bank episode count mismatch")
    require({e["episode_id"] for e in entries}==set(evaluated)==set(locked),"Behavior bank episode coverage mismatch")
    sources={r["file"]:r for r in aggregate["source_manifests"]};workers={}
    require(len(sources)==len(aggregate["source_manifests"]),"Duplicate worker source manifests")
    for relative,source in sources.items():
        path=inside(bank,relative)
        require(sha256(path)==source["sha256"],"Behavior source manifest changed")
        worker=read_json(path);seed=str(int(worker["behavior_seed"]))
        require(seed in mapping and int(seed)==int(source["seed"]),"Worker behavior seed differs")
        require(worker["checkpoint_sha256"]==mapping[seed]["sha256"],"Behavior actor is not the paired encoder checkpoint")
        require(worker["protocol_sha256"]==sha256(args.protocol) and worker["assignment_registry_sha256"]==sha256(args.registry) and
                worker["split"]=="test","Worker protocol/registry/split mismatch")
        if worker["episodes"]:
            require(worker["behavior_protocol"]["seed"]==int(seed),"Worker behavior protocol seed mismatch")
        listed={e["rollout_id"]:e for e in worker["episodes"]}
        require(len(listed)==len(worker["episodes"]),"Duplicate worker episode identity")
        workers[relative]=(worker,listed,path.parent)
    worker_ids=[episode_id for _,listed,_ in workers.values() for episode_id in listed]
    require(len(worker_ids)==len(set(worker_ids))==len(evaluated) and set(worker_ids)==set(evaluated),"Worker episodes are duplicated, missing or omitted from aggregate")
    for item in entries:
        e=evaluated[item["episode_id"]];seed=str(int(e["behavior_seed"]))
        worker,listed,folder=workers[item["source_manifest"]]
        require(item["source_manifest_sha256"]==sources[item["source_manifest"]]["sha256"],"Episode source-manifest binding differs")
        require(int(worker["behavior_seed"])==int(item["behavior_seed"])==int(seed),"Episode actor source identity differs")
        original=listed[item["episode_id"]]
        require(item["sha256"]==original["sha256"]==e["source_sha256"]==locked[item["episode_id"]],"Episode source hashes differ")
        path=inside(bank,item["file"])
        require(path==inside(folder,original["file"]) and sha256(path)==item["sha256"],"Raw episode bytes/path differ from collector provenance")
        for key in ("episode_id","task_id","map_id","behavior_seed","steps","success","collision"):
            require(item[key]==original[key]==e[key],"Episode provenance metadata differs: "+key)
        if "behavior_checkpoint_sha256" in e:
            require(e["behavior_checkpoint_sha256"]==mapping[seed]["sha256"],"Score episode actor fingerprint differs")
    return {"shared_source_seed_verified":True,"checkpoint_list_sha256":protocol["checkpoint_list_sha256"],
            "source_role_sha256":{s:mapping[s]["sha256"] for s in sorted(mapping)},
            "aggregate_sha256":sha256(aggregate_path),"worker_manifests_verified":len(workers),"episode_hashes_verified":len(entries)}


def expected_registry(path,spec):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    task_splits=defaultdict(set)
    for row in rows:
        task_splits[row["task_id"]].add(row["split"])
    require(all(len(v)==1 for v in task_splits.values()),"Task IDs overlap train/validation/test")
    test=[r for r in rows if r["split"]=="test"]
    expected={r["rollout_id"]:r for r in test}
    require(len(test)==len(expected)==spec["test_episodes"],"Test registry must contain exactly 360 unique rollouts")
    tasks=defaultdict(set)
    pairs=set()
    for row in test:
        m,b=int(row["map_id"]),int(row["behavior_seed"])
        require(m in spec["test_maps"] and b in spec["behavior_seeds"],"Unknown map/behavior seed")
        tasks[m].add(row["task_id"]);pairs.add((row["task_id"],b))
    require(all(len(tasks[m])==spec["distinct_tasks_per_map"] for m in spec["test_maps"]),"Need exactly 20 distinct task IDs per map")
    require(len(pairs)==len(test),"Duplicate task/behavior pairing")
    require(pairs==set(itertools.product(set.union(*tasks.values()),spec["behavior_seeds"])),"Incomplete crossed behavior/task design")
    return expected


def load_outputs(args,spec,identities,lock_sha,np):
    expected=expected_registry(args.registry,spec)
    outputs={};metadata=[];canonical=None;episodes=None;reference_metrics={}
    for encoder in spec["encoder_seeds"]:
        folder=args.probe_root/"evaluation"/("encoder_%02d"%encoder)
        complete=read_json(folder/"complete.json")
        require(complete.get("complete") is True and complete["encoder_seed"]==encoder,"Incomplete encoder evaluation")
        require(complete["model_lock_sha256"]==lock_sha,"Evaluation is not bound to the verified global lock")
        require(all(complete.get(k)==v for k,v in identities.items()),"Evaluation source/spec identity differs from lock")
        ee=complete["episodes"]
        require(len(ee)==len(expected) and {e["episode_id"] for e in ee}==set(expected),"Evaluation episode coverage differs from test registry")
        for e in ee:
            r=expected[e["episode_id"]]
            require((e["task_id"],int(e["map_id"]),int(e["behavior_seed"]))==(r["task_id"],int(r["map_id"]),int(r["behavior_seed"])),"Episode registry identity mismatch")
            require(0<int(e["steps"])<=900 and not(e["success"] and e["collision"]),"Invalid terminal episode metadata")
            require(e["success"] or e["collision"] or int(e["steps"])==900,"Nonterminal episode did not reach prespecified cap")
            require(len(e["source_sha256"])==64,"Missing episode source fingerprint")
        if episodes is None:
            episodes=ee
        require(ee==episodes,"Episode metadata/order differs across encoder representations")
        require(len(complete["scores"])==4 and {r["cell"] for r in complete["scores"]}==set(CELLS),"Missing/duplicate cell score files")
        metrics_path=folder/"metrics.json"
        require(sha256(metrics_path)==complete["metrics_sha256"],"Producer metric file changed")
        reference_metrics[encoder]=read_json(metrics_path)
        for item in complete["scores"]:
            path=inside(folder,item["file"])
            require(sha256(path)==item["sha256"],"Score file fingerprint mismatch")
            with np.load(path,allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in ("probabilities","logits","targets","masks","episode_index","timestep")}
            p,z=arrays["probabilities"],arrays["logits"]
            n=len(p)
            require(p.dtype==np.float32 and z.dtype==np.float32 and p.shape==z.shape==(n,3),"Invalid prediction shapes/dtypes")
            require(np.isfinite(p).all() and np.isfinite(z).all() and ((p>=0)&(p<=1)).all(),"Invalid probability/logit")
            expected_p=np.exp(-np.logaddexp(0,-z.astype(np.float64)))
            require(np.allclose(p,expected_p,atol=2e-7,rtol=1e-6),"Saved logits and probabilities disagree")
            idx={k:arrays[k] for k in ("targets","masks","episode_index","timestep")}
            require(idx["targets"].shape==idx["masks"].shape==(n,3) and idx["episode_index"].shape==idx["timestep"].shape==(n,),"Prediction and sample-index lengths differ")
            if canonical is None:
                canonical=idx
                verify_indices(episodes,canonical,np)
            require(all(np.array_equal(idx[k],canonical[k]) for k in idx),"Samples/labels/masks differ across encoder/cell")
            outputs[(encoder,item["cell"])]=(p,z)
        metadata.append({"encoder_seed":encoder,"complete_sha256":sha256(folder/"complete.json"),"test_index_sha256":complete["test_index_sha256"]})
    return outputs,episodes,canonical,metadata,reference_metrics


def verify_indices(episodes,index,np):
    n=sum(int(e["steps"]) for e in episodes)
    target,mask=index["targets"],index["masks"]
    require(target.shape==mask.shape==(n,3) and np.isin(target,[0,1]).all() and np.isin(mask,[0,1]).all(),"Invalid labels/masks")
    require(index["episode_index"].shape==index["timestep"].shape==(n,),"Invalid step index shape")
    offsets=0
    for i,e in enumerate(episodes):
        length=int(e["steps"]);where=slice(offsets,offsets+length)
        require(np.array_equal(index["episode_index"][where],np.full(length,i)) and
                np.array_equal(index["timestep"][where],np.arange(length)),"Episode-local steps are missing, duplicated, or reordered")
        remaining=length-np.arange(length)
        for j,h in enumerate(HORIZONS):
            yy=(remaining<=h)&bool(e["collision"])
            mm=yy|(remaining>=h)|bool(e["success"])
            require(np.array_equal(target[where,j],yy) and np.array_equal(mask[where,j],mm),"Targets/masks do not match terminal collision/success/censoring metadata")
        offsets+=length


def pava3(p,np):
    p=np.asarray(p,dtype=np.float64);out=p.copy()
    first=p[:,0]>p[:,1]
    m12=(p[:,0]+p[:,1])/2
    allmean=p.mean(axis=1)
    pool12=first&(m12<=p[:,2]);allpool=first&~pool12
    out[pool12,0]=m12[pool12];out[pool12,1]=m12[pool12]
    second=(~first)&(p[:,1]>p[:,2]);m23=(p[:,1]+p[:,2])/2
    pool23=second&(p[:,0]<=m23);allpool|=second&~pool23
    out[pool23,1]=m23[pool23];out[pool23,2]=m23[pool23]
    out[allpool]=allmean[allpool,None]
    require(not np.any(out[:,:-1]>out[:,1:]),"PAVA implementation returned a violation")
    return out


def draws(spec,episodes,np):
    rng=np.random.RandomState(spec["bootstrap"]["seed"])
    e_seeds=spec["encoder_seeds"];b_seeds=spec["behavior_seeds"];maps=spec["test_maps"]
    require(e_seeds==b_seeds,"Joint source identities must match across encoder/behavior roles")
    tasks={m:sorted({e["task_id"] for e in episodes if int(e["map_id"])==m}) for m in maps}
    bindex={b:i for i,b in enumerate(b_seeds)}
    task_index={t:(m,i) for m in maps for i,t in enumerate(tasks[m])}
    count=spec["bootstrap"]["draws"]
    ec=np.zeros((count,len(e_seeds)),dtype=np.int16);weight=np.zeros((count,len(episodes)),dtype=np.int16)
    for d in range(count):
        ec[d]=np.bincount(rng.randint(len(e_seeds),size=len(e_seeds)),minlength=len(e_seeds))
        bc=ec[d]  # One draw: actor and encoder are roles of the SAME checkpoint.
        tc={m:np.bincount(rng.randint(len(tasks[m]),size=len(tasks[m])),minlength=len(tasks[m])) for m in maps}
        for i,e in enumerate(episodes):
            m,t=task_index[e["task_id"]]
            weight[d,i]=bc[bindex[int(e["behavior_seed"])]]*tc[m][t]
    require(np.all(ec.sum(axis=1)==len(e_seeds)),"Encoder bootstrap multiplicity invalid")
    require(np.all(weight.sum(axis=1)==len(episodes)),"Crossed task/behavior multiplicity invalid")
    tasks_per_behavior=sum(len(v) for v in tasks.values())
    for i,b in enumerate(b_seeds):
        where=[j for j,e in enumerate(episodes) if int(e["behavior_seed"])==b]
        require(np.array_equal(weight[:,where].sum(axis=1),ec[:,i]*tasks_per_behavior),"Encoder/behavior source counts were not jointly sampled")
    return ec.astype(np.float64),weight.astype(np.float64)


def weighted_ranking(p,y,episode,weights,np,chunk_size=8):
    """Exact weighted AUROC/AP with score ties, reusing a fixed sort per score set."""
    order=np.argsort(-p,kind="mergesort");sp=p[order];sy=y[order].astype(np.float64);se=episode[order]
    starts=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1]
    auc=np.full(len(weights),np.nan);ap=auc.copy()
    if not len(p):
        return auc,ap
    for begin in range(0,len(weights),chunk_size):
        w=weights[begin:begin+chunk_size,se]
        pos=np.add.reduceat(w*sy[None,:],starts,axis=1)
        total=np.add.reduceat(w,starts,axis=1)
        cp=np.cumsum(pos,axis=1);ct=np.cumsum(total,axis=1)
        positives=cp[:,-1];negatives=ct[:,-1]-positives
        precision=np.divide(cp,ct,out=np.zeros_like(cp),where=ct>0)
        ok=(positives>0)&(negatives>0)
        with np.errstate(divide="ignore",invalid="ignore"):
            local_ap=(pos*precision).sum(axis=1)/positives
            local_auc=(pos*(negatives[:,None]-(ct-cp)+.5*(total-pos))).sum(axis=1)/(positives*negatives)
        auc[begin:begin+len(w)]=np.where(ok,local_auc,np.nan)
        ap[begin:begin+len(w)]=np.where(ok,local_ap,np.nan)
    return auc,ap


def weighted_means(values,episode,weights,np):
    sums=np.bincount(episode,weights=values,minlength=weights.shape[1])
    counts=np.bincount(episode,minlength=weights.shape[1])
    denominator=weights@counts
    return np.divide(weights@sums,denominator,out=np.full(len(weights),np.nan),where=denominator>0)


def metric_arrays(p,z,target,mask,episode,weights,np,chunk_size):
    results={};violation=np.any(p[:,:-1]>p[:,1:],axis=1).astype(float)
    vv=weighted_means(violation,episode,weights,np)
    cache={}
    for population in POPULATIONS:
        for j,h in enumerate(HORIZONS):
            valid=mask[:,2 if population=="common_m20" else j].astype(bool)
            cachekey=(j,valid.tobytes())
            if cachekey not in cache:
                pp=p[valid,j].astype(np.float64);yy=target[valid,j].astype(np.float64);ee=episode[valid]
                auc,ap=weighted_ranking(pp,yy,ee,weights,np,chunk_size)
                clipped=np.clip(pp,CLIP,1-CLIP)
                logs=-np.where(yy>0,np.log(clipped),np.log1p(-clipped))
                mm={"AUROC":auc,"AP":ap,"Brier":weighted_means((pp-yy)**2,ee,weights,np),
                    "LogLoss_clipped":weighted_means(logs,ee,weights,np),"Violation":vv}
                if z is not None:
                    zz=z[valid,j].astype(np.float64)
                    mm["LogLoss"]=weighted_means(np.logaddexp(0,zz)-yy*zz,ee,weights,np)
                cache[cachekey]=mm
            for metric,values in cache[cachekey].items():
                results[(population,h,metric)]=values
    return results


def summarize(point,distribution,np):
    valid=distribution[np.isfinite(distribution)]
    lo,hi=np.quantile(valid,[.025,.975],method="linear") if len(valid) else (math.nan,math.nan)
    return {"estimate":float(point) if math.isfinite(point) else None,
            "ci_low":float(lo) if math.isfinite(lo) else None,"ci_high":float(hi) if math.isfinite(hi) else None,
            "bootstrap_valid":len(valid),"bootstrap_invalid":len(distribution)-len(valid)}


def event_audit(episodes,index,spec,np):
    rows=[];totalpositive=0
    for i,e in enumerate(episodes):
        where=index["episode_index"]==i;valid=where&index["masks"][:,2].astype(bool)
        pos=int(index["targets"][valid,2].sum());totalpositive+=pos
        rows.append({"episode_id":e["episode_id"],"task_id":e["task_id"],"map_id":int(e["map_id"]),
                     "behavior_seed":int(e["behavior_seed"]),"steps":int(e["steps"]),"collision":int(e["collision"]),
                     "success":int(e["success"]),"h20_valid_steps":int(valid.sum()),"h20_positive_steps":pos})
    by_behavior=[];adequate=True
    for b in spec["behavior_seeds"]:
        rr=[r for r in rows if r["behavior_seed"]==b]
        n=sum(r["h20_valid_steps"] for r in rr);p=sum(r["h20_positive_steps"] for r in rr)
        ct=len({r["task_id"] for r in rr if r["collision"]})
        ok=0<p<n and ct>1;adequate&=ok
        by_behavior.append({"behavior_seed":b,"collision_tasks":ct,"positive_steps":p,"negative_steps":n-p,
                            "class_and_more_than_one_collision_task_gate":bool(ok),
                            "max_episode_share_of_positive_steps":max((r["h20_positive_steps"]/p for r in rr),default=0) if p else None,
                            "collision_tasks_by_map":{str(m):sum(r["collision"] for r in rr if r["map_id"]==m) for m in spec["test_maps"]}})
    for r in rows:
        r["share_of_all_positive_h20_steps"]=r["h20_positive_steps"]/totalpositive if totalpositive else None
    return rows,{"event_support_sufficient":bool(adequate),"by_behavior":by_behavior,
                 "distinct_collision_task_geometries":len({r["task_id"] for r in rows if r["collision"]}),
                 "positive_steps_are_not_independent_collision_events":True,
                 "concentration_cutoff":"none added after inspection; distributions are reported"}


def sample_counts(index,np):
    rows=[]
    for population in POPULATIONS:
        for j,h in enumerate(HORIZONS):
            valid=index["masks"][:,2 if population=="common_m20" else j].astype(bool)
            n=int(valid.sum());positive=int(index["targets"][valid,j].sum())
            rows.append({"population":population,"horizon":h,"n":n,"positive":positive,"negative":n-positive,
                         "episodes_with_valid_steps":len(np.unique(index["episode_index"][valid])),
                         "all_vectors_n":len(valid)})
    return rows


def analyze(args,np):
    spec=read_json(args.analysis_spec)
    require(tuple(spec["cells"])==CELLS and tuple(spec["horizons"])==HORIZONS,"Unexpected prespecified design")
    require(spec["bootstrap"]["draws"]==2000 and spec["bootstrap"]["seed"]==2026101002,"Prespecified bootstrap changed")
    lock,identities,lock_sha=verify_lock(args,spec)
    require(not args.out.exists(),"Analysis output already exists; refuse silent overwrite")
    outputs,episodes,index,metadata,references=load_outputs(args,spec,identities,lock_sha,np)
    source_roles=verify_behavior_sources(args,spec,lock_sha,episodes)
    event_rows,event_result=event_audit(episodes,index,spec,np)
    counts=sample_counts(index,np)
    for encoder,rows in references.items():
        for cell in CELLS:
            for expected in counts:
                actual=next(r for r in rows if r["cell"]==cell and r["population"]==expected["population"] and r["horizon"]==expected["horizon"])
                require(all(actual[k]==expected[k] for k in ("n","positive","negative")),"Producer sample/class counts disagree")
    encoder_counts,episode_weights=draws(spec,episodes,np)
    weights=np.vstack([np.ones(len(episodes)),episode_weights])
    all_values={};per_encoder=[];max_reference_difference=0.0
    encoder_seeds=spec["encoder_seeds"]
    for eidx,encoder in enumerate(encoder_seeds):
        for cell in CELLS:
            raw,logits=outputs[(encoder,cell)]
            transforms=("raw","cummax","pava") if cell.startswith("independent_") else ("raw",)
            if cell.startswith("ordered_"):
                require(not np.any(raw[:,:-1]>raw[:,1:]),"Ordered head unexpectedly violates ordering")
            for transform in transforms:
                p=raw if transform=="raw" else (np.maximum.accumulate(raw,axis=1) if transform=="cummax" else pava3(raw,np))
                mm=metric_arrays(p,logits if transform=="raw" else None,index["targets"],index["masks"],index["episode_index"],weights,np,args.chunk_size)
                for (population,h,metric),values in mm.items():
                    key=(cell,transform,population,h,metric)
                    if key not in all_values:
                        all_values[key]=np.full((len(encoder_seeds),len(weights)),np.nan)
                    all_values[key][eidx]=values
                    per_encoder.append({"encoder_seed":encoder,"cell":cell,"transform":transform,"population":population,
                                        "horizon":h,"metric":metric,"estimate":float(values[0]) if np.isfinite(values[0]) else None})
                    if transform=="raw" and metric!="LogLoss_clipped":
                        if metric=="Violation":
                            ref=next(r for r in references[encoder] if r["cell"]==cell and r["population"]=="all_vectors_structure_only")["strict_violation_rate"]
                        else:
                            refrow=next(r for r in references[encoder] if r["cell"]==cell and r["population"]==population and r["horizon"]==h)
                            ref=refrow["log_loss" if metric=="LogLoss" else metric]
                        if ref is None:
                            require(not np.isfinite(values[0]),"Producer undefined metric disagrees")
                        else:
                            diff=abs(float(values[0])-float(ref));max_reference_difference=max(max_reference_difference,diff)
                            require(diff<1e-9,"Independent producer-metric discrepancy: "+str(key))
                print("analyzed encoder=%d cell=%s transform=%s"%(encoder,cell,transform),flush=True)
    # Each metric is nonlinear over the resampled episode pool; only then are
    # encoders averaged with their paired bootstrap multiplicities.
    estimates={};distributions={};absolute=[]
    for key,array in sorted(all_values.items()):
        point=float(np.mean(array[:,0]))
        draws_array=np.sum(np.where(encoder_counts>0,array[:,1:].T*encoder_counts,0),axis=1)/len(encoder_seeds)
        estimates[key]=point;distributions[key]=draws_array
        absolute.append(dict(zip(("cell","transform","population","horizon","metric"),key),**summarize(point,draws_array,np)))
    contrasts=[];contrast_distributions={}
    for population,h,metric in itertools.product(POPULATIONS,HORIZONS,("AP","AUROC","Brier","LogLoss","LogLoss_clipped","Violation")):
        keys=[(cell,"raw",population,h,metric) for cell in CELLS]
        for name,coefficients in CONTRASTS.items():
            point=sum(c*estimates[k] for c,k in zip(coefficients,keys))
            dd=sum(c*distributions[k] for c,k in zip(coefficients,keys))
            contrasts.append({"contrast":name,"population":population,"horizon":h,"metric":metric,**summarize(point,dd,np)})
            contrast_distributions[(name,population,h,metric)]=dd
    posthoc=[]
    for cell,transform,population,h,metric in itertools.product(CELLS[:1]+CELLS[2:3],("cummax","pava"),POPULATIONS,HORIZONS,("AP","AUROC","Brier","LogLoss_clipped","Violation")):
        key=(cell,transform,population,h,metric);base=(cell,"raw",population,h,metric)
        point=estimates[key]-estimates[base];dd=distributions[key]-distributions[base]
        posthoc.append({"cell":cell,"transform":transform,"population":population,"horizon":h,"metric":metric,**summarize(point,dd,np)})
    primary=next(r for r in contrasts if (r["contrast"],r["population"],r["horizon"],r["metric"])==("interaction","common_m20",20,"AP"))
    supports=event_result["event_support_sufficient"] and primary["ci_low"] is not None and primary["ci_low"]>0
    status="insufficient_event_support" if not event_result["event_support_sufficient"] else ("directional_interaction_supported" if supports else "interaction_not_established")
    args.out.mkdir(parents=True,exist_ok=False)
    write_csv(args.out/"absolute_metrics.csv",absolute);write_csv(args.out/"raw_contrasts.csv",contrasts)
    write_csv(args.out/"posthoc_deltas.csv",posthoc);write_csv(args.out/"per_encoder_metrics.csv",per_encoder)
    write_csv(args.out/"sample_counts.csv",counts)
    write_csv(args.out/"episode_event_counts.csv",event_rows);dump(args.out/"event_support.json",event_result)
    arrays={};array_index=[]
    for family,items in (("absolute",distributions),("contrast",contrast_distributions)):
        for key,value in sorted(items.items()):
            name="array_%04d"%len(arrays);arrays[name]=value
            array_index.append({"array":name,"family":family,"key":list(key)})
    np.savez_compressed(args.out/"bootstrap_distributions.npz",**arrays)
    np.savez_compressed(args.out/"bootstrap_episode_multiplicities.npz",source_seed_counts=encoder_counts.astype(np.int16),encoder_counts=encoder_counts.astype(np.int16),behavior_counts=encoder_counts.astype(np.int16),episode_weights=episode_weights.astype(np.int16))
    dump(args.out/"bootstrap_array_index.json",array_index)
    result={"status":status,"primary":primary,"directional_hypothesis":"I>0","confidence_interval":"two-sided 95% percentile",
            "improvement_or_navigation_gain_claimed":False,"event_support":event_result,"bootstrap_draws":2000,"bootstrap_seed":2026101002,
            "invalid_draws_redrawn":False,"statistical_unit":"joint checkpoint source seed for encoder AND actor roles, crossed with map-stratified whole task episodes; not independently sampled encoder/behavior axes or 36 replicates",
            "point_estimator":"pool behavior/task steps separately for each encoder, then average six encoder metrics",
            "source_sha256":{"analysis_spec":sha256(args.analysis_spec),"analysis_script":sha256(Path(__file__)),"global_model_lock":lock_sha},
            "locked_identities":identities,"shared_checkpoint_provenance":source_roles,"evaluation_metadata":metadata,"producer_metric_max_abs_difference":max_reference_difference,
            "numpy_version":np.__version__,"interpretation":"A positive interaction alone is not practical improvement. Inspect all absolute cells, simple effects and intervals. Undefined-class draws are retained as invalid; finite-subset intervals are conditional. Uncertainty is conditional on common train/validation data and fixed maps. No test-driven adaptation."}
    dump(args.out/"analysis_result.json",result)
    print(json.dumps({"status":status,"primary":primary,"out":str(args.out)}))


def brute_metrics(p,y):
    pairs=list(zip(p,y));positive=sum(y);negative=len(y)-positive
    if not positive or not negative:
        return math.nan,math.nan
    auc=sum((a>b)+.5*(a==b) for a,ya in pairs if ya for b,yb in pairs if not yb)/(positive*negative)
    ap=0
    for score in set(p):
        group_positive=sum(yy for pp,yy in pairs if pp==score)
        retained=[yy for pp,yy in pairs if pp>=score]
        ap+=group_positive/positive*sum(retained)/len(retained)
    return auc,ap


def self_test(np):
    p=np.array([.1,.7,.7,.9,.4,.8]);y=np.array([0,1,0,1,1,0]);episode=np.array([0,0,1,1,2,2])
    w=np.array([[1,1,1],[2,0,3],[0,0,0]],dtype=float)
    auc,ap=weighted_ranking(p,y,episode,w,np,2)
    for d in (0,1):
        indices=[i for i,e in enumerate(episode) for _ in range(int(w[d,e]))]
        ba,bp=brute_metrics(p[indices].tolist(),y[indices].tolist())
        require(abs(auc[d]-ba)<1e-12 and abs(ap[d]-bp)<1e-12,"Weighted ties differ from independently expanded episodes")
    require(not np.isfinite(auc[2]) and not np.isfinite(ap[2]),"Invalid-class draw not preserved")
    require(np.allclose(pava3(np.array([[.8,.2,.6],[.9,.8,.1],[.1,.1,.3]]),np),np.array([[.5,.5,.6],[.6,.6,.6],[.1,.1,.3]]),atol=1e-15,rtol=0),"PAVA boundary check failed")
    # Task draws are shared, and source identities are jointly sampled for both
    # encoder and actor roles, which come from the same trained checkpoints.
    episodes=[{"behavior_seed":b,"map_id":m,"task_id":"m%d_t%d"%(m,t)} for b in (1,2) for m in (11,12,13) for t in (0,1)]
    toy={"encoder_seeds":[1,2],"behavior_seeds":[1,2],"test_maps":[11,12,13],"bootstrap":{"seed":2026101002,"draws":20}}
    ec,ww=draws(toy,episodes,np)
    require(ec.shape==(20,2) and ww.shape==(20,12),"Crossed draw shape wrong")
    for d,row in enumerate(ww):
        matrix=row.reshape(2,6)
        require(np.linalg.matrix_rank(matrix)<=1,"Behavior/task weights do not form a shared crossed product")
        require(np.array_equal(matrix.sum(axis=1)/6,ec[d]),"Actor and encoder source counts differ")
        require(all(row[[i for i,e in enumerate(episodes) if e["map_id"]==m]].sum()==4 for m in (11,12,13)),"Fixed map task totals changed")
    require(np.array_equal(ww,draws(toy,episodes,np)[1]),"Bootstrap not deterministic")
    # Exhaustive mathematical counterexample: mean(z)+mean(z) reuses the SAME
    # six source units. Its variance is 4*sigma^2/6; independent role draws
    # would omit covariance and give 2*sigma^2/6. No experimental data involved.
    z=np.arange(6,dtype=float)
    means=np.array([np.mean(z[list(indices)]) for indices in itertools.product(range(6),repeat=6)])
    joint_variance=float(np.var(2*means));independent_variance=float(2*np.var(means))
    require(abs(joint_variance-4*np.var(z)/6)<1e-12 and abs(independent_variance-2*np.var(z)/6)<1e-12,"Shared-source covariance counterexample failed")
    cells=np.array([.1,.12,.2,.25])
    require(abs(np.dot(CONTRASTS["interaction"],cells)-((.25-.2)-(.12-.1)))<1e-15,"Interaction sign is wrong")
    # Only mathematical index fixtures: one collision, one success and one
    # timeout demonstrate inclusion of the applied action and tail censoring.
    examples=[{"steps":12,"collision":True,"success":False},
              {"steps":7,"collision":False,"success":True},
              {"steps":900,"collision":False,"success":False}]
    toy_index={"episode_index":np.repeat(np.arange(3),[12,7,900]),"timestep":np.concatenate([np.arange(n) for n in (12,7,900)])}
    yy=np.zeros((919,3),dtype=bool);mm=yy.copy()
    for j,h in enumerate(HORIZONS):
        yy[max(0,12-h):12,j]=True
        mm[:12,j]=True
        mm[12:19,j]=True
        mm[19:19+900-h+1,j]=True
    toy_index.update(targets=yy,masks=mm)
    verify_indices(examples,toy_index,np)
    require(yy[7,0] and not yy[6,0] and not mm[-1,0] and mm[12:19].all(),"Action-inclusive boundary/censoring check failed")
    # Independent regression against every old-bank raw max summary. This bank
    # is already viewed; it is not used for fitting or any new test inference.
    old=ROOT/"data/shared_evaluation/scoring/shared_scores.csv.gz"
    with gzip.open(old,"rt",newline="") as f:
        groups=itertools.groupby(csv.DictReader(f),lambda r:(r["method"],int(r["seed"])))
        values=defaultdict(list)
        for (method,seed),rr in groups:
            byh=defaultdict(list)
            for r in rr:
                if int(r["label_valid"]):
                    byh[int(r["horizon"])].append((float(r["risk_conservative"]),int(r["target"])))
            for h,pairs in byh.items():
                pp=np.array([a for a,_ in pairs]);yy=np.array([b for _,b in pairs]);ee=np.zeros(len(pp),dtype=int)
                aa,bb=weighted_ranking(pp,yy,ee,np.ones((1,1)),np)
                values[(method,h,"AUROC")].append(aa[0]);values[(method,h,"AUPRC")].append(bb[0]);values[(method,h,"Brier")].append(float(np.mean((pp-yy)**2)))
    maxdiff=0
    with (old.parent/"shared_summary.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            for metric in ("AUROC","AUPRC","Brier"):
                value=values[(r["method"],int(r["horizon"]),metric)]
                diff=abs(float(np.mean(value))-float(r[metric+"_mean"]));maxdiff=max(maxdiff,diff)
                require(diff<1e-12,"Old-bank independent metric regression failed")
    print(json.dumps({"self_test":"passed","synthetic_scope":"kernel checks only, not scientific results","crossed_draws_checked":20,
                      "joint_source_counts_verified":True,"variance_counterexample":{"joint_roles":joint_variance,"independent_roles":independent_variance},
                      "old_bank_reference_max_abs_difference":maxdiff,"new_test_files_read":False,"torch_imported":False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root","--probe-dir",dest="probe_root",type=Path,default=BASE/"probe")
    parser.add_argument("--analysis-spec","--spec",dest="analysis_spec",type=Path,default=BASE/"analysis_spec.json")
    parser.add_argument("--probe-spec",type=Path,default=BASE/"probe_spec.json")
    parser.add_argument("--protocol",type=Path,default=BASE/"protocol.json")
    parser.add_argument("--registry",type=Path,default=BASE/"tasks/task_behavior_assignments.csv")
    parser.add_argument("--checkpoints",type=Path,help="Default: checkpoints.json beside protocol.json")
    parser.add_argument("--bank-root",type=Path,help="Default: bank directory beside the probe directory; provenance only, never used for fitting")
    parser.add_argument("--producer-script",type=Path,default=ROOT/"scripts/frozen_risk_probe.py")
    parser.add_argument("--out",type=Path,default=BASE/"analysis")
    parser.add_argument("--chunk-size",type=int,default=8,help="Only controls temporary-memory use; never changes draws or estimand")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args();require(args.chunk_size>0,"chunk-size must be positive")
    import numpy as np
    if args.self_test:
        self_test(np)
    else:
        analyze(args,np)


if __name__=="__main__":
    main()
