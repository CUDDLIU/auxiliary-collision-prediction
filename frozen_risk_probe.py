#!/usr/bin/env python3
"""Prespecified frozen-Q1 2x2 probe. Python 3.8; real execution needs torch.

Stages are deliberately separate: extract(train/val), train(each encoder), lock
(all 24 heads), extract(test), evaluate(each encoder). ``purge`` removes only a
generated feature.npy after its consumer finishes, preserving hashes and indices.
No stage changes SAC.py, an encoder, an actor, or a source episode. Test NPZ files
are not opened until a complete model lock exists. Synthetic selftest is not a
scientific result. See experiments/loss_ordering_v1_3/probe_spec.json for the budget.
The experiment-only ordered forward expands the three-horizon prefix explicitly
to avoid unsupported CUDA cumsum under strict deterministic algorithms.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

HORIZONS = (5, 10, 20)
CELLS = ("independent_balanced", "ordered_balanced", "independent_proper", "ordered_proper")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(args):
    return {"spec_sha256": sha256(args.spec), "script_sha256": sha256(Path(__file__)),
            "registry_sha256": sha256(args.registry), "protocol_sha256": sha256(args.protocol),
            "analysis_spec_sha256": sha256(args.analysis_spec)}


def check_identity(record, args):
    for key, value in identity(args).items():
        require(record.get(key) == value, "Locked input/code identity differs: " + key)


def registry(args, spec):
    with args.registry.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"rollout_id", "task_id", "split", "map_id", "behavior_seed"}
    require(rows and required <= set(rows[0]), "Invalid task_behavior_assignments.csv schema")
    require(len({r["rollout_id"] for r in rows}) == len(rows), "Duplicate rollout_id")
    task_splits = {}
    for row in rows:
        require(row["split"] in ("train", "val", "test"), "Formal registry must exclude smoke")
        require(int(row["behavior_seed"]) in spec["behavior_seeds"], "Unknown behavior seed")
        if row["task_id"] in task_splits:
            require(task_splits[row["task_id"]] == row["split"], "Task crosses split boundary")
        task_splits[row["task_id"]] = row["split"]
    for split, budget in spec["episode_budget"].items():
        selected = [r for r in rows if r["split"] == split]
        require(len(selected) == budget, "Wrong episode budget for " + split)
        pairs = {(r["task_id"], int(r["behavior_seed"])) for r in selected}
        tasks = {r["task_id"] for r in selected}
        require(len(pairs) == len(selected) == len(tasks) * 6,
                "Each split task must have all six behavior seeds exactly once")
    return rows


def context(args):
    spec = read_json(args.spec)
    require(tuple(spec["horizons"]) == HORIZONS and tuple(spec["cells"]) == CELLS,
            "Unsupported spec horizons or cells")
    require(spec["encoder_seeds"] == list(range(1, 7)), "Expected six frozen encoders")
    rows = registry(args, spec)
    args.out.mkdir(parents=True, exist_ok=True)
    return spec, rows


def cache_dir(args, split=None):
    return args.out / "cache" / ("encoder_%02d" % args.encoder_seed) / (split or args.split)


def models_dir(args):
    return args.out / "models" / ("encoder_%02d" % args.encoder_seed)


def source_module(args, spec):
    path = args.sac_dir / "SAC.py"
    require(sha256(path) == spec["source"]["SAC_py_sha256"], "Original SAC.py hash mismatch")
    module_spec = importlib.util.spec_from_file_location("probe_original_sac", str(path))
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(args.sac_dir.resolve()))
    module_spec.loader.exec_module(module)
    return module


def torch_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    return torch, device


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verified_checkpoint_audit(args):
    protocol = read_json(args.protocol)
    binding = protocol["checkpoint_identity_audit"]
    relative = Path(binding["file"])
    root = args.protocol.parent.resolve()
    path = (root / relative).resolve()
    require(not relative.is_absolute() and root in path.parents and sha256(path) == binding["sha256"],
            "Checkpoint identity audit file/binding differs")
    audit = read_json(path)
    require(audit.get("schema") == "lira_checkpoint_identity_audit_v1" and
            audit.get("status") == "current_fixed_source_identity_verified", "Checkpoint identity audit is not verified")
    records = audit.get("checkpoints", [])
    by_seed = {int(r["seed"]): r for r in records}
    require(len(records) == 6 and set(by_seed) == set(range(1, 7)), "Audit must contain six unique source seeds")
    for seed, record in by_seed.items():
        hashes = [record["sha256"], record["roles"]["actor"]["canonical_tensors_sha256"],
                  record["roles"]["Q1"]["canonical_tensors_sha256"]]
        require(all(len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in hashes), "Invalid source fingerprint")
        require(record["sha256"] == record["repeated_sha256"] == protocol["checkpoint_role_mapping"][str(seed)]["sha256"] and
                record.get("stable_file_read") is True and record.get("structure_and_current_source_identity_verified") is True and
                record.get("independent_domain", {}).get("agreement") is True and
                record["roles"]["actor"].get("payload_key") == "policy" and record["roles"]["Q1"].get("payload_key") == "q1",
                "Audit source role/whole-file identity differs")
    for role in ("actor", "Q1"):
        require(len({r["roles"][role]["canonical_tensors_sha256"] for r in records}) == 6,
                "Audit repeats a source role fingerprint")
    return by_seed, binding["sha256"]


def load_q1(args, module, torch, device):
    audited, audit_sha = verified_checkpoint_audit(args)
    audited_source = audited[args.encoder_seed]
    entries = read_json(args.checkpoints)
    require(isinstance(entries, list) and len(entries) == 6, "Need all six ZeroAux checkpoints")
    require({int(e["seed"]) for e in entries} == set(range(1, 7)), "Checkpoint seed list mismatch")
    entry = next(e for e in entries if int(e["seed"]) == args.encoder_seed)
    path = Path(entry["path"]).expanduser()
    if not path.is_absolute():
        path = args.checkpoints.parent / path
    require(path.resolve() == Path(audited_source["listed_path"]).resolve() and
            entry.get("sha256") == audited_source["sha256"], "Encoder source differs from identity audit")
    stat = path.stat()
    require(sha256(path) == audited_source["sha256"], "Whole checkpoint SHA mismatch")
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False  # Trusted user-supplied legacy checkpoint.
    saved = torch.load(str(path), **kwargs)
    cfg = saved["args"] if isinstance(saved["args"], dict) else vars(saved["args"])
    args_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    require(args_sha == audited_source["checkpoint_args_sha256"], "Loaded checkpoint configuration differs from identity audit")
    require(int(cfg.get("seed", -1)) == args.encoder_seed, "Checkpoint seed mismatch")
    require(not cfg.get("use_multi_horizon_risk", False), "Q1 must be ZeroAux, not risk-trained")
    require(float(cfg.get("collision_aux_weight", -1)) == 0.0, "Checkpoint is not zero-weight auxiliary")
    require(int(cfg.get("keys_num", cfg.get("key_num", 10))) == 10, "History length mismatch")
    require(cfg.get("lidar_repr") == "multi_v1" and cfg.get("temporal_backbone", "transformer") == "transformer",
            "Expected LiRA Transformer representation")
    common = {"state_dim": 76, "action_dim": 2, "lidar_repr": cfg["lidar_repr"],
              "lidar_max_range": float(cfg.get("lidar_max_range", 10.0)),
              "temporal_backbone": cfg.get("temporal_backbone", "transformer"),
              "use_collision_aux": bool(cfg.get("use_collision_aux", False)),
              "use_multi_horizon_risk": False}
    q = module.LiRASACQ(**common)
    q.load_state_dict(saved["q1"], strict=True)
    del saved
    actual_q1_sha = tensor_fingerprint(q.state_dict())
    require(actual_q1_sha == audited_source["roles"]["Q1"]["canonical_tensors_sha256"],
            "Actually loaded Q1 tensor identity differs from audit")
    after = path.stat()
    require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Checkpoint changed during encoder load")
    q.eval().requires_grad_(False).to(device)
    require(q.fc4.in_features == 1024, "Unexpected hidden width")
    info = {"encoder_seed": args.encoder_seed, "checkpoint_name": path.name,
            "checkpoint_size_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
            "checkpoint_file_sha256": entry.get("sha256"), "checkpoint_list_sha256": sha256(args.checkpoints),
            "q1_tensors_sha256": actual_q1_sha, "constructor_kwargs": common,
            "checkpoint_args_sha256": args_sha, "checkpoint_identity_audit_sha256": audit_sha,
            "audited_actor_tensors_sha256": audited_source["roles"]["actor"]["canonical_tensors_sha256"],
            "loaded_q1_identity_verified_before_bank_read": True}
    return q, info


def labels_masks(collision, success, np):
    n = len(collision)
    targets = np.zeros((n, 3), dtype=np.int8)
    masks = np.zeros((n, 3), dtype=bool)
    for t in range(n):
        for j, h in enumerate(HORIZONS):
            targets[t, j] = int(np.any(collision[t:min(n, t + h)]))
            masks[t, j] = targets[t, j] or n - t >= h or success
    return targets, masks


def verified_lock(args):
    path = args.out / "model_lock.json"
    require(path.is_file(), "Test access forbidden: all 24 models must be locked first")
    lock = read_json(path)
    require(lock.get("status") == "locked" and len(lock.get("models", [])) == 24, "Incomplete lock")
    check_identity(lock, args)
    for item in lock["models"]:
        require(sha256(args.out / item["path"]) == item["sha256"], "Locked model changed")
    return lock


def load_bank(args, rows, np):
    # Caller enforces test lock BEFORE this function can open a test manifest/NPZ.
    folder = args.bank_root / args.split
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("complete") is True, "Collection split not marked complete")
    require(manifest.get("protocol_sha256") == sha256(args.protocol), "Collection protocol differs")
    require(manifest.get("assignment_registry_sha256", manifest.get("registry_sha256")) == sha256(args.registry),
            "Collection assignment registry differs")
    entries = manifest["episodes"]
    expected = {r["rollout_id"]: r for r in rows if r["split"] == args.split}
    files = {}
    for item in entries:
        episode_id = item.get("episode_id", item.get("rollout_id", Path(item["file"]).stem))
        require(episode_id not in files and episode_id in expected, "Duplicate/unregistered bank episode")
        files[episode_id] = item
    require(set(files) == set(expected), "Split bank episodes differ from prespecified registry")
    bank = []
    for episode_id in sorted(expected):
        item, row = files[episode_id], expected[episode_id]
        path = (folder / item["file"]).resolve()
        require(folder.resolve() in path.parents, "Episode file must remain inside its split folder")
        file_sha = sha256(path)
        require(file_sha == item["sha256"], "Episode SHA mismatch: " + episode_id)
        with np.load(path, allow_pickle=False) as saved:
            needed = {"states", "actions", "collision", "success", "task_id", "map_id", "behavior_seed", "split"}
            require(needed <= set(saved.files), "Missing bank arrays: " + str(needed - set(saved.files)))
            arrays = {k: saved[k].copy() for k in needed}
            for key in ("task_id", "map_id", "behavior_seed", "split", "success"):
                require(arrays[key].shape == (), "Metadata must be scalar: " + key)
            require(str(arrays["task_id"].item()) == row["task_id"] and
                    int(arrays["map_id"].item()) == int(row["map_id"]) and
                    int(arrays["behavior_seed"].item()) == int(row["behavior_seed"]) and
                    str(arrays["split"].item()) == args.split, "Episode registry identity mismatch")
            for key in ("rollout_id", "episode_id"):
                if key in saved:
                    require(str(saved[key].item()) == episode_id, "Embedded episode ID mismatch")
        states, actions, collision = (arrays[k] for k in ("states", "actions", "collision"))
        n = len(states)
        require(states.dtype == np.float32 and states.shape == (n, 76) and 0 < n <= 900,
                "Invalid states shape/dtype/episode length")
        require(actions.dtype == np.float32 and actions.shape == (n, 2), "Invalid actions shape/dtype")
        require(np.isfinite(states).all() and np.isfinite(actions).all() and (np.abs(actions) <= 1).all(),
                "Nonfinite state or invalid normalized action")
        require(collision.shape == (n,) and np.isin(collision, [0, 1]).all(), "Invalid collision vector")
        require(arrays["success"].item() in (True, False, 0, 1), "Invalid success scalar")
        success = bool(arrays["success"].item())
        require(collision.sum() <= 1 and (not collision.sum() or (collision[-1] and not success)),
                "Collision must be terminal and exclude success")
        targets, masks = labels_masks(collision, success, np)
        bank.append({"episode_id": episode_id, "task_id": row["task_id"], "map_id": int(row["map_id"]),
                     "behavior_seed": int(row["behavior_seed"]), "source_sha256": file_sha,
                     "states": states, "actions": actions, "targets": targets, "masks": masks,
                     "steps": n, "success": success, "collision": bool(collision.sum())})
    return bank, sha256(manifest_path)


def disk_guard(args, spec, required_bytes):
    existing = sum(p.stat().st_size for p in (args.out / "cache").rglob("features.npy")) if (args.out / "cache").exists() else 0
    resources = spec["resources"]
    require(existing + required_bytes <= resources["feature_cache_limit_bytes"],
            "8 GiB feature-cache ceiling would be exceeded; purge completed encoder caches")
    output_bytes = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    require(output_bytes + required_bytes <= resources["generated_output_limit_bytes"],
            "8 GiB generated-output ceiling would be exceeded")
    require(shutil.disk_usage(args.out).free >= required_bytes + resources["minimum_free_reserve_bytes"],
            "Insufficient free disk for extraction plus 1 GiB reserve")


def extract(args, np):
    spec, rows = context(args)
    lock = verified_lock(args) if args.split == "test" else None
    require(args.split == "test" or not (args.out / "model_lock.json").exists(),
            "Train/val extraction forbidden after lock")
    folder = cache_dir(args)
    require(not folder.exists(), "Cache already exists; never overwrite a recorded extraction")
    # Resolve and verify the actual loaded Q1 before reading any split-bank
    # observation/index/cache. For test, the global head lock was checked above.
    torch, device = torch_runtime(args)
    module = source_module(args, spec)
    q, encoder = load_q1(args, module, torch, device)
    if lock:
        expected = lock["encoders"][str(args.encoder_seed)]
        require(encoder["q1_tensors_sha256"] == expected["q1_tensors_sha256"], "Test encoder differs from locked encoder")
        require(encoder["checkpoint_args_sha256"] == expected["checkpoint_args_sha256"], "Test checkpoint args differ")
    bank, bank_sha = load_bank(args, rows, np)
    if lock:
        # First post-lock extraction commits this bank. All later encoders must
        # score the same source bytes; no adaptation after seeing earlier scores.
        bank_lock_path = args.out / "test_bank_lock.json"
        bank_identity = {"model_lock_sha256": sha256(args.out / "model_lock.json"),
                         "split_bank_manifest_sha256": bank_sha,
                         "episodes": [{"episode_id": e["episode_id"], "source_sha256": e["source_sha256"]} for e in bank]}
        if bank_lock_path.exists():
            require(read_json(bank_lock_path) == bank_identity, "Test bank changed between encoders")
        else:
            write_json(bank_lock_path, bank_identity)
    n = sum(e["steps"] for e in bank)
    disk_guard(args, spec, n * 1024 * 4 + n * 32 + 1048576)
    folder.mkdir(parents=True)
    features = np.lib.format.open_memmap(folder / "features.npy", mode="w+", dtype=np.float32, shape=(n, 1024))
    offsets = np.cumsum([0] + [e["steps"] for e in bank])
    targets = np.concatenate([e["targets"] for e in bank])
    masks = np.concatenate([e["masks"] for e in bank])
    episode_index = np.repeat(np.arange(len(bank), dtype=np.int32), [e["steps"] for e in bank])
    timestep = np.concatenate([np.arange(e["steps"], dtype=np.int32) for e in bank])
    np.savez_compressed(folder / "index.npz", targets=targets, masks=masks, episode_index=episode_index, timestep=timestep)
    groups = {}
    for e, episode in enumerate(bank):
        for t in range(episode["steps"]):
            groups.setdefault(min(t + 1, 10), []).append((e, t))
    captured = []
    def capture_hidden(_module, inputs):
        captured.append(inputs[0].detach())
    hook = q.fc4.register_forward_pre_hook(capture_hidden)
    batch_size = spec["resources"]["extraction_batch_size"]
    written = 0
    try:
        with torch.no_grad():
            for length, items in sorted(groups.items()):
                for start in range(0, len(items), batch_size):
                    batch = items[start:start + batch_size]
                    states = np.stack([bank[e]["states"][t-length+1:t+1] for e, t in batch])
                    actions = np.stack([bank[e]["actions"][t] for e, t in batch])
                    captured.clear()
                    q(torch.from_numpy(states).to(device), torch.from_numpy(actions).to(device))
                    require(len(captured) == 1, "Hidden hook must fire exactly once per original Q1 call")
                    hidden = captured[0].cpu().numpy()
                    require(hidden.shape == (len(batch), 1024) and np.isfinite(hidden).all(), "Invalid hidden features")
                    features[[offsets[e] + t for e, t in batch]] = hidden
                    written += len(batch)
                print("extract encoder=%d split=%s history=%d rows=%d/%d" %
                      (args.encoder_seed, args.split, length, written, n), flush=True)
        require(written == n, "Feature extraction incomplete")
        require(all(not p.requires_grad and p.grad is None for p in q.parameters()), "Encoder gradient found")
        after = tensor_fingerprint(q.state_dict())
        require(after == encoder["q1_tensors_sha256"], "Frozen encoder weights changed")
        features.flush()
        del features
        episodes = [{k: v for k, v in e.items() if k not in ("states", "actions", "targets", "masks")} for e in bank]
        metadata = dict(identity(args), schema="lira_probe_feature_cache_v1", complete=True,
                        created_utc=utc(), split=args.split, encoder=encoder, q1_after_sha256=after,
                        no_encoder_gradients=True, samples=n, hidden_dimension=1024, dtype="float32",
                        split_bank_manifest_sha256=bank_sha, features_sha256=sha256(folder / "features.npy"),
                        index_sha256=sha256(folder / "index.npz"), episodes=episodes,
                        torch_version=torch.__version__, numpy_version=np.__version__, device=str(device),
                        common_valid_samples=int(masks[:, 2].sum()))
        write_json(folder / "manifest.json", metadata)
    finally:
        hook.remove()


def read_cache(args, split, np, need_features=True):
    folder = cache_dir(args, split)
    meta = read_json(folder / "manifest.json")
    check_identity(meta, args)
    require(meta["complete"] and meta["split"] == split and meta["encoder"]["encoder_seed"] == args.encoder_seed,
            "Cache identity mismatch")
    require(sha256(folder / "index.npz") == meta["index_sha256"], "Cache index changed")
    with np.load(folder / "index.npz", allow_pickle=False) as z:
        index = {k: z[k].copy() for k in z.files}
    features = None
    if need_features:
        require(sha256(folder / "features.npy") == meta["features_sha256"], "Features changed")
        features = np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
        require(features.shape == (meta["samples"], 1024) and features.dtype == np.float32, "Feature layout mismatch")
    return features, index, meta


def ranking_metrics(probability, targets, np):
    p, y = np.asarray(probability, dtype=np.float64), np.asarray(targets, dtype=np.int64)
    positives, n = int(y.sum()), len(y)
    negatives = n - positives
    if not positives or not negatives:
        return None, None
    order = np.argsort(p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    counts = np.diff(np.r_[starts, n])
    positive_groups = np.add.reduceat(sorted_y, starts)
    ranks = starts + (counts + 1) / 2.0
    auc = (np.dot(positive_groups, ranks) - positives * (positives + 1) / 2.0) / (positives * negatives)
    suffix_positive = np.cumsum(positive_groups[::-1])[::-1]
    suffix_count = np.cumsum(counts[::-1])[::-1]
    ap = np.sum(positive_groups / positives * suffix_positive / suffix_count)
    return float(auc), float(ap)


def metric_rows(probabilities, logits, targets, masks, np):
    result = []
    for population in ("common_m20", "per_horizon_sensitivity"):
        for j, horizon in enumerate(HORIZONS):
            valid = masks[:, 2] if population == "common_m20" else masks[:, j]
            p, y, z = probabilities[valid, j], targets[valid, j], logits[valid, j].astype(np.float64)
            auc, ap = ranking_metrics(p, y, np)
            result.append({"population": population, "horizon": horizon, "n": len(y),
                           "positive": int(y.sum()), "negative": int(len(y)-y.sum()), "AUROC": auc, "AP": ap,
                           "Brier": float(np.mean((p.astype(np.float64)-y)**2)) if len(y) else None,
                           "log_loss": float(np.mean(np.logaddexp(0, z)-y*z)) if len(y) else None})
    result.append({"population": "all_vectors_structure_only", "horizon": None, "n": len(probabilities),
                   "strict_violation_rate": float(np.mean(np.any(probabilities[:, :-1] > probabilities[:, 1:], axis=1)))})
    return result


def class_gate(index, split, np):
    require(index["masks"][index["masks"][:, 2]].all(), "m20-valid rows must be valid at all horizons")
    y = index["targets"][index["masks"][:, 2]]
    require(len(y) > 0, "No common valid samples in " + split)
    for j, h in enumerate(HORIZONS):
        require(0 < int(y[:, j].sum()) < len(y), "Stop: %s h%d lacks one class; do not inspect test or extend sample" % (split, h))


def strict_deterministic_ordered_head_class(module, torch):
    """Keep the original Linear parameters; expand its fixed three-horizon sum.

    CUDA cumsum is unsupported with strict deterministic algorithms in the
    execution environment. For [5, 10, 20], its two-increment prefix is exactly
    [d1, d1+d2]. Retain first+(d1+d2), rather than changing the association to
    (first+d1)+d2. The inherited constructor and state_dict names stay intact.
    """
    class StrictDeterministicOrderedHead(module.MonotonicMultiHorizonRiskHead):
        def forward(self, hidden):
            raw = self.projection(hidden)
            require(raw.size(-1) == 3, "Deterministic ordered forward requires the three fixed horizons")
            first = raw[..., :1]
            increments = torch.nn.functional.softplus(raw[..., 1:])
            d1, d2 = increments[..., :1], increments[..., 1:2]
            logits = torch.cat((first, first + d1, first + (d1 + d2)), dim=-1)
            return logits, torch.sigmoid(logits)
    return StrictDeterministicOrderedHead


def initialized_heads(module, spec, seed, torch, np, device):
    random = np.random.RandomState(spec["initialization"]["seed_base"] + seed)
    weight = random.normal(0.0, spec["initialization"]["weight_std"], (3, 1024)).astype(np.float32)
    p = np.asarray(spec["initialization"]["zero_input_probabilities"], dtype=np.float64)
    logits = np.log(p / (1-p))
    ordered_bias = np.r_[logits[0], np.log(np.expm1(np.diff(logits)))].astype(np.float32)
    heads = {}
    for cell in CELLS:
        ordered = cell.startswith("ordered_")
        cls = strict_deterministic_ordered_head_class(module, torch) if ordered else module.IndependentMultiHorizonRiskHead
        head = cls(1024, 3).to(device)
        with torch.no_grad():
            head.projection.weight.copy_(torch.from_numpy(weight).to(device))
            head.projection.bias.copy_(torch.from_numpy(ordered_bias if ordered else logits.astype(np.float32)).to(device))
        heads[cell] = head
    return heads


def predict(head, features, torch, np, device, batch_size):
    head.eval()
    logits = np.empty((len(features), 3), dtype=np.float32)
    probabilities = np.empty_like(logits)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(np.array(features[start:start+batch_size], copy=True)).to(device)
            z, p = head(x)
            logits[start:start+batch_size], probabilities[start:start+batch_size] = z.cpu().numpy(), p.cpu().numpy()
    require(np.isfinite(logits).all() and np.isfinite(probabilities).all(), "Nonfinite prediction")
    return logits, probabilities


def save_head(path, head, epoch, ap, np):
    np.savez(path, weight=head.projection.weight.detach().cpu().numpy(),
             bias=head.projection.bias.detach().cpu().numpy(), selected_epoch=np.array(epoch),
             validation_h20_AP=np.array(ap))


def train(args, np):
    spec, _ = context(args)
    require(not (args.out / "model_lock.json").exists(), "Cannot fit or replace heads after test lock")
    folder = models_dir(args)
    require(not folder.exists(), "Training output already exists; no silent retry or overwrite")
    xf, ti, tm = read_cache(args, "train", np)
    vf, vi, vm = read_cache(args, "val", np)
    require(tm["encoder"] == vm["encoder"], "Train/val frozen encoder identity differs")
    require({e["task_id"] for e in tm["episodes"]}.isdisjoint({e["task_id"] for e in vm["episodes"]}),
            "Train and validation tasks overlap")
    class_gate(ti, "train", np)
    class_gate(vi, "val", np)
    torch, device = torch_runtime(args)
    module = source_module(args, spec)
    heads = initialized_heads(module, spec, args.encoder_seed, torch, np, device)
    opt = spec["optimization"]
    optimizers = {cell: torch.optim.Adam(head.parameters(), lr=opt["learning_rate"],
                                       betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])
                  for cell, head in heads.items()}
    folder.mkdir(parents=True)
    train_indices = np.flatnonzero(ti["masks"][:, 2])
    best = {cell: {"AP": -1.0, "epoch": None} for cell in CELLS}
    initial = {}
    initial_probabilities = {}
    infer_batch = spec["resources"]["inference_batch_size"]
    for cell, head in heads.items():
        z, p = predict(head, xf, torch, np, device, infer_batch)
        initial_probabilities[cell] = p
        initial[cell] = {"probability_mean_all_train": p.mean(axis=0).tolist(),
                         "metrics": metric_rows(p, z, ti["targets"], ti["masks"], np),
                         "parameters_sha256": tensor_fingerprint(head.state_dict())}
    initial["max_absolute_initial_probability_difference"] = float(np.max(np.abs(
        initial_probabilities["independent_balanced"]-initial_probabilities["ordered_balanced"])))
    del initial_probabilities
    write_json(folder / "initialization.json", initial)
    batch_fields = ["epoch", "batch", "cell", "size", "row_order_sha256", "objective", "gradient_l2"]
    batch_fields += ["h%d_%s" % (h, cls) for h in HORIZONS for cls in ("positive", "negative")]
    history = []
    with (folder / "batches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=batch_fields)
        writer.writeheader()
        for epoch in range(1, opt["epochs"] + 1):
            shuffled = train_indices[np.random.RandomState(opt["shuffle_seed"] + epoch).permutation(len(train_indices))]
            epoch_hash = hashlib.sha256(shuffled.astype("<i8").tobytes()).hexdigest()
            for batch_no, start in enumerate(range(0, len(shuffled), opt["batch_size"]), 1):
                indices = shuffled[start:start+opt["batch_size"]]
                x = torch.from_numpy(np.array(xf[indices], copy=True)).to(device)
                targets_np = ti["targets"][indices].astype(np.float32)
                y = torch.from_numpy(targets_np).to(device)
                mask = torch.ones_like(y)  # Common m20 rows only; all horizons share population.
                counts = {}
                for j, h in enumerate(HORIZONS):
                    counts["h%d_positive" % h] = int(targets_np[:, j].sum())
                    counts["h%d_negative" % h] = len(indices)-counts["h%d_positive" % h]
                order_hash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
                for cell, head in heads.items():
                    head.train()
                    optimizers[cell].zero_grad(set_to_none=True)
                    z, _ = head(x)
                    loss = (module.balanced_binary_cross_entropy_with_logits(z, y, mask)
                            if cell.endswith("_balanced") else
                            (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")*mask).sum()/mask.sum())
                    require(bool(torch.isfinite(loss)), "Nonfinite objective in " + cell)
                    loss.backward()
                    gradients = [p.grad for p in head.parameters()]
                    require(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients), "Invalid gradient in " + cell)
                    norm = math.sqrt(sum(float((g.detach().double()**2).sum()) for g in gradients))
                    optimizers[cell].step()
                    require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), "Nonfinite fitted head")
                    writer.writerow(dict(counts, epoch=epoch, batch=batch_no, cell=cell, size=len(indices),
                                         row_order_sha256=order_hash, objective=float(loss.detach()), gradient_l2=norm))
            stream.flush()
            for cell, head in heads.items():
                tz, tp = predict(head, xf, torch, np, device, infer_batch)
                vz, vp = predict(head, vf, torch, np, device, infer_batch)
                validation = metric_rows(vp, vz, vi["targets"], vi["masks"], np)
                ap = next(r["AP"] for r in validation if r["population"] == "common_m20" and r["horizon"] == 20)
                require(ap is not None and math.isfinite(ap), "Undefined validation AP")
                history.append({"epoch": epoch, "cell": cell, "row_order_sha256": epoch_hash,
                                "train": metric_rows(tp, tz, ti["targets"], ti["masks"], np), "val": validation})
                if ap > best[cell]["AP"]:
                    best[cell] = {"AP": ap, "epoch": epoch}
                    save_head(folder / (cell + "_selected.npz"), head, epoch, ap, np)
                print("train encoder=%d epoch=%d cell=%s val_h20_AP=%.8f" %
                      (args.encoder_seed, epoch, cell, ap), flush=True)
            write_json(folder / "history.json", history)
    for cell, head in heads.items():
        torch.save({"head": head.state_dict(), "optimizer": optimizers[cell].state_dict(), "epoch": opt["epochs"]},
                   str(folder / (cell + "_final_training_state.pt")))
    report = dict(identity(args), schema="lira_probe_training_v1", complete=True,
                  encoder_seed=args.encoder_seed, encoder=tm["encoder"], finished_utc=utc(),
                  train_cache_manifest_sha256=sha256(cache_dir(args, "train") / "manifest.json"),
                  val_cache_manifest_sha256=sha256(cache_dir(args, "val") / "manifest.json"),
                  train_index_sha256=tm["index_sha256"], val_index_sha256=vm["index_sha256"],
                  train_features_sha256=tm["features_sha256"], val_features_sha256=vm["features_sha256"],
                  epochs=opt["epochs"], common_train_samples=len(train_indices),
                  batch_log_sha256=sha256(folder / "batches.csv"), history_sha256=sha256(folder / "history.json"),
                  initialization_sha256=sha256(folder / "initialization.json"),
                  selected={cell: dict(best[cell], sha256=sha256(folder / (cell + "_selected.npz"))) for cell in CELLS},
                  torch_version=torch.__version__, device=str(device),
                  encoder_not_loaded_during_head_training=True, test_access=False)
    write_json(folder / "complete.json", report)


def lock_models(args, np):
    spec, _ = context(args)
    path = args.out / "model_lock.json"
    require(not path.exists(), "Model lock already exists; no overwrite")
    records, encoders, model_rows = [], {}, []
    population_identity = {}
    for seed in spec["encoder_seeds"]:
        args.encoder_seed = seed
        folder = models_dir(args)
        report = read_json(folder / "complete.json")
        check_identity(report, args)
        require(report["complete"] and report["epochs"] == spec["optimization"]["epochs"] and report["test_access"] is False,
                "Incomplete training budget or premature test access")
        for split in ("train", "val"):
            _, _, meta = read_cache(args, split, np, need_features=False)
            require(sha256(cache_dir(args, split) / "manifest.json") == report[split+"_cache_manifest_sha256"], "Cache metadata changed")
            require(meta["encoder"] == report["encoder"], "Encoder mismatch")
            population = {"index_sha256": meta["index_sha256"], "episodes": meta["episodes"],
                          "split_bank_manifest_sha256": meta["split_bank_manifest_sha256"]}
            if split in population_identity:
                require(population == population_identity[split], "Encoders did not use identical split samples")
            population_identity[split] = population
        for filename, field in (("batches.csv", "batch_log_sha256"), ("history.json", "history_sha256"),
                                ("initialization.json", "initialization_sha256")):
            require(sha256(folder / filename) == report[field], "Training log changed")
        history = read_json(folder / "history.json")
        require(len(history) == len(CELLS) * spec["optimization"]["epochs"], "Training history budget mismatch")
        for cell in CELLS:
            cell_history = sorted([r for r in history if r["cell"] == cell], key=lambda r: r["epoch"])
            require([r["epoch"] for r in cell_history] == list(range(1, spec["optimization"]["epochs"]+1)),
                    "Missing or duplicate epoch in cell history")
            candidates = [(next(m["AP"] for m in r["val"] if m["population"] == "common_m20" and m["horizon"] == 20),
                           r["epoch"]) for r in cell_history]
            expected_ap, expected_epoch = max(candidates, key=lambda pair: (pair[0], -pair[1]))
            require(report["selected"][cell]["epoch"] == expected_epoch and report["selected"][cell]["AP"] == expected_ap,
                    "Selection does not follow validation AP / earliest tie rule")
            model = folder / (cell + "_selected.npz")
            require(sha256(model) == report["selected"][cell]["sha256"], "Selected model changed")
            with np.load(model, allow_pickle=False) as saved:
                require(saved["weight"].shape == (3, 1024) and saved["bias"].shape == (3,) and
                        np.isfinite(saved["weight"]).all() and np.isfinite(saved["bias"]).all(), "Invalid saved head")
                require(int(saved["selected_epoch"]) == expected_epoch and float(saved["validation_h20_AP"]) == expected_ap,
                        "Model file selection metadata differs from training history")
            model_rows.append({"encoder_seed": seed, "cell": cell, "path": str(model.relative_to(args.out)),
                               "sha256": sha256(model), "selected_epoch": report["selected"][cell]["epoch"],
                               "validation_h20_AP": report["selected"][cell]["AP"]})
        encoders[str(seed)] = report["encoder"]
        records.append({"encoder_seed": seed, "report_sha256": sha256(folder / "complete.json")})
    result = dict(identity(args), schema="lira_probe_model_lock_v1", status="locked", created_utc=utc(),
                  models=model_rows, encoders=encoders, training_records=records,
                  population_identity=population_identity,
                  test_data_read=False, instruction="Immutable: any later model/spec/code change creates a new exploratory study.")
    write_json(path, result)
    print("LOCKED 24 selected readouts; test extraction is now permitted", flush=True)


def evaluate(args, np):
    spec, _ = context(args)
    lock = verified_lock(args)
    folder = args.out / "evaluation" / ("encoder_%02d" % args.encoder_seed)
    require(not folder.exists(), "Evaluation output exists; do not overwrite test results")
    xf, index, meta = read_cache(args, "test", np)
    require(meta["encoder"]["q1_tensors_sha256"] == lock["encoders"][str(args.encoder_seed)]["q1_tensors_sha256"],
            "Test representation differs from model lock")
    torch, device = torch_runtime(args)
    module = source_module(args, spec)
    heads = initialized_heads(module, spec, args.encoder_seed, torch, np, device)
    folder.mkdir(parents=True)
    rows, files = [], []
    for cell, head in heads.items():
        model = models_dir(args) / (cell + "_selected.npz")
        with np.load(model, allow_pickle=False) as saved:
            with torch.no_grad():
                head.projection.weight.copy_(torch.from_numpy(saved["weight"]).to(device))
                head.projection.bias.copy_(torch.from_numpy(saved["bias"]).to(device))
        logits, probabilities = predict(head, xf, torch, np, device, spec["resources"]["inference_batch_size"])
        score_path = folder / (cell + "_scores.npz")
        np.savez_compressed(score_path, logits=logits, probabilities=probabilities,
                            targets=index["targets"], masks=index["masks"],
                            episode_index=index["episode_index"], timestep=index["timestep"])
        files.append({"cell": cell, "file": score_path.name, "sha256": sha256(score_path)})
        rows.extend(dict(r, cell=cell, encoder_seed=args.encoder_seed) for r in metric_rows(
            probabilities, logits, index["targets"], index["masks"], np))
    write_json(folder / "metrics.json", rows)
    write_json(folder / "complete.json", dict(identity(args), complete=True, encoder_seed=args.encoder_seed,
               model_lock_sha256=sha256(args.out / "model_lock.json"),
               test_cache_manifest_sha256=sha256(cache_dir(args, "test") / "manifest.json"),
               test_features_sha256=meta["features_sha256"], test_index_sha256=meta["index_sha256"],
               episodes=meta["episodes"], scores=files, metrics_sha256=sha256(folder / "metrics.json"),
               finished_utc=utc(), inference_only=True,
               interpretation="Report all four cells and simple effects; crossed whole-episode uncertainty is computed separately."))
    print("Scored four locked cells for encoder %d; no model selection on test" % args.encoder_seed, flush=True)


def purge(args, np):
    context(args)
    if args.split == "test":
        verified_lock(args)
    folder = cache_dir(args)
    _, _, metadata = read_cache(args, args.split, np, need_features=False)
    if args.split == "test":
        report_path = args.out / "evaluation" / ("encoder_%02d" % args.encoder_seed) / "complete.json"
        report = read_json(report_path)
        require(report["complete"] and report["test_features_sha256"] == metadata["features_sha256"], "Test scoring incomplete")
    else:
        report = read_json(models_dir(args) / "complete.json")
        check_identity(report, args)
        require(report["complete"] and report[args.split+"_features_sha256"] == metadata["features_sha256"], "Four-cell training incomplete")
    path = folder / "features.npy"
    require(path.is_file() and sha256(path) == metadata["features_sha256"], "Only verified generated features may be purged")
    size = path.stat().st_size
    path.unlink()
    write_json(folder / "purged.json", {"file": "features.npy", "sha256": metadata["features_sha256"],
                                       "bytes": size, "purged_utc": utc(), "source_episodes_untouched": True})
    print("Purged generated feature cache only: %d bytes; index and provenance retained" % size)


def synthetic_pipeline(args, np):
    """Tiny disposable integration fixture; no actor, encoder checkpoint or bank.

    Uses two epochs of 24 synthetic rows, never the formal budget/data. Exercising
    the actual train/lock/evaluate/purge functions catches stage/schema mistakes.
    Every file is inside a TemporaryDirectory removed before selftest returns.
    """
    synthetic = argparse.Namespace(**vars(args))
    with tempfile.TemporaryDirectory(prefix="probe-synthetic-pipeline-") as temporary:
        root = Path(temporary)
        synthetic.out = root / "outputs"
        synthetic.spec = root / "synthetic_spec.json"
        synthetic.protocol = root / "synthetic_protocol.json"
        synthetic.analysis_spec = root / "synthetic_analysis.json"
        synthetic.registry = root / "synthetic_assignments.csv"
        spec = read_json(args.spec)
        spec["status"] = "disposable_synthetic_integration_test_not_scientific_results"
        spec["optimization"]["epochs"] = 2
        spec["optimization"]["batch_size"] = 7
        spec["episode_budget"] = {split: 6 for split in ("train", "val", "test")}
        write_json(synthetic.spec, spec)
        write_json(synthetic.protocol, {"synthetic": True})
        write_json(synthetic.analysis_spec, {"synthetic": True})
        with synthetic.registry.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rollout_id", "task_id", "split", "map_id", "behavior_seed"])
            for split in ("train", "val", "test"):
                for behavior in range(1, 7):
                    writer.writerow(["%s_behavior%d" % (split, behavior), split, split, 3, behavior])
        def fixture(seed, split):
            synthetic.encoder_seed, synthetic.split = seed, split
            directory = cache_dir(synthetic)
            directory.mkdir(parents=True)
            x = np.random.RandomState(100+seed).normal(0, .1, (24, 1024)).astype(np.float32)
            np.save(directory / "features.npy", x)
            y = np.tile(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int8), (6, 1))
            np.savez_compressed(directory / "index.npz", targets=y, masks=np.ones_like(y, dtype=bool),
                                episode_index=np.repeat(np.arange(6, dtype=np.int32), 4),
                                timestep=np.tile(np.arange(4, dtype=np.int32), 6))
            episodes = [{"episode_id": "%s_behavior%d" % (split, b), "task_id": split,
                         "behavior_seed": b, "map_id": 3, "steps": 4, "success": False,
                         "collision": True, "source_sha256": "synthetic"} for b in range(1, 7)]
            metadata = dict(identity(synthetic), complete=True, split=split, samples=24,
                            encoder={"encoder_seed": seed, "q1_tensors_sha256": "synthetic_encoder_%d" % seed,
                                     "checkpoint_args_sha256": "synthetic"},
                            index_sha256=sha256(directory / "index.npz"),
                            features_sha256=sha256(directory / "features.npy"), episodes=episodes,
                            split_bank_manifest_sha256="synthetic_" + split)
            write_json(directory / "manifest.json", metadata)
        for seed in range(1, 7):
            fixture(seed, "train")
            fixture(seed, "val")
            train(synthetic, np)
            for split in ("train", "val"):
                synthetic.split = split
                purge(synthetic, np)
        lock_models(synthetic, np)
        assert len(verified_lock(synthetic)["models"]) == 24
        for seed in range(1, 7):
            fixture(seed, "test")
            evaluate(synthetic, np)
            purge(synthetic, np)
        # Any post-lock head tampering must prevent further test access.
        model = synthetic.out / read_json(synthetic.out / "model_lock.json")["models"][0]["path"]
        with model.open("ab") as f:
            f.write(b"synthetic_tamper")
        try:
            verified_lock(synthetic)
        except ValueError as exc:
            assert "changed" in str(exc)
        else:
            raise AssertionError("Changed selected model did not block test access")
    print("Disposable 24-head/two-epoch synthetic train-lock-evaluate-purge and tamper rejection: PASS")


def selftest(args, np):
    y, m = labels_masks(np.array([0, 0, 0, 0, 0, 1]), False, np)
    assert y[:, 0].tolist() == [0, 1, 1, 1, 1, 1] and m.all()
    y, m = labels_masks(np.zeros(21, dtype=np.int8), False, np)
    assert m[:, 2].sum() == 2 and not y.any()
    assert labels_masks(np.zeros(2), True, np)[1].all()
    assert ranking_metrics([.2, .8], [0, 1], np) == (1.0, 1.0)
    assert ranking_metrics([.2, .2], [0, 1], np) == (.5, .5)
    assert ranking_metrics([.8, .2], [0, 1], np) == (0.0, .5)
    assert ranking_metrics([.2], [0], np) == (None, None)
    p = np.array([.05, .06, .07])
    l = np.log(p/(1-p))
    b = np.r_[l[0], np.log(np.expm1(np.diff(l)))]
    reconstructed = np.r_[b[0], b[0] + np.cumsum(np.logaddexp(0, b[1:]))]
    assert np.allclose(reconstructed, l)
    with tempfile.TemporaryDirectory(prefix="probe-selftest-") as temporary:
        original_out = args.out
        args.out = Path(temporary)
        try:
            verified_lock(args)
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError("Missing test lock was accepted")
        args.out = original_out
    if args.with_torch:
        spec = read_json(args.spec)
        torch, device = torch_runtime(args)
        module = source_module(args, spec)
        heads = initialized_heads(module, spec, 1, torch, np, device)
        zero = torch.zeros(4, 1024, device=device)
        for cell, head in heads.items():
            z, prob = head(zero)
            assert np.allclose(prob.detach().cpu().numpy(), p[None, :], atol=1e-7)
        z = torch.tensor([[0., 1., -1.], [2., -2., 0.], [-3., 3., 2.]], device=device, requires_grad=True)
        y = torch.tensor([[0., 0., 1.], [1., 0., 1.], [1., 0., 1.]], device=device)
        element = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
        expected = ((element[0, 0]+element[1:, 0].mean())/2+element[:, 1].mean()+element[:, 2].mean())/3
        actual = module.balanced_binary_cross_entropy_with_logits(z, y)
        assert torch.allclose(expected, actual)
        assert not torch.allclose(actual, element.mean())
        assert float(module.balanced_binary_cross_entropy_with_logits(z, y, torch.zeros_like(y))) == 0
        mask = torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]], device=device)
        assert torch.allclose(module.balanced_binary_cross_entropy_with_logits(z, y, mask), element[:, 1].mean())
        ordered = heads["ordered_balanced"]
        frozen = torch.randn(12, 1024, device=device)
        z, prob = ordered(frozen)
        assert bool((prob[:, 1:] >= prob[:, :-1]).all())
        assert not frozen.requires_grad
        z.sum().backward()
        assert all(param.grad is not None for param in ordered.parameters())
        assert frozen.grad is None
        if args.pipeline:
            synthetic_pipeline(args, np)
    print("Synthetic masks, tied AP/AUROC, initialization and test-access gate: PASS. No scientific training/results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "train", "lock", "evaluate", "purge", "selftest"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/loss_ordering_v1_3/probe_spec.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/loss_ordering_v1_3/protocol.json"))
    parser.add_argument("--analysis-spec", type=Path, default=Path("experiments/loss_ordering_v1_3/analysis_spec.json"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/loss_ordering_v1_3/tasks/task_behavior_assignments.csv"))
    parser.add_argument("--out", type=Path, default=Path("experiments/loss_ordering_v1_3/probe"))
    parser.add_argument("--sac-dir", type=Path, default=Path("server_snapshot/LiRA-SAC/SAC-robot-navigation-CL/SAC"))
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--bank-root", type=Path)
    parser.add_argument("--encoder-seed", type=int, choices=range(1, 7))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--with-torch", action="store_true", help="Run additional synthetic tests against original torch heads/loss")
    parser.add_argument("--pipeline", action="store_true", help="With --with-torch, run a disposable tiny 24-head stage integration test")
    args = parser.parse_args()
    if args.stage not in ("selftest", "lock") and args.encoder_seed is None:
        parser.error("--encoder-seed is required")
    if args.stage in ("extract", "purge") and args.split is None:
        parser.error("--split is required")
    if args.stage == "extract" and (args.checkpoints is None or args.bank_root is None):
        parser.error("extract requires --checkpoints and --bank-root")
    if args.pipeline and (args.stage != "selftest" or not args.with_torch):
        parser.error("--pipeline requires selftest --with-torch")
    import numpy as np
    stages = {"extract": extract, "train": train, "lock": lock_models, "evaluate": evaluate,
              "purge": purge, "selftest": selftest}
    stages[args.stage](args, np)


if __name__ == "__main__":
    main()
