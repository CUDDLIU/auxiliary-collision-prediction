#!/usr/bin/env python3
"""Portable, identity-bound adapter to the preserved frozen-readout producer.

No collection or encoder training. Only loader, action-history extraction,
resource accounting and identity metadata differ from the preserved producer.
The fitting, initialization, validation selection and lock functions are reused.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import time
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FEATURE_LIMIT = 3 * 1024**3
OUTPUT_LIMIT = 8 * 1024**3
RESERVE = 2 * 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def relative_file(root, name):
    p = Path(name)
    require(not p.is_absolute(), 'Binding paths must be relative')
    return (root / p).resolve()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_binding(path):
    path = Path(path).resolve()
    binding = read_json(path)
    require(binding.get('schema') == 'lira_anonymous_review_binding_v1', 'Unexpected binding schema')
    require(binding.get('interpretation') == 'post_hoc_input_convention_sensitivity',
            'This is a post-hoc sensitivity, not a replacement primary study')
    require(binding.get('identity_chain') == 'anonymous_review_derivative_v1', 'Anonymous review identity chain required')
    files = {}
    for name, record in binding['files'].items():
        p = relative_file(path.parent, record['path'])
        require(p.is_file() and sha256(p) == record['sha256'], 'Bound file changed: ' + name)
        files[name] = p
    expected_code = {p.name: sha256(p) for p in [HERE / 'portable_readout.py', HERE / 'run_offline.py', HERE / 'seal_binding.py']}
    require(binding.get('code_sha256') == expected_code, 'Wrapper/runner/sealer source identity differs')
    required = {'producer', 'analyzer', 'protocol', 'probe_spec', 'analysis_spec',
                'registry', 'checkpoints', 'source_audit', 'original_model_lock',
                'exports', 'amendment_protocol', 'source_provenance', 'bank_train', 'bank_val', 'bank_test', 'review_manifest'}
    require(required <= set(files), 'Incomplete input/source binding')
    review = read_json(files['review_manifest'])
    require(review.get('schema') == 'lira_anonymous_review_core_v1' and review.get('status') == 'complete', 'Anonymous review manifest incomplete')
    require(review.get('anonymous_neural_replay') == 'not_executed', 'Reference package must not claim an anonymous neural execution')
    verified_exports(files['exports'])
    original = read_json(files['protocol'])
    require(sha256(files['source_audit']) == original['checkpoint_identity_audit']['sha256'], 'Original source audit mismatch')
    require(sha256(files['source_provenance']) == original['checkpoint_provenance_sha256'], 'Original source provenance mismatch')
    require(sha256(files['checkpoints']) == original['checkpoint_list_sha256'], 'Original checkpoint list mismatch')
    require(sha256(files['registry']) == original['registry_sha256'], 'Original registry mismatch')
    require(sha256(files['probe_spec']) == original['probe_spec_sha256'], 'Original probe specification mismatch')
    require(sha256(files['analysis_spec']) == original['analysis_spec_sha256'], 'Original analysis specification mismatch')
    require(sha256(files['producer']) == original['execution_source_sha256']['frozen_risk_probe.py'], 'Original producer source mismatch')
    require(sha256(files['analyzer']) == original['execution_source_sha256']['analyze_frozen_probe.py'], 'Original analyzer source mismatch')
    original_lock = read_json(files['original_model_lock'])
    require(original_lock.get('status') == 'locked' and len(original_lock.get('models', [])) == 24 and
            original_lock['protocol_sha256'] == sha256(files['protocol']), 'Original model lock mismatch')
    require(original_lock['script_sha256'] == sha256(files['producer']) and
            original_lock['spec_sha256'] == sha256(files['probe_spec']), 'Original lock source/spec mismatch')
    directories = {k: relative_file(path.parent, v) for k, v in binding['directories'].items()}
    require({'bank_root', 'sac_dir', 'baseline_probe'} <= set(directories), 'Missing input directories')
    source = directories['sac_dir'] / 'SAC.py'
    require(sha256(source) == read_json(files['probe_spec'])['source']['SAC_py_sha256'], 'Original SAC source mismatch')
    output = relative_file(path.parent, binding['output_root'])
    require(path.parent in output.parents and output != path.parent, 'Outputs must stay in a new binding subdirectory')
    for p in directories.values():
        require(output != p and p not in output.parents, 'Output directory overlaps preserved inputs')
    require(binding['execution']['physical_gpu'] == 0 and binding['execution']['workers'] == 1,
            'Only GPU0 and one worker are supported')
    require(binding.get('authorized_modes') == ['broadcast', 'aligned'], 'Both authorized modes must be fixed in order')
    maximum = binding['execution']['max_runtime_seconds']
    require(isinstance(maximum, (int, float)) and math.isfinite(maximum) and 30 < maximum <= 7200, 'Runtime must be finite, exceed the 30-second cleanup reserve and be at most 7200 seconds')
    if 'absolute_deadline_unix' in binding['execution']:
        deadline = binding['execution']['absolute_deadline_unix']
        require(isinstance(deadline, (int, float)) and math.isfinite(deadline) and deadline > 0, 'Deadline must be finite and positive')
    require(binding.get('resources') == {'feature_limit_bytes_per_mode': FEATURE_LIMIT,
            'output_limit_bytes_total': OUTPUT_LIMIT, 'minimum_free_reserve_bytes': RESERVE}, 'Resource limits must be fixed')
    return binding, files, directories, output


def action_batch(bank, batch, length, mode, np):
    require(mode in ('broadcast', 'aligned'), 'Unknown input mode')
    if mode == 'broadcast':
        return np.stack([bank[e]['actions'][t] for e, t in batch])
    result = np.stack([bank[e]['actions'][t-length+1:t+1] for e, t in batch])
    require(result.shape == (len(batch), length, 2), 'Aligned actions must match every observation token')
    return result


def verified_exports(path):
    path = Path(path)
    require(not (path.parent / 'STOPPED.json').exists(), 'Stopped Q1 export must not be consumed')
    completion = read_json(path.parent / 'COMPLETE.json')
    require(completion.get('status') == 'complete' and completion.get('sources') == 6 and
            completion.get('manifest_sha256') == sha256(path), 'Incomplete Q1 export/manifest binding')
    exports = read_json(path)
    require(exports.get('schema') == 'lira_anonymous_q1_export_v1' and exports.get('status') == 'complete', 'Invalid Q1 export schema/status')
    return exports


def portable_load_q1(args, module, torch, device):
    p = args._producer
    audited, audit_sha = p.verified_checkpoint_audit(args)
    source = audited[args.encoder_seed]
    listed = p.read_json(args.checkpoints)
    original = next(e for e in listed if int(e['seed']) == args.encoder_seed)
    exports = verified_exports(args._files['exports'])
    require(exports['source_checkpoint_list_sha256'] == p.sha256(args.checkpoints) and
            exports['source_audit_sha256'] == audit_sha, 'Portable export source/audit binding mismatch')
    entries = exports.get('checkpoints', [])
    require(len(entries) == 6 and {int(e['seed']) for e in entries} == set(range(1, 7)), 'Six unique Q1 exports required')
    entry = next(e for e in entries if int(e['seed']) == args.encoder_seed)
    require(entry['source_checkpoint_sha256'] == source['sha256'] == original['sha256'], 'Q1 source checkpoint mismatch')
    require(entry['original_checkpoint_args_sha256'] == source['checkpoint_args_sha256'], 'Original export args ancestry mismatch')
    require(entry['q1_tensors_sha256'] == source['roles']['Q1']['canonical_tensors_sha256'], 'Export Q1 audit mismatch')
    path = relative_file(args._files['exports'].parent, entry['file'])
    require(args._files['exports'].parent in path.parents, 'Q1 export escapes its manifest directory')
    # New anonymous whole-file/config identities are distinct from the preserved audit.
    # Same read-only handle: stat/hash -> safe load -> hash/stat; no path swap window.
    require('weights_only' in inspect.signature(torch.load).parameters, 'Safe Torch weights_only loader required')
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        stat = os.fstat(stream.fileno())
        def open_digest():
            stream.seek(0)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
            return h.hexdigest()
        require(stat.st_size == entry['bytes'] and open_digest() == entry['sha256'], 'Portable Q1 file hash/size mismatch')
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        after_handle, after_path = os.fstat(stream.fileno()), path.stat()
        identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
        require(open_digest() == entry['sha256'] and identity(stat) == identity(after_handle) == identity(after_path), 'Anonymous Q1 export changed during read')
    require(set(saved) == {'q1', 'args'} and isinstance(saved['args'], dict), 'Export payload must contain q1 and plain args only')
    cfg = saved['args']
    actual_args = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    require(actual_args == entry['checkpoint_args_sha256'], 'Loaded portable args hash differs')
    require(int(cfg.get('seed', -1)) == args.encoder_seed and
            not cfg.get('use_multi_horizon_risk', False) and
            float(cfg.get('collision_aux_weight', -1)) == 0.0, 'Expected matching ZeroAux checkpoint')
    require(int(cfg.get('keys_num', cfg.get('key_num', 10))) == 10 and cfg.get('lidar_repr') == 'multi_v1' and
            cfg.get('temporal_backbone', 'transformer') == 'transformer', 'Unexpected encoder representation/history')
    common = {'state_dim': 76, 'action_dim': 2, 'lidar_repr': cfg['lidar_repr'],
              'lidar_max_range': float(cfg.get('lidar_max_range', 10.0)),
              'temporal_backbone': cfg.get('temporal_backbone', 'transformer'),
              'use_collision_aux': bool(cfg.get('use_collision_aux', False)), 'use_multi_horizon_risk': False}
    require(common == entry['constructor_kwargs'], 'Export constructor differs from checkpoint args')
    minimal = dict(common, seed=args.encoder_seed, collision_aux_weight=0.0, keys_num=10)
    require(cfg == minimal and cfg == entry['minimal_cfg'], 'Anonymous args must contain only the minimal constructor/configuration')
    q = module.LiRASACQ(**common)
    q.load_state_dict(saved['q1'], strict=True)
    del saved
    actual_q1 = p.tensor_fingerprint(q.state_dict())
    require(actual_q1 == entry['q1_tensors_sha256'], 'Actually loaded Q1 tensor identity mismatch')
    after = path.stat()
    require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Export changed during read')
    q.eval().requires_grad_(False).to(device)
    require(q.fc4.in_features == 1024, 'Unexpected Q1 feature width')
    info = {'encoder_seed': args.encoder_seed, 'checkpoint_name': Path(original['path']).name,
            'checkpoint_size_bytes': original['bytes'], 'checkpoint_file_sha256': original['sha256'],
            'checkpoint_list_sha256': p.sha256(args.checkpoints), 'q1_tensors_sha256': actual_q1,
            'constructor_kwargs': common, 'checkpoint_args_sha256': actual_args,
            'original_checkpoint_args_sha256': entry['original_checkpoint_args_sha256'],
            'review_identity_chain': 'anonymous_review_derivative_v1',
            'checkpoint_identity_audit_sha256': audit_sha,
            'audited_actor_tensors_sha256': source['roles']['actor']['canonical_tensors_sha256'],
            'loaded_q1_identity_verified_before_bank_read': True,
            'portable_file': entry['file'], 'portable_sha256': entry['sha256'],
            'portable_bytes': entry['bytes'], 'input_mode': args.input_mode}
    return q, info


def output_guard(args, required_bytes):
    outputs = sum(p.stat().st_size for p in args.out.parent.rglob('*') if p.is_file())
    require(outputs + required_bytes <= OUTPUT_LIMIT, '8 GiB total new-output limit would be exceeded')
    require(shutil.disk_usage(args.out.parent).free >= required_bytes + RESERVE, 'Need allocation space plus 2 GiB free reserve')


def disk_guard(args, spec, required_bytes):
    existing = sum(p.stat().st_size for p in (args.out / 'cache').rglob('features.npy')) if (args.out / 'cache').exists() else 0
    require(existing + required_bytes <= FEATURE_LIMIT, '3 GiB per-mode retained feature limit would be exceeded')
    output_guard(args, required_bytes)


def install_producer(args):
    p = import_file('preserved_frozen_probe', args._files['producer'])
    original_identity = p.identity
    def identity(a):
        result = original_identity(a)
        result['script_sha256'] = sha256(HERE / 'portable_readout.py')
        result.update(wrapper_sha256=result['script_sha256'], original_producer_sha256=sha256(args._files['producer']),
                      original_analyzer_sha256=sha256(args._files['analyzer']),
                      runner_sha256=sha256(HERE / 'run_offline.py'), binding_sha256=sha256(args.binding),
                      amendment_protocol_sha256=sha256(args._files['amendment_protocol']),
                      export_manifest_sha256=sha256(args._files['exports']),
                      source_audit_sha256=sha256(args._files['source_audit']), input_mode=args.input_mode,
                      interpretation='post_hoc_input_convention_sensitivity')
        return result
    p.identity = identity
    p.load_q1 = portable_load_q1
    p.disk_guard = disk_guard
    original_load_bank = p.load_bank
    def load_bank(a, rows, np):
        manifest = a.bank_root / a.split / 'manifest.json'
        require(sha256(manifest) == sha256(args._files['bank_' + a.split]), 'Original bank manifest identity mismatch')
        return original_load_bank(a, rows, np)
    p.load_bank = load_bank
    # A single explicit expression replacement leaves all original extraction,
    # label/mask, grouping, hook, no-gradient, hash and test-lock checks intact.
    old = 'actions = np.stack([bank[e]["actions"][t] for e, t in batch])'
    new = 'actions = action_batch(bank, batch, length, args.input_mode, np)'
    extract_source = inspect.getsource(p.extract)
    require(extract_source.count(old) == 1, 'Original extraction adapter anchor differs')
    p.__dict__['action_batch'] = action_batch
    exec(compile(extract_source.replace(old, new), str(HERE / 'portable_readout.py') + ':extract_adapter', 'exec'), p.__dict__)
    args._producer = p
    return p


def adapt_behavior_verifier(m):
    src = inspect.getsource(m.verify_behavior_sources)
    old = 'aggregate.get("model_lock_sha256")==lock_sha'
    new = 'aggregate.get("model_lock_sha256")==sha256(args.original_model_lock)'
    require(src.count(old) == 1, 'Original analysis collection-lock anchor differs')
    exec(compile(src.replace(old, new), str(HERE / 'portable_readout.py') + ':analysis_adapter', 'exec'), m.__dict__)
    return m.verify_behavior_sources


def analysis_adapter(args, np):
    p = args._producer
    p.verified_lock(args)
    m = import_file('preserved_independent_analysis', args._files['analyzer'])
    original_verify = m.verify_lock
    def verify_lock(a, spec):
        result = original_verify(a, spec)
        p.check_identity(result[0], args)
        return result
    m.verify_lock = verify_lock
    # The original data were collected before this sensitivity study. Verify
    # their old collection lock and the new scoring lock independently.
    adapt_behavior_verifier(m)
    a = SimpleNamespace(probe_root=args.out, analysis_spec=args.analysis_spec,
                        probe_spec=args.spec, protocol=args.protocol, registry=args.registry,
                        checkpoints=args.checkpoints, bank_root=args.bank_root,
                        producer_script=HERE / 'portable_readout.py', out=args.out / 'analysis',
                        chunk_size=8, original_model_lock=args._files['original_model_lock'])
    m.analyze(a, np)
    result_path = a.out / 'analysis_result.json'
    result = read_json(result_path)
    result.update(study_role='post_hoc_input_convention_sensitivity', input_mode=args.input_mode,
                  original_primary_unchanged=True, independent_confirmatory_claim=False)
    p.write_json(result_path, result)
    p.write_json(a.out / 'sensitivity_identity.json', p.identity(args))


def arrays_bitwise_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order='C') == right.tobytes(order='C')


def compare_broadcast(args, np):
    require(args.input_mode == 'broadcast', 'Strict reproduction comparison applies to broadcast mode')
    args._producer.verified_lock(args)
    baseline = args._directories['baseline_probe']
    mismatches = []
    verified = []
    for seed in range(1, 7):
        folder = 'encoder_%02d' % seed
        for split in ('train', 'val', 'test'):
            relative = Path('cache') / folder / split / 'manifest.json'
            old_path = baseline / relative
            require(sha256(old_path) == args._binding['baseline_sha256'].get(str(relative)), 'Baseline feature receipt changed')
            old = read_json(old_path)
            new = read_json(args.out / 'cache' / folder / split / 'manifest.json')
            if old['features_sha256'] != new['features_sha256']:
                mismatches.append('features:' + folder + ':' + split)
            verified.append('features:' + folder + ':' + split)
        for cell in args._producer.CELLS:
            for relative in [Path('models') / folder / (cell + '_selected.npz'),
                             Path('evaluation') / folder / (cell + '_scores.npz')]:
                old_path, new_path = baseline / relative, args.out / relative
                # Root's sealed inventory binds every baseline reference read.
                key = str(relative)
                expected = args._binding['baseline_sha256'].get(key)
                require(expected and sha256(old_path) == expected, 'Baseline reference changed: ' + key)
                with np.load(old_path, allow_pickle=False) as a, np.load(new_path, allow_pickle=False) as b:
                    if set(a.files) != set(b.files) or any(not arrays_bitwise_equal(a[k], b[k]) for k in a.files):
                        mismatches.append(key)
                verified.append(key)
    result = dict(args._producer.identity(args), status='passed' if not mismatches else 'failed',
                  strict_array_equality=True, exact_feature_hashes=True, checked=len(verified), mismatches=mismatches)
    path = args.out / 'broadcast_reproduction.json'
    require(not path.exists(), 'Reproduction receipt already exists')
    args._producer.write_json(path, result)
    require(not mismatches, 'Broadcast reproduction differs; aligned sensitivity must not start')
    return result


def build_args(binding_path, mode, stage, seed=None, split=None, device='cuda:0'):
    binding, files, directories, output = load_binding(binding_path)
    a = SimpleNamespace(binding=Path(binding_path).resolve(), input_mode=mode, stage=stage,
                        encoder_seed=seed, split=split, device=device, out=output / mode,
                        spec=files['probe_spec'], protocol=files['protocol'], analysis_spec=files['analysis_spec'],
                        registry=files['registry'], checkpoints=files['checkpoints'],
                        sac_dir=directories['sac_dir'], bank_root=directories['bank_root'],
                        _binding=binding, _files=files, _directories=directories)
    require(mode in ('broadcast', 'aligned'), 'Unknown input mode')
    if mode == 'aligned':
        receipt = output / 'broadcast' / 'broadcast_reproduction.json'
        require(receipt.is_file(), 'Aligned sensitivity requires a completed strict broadcast reproduction')
        r = read_json(receipt)
        require(r.get('status') == 'passed' and r.get('binding_sha256') == sha256(a.binding) and
                r.get('wrapper_sha256') == sha256(HERE / 'portable_readout.py'), 'Broadcast receipt identity mismatch')
    return a


def validate_runtime_versions(torch, np):
    require(torch.__version__ == '2.4.1+cu121', 'Formal runtime requires Torch 2.4.1+cu121')
    require(np.__version__ == '1.23.5', 'Formal runtime requires NumPy 1.23.5')


def verify_runner_receipt(args):
    expected = (args.out.parent / 'runner_state.json').resolve()
    supplied = os.environ.get('LIRA_OFFLINE_RUNNER_RECEIPT', '')
    nonce = os.environ.get('LIRA_OFFLINE_RUNNER_CAPABILITY', '')
    require(supplied and Path(supplied).resolve() == expected and nonce,
            'Formal stages require the bounded runner capability receipt')
    state = read_json(expected)
    require(state.get('status') == 'running' and state.get('pid') == os.getppid(),
            'Formal stage parent must be the live bounded runner')
    require(state.get('binding_sha256') == sha256(args.binding) and
            state.get('capability_sha256') == hashlib.sha256(nonce.encode()).hexdigest(),
            'Runner capability/binding identity differs')
    start, deadline, compute = (state.get(k) for k in
                               ('started_wall_unix', 'deadline_wall_unix', 'compute_deadline_wall_unix'))
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, deadline, compute)),
            'Runner receipt has invalid clocks')
    require(state.get('cleanup_reserve_seconds') == 30 and compute == deadline - 30 and
            deadline <= start + args._binding['execution']['max_runtime_seconds'] and
            start <= time.time() < compute, 'Runner compute window expired or differs from the fixed budget')
    if 'absolute_deadline_unix' in args._binding['execution']:
        require(deadline <= args._binding['execution']['absolute_deadline_unix'], 'Runner exceeds absolute deadline')
    current = state.get('stages', [])
    require(current and current[-1].get('status') == 'running', 'Runner has no active stage')
    expected_stage = (args.input_mode, args.stage, args.encoder_seed, args.split)
    actual_stage = tuple(current[-1].get(k) for k in ('mode', 'stage', 'encoder_seed', 'split'))
    require(actual_stage == expected_stage, 'Runner current-stage capability differs')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('stage', choices=('verify', 'extract', 'train', 'lock', 'evaluate', 'analysis', 'compare'))
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--input-mode', choices=('broadcast', 'aligned'), required=True)
    parser.add_argument('--encoder-seed', type=int, choices=range(1, 7))
    parser.add_argument('--split', choices=('train', 'val', 'test'))
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    cli = parser.parse_args()
    if cli.stage in ('extract', 'train', 'evaluate') and cli.encoder_seed is None:
        parser.error('This stage requires --encoder-seed')
    if cli.stage == 'extract' and cli.split is None:
        parser.error('Extraction requires --split')
    args = build_args(cli.binding, cli.input_mode, cli.stage, cli.encoder_seed, cli.split, cli.device)
    p = install_producer(args)
    if cli.stage == 'verify':
        print(json.dumps(p.identity(args), sort_keys=True))
        return
    verify_runner_receipt(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_guard(args, {'analysis': 128 * 1024**2, 'train': 32 * 1024**2,
                        'evaluate': 64 * 1024**2}.get(cli.stage, 2 * 1024**2))
    import numpy as np
    import torch
    validate_runtime_versions(torch, np)
    if cli.stage == 'analysis':
        analysis_adapter(args, np)
    elif cli.stage == 'compare':
        compare_broadcast(args, np)
    else:
        {'extract': p.extract, 'train': p.train, 'lock': p.lock_models, 'evaluate': p.evaluate}[cli.stage](args, np)


if __name__ == '__main__':
    main()
