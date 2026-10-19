#!/usr/bin/env python3
"""Identity-bound aligned frozen-readout follow-up; called by one root supervisor.

This adapter never collects episodes or updates a Q1/actor. It reuses the sealed
original fitting and independent analysis functions. The extraction expression
uses recorded action histories. Epoch heads and validation geometry-deletion
diagnostics are added without changing full-validation model selection.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
FEATURE_LIMIT = 8 * 1024**3
OUTPUT_LIMIT = 10 * 1024**3
RESERVE = 3 * 1024**3
FORMAL_ACTION_LIMIT = 300000
PRODUCER_SHA = '0261d984fbb68278c3de75876495ee390826a24a46bab935d7c0ec3ee03ce67a'
ANALYZER_SHA = 'f6c0bad71630948488494e59e70d165fb8dca5d42acd320f67b7c6586dcae1bf'
LOADER_SHA = '85c95d4bb46e9d6897847390da58258548dcee756f08e5e74bb6689fa1224b38'
SAC_SHA = 'c2b99d4e65fad93b80624612103592c1721e07fe5e788bf5b4b11b5f2f24701b'
CELLS = ('independent_balanced', 'ordered_balanced', 'independent_proper', 'ordered_proper')
GATES = {'train': (30, 20), 'val': (20, 12), 'test': (20, 12)}
ROLE = 'new_dataset_frozen_probe_followup'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def relative_inside(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    require(not relative.is_absolute() and root.resolve() in path.parents,
            'Path must be relative and remain inside the sealed root')
    return path


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_new_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify_manifest(root):
    manifest_path = root / 'code_manifest.json'
    manifest = read_json(manifest_path)
    require(manifest.get('schema') == 'lira_event_coverage_code_manifest_v1',
            'Unexpected code manifest schema')
    records = manifest.get('files', [])
    require(records, 'Empty code manifest')
    paths = {}
    for record in records:
        path = relative_inside(root, record['path'])
        require(path not in paths and path != manifest_path.resolve(),
                'Duplicate/self-referential code manifest path')
        require(path.is_file() and sha256(path) == record['sha256'],
                'Sealed manifest input changed: ' + record['path'])
        if 'bytes' in record:
            require(path.stat().st_size == record['bytes'], 'Sealed manifest size differs')
        paths[path] = record['sha256']
    require(SELF in paths and (root / 'binding.json').resolve() in paths,
            'Code manifest must bind this adapter and binding.json')
    return manifest_path, paths


def verify_probe_spec(spec):
    require(tuple(spec['horizons']) == (5, 10, 20) and tuple(spec['cells']) == CELLS,
            'The original three-horizon four-cell design must be retained')
    require(spec['encoder_seeds'] == spec['behavior_seeds'] == list(range(1, 7)),
            'Exactly six paired frozen checkpoint identities are required')
    require(spec['episode_budget'] == {'train': 540, 'val': 360, 'test': 360},
            'Follow-up episode budgets must be 540/360/360')
    require(spec['source']['SAC_py_sha256'] == SAC_SHA and spec['source']['hidden_dimension'] == 1024,
            'Original source and feature width must be retained')
    init = spec['initialization']
    require(init['seed_base'] == 20261010 and init['weight_std'] == 0.0001 and
            init['zero_input_probabilities'] == [0.05, 0.06, 0.07],
            'Initialization differs from the original frozen probe')
    expected = {'optimizer': 'Adam', 'learning_rate': 0.001, 'betas': [0.9, 0.999],
                'eps': 1e-8, 'weight_decay': 0.0, 'batch_size': 1024, 'epochs': 20,
                'drop_last': False, 'shuffle_seed': 20261011, 'gradient_clipping': None,
                'schedule': None, 'early_stopping': False, 'mixed_precision': False}
    require(all(spec['optimization'].get(k) == v for k, v in expected.items()),
            'The fixed original optimizer, initialization and twenty-epoch budget cannot change')
    resources = spec['resources']
    require(resources['feature_cache_limit_bytes'] == FEATURE_LIMIT and
            resources['generated_output_limit_bytes'] == OUTPUT_LIMIT and
            resources['minimum_free_reserve_bytes'] == RESERVE and
            resources['extraction_batch_size'] == 32 and resources['inference_batch_size'] == 4096,
            'Fixed resource ceilings/batch sizes differ')


def load_binding(root):
    root = Path(root).resolve()
    path = root / 'binding.json'
    binding = read_json(path)
    require(binding.get('schema') == 'lira_event_coverage_probe_binding_v1' and
            binding.get('input_mode') == 'aligned' and binding.get('study_role') == ROLE,
            'Binding must identify the fixed aligned new-dataset follow-up')
    manifest_path, manifest_files = verify_manifest(root)
    files = {}
    for name, record in binding['files'].items():
        source = relative_inside(root, record['path'])
        require(source.is_file() and sha256(source) == record['sha256'],
                'Bound follow-up input changed: ' + name)
        files[name] = source
    required = {'producer', 'analyzer', 'portable_loader', 'protocol', 'probe_spec', 'analysis_spec',
                'registry', 'checkpoints', 'source_audit', 'source_provenance', 'exports'}
    require(required <= set(files), 'Incomplete follow-up binding')
    require(files['protocol'] == (root / 'protocol.json').resolve(), 'Protocol must be ROOT/protocol.json')
    require(files['source_provenance'] == (root / 'preflight/source_provenance.json').resolve(),
            'Original analyzer requires preflight/source_provenance.json')
    require(sha256(files['producer']) == PRODUCER_SHA and sha256(files['analyzer']) == ANALYZER_SHA and
            sha256(files['portable_loader']) == LOADER_SHA, 'Preserved source identity differs')
    for name, source in files.items():
        if name != 'exports':
            require(source in manifest_files, 'Bound input absent from root code manifest: ' + name)
    protocol = read_json(files['protocol'])
    for name, key in [('probe_spec', 'probe_spec_sha256'), ('analysis_spec', 'analysis_spec_sha256'),
                      ('registry', 'registry_sha256'), ('checkpoints', 'checkpoint_list_sha256'),
                      ('source_provenance', 'checkpoint_provenance_sha256')]:
        require(protocol[key] == sha256(files[name]), 'Protocol input binding differs: ' + name)
    audit_binding = protocol['checkpoint_identity_audit']
    require(relative_inside(root, audit_binding['file']) == files['source_audit'] and
            audit_binding['sha256'] == sha256(files['source_audit']), 'Q1/actor audit binding differs')
    if protocol.get('engineering_revision')=='v1.2':
        import source_bridge
        source_bridge.verify(root)
    else:
        require(protocol['same_checkpoint_for_encoder_and_behavior'] is True,
                'Joint-source analysis requires paired actor and encoder identities')
    require(protocol['execution_source_sha256']['frozen_risk_probe.py'] == PRODUCER_SHA and
            protocol['execution_source_sha256']['analyze_frozen_probe.py'] == ANALYZER_SHA,
            'Protocol preserved source binding differs')
    spec, analysis = read_json(files['probe_spec']), read_json(files['analysis_spec'])
    verify_probe_spec(spec)
    require(analysis['test_maps'] == [11, 12, 13] and analysis['distinct_tasks_per_map'] == 20 and
            analysis['test_episodes'] == 360 and analysis['bootstrap']['draws'] == 2000 and
            analysis['bootstrap']['seed'] == 2026101002,
            'Fixed stratified held-out analysis design differs')
    directories = {k: relative_inside(root, v) for k, v in binding['directories'].items()}
    require({'bank_root', 'sac_dir'} <= set(directories), 'Missing bank/SAC input directories')
    sac = directories['sac_dir'] / 'SAC.py'
    require(sha256(sac) == SAC_SHA and sac.resolve() in manifest_files, 'Original SAC source differs/unsealed')
    output = relative_inside(root, binding['output_root'])
    for source in files.values():
        require(output != source and output not in source.parents, 'Output overlaps a sealed input file')
    for directory in directories.values():
        require(output != directory and output not in directory.parents and directory not in output.parents,
                'Output overlaps bank or SAC inputs')
    loader = import_file('followup_preserved_portable_loader', files['portable_loader'])
    exports = loader.verified_exports(files['exports'])
    require(exports['source_checkpoint_list_sha256'] == sha256(files['checkpoints']) and
            exports['source_audit_sha256'] == sha256(files['source_audit']),
            'Portable Q1 manifest source/audit binding differs')
    entries = exports.get('checkpoints', [])
    require(len(entries) == 6 and {int(e['seed']) for e in entries} == set(range(1, 7)),
            'Six unique portable Q1 export metadata entries required')
    for entry in entries:
        export = relative_inside(files['exports'].parent, entry['file'])
        require(export.is_file() and export.stat().st_size == entry['bytes'], 'Missing/incomplete portable Q1 file')
        require(entry['source_checkpoint_sha256'] == protocol['checkpoint_role_mapping'][str(entry['seed'])]['sha256'],
                'Portable Q1 source checkpoint identity differs')
    return binding, files, directories, output, manifest_path, loader


def build_args(root, stage, encoder_seed=None, split=None):
    root = Path(root).resolve()
    binding, files, directories, output, manifest_path, loader = load_binding(root)
    return SimpleNamespace(root=root, binding=root / 'binding.json', stage=stage,
                           encoder_seed=encoder_seed, split=split, device='cuda:0', input_mode='aligned',
                           out=output, spec=files['probe_spec'], protocol=files['protocol'],
                           analysis_spec=files['analysis_spec'], registry=files['registry'],
                           checkpoints=files['checkpoints'], sac_dir=directories['sac_dir'],
                           bank_root=directories['bank_root'], code_manifest=manifest_path,
                           _binding=binding, _files=files, _directories=directories, _loader=loader)


def verify_runner_receipt(args, now=None):
    supplied = os.environ.get('LIRA_B_RUNNER_RECEIPT', '')
    nonce = os.environ.get('LIRA_B_RUNNER_CAPABILITY', '')
    require(supplied and Path(supplied).resolve() == (args.root / 'state.json').resolve() and nonce,
            'Formal probe stages require the root supervisor capability')
    state = read_json(args.root / 'state.json')
    require(state.get('schema') == 'lira_event_coverage_v1' and state.get('status') == 'running' and
            state.get('pid') == os.getppid(), 'Formal parent must be the live root supervisor')
    require(state.get('protocol_sha256') == sha256(args.protocol) and
            state.get('code_manifest_sha256') == sha256(args.code_manifest) and
            state.get('capability_sha256') == hashlib.sha256(nonce.encode()).hexdigest(),
            'Supervisor protocol/code/capability binding differs')
    deadline = state.get('compute_deadline_unix')
    require(isinstance(deadline, (int, float)) and not isinstance(deadline, bool) and
            math.isfinite(deadline) and (time.time() if now is None else now) < deadline,
            'Root supervisor compute deadline expired/invalid')
    current = state.get('current_stage') or {}
    expected = {'kind': 'probe', 'stage': args.stage, 'encoder_seed': args.encoder_seed,
                'split': args.split, 'status': 'running'}
    require(all(current.get(k) == v for k, v in expected.items()), 'Supervisor current stage differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Only physical GPU0 is authorized')
    return state


def tree_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file()) if root.exists() else 0


def output_guard(args, required_bytes):
    existing = tree_bytes(args.out) + tree_bytes(args.bank_root)
    require(existing + required_bytes <= OUTPUT_LIMIT, '10 GiB generated bank/probe output ceiling exceeded')
    require(shutil.disk_usage(args.root).free >= required_bytes + RESERVE,
            'Allocation must retain at least 3 GiB free disk')


def disk_guard(args, spec, required_bytes):
    cached = sum(p.stat().st_size for p in (args.out / 'cache').rglob('features.npy')) if (args.out / 'cache').exists() else 0
    require(cached + required_bytes <= FEATURE_LIMIT, '8 GiB retained feature ceiling exceeded')
    output_guard(args, required_bytes)


def support_summary(episodes, index, split, np):
    require(split in GATES, 'Unknown support split')
    required_ep, required_tasks = GATES[split]
    tasks = {(int(e['map_id']), e['task_id']) for e in episodes if bool(e['collision'])}
    collision_ep = sum(bool(e['collision']) for e in episodes)
    valid = index['masks'][:, 2].astype(bool)
    require(index['masks'][valid].all(), 'Common m20 rows must be valid at all three horizons')
    by_behavior = []
    for seed in range(1, 7):
        ee = [e for e in episodes if int(e['behavior_seed']) == seed]
        collision_tasks = {(int(e['map_id']), e['task_id']) for e in ee if bool(e['collision'])}
        episode_ids = [i for i, e in enumerate(episodes) if int(e['behavior_seed']) == seed]
        behavior_valid = valid & np.isin(index['episode_index'], episode_ids)
        behavior_counts = []
        for j, horizon in enumerate((5, 10, 20)):
            y = index['targets'][behavior_valid, j]
            pos = int(y.sum())
            behavior_counts.append({'horizon': horizon, 'positive': pos, 'negative': len(y) - pos,
                                    'both_classes': 0 < pos < len(y)})
        by_behavior.append({'behavior_seed': seed, 'collision_episodes': sum(bool(e['collision']) for e in ee),
                            'different_collision_tasks': len(collision_tasks),
                            'at_least_two_collision_tasks': len(collision_tasks) >= 2,
                            'common_m20_class_counts': behavior_counts,
                            'all_horizons_have_both_classes': all(r['both_classes'] for r in behavior_counts)})
    counts = []
    for j, horizon in enumerate((5, 10, 20)):
        y = index['targets'][valid, j]
        pos = int(y.sum())
        counts.append({'horizon': horizon, 'positive': pos, 'negative': len(y) - pos,
                       'both_classes': 0 < pos < len(y)})
    passed = (collision_ep >= required_ep and len(tasks) >= required_tasks and
              all(x['at_least_two_collision_tasks'] and x['all_horizons_have_both_classes'] for x in by_behavior) and
              all(x['both_classes'] for x in counts))
    return {'split': split, 'passed': bool(passed), 'collision_episodes': collision_ep,
            'different_collision_geometries': len(tasks), 'minimum_collision_episodes': required_ep,
            'minimum_collision_geometries': required_tasks, 'by_behavior': by_behavior,
            'common_m20_class_counts': counts, 'no_adaptive_top_up': True}


def index_from_bank(bank, np):
    return {'targets': np.concatenate([e['targets'] for e in bank]),
            'masks': np.concatenate([e['masks'] for e in bank]),
            'episode_index': np.repeat(np.arange(len(bank), dtype=np.int32), [e['steps'] for e in bank])}


def checked_support(args, episodes, index, split, np):
    result = support_summary(episodes, index, split, np)
    require(result['passed'], 'Stop: insufficient fixed-budget event/class support for ' + split)
    return result


def pin_bank(args, split, bank_sha):
    path = args.out / 'bank_receipts' / (split + '.json')
    record = {'protocol_sha256': sha256(args.protocol), 'registry_sha256': sha256(args.registry),
              'split': split, 'manifest_sha256': bank_sha}
    if path.exists():
        require(read_json(path) == record, 'Bank changed after first frozen-readout access')
    else:
        write_new_json(path, record)


def loo_epoch_statistics(probabilities, index, episodes, rank_fn, np):
    """Delete each collision geometry and ALL of its six behavior episodes."""
    geometry_by_episode = [(int(e['map_id']), e['task_id']) for e in episodes]
    geometries = sorted({geometry_by_episode[i] for i, e in enumerate(episodes) if bool(e['collision'])})
    common = index['masks'][:, 2].astype(bool)
    require(bool(common.any()), 'Geometry-deletion diagnostic requires common-valid validation rows')
    # Sort once. After deletion, remove empty tie groups, producing exactly the
    # same group counts used by the preserved ranking_metrics on that subset.
    probability = probabilities[common, 2].astype(np.float64)
    target = index['targets'][common, 2].astype(np.int64)
    episode_index = index['episode_index'][common]
    order = np.argsort(probability, kind='mergesort')
    sorted_p, sorted_y, sorted_e = probability[order], target[order], episode_index[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    group_counts = np.diff(np.r_[starts, len(sorted_p)])
    group_positive = np.add.reduceat(sorted_y, starts)
    group_index = np.repeat(np.arange(len(starts)), group_counts)
    rows = []
    for geometry in geometries:
        deleted = [i for i, g in enumerate(geometry_by_episode) if g == geometry]
        require(len(deleted) == 6 and {int(episodes[i]['behavior_seed']) for i in deleted} == set(range(1, 7)),
                'LOO geometry must remove exactly six different behavior episodes')
        removed = np.isin(sorted_e, deleted)
        counts = group_counts - np.bincount(group_index[removed], minlength=len(starts))
        positives = group_positive - np.bincount(group_index[removed], weights=sorted_y[removed],
                                                minlength=len(starts)).astype(np.int64)
        retained = counts > 0
        counts, positives = counts[retained], positives[retained]
        n, positive = int(counts.sum()), int(positives.sum())
        ap = None
        if 0 < positive < n:
            suffix_positive = np.cumsum(positives[::-1])[::-1]
            suffix_count = np.cumsum(counts[::-1])[::-1]
            ap = float(np.sum(positives / positive * suffix_positive / suffix_count))
        rows.append({'map_id': geometry[0], 'task_id': geometry[1],
                     'removed_episode_ids': [episodes[i]['episode_id'] for i in deleted],
                     'remaining_common_samples': n, 'positive': positive,
                     'negative': n - positive, 'h20_AP': ap,
                     'class_defined': ap is not None})
    return rows


def record_epoch(args, folder, head, cell, epoch, ap, probabilities, index, episodes, np):
    verify_runner_receipt(args)
    from event_budget import execution_approval
    execution_approval(args.root)
    p = args._producer
    epoch_dir = folder / 'epoch_heads'
    epoch_dir.mkdir(exist_ok=True)
    target = epoch_dir / ('%s_epoch_%02d.npz' % (cell, epoch))
    require(not target.exists(), 'Epoch head already recorded; no silent epoch retry')
    p.save_head(target, head, epoch, ap, np)
    rows = loo_epoch_statistics(probabilities, index, episodes, p.ranking_metrics, np)
    record = {'encoder_seed': args.encoder_seed, 'cell': cell, 'epoch': epoch,
              'full_validation_h20_AP': ap, 'epoch_head': str(target.relative_to(folder)),
              'epoch_head_sha256': sha256(target), 'leave_one_collision_geometry': rows,
              'formal_selection_uses_full_validation_only': True}
    write_new_json(folder / 'epoch_diagnostics' / ('%s_epoch_%02d.json' % (cell, epoch)), record)


def summarize_loo(folder, cells, epochs):
    selected = {}
    for cell in cells:
        records = [read_json(folder / 'epoch_diagnostics' / ('%s_epoch_%02d.json' % (cell, epoch)))
                   for epoch in range(1, epochs + 1)]
        full = max(records, key=lambda r: (r['full_validation_h20_AP'], -r['epoch']))
        keys = [(r['map_id'], r['task_id']) for r in records[0]['leave_one_collision_geometry']]
        results = []
        for geometry in keys:
            candidates = []
            for record in records:
                row = next(r for r in record['leave_one_collision_geometry']
                           if (r['map_id'], r['task_id']) == geometry)
                if row['h20_AP'] is not None:
                    candidates.append((row['h20_AP'], record['epoch']))
            best = max(candidates, key=lambda pair: (pair[0], -pair[1])) if candidates else (None, None)
            results.append({'map_id': geometry[0], 'task_id': geometry[1], 'selected_epoch': best[1],
                            'validation_h20_AP': best[0], 'valid_epochs': len(candidates),
                            'invalid_epochs': epochs - len(candidates),
                            'selected_epoch_changed': best[1] != full['epoch'] if best[1] is not None else None})
        selected[cell] = {'formal_selected_epoch': full['epoch'],
                          'formal_validation_h20_AP': full['full_validation_h20_AP'],
                          'geometry_deletions': results,
                          'changed_selection_count': sum(r['selected_epoch_changed'] is True for r in results)}
    return {'schema': 'lira_validation_geometry_deletion_v1', 'selection_diagnostic_only': True,
            'formal_models_replaced': False, 'extra_training_epochs': 0, 'cells': selected}


def install_producer(args):
    p = import_file('followup_preserved_producer', args._files['producer'])
    original_registry = p.registry
    def registry(a, spec):
        rows = original_registry(a, spec)
        for split, count in [('train', 30), ('val', 20), ('test', 20)]:
            selected = [r for r in rows if r['split'] == split]
            require({int(r['map_id']) for r in selected} == {11, 12, 13},
                    'Follow-up tasks must remain in Maps11--13')
            for map_id in (11, 12, 13):
                require(len({r['task_id'] for r in selected if int(r['map_id']) == map_id}) == count,
                        'Wrong per-map fixed task geometry budget')
        return rows
    p.registry = registry
    original_identity = p.identity
    def identity(a):
        result = original_identity(a)
        result.update(script_sha256=sha256(SELF), adapter_sha256=sha256(SELF),
                      original_producer_sha256=PRODUCER_SHA, original_analyzer_sha256=ANALYZER_SHA,
                      portable_loader_sha256=LOADER_SHA, binding_sha256=sha256(a.binding),
                      code_manifest_sha256=sha256(a.code_manifest),
                      export_manifest_sha256=sha256(a._files['exports']),
                      source_audit_sha256=sha256(a._files['source_audit']),
                      input_mode='aligned', study_role=ROLE)
        if read_json(a.protocol).get('engineering_revision')=='v1.2':
            import source_bridge
            proof=source_bridge.verify(a.root)
            result.update(prospective_source_bridge_sha256=proof['bridge_sha256'],historical_full_file_equivalence_verified=False)
        return result
    p.identity = identity
    p.load_q1 = args._loader.portable_load_q1
    p.disk_guard = disk_guard
    original_verified_lock = p.verified_lock
    def verified_lock(a):
        lock = original_verified_lock(a)
        require({(int(r['encoder_seed']), r['cell']) for r in lock['models']} ==
                {(seed, cell) for seed in range(1, 7) for cell in CELLS},
                'Global lock must contain all twenty-four unique frozen readouts')
        reports = lock.get('training_records', [])
        require(len(reports) == 6 and {int(r['encoder_seed']) for r in reports} == set(range(1, 7)),
                'Global lock lacks six unique completed training reports')
        for record in reports:
            report = a.out / 'models' / ('encoder_%02d' % int(record['encoder_seed'])) / 'complete.json'
            require(sha256(report) == record['report_sha256'], 'Locked training report changed')
        for record in lock['models']:
            relative_inside(a.out, record['path'])
        return lock
    p.verified_lock = verified_lock
    original_load_bank = p.load_bank
    def load_bank(a, rows, np):
        bank, bank_sha = original_load_bank(a, rows, np)
        checked_support(a, bank, index_from_bank(bank, np), a.split, np)
        if a.split == 'test':
            aggregate = read_json(a.bank_root / 'test' / 'manifest.json')
            require(aggregate.get('model_lock_sha256') == sha256(a.out / 'model_lock.json'),
                    'Follow-up test collection must bind the new global model lock')
        pin_bank(a, a.split, bank_sha)
        return bank, bank_sha
    p.load_bank = load_bank
    old = 'actions = np.stack([bank[e]["actions"][t] for e, t in batch])'
    new = 'actions = aligned_actions(bank, batch, length, np)'
    source = inspect.getsource(p.extract)
    require(source.count(old) == 1, 'Original action-history extraction anchor changed')
    p.__dict__['aligned_actions'] = aligned_actions
    exec(compile(source.replace(old, new), str(SELF) + ':aligned_extract', 'exec'), p.__dict__)
    source = inspect.getsource(p.train)
    anchor = '                history.append({"epoch": epoch, "cell": cell, "row_order_sha256": epoch_hash,'
    require(source.count(anchor) == 1, 'Original epoch-diagnostic insertion anchor changed')
    insertion = ('                record_epoch(args, folder, head, cell, epoch, ap, vp, vi, vm["episodes"], np)\n')
    p.__dict__['record_epoch'] = record_epoch
    exec(compile(source.replace(anchor, insertion + anchor), str(SELF) + ':diagnostic_train', 'exec'), p.__dict__)
    original_train = p.train
    def train(a, np):
        # Fit is impossible until BOTH fixed train and validation populations pass.
        for split in ('train', 'val'):
            _, index, metadata = p.read_cache(a, split, np, need_features=False)
            checked_support(a, metadata['episodes'], index, split, np)
        original_train(a, np)
        folder = p.models_dir(a)
        summary = summarize_loo(folder, CELLS, 20)
        write_new_json(folder / 'validation_geometry_deletion.json', dict(identity(a), **summary))
        report_path = folder / 'complete.json'
        report = read_json(report_path)
        record_paths = sorted((folder / 'epoch_diagnostics').glob('*.json'))
        require(len(record_paths) == 80, 'Each of four cells must retain all twenty epoch diagnostics')
        report['epoch_diagnostics'] = [{'file': str(path.relative_to(folder)), 'sha256': sha256(path)} for path in record_paths]
        report['validation_geometry_deletion_sha256'] = sha256(folder / 'validation_geometry_deletion.json')
        p.write_json(report_path, report)
    p.train = train
    original_lock = p.lock_models
    def lock_models(a, np):
        for seed in range(1, 7):
            verify_epoch_artifacts(a, seed, np)
        original_lock(a, np)
    p.lock_models = lock_models
    args._producer = p
    return p


def aligned_actions(bank, batch, length, np):
    result = np.stack([bank[e]['actions'][t-length+1:t+1] for e, t in batch])
    require(result.shape == (len(batch), length, 2), 'Action tokens must align with observation history')
    return result


def verify_epoch_artifacts(args, seed, np):
    folder = args.out / 'models' / ('encoder_%02d' % seed)
    report = read_json(folder / 'complete.json')
    args._producer.check_identity(report, args)
    records = report.get('epoch_diagnostics', [])
    expected = {'%s_epoch_%02d.json' % (cell, epoch) for cell in CELLS for epoch in range(1, 21)}
    require(len(records) == 80 and {Path(r['file']).name for r in records} == expected,
            'Missing or duplicate epoch diagnostic records')
    for item in records:
        path = relative_inside(folder, item['file'])
        require(sha256(path) == item['sha256'], 'Epoch diagnostic changed')
        record = read_json(path)
        require(record['encoder_seed'] == seed and record['formal_selection_uses_full_validation_only'] is True,
                'Epoch diagnostic identity/role differs')
        head = relative_inside(folder, record['epoch_head'])
        require(sha256(head) == record['epoch_head_sha256'], 'Epoch head changed')
        with np.load(head, allow_pickle=False) as saved:
            require(saved['weight'].shape == (3, 1024) and saved['bias'].shape == (3,) and
                    np.isfinite(saved['weight']).all() and np.isfinite(saved['bias']).all() and
                    int(saved['selected_epoch']) == record['epoch'] and
                    float(saved['validation_h20_AP']) == record['full_validation_h20_AP'],
                    'Invalid retained epoch head/metadata')
    summary_path = folder / 'validation_geometry_deletion.json'
    require(sha256(summary_path) == report['validation_geometry_deletion_sha256'],
            'Geometry-deletion summary changed')
    require(read_json(summary_path) == dict(args._producer.identity(args), **summarize_loo(folder, CELLS, 20)),
            'Geometry-deletion selected-epoch summary is inconsistent')


def analysis_adapter(args, np):
    p = args._producer
    p.verified_lock(args)
    for seed in range(1, 7):
        verify_epoch_artifacts(args, seed, np)
    m = import_file('followup_preserved_analyzer', args._files['analyzer'])
    if read_json(args.protocol).get('engineering_revision')=='v1.2':
        import source_bridge
        source_bridge.verify(args.root)
        # Preserve the old encoder/export chain; only behavior whole-file IDs
        # use the newly audited inference snapshots. The source-seed coupling is
        # justified by exact actor/Q1/Q2/config canonical equality, never bytes.
        source=inspect.getsource(m.verify_source_roles)
        old='require(protocol.get("same_checkpoint_for_encoder_and_behavior") is True,"Joint-source bootstrap requires shared checkpoint roles")'
        new='require(protocol.get("paired_source_roles_verified_by_canonical_bridge") is True,"Canonical source-role bridge is required")'
        require(source.count(old)==1,'Original source-role verification anchor changed')
        exec(compile(source.replace(old,new),str(SELF)+':canonical_encoder_roles','exec'),m.__dict__)
        source=inspect.getsource(m.verify_behavior_sources)
        old='protocol=read_json(args.protocol);mapping=protocol["checkpoint_role_mapping"]'
        new='protocol=read_json(args.protocol);mapping=protocol["behavior_checkpoint_role_mapping"]'
        require(source.count(old)==1,'Original behavior source mapping anchor changed')
        exec(compile(source.replace(old,new),str(SELF)+':prospective_behavior_roles','exec'),m.__dict__)
    original_verify = m.verify_lock
    def verify_lock(a, spec):
        result = original_verify(a, spec)
        p.check_identity(result[0], args)
        return result
    m.verify_lock = verify_lock
    original_event_audit = m.event_audit
    def event_audit(episodes, index, spec, np):
        rows, result = original_event_audit(episodes, index, spec, np)
        support = checked_support(args, episodes, index, 'test', np)
        result['fixed_followup_support_gate'] = support
        result['event_support_sufficient'] = bool(result['event_support_sufficient'] and support['passed'])
        return rows, result
    m.event_audit = event_audit
    a = SimpleNamespace(probe_root=args.out, analysis_spec=args.analysis_spec, probe_spec=args.spec,
                        protocol=args.protocol, registry=args.registry, checkpoints=args.checkpoints,
                        bank_root=args.bank_root, producer_script=SELF, out=args.out / 'analysis', chunk_size=8)
    m.analyze(a, np)
    path = a.out / 'analysis_result.json'
    result = read_json(path)
    result.update(study_role=ROLE, input_mode='aligned', original_study_unchanged=True,
                  navigation_or_end_to_end_training_claimed=False,
                  validation_geometry_deletion_is_selection_diagnostic_only=True)
    if read_json(args.protocol).get('engineering_revision')=='v1.2':
        result.update(source_pairing_basis='Exact actor/Q1/Q2/schema/full-args canonical bridge; encoder export and prospective inference snapshot byte identities remain distinct.',historical_full_file_equivalence_verified=False)
    p.write_json(path, result)
    p.write_json(a.out / 'followup_identity.json', p.identity(args))


def validate_formal_runtime(torch, np):
    require(torch.__version__ == '2.4.1+cu121' and np.__version__ == '1.23.5',
            'Formal stages require original Torch 2.4.1+cu121 / NumPy 1.23.5')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Formal stages require exactly the one visible physical GPU0')


def check_action_budget(args):
    total = 0
    for split in ('train', 'val', 'test'):
        manifest = args.bank_root / split / 'manifest.json'
        if split == 'test' and not (args.out / 'model_lock.json').is_file():
            require(not manifest.exists(), 'Formal test collection exists before global head lock')
            continue
        if manifest.is_file():
            total += sum(int(e['steps']) for e in read_json(manifest)['episodes'])
    require(total <= FORMAL_ACTION_LIMIT, 'Formal recorded control-action budget exceeds 300000')


def claim_stage(args):
    tag = args.stage + ('_encoder_%02d' % args.encoder_seed if args.encoder_seed is not None else '')
    tag += '_' + args.split if args.split else ''
    write_new_json(args.out / 'stage_attempts' / (tag + '.json'),
                   dict(args._producer.identity(args), stage=args.stage, encoder_seed=args.encoder_seed,
                        split=args.split, parent_pid=os.getppid(), started_unix=time.time(),
                        instruction='Permanent attempt claim; failure is terminal, not permission to retry.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('stage', choices=('verify', 'extract', 'train', 'lock', 'evaluate', 'analysis'))
    parser.add_argument('--encoder-seed', type=int, choices=range(1, 7))
    parser.add_argument('--split', choices=('train', 'val', 'test'))
    cli = parser.parse_args()
    if cli.stage in ('extract', 'train', 'evaluate') and cli.encoder_seed is None:
        parser.error('This stage requires --encoder-seed')
    if cli.stage == 'extract' and cli.split is None:
        parser.error('Extraction requires --split')
    if cli.stage != 'extract' and cli.split is not None:
        parser.error('--split is only valid for extraction')
    if cli.stage in ('verify', 'lock', 'analysis') and cli.encoder_seed is not None:
        parser.error('This stage takes no encoder seed')
    args = build_args(cli.root, cli.stage, cli.encoder_seed, cli.split)
    p = install_producer(args)
    if cli.stage == 'verify':
        # Verification reads sealed metadata only; no output creation/torch/bank NPZ.
        p.registry(args, read_json(args.spec))
        print(json.dumps(p.identity(args), sort_keys=True))
        return
    verify_runner_receipt(args)
    import numpy as np
    import torch
    validate_formal_runtime(torch, np)
    if cli.split == 'test' or cli.stage in ('evaluate', 'analysis'):
        p.verified_lock(args)
    check_action_budget(args)
    output_guard(args, {'analysis': 128 * 1024**2, 'train': 32 * 1024**2,
                        'evaluate': 64 * 1024**2}.get(cli.stage, 2 * 1024**2))
    claim_stage(args)
    if cli.stage == 'analysis':
        analysis_adapter(args, np)
    else:
        {'extract': p.extract, 'train': p.train, 'lock': p.lock_models, 'evaluate': p.evaluate}[cli.stage](args, np)
    # The preserved lock loop temporarily visits all six encoder seeds.
    args.encoder_seed, args.split = cli.encoder_seed, cli.split
    verify_runner_receipt(args)


if __name__ == '__main__':
    main()
