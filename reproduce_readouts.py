#!/usr/bin/env python3
"""Recompute frozen-readout point metrics and saved-bootstrap intervals.
No neural fitting, simulator calls, or bootstrap resampling is performed.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def reproduce(root, analyzer_path, out):
    spec = importlib.util.spec_from_file_location('recorded_analyzer', analyzer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {}
    reference_index = None
    for seed in range(1, 7):
        for cell in module.CELLS:
            path = root / 'evaluation' / ('encoder_%02d' % seed) / (cell + '_scores.npz')
            with np.load(path, allow_pickle=False) as z:
                p, logits, targets, masks, episodes = [z[k] for k in ('probabilities', 'logits', 'targets', 'masks', 'episode_index')]
                current_index = tuple(z[k] for k in ('targets', 'masks', 'episode_index', 'timestep'))
                if reference_index is None:
                    reference_index = tuple(x.copy() for x in current_index)
                elif not all(np.array_equal(a, b) for a, b in zip(reference_index, current_index)):
                    raise ValueError('Sample/label identity differs: ' + str(path))
            weights = np.ones((1, int(episodes.max()) + 1))
            transforms = ('raw', 'cummax', 'pava') if cell.startswith('independent_') else ('raw',)
            for transform in transforms:
                pp = p if transform == 'raw' else (np.maximum.accumulate(p, axis=1) if transform == 'cummax' else module.pava3(p, np))
                mm = module.metric_arrays(pp, logits if transform == 'raw' else None, targets, masks, episodes, weights, np, 1)
                for (population, horizon, metric), value in mm.items():
                    key = (cell, transform, population, horizon, metric)
                    values.setdefault(key, []).append(float(value[0]))
    estimates = {key: float(np.mean(v)) for key, v in values.items()}
    draws_index = json.loads((root / 'analysis/bootstrap_array_index.json').read_text())
    lookup = {(r['family'], tuple(r['key'])): r['array'] for r in draws_index}
    checked = []
    max_error = 0.0
    with np.load(root / 'analysis/bootstrap_distributions.npz', allow_pickle=False) as saved:
        for family, filename in [('absolute', 'absolute_metrics.csv'), ('contrast', 'raw_contrasts.csv')]:
            for row in rows(root / 'analysis' / filename):
                population, horizon, metric = row['population'], int(row['horizon']), row['metric']
                if family == 'absolute':
                    key = (row['cell'], row['transform'], population, horizon, metric)
                    point = estimates[key]
                else:
                    key = (row['contrast'], population, horizon, metric)
                    point = sum(c * estimates[(cell, 'raw', population, horizon, metric)] for c, cell in zip(module.CONTRASTS[row['contrast']], module.CELLS))
                dd = saved[lookup[(family, key)]]
                finite = dd[np.isfinite(dd)]
                low, high = np.quantile(finite, [.025, .975], method='linear')
                errors = [abs(point - float(row['estimate'])), abs(low - float(row['ci_low'])), abs(high - float(row['ci_high']))]
                max_error = max(max_error, *errors)
                if max(errors) > 1e-9:
                    raise ValueError('Reference mismatch: ' + str(key) + ' errors=' + str(errors))
                if len(finite) != int(row['bootstrap_valid']) or len(dd) - len(finite) != int(row['bootstrap_invalid']):
                    raise ValueError('Bootstrap count mismatch: ' + str(key))
                checked.append(dict(family=family, key=list(key), estimate=point, ci_low=float(low), ci_high=float(high)))
    with np.load(root / 'analysis/bootstrap_episode_multiplicities.npz', allow_pickle=False) as z:
        if not np.array_equal(z['encoder_counts'], z['behavior_counts']):
            raise ValueError('Encoder and behaviour source counts are not paired')
    result = dict(status='PASS', source=str(root), selected_heads=24, estimates_and_intervals_checked=len(checked), max_abs_error=max_error, neural_execution=False, new_bootstrap_draws=False, results=checked)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True, help='Directory containing evaluation/ and analysis/')
    parser.add_argument('--analyzer', type=Path, default=Path(__file__).with_name('analyze_frozen_probe.py'))
    parser.add_argument('--out', type=Path, default=Path('reproduced_readouts.json'))
    args = parser.parse_args()
    reproduce(args.study, args.analyzer, args.out)
