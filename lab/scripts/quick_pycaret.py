"""Small zero-label PyCaret screen; whole-benchmark families held out per fold.

This is a separate user-requested local diagnostic, not execution of a lab step.
See --help. Never reads answers, grading fields, item_features, or response traces.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import signal
import time
from pathlib import Path

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def text_features(text):
    text = '' if pd.isna(text) else str(text)
    n = max(len(text), 1)
    return {
        'log_chars': np.log1p(len(text)),
        'log_words': np.log1p(len(text.split())),
        'log_lines': np.log1p(text.count('\n')),
        'digit_fraction': sum(c.isdigit() for c in text) / n,
        'nonascii_fraction': sum(ord(c) > 127 for c in text) / n,
        'math_fraction': sum(c in '=+*/^<>∑∫√' for c in text) / n,
        'question_marks': min(text.count('?'), 30),
        'code_fences': min(text.count('```'), 20),
        'choice_markers': min(len(re.findall(r'(?:^|\n)\s*[A-E][.)]', text)), 10),
    }


def safe_feature_names(frame):
    """Rename encoded columns only; values/order stay fixed (LightGBM JSON rule)."""
    if isinstance(frame, pd.DataFrame):
        frame = frame.copy()
        frame.columns = [f'feature_{i}' for i in range(frame.shape[1])]
    return frame


def prepare(args):
    inv = json.loads((args.aims / 'data/inventory.json').read_text())
    root = args.aims / 'data/measurement-db'
    eligible = []
    for name, info in sorted(inv['dirs'].items()):
        d = root / name / 'formatted_tables'
        if info.get('eligible') and all((d / f'{t}.parquet').exists()
                                       for t in ['subjects', 'items']):
            r = next((d / f'{t}.parquet' for t in ['responses', 'response']
                      if (d / f'{t}.parquet').exists()), None)
            if r is not None:
                eligible.append((name, d, r))
    # Reserve an untouched name-selected lockbox before reading any outcomes.
    eligible.sort(key=lambda x: digest('pycaret-lockbox-42:' + x[0]))
    reserved = [x[0] for x in eligible[:max(1, len(eligible) // 10)]]
    existing = args.code / 'lab/results/splits.json'
    canonical_lockbox = []
    if existing.exists():
        canonical_lockbox = json.loads(existing.read_text())['lockbox']
    candidates = [x for x in eligible if x[0] not in reserved + canonical_lockbox]
    candidates.sort(key=lambda x: digest('pycaret-screen-42:' + x[0]))
    selected = candidates[:args.benchmarks]
    if args.selection_manifest:
        frozen = json.loads(args.selection_manifest.read_text())
        names = frozen['selected_benchmarks']
        lookup = {x[0]: x for x in eligible}
        assert not set(names).intersection(canonical_lockbox + reserved)
        selected = [lookup[name] for name in names]
    frames, audit = [], []
    for name, d, response_path in selected:
        subjects = pd.read_parquet(d / 'subjects.parquet')
        subjects = subjects.sort_values('subject_id').sample(
            n=min(args.subjects, len(subjects)), random_state=42)
        items = pd.read_parquet(d / 'items.parquet', columns=['item_id', 'content'])
        items = items.sort_values('item_id').sample(
            n=min(args.items, len(items)), random_state=42)
        # No outcome is involved in item/subject sampling.
        items['fingerprint'] = items.content.map(
            lambda x: digest(' '.join(str(x).lower().split())) if pd.notna(x)
            and len(str(x).strip()) >= 40 else '')
        features = pd.DataFrame([text_features(x) for x in items.content], index=items.index)
        items = pd.concat([items.drop(columns='content'), features], axis=1)
        chunks = []
        for batch in pq.ParquetFile(response_path).iter_batches(
                batch_size=131072, columns=['subject_id', 'item_id', 'response']):
            r = batch.to_pandas()
            r = r[r.subject_id.isin(subjects.subject_id) & r.item_id.isin(items.item_id)]
            r = r[r.response.isin([0.0, 1.0])]
            if len(r):
                chunks.append(r)
        if not chunks:
            audit.append({'benchmark': name, 'status': 'no sampled binary responses'})
            continue
        r = pd.concat(chunks, ignore_index=True)
        # One seeded random recorded trial per subject/item cell: repeats cannot leak.
        r = r.sample(frac=1, random_state=42).drop_duplicates(['subject_id', 'item_id'])
        fields = ['subject_id', 'normalized_name', 'provider', 'harness', 'reasoning_effort']
        s = subjects.reindex(columns=fields).fillna('')
        r = r.merge(s, on='subject_id', validate='many_to_one').merge(
            items, on='item_id', validate='many_to_one')
        r['benchmark'] = name
        r['pair'] = name + ':' + r.subject_id.astype(str)
        r['target'] = r.pop('response').astype(int)
        frames.append(r)
        audit.append({'benchmark': name, 'rows': len(r), 'subjects': r.subject_id.nunique()})
        print('PREP', name, len(r), flush=True)
    df = pd.concat(frames, ignore_index=True)
    # Keep identical normalized content in one source benchmark only; also collapse
    # same-item repeats across different subjects only through shared group splits.
    nonempty = df[df.fingerprint.ne('')]
    owners = nonempty.groupby('fingerprint').benchmark.min()
    keep = df.fingerprint.eq('') | df.benchmark.eq(df.fingerprint.map(owners))
    removed = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    # Conservative slug family rule catches common version/language suffixes.
    def family(name):
        name = re.sub(r'[_-](?:v?\d+(?:[._-]\d+)*|en|zh|test|val|dev)$', '', name)
        return name
    df['family'] = df.benchmark.map(family)
    assert not df[df.fingerprint.ne('')].groupby('fingerprint').benchmark.nunique().gt(1).any()
    df.to_parquet(args.output / 'sample.parquet', index=False)
    manifest = {'dataset_revision': inv['revision'], 'seed': 42,
                'reserved_unopened_benchmarks': reserved, 'selected_benchmarks': [x[0] for x in selected],
                'audit': audit, 'cross_benchmark_duplicate_rows_removed': removed,
                'rows': len(df), 'benchmarks': df.benchmark.nunique(),
                'sampling': 'up to 6 subjects x 80 items per benchmark; one trial per cell',
                'scope': 'exploratory zero-label feature screen; not competition ALC',
                'python': platform.python_version(), 'script_sha256': digest(Path(__file__).read_text())}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return df


def run(args):
    sample = pd.read_parquet(args.output / 'sample.parquet')
    canonical = args.code / 'lab/results/splits.json'
    if canonical.exists():
        lockbox = json.loads(canonical.read_text())['lockbox']
        assert not set(sample.benchmark).intersection(lockbox), 'Canonical lockbox overlap: do not run.'
    from pycaret.classification import ClassificationExperiment
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import FunctionTransformer
    import category_encoders as ce

    df = sample
    excluded = {'subject_id', 'item_id', 'fingerprint', 'benchmark', 'family', 'pair', 'target'}
    features = [x for x in df if x not in excluded]
    models = ['dummy', 'lr', 'dt', 'rf', 'et', 'gbc', 'lightgbm', 'nb', 'knn', 'ada']
    outputs, attempts, split_records = [], [], []
    if args.resume:
        outputs = [pd.read_parquet(args.output / 'predictions.parquet')]
        attempts = json.loads((args.output / 'attempts.json').read_text())
    def time_limit(signum, frame):
        raise TimeoutError('Quick-screen per-model limit: 120 seconds')
    signal.signal(signal.SIGALRM, time_limit)
    start = time.time()
    for fold, (tr, va) in enumerate(GroupKFold(3).split(df, groups=df.family)):
        train, valid = df.iloc[tr], df.iloc[va]
        assert set(train.benchmark).isdisjoint(valid.benchmark)
        assert set(train.family).isdisjoint(valid.family)
        assert (set(train.fingerprint) & set(valid.fingerprint)) <= {''}
        split_records.append({'fold': fold, 'train_benchmarks': sorted(train.benchmark.unique()),
                              'validation_benchmarks': sorted(valid.benchmark.unique())})
        (args.output / 'splits.json').write_text(json.dumps(split_records, indent=2))
        for name, probability in [('constant_0.5', .5), ('training_mean', train.target.mean())]:
            if args.resume and any(((o.model == name) & (o.fold == fold)).any() for o in outputs):
                continue
            control = valid[['benchmark', 'pair', 'target']].copy()
            control['prediction'] = probability
            control['model'] = name
            control['fold'] = fold
            outputs.append(control)
        e = ClassificationExperiment()
        e.setup(data=train[features + ['target']].reset_index(drop=True),
                test_data=valid[features + ['target']].reset_index(drop=True), target='target',
                index=False, session_id=42, n_jobs=2, use_gpu=False, html=False,
                verbose=False, system_log=str(args.output / f'pycaret-fold-{fold}.log'),
                log_experiment=False, memory=False, normalize=True,
                custom_pipeline=FunctionTransformer(safe_feature_names),
                max_encoding_ohe=100000, encoding_method=ce.OneHotEncoder(handle_unknown='value'),
                fold=2)
        for name in models:
            if any(a['fold'] == fold and a['model'] == name and a['status'] == 'ok' for a in attempts):
                continue
            t = time.time()
            print('FIT', fold, name, flush=True)
            try:
                signal.alarm(120)
                # No random-row cross-validation or tuning. The explicit test_data is
                # the outer held-out benchmark fold; preprocessing fits training only.
                model = e.create_model(name, cross_validation=False, verbose=False)
                predicted = e.predict_model(model, data=valid[features + ['target']].reset_index(drop=True),
                                            raw_score=True, round=8, verbose=False)
                p = predicted['prediction_score_1'].to_numpy(dtype=float)
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                result = valid[['benchmark', 'pair', 'target']].copy()
                result['prediction'] = p
                result['model'] = name
                result['fold'] = fold
                outputs.append(result)
                record = {'fold': fold, 'model': name, 'status': 'ok', 'seconds': time.time() - t,
                          'estimator': repr(model), 'parameters': repr(model.get_params())}
            except Exception as exc:
                record = {'fold': fold, 'model': name, 'status': 'failed',
                          'error': repr(exc), 'seconds': time.time() - t}
            finally:
                signal.alarm(0)
            attempts.append(record)
            print('DONE', {k: v for k, v in record.items() if k not in ['estimator', 'parameters']}, flush=True)
            pd.concat(outputs, ignore_index=True).to_parquet(args.output / 'predictions.parquet', index=False)
            (args.output / 'attempts.json').write_text(json.dumps(attempts, indent=2))
    out = pd.concat(outputs, ignore_index=True)
    rows = []
    for name, g in out.groupby('model'):
        if g.fold.nunique() != 3:
            continue
        y, p = g.target, g.prediction
        sq = (p - y) ** 2
        pairs = g.assign(sq=sq).groupby(['benchmark', 'pair']).sq.mean()
        rows.append({'model': name, 'pair_brier': pairs.mean(), 'row_brier': sq.mean(),
                     'benchmark_brier': pairs.groupby(level=0).mean().mean(),
                     'auc': roc_auc_score(y, p), 'accuracy': ((p >= .5) == y).mean(),
                     'log_loss': log_loss(y, np.clip(p, 1e-6, 1-1e-6)),
                     'fit_seconds': sum(a['seconds'] for a in attempts if a['model'] == name),
                     'folds': 3, 'rows': len(g), 'pairs': len(pairs)})
    score = pd.DataFrame(rows).sort_values('pair_brier')
    score.to_csv(args.output / 'leaderboard.csv', index=False)
    (args.output / 'splits.json').write_text(json.dumps(split_records, indent=2))
    (args.output / 'runtime.json').write_text(json.dumps({'total_seconds': time.time()-start,
                                                        'features': features}, indent=2))
    print(score.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aims', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmarks', type=int, default=60)
    parser.add_argument('--subjects', type=int, default=6)
    parser.add_argument('--items', type=int, default=80)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Retry failed fits, preserving successful predictions/attempts')
    parser.add_argument('--selection-manifest', type=Path, help='Freeze benchmark selection for exact regeneration')
    args = parser.parse_args()
    args.aims = args.aims.resolve()
    args.code = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / 'sample.parquet').exists():
        prepare(args)
    if not args.prepare_only:
        run(args)
