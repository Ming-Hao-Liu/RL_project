"""Matched pruning timing and analysis of frozen, existing large tasks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter, process_time


THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
SCIENCE = ('model_value', 'lower', 'upper', 'expected_utility', 'expected_tests',
           'success_probability')
COUNTS = ('evaluations', 'expansions', 'cache_hits', 'action_scans')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(task_ids, repeats=3, seed=2026111704):
    """Balance first-method order within every round; freeze before timing."""
    ids = sorted(task_ids)
    if len(ids) % 2 or len(ids) != len(set(ids)) or not ids:
        raise ValueError('require an even number of unique tasks')
    rng = random.Random(seed)
    rows = []
    for repeat in range(repeats):
        order = list(ids)
        rng.shuffle(order)
        first = [True] * (len(ids) // 2) + [False] * (len(ids) // 2)
        rng.shuffle(first)
        rows.extend(dict(task_id=task, repeat=repeat, order=[flag, not flag])
                    for task, flag in zip(order, first))
    return rows


def pending_pairs(protocol_schedule, records):
    """Resume only complete pairs; malformed or partial records are rejected."""
    expected = {(p['task_id'], p['repeat']) for p in protocol_schedule}
    seen = {}
    for row in records:
        pair = row['task_id'], row['repeat']
        flag = row['prune']
        if pair not in expected or not isinstance(flag, bool) or flag in seen.get(pair, set()):
            raise ValueError('unexpected or duplicate timing record')
        seen.setdefault(pair, set()).add(flag)
    if any(flags != {False, True} for flags in seen.values()):
        raise ValueError('incomplete timing pair')
    return [pair for pair in protocol_schedule if (pair['task_id'], pair['repeat']) not in seen]


def same_science(left, right, tolerance=1.12e-10, compare_scans=True):
    if left['policy'] != right['policy'] or left['status'] != right['status']:
        raise ValueError('policy or termination status mismatch')
    for name in SCIENCE:
        if not math.isfinite(left[name]) or not math.isfinite(right[name]) or abs(left[name] - right[name]) > tolerance:
            raise ValueError(f'{name} mismatch')
    for name in COUNTS:
        if name != 'action_scans' or compare_scans:
            if left[name] != right[name]:
                raise ValueError(f'{name} mismatch')


def audit(source):
    """Validate saved policies and exact cohort identity, without planning."""
    from rl_retest.experiments.records import read_json, read_jsonl, policy_from_data
    from rl_retest.experiments.tasks import Task, contexts
    from rl_retest.evaluation import evaluate
    manifest = read_json(source / 'manifest.json')
    tasks = read_jsonl(source / 'tasks.jsonl')
    jobs = read_jsonl(source / 'jobs.jsonl')
    original = source.parent / 'large'
    selected = {t['task_id']: t for t in read_jsonl(original / 'tasks.jsonl')
                if (t['parameters']['k'], t['parameters']['horizon'], t['parameters']['n']) == (40, 6, 0)}
    if len(tasks) != 20 or len(jobs) != 40 or len(selected) != 20:
        raise ValueError('expected 20 existing tasks and 40 jobs')
    if {t['task_id']: t for t in tasks} != selected:
        raise ValueError('tasks differ from frozen large cohort')
    if manifest['suite'] != 'pruning':
        raise ValueError('expected pruning suite')
    for name in ('tasks', 'jobs'):
        if sha(source / f'{name}.jsonl') != manifest[f'{name}_sha256']:
            raise ValueError(f'{name} digest mismatch')
    old_config = read_json(original / 'manifest.json')['config']
    if manifest['config']['model'] != old_config['model']:
        raise ValueError('model differs from frozen large cohort')
    files = {source / 'records' / f"{j['job_id']}.jsonl" for j in jobs}
    if files != set((source / 'records').glob('*.jsonl')):
        raise ValueError('missing or unexpected job files')
    pairs = {task: {} for task in selected}
    inputs = [source/'manifest.json', source/'tasks.jsonl', source/'jobs.jsonl', *sorted(files),
              original/'manifest.json', original/'tasks.jsonl']
    for job in jobs:
        if (job['method'], job['memory'], job['budget'], job['epsilon']) != ('bounded', 'correct', 10000, 0.01):
            raise ValueError('unexpected method or budget')
        flag = job['prune']; task_id = job['task_id']
        if not isinstance(flag, bool) or task_id not in pairs or flag in pairs[task_id]:
            raise ValueError('duplicate or invalid method pair')
        rows = read_jsonl(source / 'records' / f"{job['job_id']}.jsonl")
        if len(rows) != 1:
            raise ValueError('expected one record per job')
        row = rows[0]
        if any(row.get(k) != v for k, v in job.items()):
            raise ValueError('record/job mismatch')
        if row['status'] not in ('complete', 'tolerance', 'budget_exhausted'):
            raise ValueError('invalid completion status')
        task = Task.from_data(selected[task_id]); problem, truth = contexts(task, manifest['config'])
        result = evaluate(policy_from_data(row['policy']), truth)
        for name in ('expected_utility', 'expected_tests', 'success_probability'):
            if abs(getattr(result, name) - row[name]) > 1.12e-10:
                raise ValueError('fixed policy evaluation mismatch')
        if abs(result.expected_utility - row['lower']) > 1.12e-10 or abs(row['model_value'] - row['lower']) > 1.12e-10:
            raise ValueError('policy does not attain lower bound')
        if row['upper'] + 1.12e-10 < row['lower'] or row['evaluations'] > 10000:
            raise ValueError('invalid bound or budget')
        if row['status'] == 'tolerance' and row['upper']-row['lower'] > 0.01+1.12e-10:
            raise ValueError('invalid tolerance status')
        pairs[task_id][flag] = row
    for pair in pairs.values():
        if set(pair) != {False, True}:
            raise ValueError('incomplete method pair')
        same_science(pair[True], pair[False], compare_scans=False)
    old_rows = {}
    for path in sorted((original/'records').glob('*.jsonl')):
        for row in read_jsonl(path):
            if (row['task_id'] in selected and row['method'] == 'bounded'
                    and row.get('budget') == 10000 and row.get('epsilon') == 0.01):
                if row['task_id'] in old_rows:
                    raise ValueError('duplicate frozen reference')
                old_rows[row['task_id']] = row
                inputs.append(path)
    if set(old_rows) != set(selected):
        raise ValueError('missing frozen references')
    for task_id, old in old_rows.items():
        same_science(pairs[task_id][True], old)
    return manifest, selected, pairs, {str(p): sha(p) for p in sorted(set(inputs))}


def timing_protocol(tasks):
    return dict(repeats=3, order_seed=2026111704, bootstrap_seed=2026111705,
                    bootstrap_samples=5000, budget=10000, epsilon=0.01,
                    schedule=schedule(tasks), task_ids=sorted(tasks),
                    primary='geometric mean of per-task median wall-time on/off ratios',
                    scope='existing 20 K40 B6 n0 tasks; serial; fresh caches; context/evaluation/I/O excluded')


def measure(source, out):
    from dataclasses import asdict
    from rl_retest.experiments.records import read_json, dump_json, policy_to_data
    from rl_retest.experiments.tasks import Task, contexts
    from rl_retest.planners.bounded import solve_bounded
    from rl_retest.evaluation import evaluate
    manifest, tasks, pairs, inputs = audit(source)
    protocol = timing_protocol(tasks)
    environment = dict(platform=platform.platform(), python=platform.python_version(),
                       threads={name: os.environ.get(name) for name in THREADS})
    if any(v != '1' for v in environment['threads'].values()):
        raise ValueError('all numerical thread limits must be 1')
    source_hashes = {'script': sha(__file__), 'uv.lock': sha(Path(__file__).resolve().parents[1]/'uv.lock')}
    from rl_retest.experiments.runner import _source_digest
    source_hashes['production'] = _source_digest()
    if manifest['lock_sha256'] != source_hashes['uv.lock']:
        raise ValueError('dependency lock differs from prepared run')
    if manifest['source_sha256'] != source_hashes['production']:
        raise ValueError('production sources differ from prepared run')
    if out.exists():
        data = read_json(out)
        for key, value in [('protocol', protocol), ('environment', environment),
                           ('source_hashes', source_hashes), ('inputs_sha256', inputs)]:
            if data[key] != value:
                raise ValueError(f'cannot resume: {key} changed')
    else:
        data = dict(protocol=protocol, environment=environment, source_hashes=source_hashes,
                    inputs_sha256=inputs, commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    sessions=[], records=[])
        dump_json(out, data)
    pending = pending_pairs(protocol['schedule'], data['records'])
    if not pending:
        print('120 existing matched measurements; nothing to resume')
        return
    session = len(data['sessions'])
    data['sessions'].append(dict(id=session, utc=datetime.now(timezone.utc).isoformat(), load=os.getloadavg()))
    dump_json(out, data)
    for item in pending:
        task_id = item['task_id']
        problem, truth = contexts(Task.from_data(tasks[task_id]), manifest['config'])
        completed = []
        for position, flag in enumerate(item['order']):
            wall_start = perf_counter(); cpu_start = process_time()
            plan = solve_bounded(problem, 10000, 0.01, prune=flag)
            cpu = process_time() - cpu_start; wall = perf_counter() - wall_start
            evaluation_start = perf_counter(); result = evaluate(plan.policy, truth)
            evaluation_wall = perf_counter() - evaluation_start
            current = dict(policy=policy_to_data(plan.policy), model_value=plan.model_value,
                           lower=plan.lower, upper=plan.upper, status=plan.status,
                           **{name: getattr(plan, name) for name in COUNTS}, **asdict(result))
            same_science(current, pairs[task_id][flag])
            policy_hash = hashlib.sha256(json.dumps(current.pop('policy'), sort_keys=True).encode()).hexdigest()
            completed.append(dict(task_id=task_id, repeat=item['repeat'], prune=flag, position=position,
                                  session=session, planning_wall=wall, planning_cpu=cpu,
                                  evaluation_wall=evaluation_wall, policy_sha256=policy_hash,
                                  matches_saved=True, **current))
        data['records'].extend(completed)
        dump_json(out, data)
    print('120 matched timings; policies, bounds, values and counts match saved records')


def summarize(rows, seed=2026111705):
    """Bootstrap task pairs, never pool repeated timings as independent tasks."""
    import numpy as np
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('require unique task pairs')
    on, off = (np.array([r[f'{side}_wall'] for r in rows]) for side in ('on', 'off'))
    if not np.all(np.isfinite(on) & np.isfinite(off) & (on > 0) & (off > 0)):
        raise ValueError('wall times must be positive and finite')
    rng = np.random.default_rng(seed); indices = rng.integers(0, len(rows), (5000, len(rows)))
    log_ratio = np.log(on/off); delta = (on-off)*1000
    output = []
    for metric, value, draws, unit in [
            ('wall_ratio', np.exp(log_ratio.mean()), np.exp(log_ratio[indices].mean(axis=1)), 'on/off'),
            ('mean_wall_delta_ms', delta.mean(), delta[indices].mean(axis=1), 'ms')]:
        lo, hi = np.quantile(draws, [0.025, 0.975])
        output.append(dict(metric=metric, estimate=float(value), ci_low=float(lo), ci_high=float(hi),
                           unit=unit, task_count=len(rows)))
    for side in ('on', 'off'):
        for clock in ('wall', 'cpu'):
            values = np.array([r[f'{side}_{clock}'] for r in rows])*1000
            for label, quantile in [('median', 0.5), ('p90', 0.9)]:
                output.append(dict(metric=f'{side}_{clock}_{label}', estimate=float(np.quantile(values, quantile)),
                                   ci_low=None, ci_high=None, unit='ms', task_count=len(rows)))
    return output


def analyze(source, out):
    import numpy as np
    from rl_retest.experiments.records import read_json, dump_json
    from scripts.analyze_study import write_csv
    manifest, tasks, pairs, inputs = audit(source)
    timing = read_json(source/'timing.json')
    expected_schedule = schedule(tasks)
    if timing['protocol'] != timing_protocol(tasks) or timing['inputs_sha256'] != inputs:
        raise ValueError('timing protocol or input hashes differ')
    if (timing['source_hashes']['production'] != manifest['source_sha256']
            or timing['source_hashes']['uv.lock'] != manifest['lock_sha256']
            or any(timing['environment']['threads'].get(k) != '1' for k in THREADS)):
        raise ValueError('timing source, dependency or thread contract mismatch')
    if pending_pairs(expected_schedule, timing['records']):
        raise ValueError('incomplete matched timing')
    input_hashes = {**inputs, str(source/'timing.json'): sha(source/'timing.json')}
    by_task = {task: {True: [], False: []} for task in tasks}
    for row in timing['records']:
        saved = pairs[row['task_id']][row['prune']]
        policy_hash = hashlib.sha256(json.dumps(saved['policy'], sort_keys=True).encode()).hexdigest()
        if row['policy_sha256'] != policy_hash or not row['matches_saved']:
            raise ValueError('timing policy mismatch')
        same_science({**row, 'policy': saved['policy']}, saved)
        item = next(p for p in expected_schedule if (p['task_id'],p['repeat']) == (row['task_id'],row['repeat']))
        if item['order'][row['position']] != row['prune']:
            raise ValueError('timing order mismatch')
        by_task[row['task_id']][row['prune']].append(row)
    rows = []
    for task_id in sorted(tasks):
        a, b = pairs[task_id][True], pairs[task_id][False]
        row = dict(task_id=task_id, policies_equal=a['policy']==b['policy'])
        for side, flag in [('on', True), ('off', False)]:
            for clock in ('wall', 'cpu'):
                row[f'{side}_{clock}'] = float(np.median([r[f'planning_{clock}'] for r in by_task[task_id][flag]]))
            for name in (*SCIENCE, *COUNTS, 'status'):
                row[f'{side}_{name}'] = pairs[task_id][flag][name]
            row[f'{side}_width'] = pairs[task_id][flag]['upper']-pairs[task_id][flag]['lower']
        row.update(wall_ratio=row['on_wall']/row['off_wall'],
                   wall_delta_ms=1000*(row['on_wall']-row['off_wall']),
                   action_scans_delta=a['action_scans']-b['action_scans'])
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    for side in ('on', 'off'):
        for name in COUNTS:
            summary.append(dict(metric=f'{side}_mean_{name}', estimate=float(np.mean([r[f'{side}_{name}'] for r in rows])),
                                ci_low=None, ci_high=None, unit='count', task_count=20))
        summary.append(dict(metric=f'{side}_tolerance_tasks', estimate=sum(r[f'{side}_width'] <= 0.01 for r in rows),
                            ci_low=None, ci_high=None, unit='tasks', task_count=20))
    write_csv(out/'paired_tasks.csv', rows); write_csv(out/'summary.csv', summary)
    dump_json(out/'verification.json', dict(valid=True, tasks=20, policies=40, matched_measurements=120,
              fixed_policy_evaluations=40, original_pruned_matches=20, pair_science_matches=20,
              exact_optimum_checks=0, numerical_tolerance=1.12e-10))
    dump_json(out/'input_sha256.json', input_hashes)
    dump_json(out/'provenance.json', dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              script_sha256=sha(__file__), generation_commit=manifest['git_commit'],
              timing_commit=timing['commit'], protocol=timing['protocol'],
              interpretation='Exploratory existing-cohort ablation; action_scans counts non-skipped candidate/backup processing, not all loops or operations'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    off = np.array([r['off_wall']*1000 for r in rows]); on = np.array([r['on_wall']*1000 for r in rows])
    axes[0].scatter(off, on); low=min(off.min(),on.min())*.95; high=max(off.max(),on.max())*1.05
    axes[0].plot([low, high], [low, high], '--', color='gray')
    axes[0].set(xlabel='Explicit pruning OFF (ms)', ylabel='Explicit pruning ON (ms)', title='Matched task median wall time')
    axes[1].bar(range(20), [r['action_scans_delta'] for r in rows])
    axes[1].set(xlabel='Task (fixed ID order)', ylabel='Counted action processing: ON - OFF', title='Not total loop iterations or CPU work')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps(summary[:2], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('time', 'analyze'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for name in THREADS:
        os.environ[name] = '1'
    if args.command == 'time':
        measure(args.source, args.out)
    else:
        analyze(args.source, args.out)


if __name__ == '__main__':
    main()
