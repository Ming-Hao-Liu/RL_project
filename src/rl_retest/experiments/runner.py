"""Freeze, execute, resume and verify experiment jobs."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ..evaluation import evaluate, evaluate_resampling
from ..model import FALLBACK
from ..model import best_stop
from ..bounds import perfect_information
from ..policy import Stop
from .records import dump_json, dump_jsonl, policy_from_data, policy_to_data, read_json, read_jsonl
from .tasks import ExperimentSpec, Task, contexts, create_tasks, jobs_for


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _source_digest():
    package = Path(__file__).resolve().parents[1]
    selected = [package / name for name in ('model.py', 'policy.py', 'bounds.py', 'evaluation.py')]
    selected += sorted((package / 'planners').glob('*.py'))
    selected += sorted((package / 'experiments').glob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(selected):
        digest.update(str(path.relative_to(package)).encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(config_path, suite, out):
    root = Path(out)
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'run already exists: {root}')
    config = read_json(config_path)
    if config['schema_version'] != 1 or suite not in config['suites']:
        raise ValueError('unsupported configuration or suite')
    tasks = create_tasks(ExperimentSpec(config, suite))
    jobs = [job for task in tasks for job in jobs_for(task, config, suite)]
    root.mkdir(parents=True, exist_ok=True)
    (root / 'records').mkdir(exist_ok=True)
    dump_jsonl(root / 'tasks.jsonl', [task.to_data() for task in tasks])
    dump_jsonl(root / 'jobs.jsonl', jobs)
    lock = Path(__file__).resolve().parents[3] / 'uv.lock'
    manifest = {'schema_version': 1, 'suite': suite, 'config': config, 'task_count': len(tasks),
                'job_count': len(jobs), 'tasks_sha256': _digest((root / 'tasks.jsonl').read_bytes()),
                'jobs_sha256': _digest((root / 'jobs.jsonl').read_bytes()), 'git_commit': _git_commit(),
                'source_sha256': _source_digest(),
                'config_sha256': _digest(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()),
                'lock_sha256': _digest(lock.read_bytes()) if lock.exists() else None,
                'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                'cpu': platform.processor(), 'cpu_count': os.cpu_count(),
                'thread_limits': {name: os.environ.get(name) for name in
                                  ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    dump_json(root / 'manifest.json', manifest)
    return manifest


def _load(root):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    if manifest.get('config_sha256') != _digest(json.dumps(manifest['config'], sort_keys=True, separators=(',', ':')).encode()):
        raise ValueError('configuration snapshot does not match manifest')
    if manifest.get('source_sha256') != _source_digest():
        raise ValueError('planning or evaluation source has changed; prepare a new run')
    lock = Path(__file__).resolve().parents[3] / 'uv.lock'
    if lock.exists() and manifest.get('lock_sha256') != _digest(lock.read_bytes()):
        raise ValueError('dependency lock has changed; prepare a new run')
    for name in ('tasks','jobs'):
        if _digest((root / f'{name}.jsonl').read_bytes()) != manifest[f'{name}_sha256']:
            raise ValueError(f'{name} file does not match manifest')
    tasks = {row['task_id']: Task.from_data(row) for row in read_jsonl(root / 'tasks.jsonl')}
    return manifest, tasks, read_jsonl(root / 'jobs.jsonl')


def _plan(problem, job):
    from ..planners.dynamic import solve_exact, solve_stop, solve_lookahead, solve_iterative
    from ..planners.blinkered import solve_blinkered
    from ..planners.bounded import solve_bounded
    method = job['method']
    if method == 'exact': return solve_exact(problem)
    if method == 'stop': return solve_stop(problem)
    if method == 'myopic': return solve_lookahead(problem, 1)
    if method == 'depth2': return solve_lookahead(problem, 2)
    if method == 'depth3': return solve_lookahead(problem, 3)
    if method == 'blinkered': return solve_blinkered(problem)
    if method == 'iterative': return solve_iterative(problem, job['budget'])
    if method == 'bounded': return solve_bounded(problem, job['budget'], job['epsilon'], prune=job.get('prune', True))
    raise ValueError(f'unknown method: {method}')


def _evaluate(policy, true_context, task, scenario):
    if scenario == 'type_resampling':
        return evaluate_resampling(policy, true_context, task.priors)
    return evaluate(policy, true_context)


def _execute(payload):
    task_data, config, job, contended = payload
    task = Task.from_data(task_data)
    start = time.perf_counter()
    problem, truth = contexts(task, config, job['memory'], job.get('assumed_current'))
    if problem is None:
        from ..planners.common import PlanResult
        empty = PlanResult(Stop(FALLBACK), 0.0, None, None, 0, 0, 0, 0, 'incompatible_history')
        keys = ([(budget, epsilon) for budget in job['budgets'] for epsilon in job['epsilons']]
                if job['method'] == 'bounded_trace' else
                [(budget, None) for budget in job['budgets']] if job['method'] == 'iterative_trace' else [None])
        plans = [(key, empty, time.perf_counter() - start) for key in keys]
    elif job['method'] == 'iterative_trace':
        from ..planners.dynamic import solve_iterative_trace
        initialization = time.perf_counter() - start
        snapshots = solve_iterative_trace(problem, job['budgets'])
        plans = [((budget, None), plan, initialization + elapsed) for budget, (plan, elapsed) in sorted(snapshots.items())]
    elif job['method'] == 'bounded_trace':
        from ..planners.bounded import solve_bounded_trace
        initialization = time.perf_counter() - start
        snapshots = solve_bounded_trace(problem, job['budgets'], job['epsilons'])
        plans = [(key, plan, initialization + elapsed) for key, (plan, elapsed) in sorted(snapshots.items())]
    else:
        plan = _plan(problem, job)
        plans = [(None, plan, time.perf_counter() - start)]
    records = []
    for key, plan, planning_seconds in plans:
        evaluation_start = time.perf_counter()
        actual = _evaluate(plan.policy, truth, task, job.get('scenario'))
        stop_value = None if problem is None else best_stop(problem.belief, problem.spec)[0]
        perfect_value = None if problem is None else perfect_information(problem.belief, problem.spec)
        evaluation_seconds = time.perf_counter() - evaluation_start
        record = {'schema_version': 1, **job, 'parameters': task.parameters, 'status': plan.status,
            'policy': policy_to_data(plan.policy), 'model_value': None if problem is None else plan.model_value,
            'lower': plan.lower, 'upper': plan.upper, 'evaluations': plan.evaluations,
            'cache_hits': plan.cache_hits, 'expansions': plan.expansions, 'action_scans': plan.action_scans,
            'planning_seconds': planning_seconds, 'evaluation_seconds': evaluation_seconds,
            'timing_contended': contended, 'initial_stop': stop_value,
            'perfect_information': perfect_value, **asdict(actual)}
        if key is not None:
            if job['method'] == 'bounded_trace':
                record.update(method='bounded', budget=key[0], epsilon=key[1])
            else:
                record.update(method='iterative', budget=key[0])
        records.append(record)
        for alias in job.get('aliases', []):
            records.append({**record, **alias, 'alias_of': {'method': job['method'], 'memory': job['memory']}})
    return records


def run(root, workers=1):
    if workers < 1:
        raise ValueError('workers must be positive')
    root = Path(root)
    manifest, tasks, jobs = _load(root)
    pending = []
    for job in jobs:
        path = root / 'records' / f"{job['job_id']}.jsonl"
        if path.exists() and read_jsonl(path)[0].get('status') != 'failed':
            continue
        pending.append(job)
    def payload(job):
        return tasks[job['task_id']].to_data(), manifest['config'], job, workers > 1
    failures = []
    with (root / 'run.log').open('a', encoding='utf-8') as log:
        log.write(json.dumps({'event':'start', 'jobs':len(pending), 'workers':workers, 'time':time.time()}) + '\n')
    def save(job, result):
        dump_jsonl(root / 'records' / f"{job['job_id']}.jsonl", result if isinstance(result, list) else [result])
        if isinstance(result, dict) and result.get('status') == 'failed':
            failures.append(result)
    if workers == 1:
        for job in pending:
            try:
                result = _execute(payload(job))
            except Exception as exc:
                result = {'schema_version':1, **job, 'status':'failed', 'error':f'{type(exc).__name__}: {exc}'}
            save(job,result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Limit queued tasks so large studies do not retain tens of thousands of futures.
            for start in range(0, len(pending), workers * 8):
                batch = pending[start:start + workers * 8]
                futures = [pool.submit(_execute, payload(job)) for job in batch]
                for job, future in zip(batch, futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'schema_version':1, **job, 'status':'failed', 'error':f'{type(exc).__name__}: {exc}'}
                    save(job,result)
    with (root / 'run.log').open('a', encoding='utf-8') as log:
        log.write(json.dumps({'event':'finish', 'jobs':len(pending), 'failures':failures, 'time':time.time()}) + '\n')
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; inspect records and run.log")
    return len(pending)


def verify(root):
    root = Path(root)
    manifest, tasks, jobs = _load(root)
    errors = []
    exact = {}
    def model_key(row):
        return row['task_id'], row['memory'], row.get('assumed_current')
    for file in (root / 'records').glob('*.jsonl'):
        for row in read_jsonl(file):
            if row.get('method') == 'exact' and row.get('model_value') is not None:
                exact[model_key(row)] = row['model_value']
    certificate_checks = 0
    for job in jobs:
        path = root / 'records' / f"{job['job_id']}.jsonl"
        if not path.exists():
            errors.append(f"{job['job_id']}: missing")
            continue
        records = read_jsonl(path)
        if not records:
            errors.append(f"{job['job_id']}: empty record")
            continue
        if records[0]['status'] == 'failed':
            errors.append(f"{job['job_id']}: {records[0]['error']}")
            continue
        if job['method'] == 'bounded_trace':
            expected = {(budget, epsilon) for budget in job['budgets'] for epsilon in job['epsilons']}
            actual_keys = [(record.get('budget'),record.get('epsilon')) for record in records]
            if len(actual_keys) != len(expected) or set(actual_keys) != expected:
                errors.append(f"{job['job_id']}: incomplete or duplicate checkpoints")
        elif job['method'] == 'iterative_trace':
            actual_budgets = [record.get('budget') for record in records]
            if len(actual_budgets) != len(job['budgets']) or set(actual_budgets) != set(job['budgets']):
                errors.append(f"{job['job_id']}: incomplete or duplicate checkpoints")
        else:
            expected = {(job['method'],job['memory'])} | {(a['method'],a['memory']) for a in job.get('aliases', [])}
            keys = [(record.get('method'),record.get('memory')) for record in records]
            if len(keys) != len(expected) or set(keys) != expected:
                errors.append(f"{job['job_id']}: incomplete or duplicate aliases")
        task = tasks[job['task_id']]
        for record in records:
            if record['job_id'] != job['job_id'] or record['task_id'] != job['task_id']:
                errors.append(f"{job['job_id']}: mismatched record")
                continue
            problem, truth = contexts(task, manifest['config'], record['memory'], job.get('assumed_current'))
            try:
                policy = policy_from_data(record['policy'])
                actual = _evaluate(policy,truth,task,job.get('scenario'))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{job['job_id']}: invalid policy: {exc}")
                continue
            for name,value in asdict(actual).items():
                if abs(value - record[name]) > 1e-10:
                    errors.append(f"{job['job_id']}: {name} mismatch")
            if problem is not None:
                model_context = type(truth)(problem.belief, problem.spec)
                model_value = evaluate(policy,model_context).expected_utility
                if abs(model_value-record['model_value']) > 1e-10:
                    errors.append(f"{job['job_id']}: model value mismatch")
                if record['lower'] is not None and abs(model_value-record['lower']) > 1e-10:
                    errors.append(f"{job['job_id']}: lower bound mismatch")
                if record['upper'] is not None and model_value > record['upper'] + 1e-10:
                    errors.append(f"{job['job_id']}: upper bound below policy")
            reference = exact.get(model_key(record))
            if record.get('method') == 'bounded' and reference is not None:
                certificate_checks += 1
                tolerance = 1e-10 * max(1, truth.spec.reward + truth.spec.horizon * truth.spec.cost)
                if not record['lower'] - tolerance <= reference <= record['upper'] + tolerance:
                    errors.append(f"{job['job_id']}: exact value outside certificate")
            if 'budget' in record and record['evaluations'] > record['budget']:
                errors.append(f"{job['job_id']}: budget exceeded")
            if (record['status'] == 'tolerance' and record['upper'] is not None and
                    record['upper'] - record['lower'] > record['epsilon'] * truth.spec.reward + 1e-10):
                errors.append(f"{job['job_id']}: tolerance not achieved")
    return {'expected':len(jobs), 'errors':errors, 'valid':not errors, 'certificate_checks':certificate_checks}
