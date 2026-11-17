"""Frozen mechanism and finite-calibration experiments; per-job atomic resume."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import numpy as np
from rl_retest.model import ModelSpec, PlanningProblem, EvaluationContext, best_stop
from rl_retest.experiments.records import dump_json, dump_jsonl, read_json, read_jsonl, policy_to_data
from rl_retest.experiments.tasks import Task, contexts
from rl_retest.experiments.calibration import calibrate, plugin_problem
from rl_retest.planners.dynamic import solve_exact, solve_lookahead
from rl_retest.planners.blinkered import solve_blinkered
from rl_retest.planners.bounded import solve_bounded, solve_bounded_warm
from rl_retest.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__), ROOT/'configs/research.json']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def record(problem, method, truth=None, budget=10000, epsilon=0):
    start = perf_counter()
    seed = None
    if method == 'exact':
        plan = solve_exact(problem)
    elif method == 'myopic':
        plan = solve_lookahead(problem, 1)
    elif method == 'blinkered':
        plan = solve_blinkered(problem)
    elif method == 'cold':
        plan = solve_bounded(problem, budget, epsilon)
    elif method == 'warm':
        plan, seed = solve_bounded_warm(problem, budget, epsilon)
    else:
        raise ValueError(method)
    planning_seconds = perf_counter()-start
    start = perf_counter()
    true = truth or EvaluationContext(problem.belief, problem.spec)
    ev = evaluate(plan.policy, true)
    model = evaluate(plan.policy, EvaluationContext(problem.belief, problem.spec))
    assert abs(model.expected_utility-plan.model_value) <= 1e-10
    if truth is None:
        assert abs(ev.expected_utility-plan.model_value) <= 1e-10
    if method in ('warm', 'cold'):
        assert plan.evaluations <= budget
    return {**asdict(plan), 'policy': policy_to_data(plan.policy), **asdict(ev),
            'method': method, 'initialization': seed, 'planning_seconds': planning_seconds,
            'evaluation_seconds': perf_counter()-start}


def prepare(source, out, protocol):
    existing = out/'manifest.json'
    if existing.exists():
        manifest = read_json(existing)
        assert manifest['protocol'] == protocol and manifest['source_files'] == fingerprint()
        assert manifest['main_tasks_sha256'] == hashlib.sha256((source/'main/tasks.jsonl').read_bytes()).hexdigest()
        return
    selection = protocol['calibration']['selection']
    tasks = [r for r in read_jsonl(source/'main/tasks.jsonl') if all(
        r['parameters'][k] in v if isinstance(v, list) else r['parameters'][k] == v
        for k,v in selection.items())]
    assert len(tasks) == 36
    jobs = []
    grid = protocol['grid']
    pairs = list(itertools.product(np.linspace(grid['prior_start'], grid['prior_stop'], grid['points']),
                                   np.linspace(grid['cost_start'], grid['cost_stop'], grid['points'])))
    pairs += [tuple(x) for x in grid['anchors'] if tuple(x) not in pairs]
    for j,(q,cost) in enumerate(pairs):
        for h in grid['horizons']:
            jobs.append(dict(id=f'grid-{j:04}-{h}', group='grid', q=float(q), cost=float(cost), horizon=h))
    for k,h,profile in itertools.product(protocol['deep']['k'], protocol['deep']['horizons'], protocol['deep']['profiles']):
        jobs.append(dict(id=f'deep-{k}-{h}-{profile["name"]}', group='deep', k=k, horizon=h, **profile))
    for m,rep in itertools.product(protocol['calibration']['sizes'], range(protocol['calibration']['repeats'])):
        jobs.append(dict(id=f'calibration-{m}-{rep:02}', group='calibration', size=m, repeat=rep))
    manifest = {'schema_version':1, 'protocol': protocol, 'source_files': fingerprint(),
                'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'main_tasks_sha256': hashlib.sha256((source/'main/tasks.jsonl').read_bytes()).hexdigest(),
                'platform': platform.platform(), 'cpus': os.cpu_count(), 'load': os.getloadavg(),
                'workers': protocol['workers'], 'threads': {k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
    dump_jsonl(out/'tasks.jsonl', tasks)
    dump_jsonl(out/'jobs.jsonl', jobs)
    for name in ('tasks','jobs'):
        manifest[f'{name}_sha256'] = hashlib.sha256((out/f'{name}.jsonl').read_bytes()).hexdigest()
    dump_json(out/'manifest.json', manifest)


def execute(args):
    job, out, protocol, tasks, config = args
    path = out/'records'/f'{job["id"]}.json'
    if path.exists():
        return job['id']
    if job['group'] in ('grid','deep'):
        qs = [job['q']] if job['group']=='grid' else job['q'][:job['k']]
        problem = PlanningProblem(tuple((0.,q,1-q) for q in qs),
            ModelSpec(1.,job['cost'],job['horizon'],.5,(0.,.8,.05),()))
        methods = ('exact','myopic') if job['group']=='grid' else ('exact','myopic','blinkered','cold','warm')
        records = [record(problem, m, budget=protocol['deep']['budget'], epsilon=protocol['deep']['epsilon']) for m in methods]
        exact = records[0]['expected_utility']
        for r in records:
            assert r['expected_utility'] <= exact+1e-10
            if r['upper'] is not None:
                assert r['lower']-1e-10 <= exact <= r['upper']+1e-10
        result = {'job': job, 'problem': asdict(problem), 'records': records}
    else:
        cal = calibrate(protocol['calibration']['seed'], job['size'], job['repeat'])
        rows = []
        for raw in tasks:
            task = Task.from_data(raw)
            _,truth = contexts(task,config)
            problem = plugin_problem(task,truth.spec,cal['estimates'])
            for method in ('exact','myopic'):
                rows.append({'task_id':task.task_id, 'component':'both',
                             'planning_belief':problem.belief, 'planning_spec':asdict(problem.spec),
                             **record(problem,method,truth)})
            diag=protocol['calibration']['diagnostic_slice']
            if job['size'] in diag['sizes'] and all(
                task.parameters[k] in v if isinstance(v,list) else task.parameters[k]==v
                for k,v in diag.items() if k!='sizes'):
                for component in ('history','future'):
                    problem=plugin_problem(task,truth.spec,cal['estimates'],component)
                    rows.append({'task_id':task.task_id, 'component':component,
                                 'planning_belief':problem.belief, 'planning_spec':asdict(problem.spec),
                                 **record(problem,'exact',truth)})
        result = {'job': job, 'calibration': cal, 'records': rows}
    dump_json(path,result)
    return job['id']


def run(source,out,workers,group):
    protocol=read_json(ROOT/'configs/research.json')
    prepare(source,out,protocol)
    jobs=read_jsonl(out/'jobs.jsonl')
    tasks=read_jsonl(out/'tasks.jsonl')
    config=read_json(source/'main/manifest.json')['config']
    # True-table references are saved once, not recomputed in calibration repeats.
    if not (out/'truth.json').exists():
        truth={}
        for raw in tasks:
            t=Task.from_data(raw);problem,ctx=contexts(t,config)
            truth[t.task_id]=record(problem,'exact',ctx)
        dump_json(out/'truth.json',truth)
    pending=[j for j in jobs if (group=='all' or j['group']==group) and not (out/'records'/f'{j["id"]}.json').exists()]
    args=[(j,out,protocol,tasks,config) for j in pending]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,name in enumerate(pool.map(execute,args)):
            if (i+1)%50==0 or i+1==len(args):print(f'{i+1}/{len(args)} {name}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data')
    p.add_argument('--out',type=Path,default=ROOT/'data/research')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--group',choices=('all','grid','deep','calibration'),default='all')
    a=p.parse_args();run(a.source,a.out,a.workers,a.group)
