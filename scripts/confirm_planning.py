"""Frozen independent comparison of root incumbents and planning costs."""
import argparse
from dataclasses import replace, asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter, process_time

from rl_retest.experiments.tasks import ExperimentSpec, create_tasks, contexts
from rl_retest.experiments.records import policy_to_data
from rl_retest.experiments.runner import _source_digest
from rl_retest.planners.dynamic import solve_lookahead
from rl_retest.planners.blinkered import solve_blinkered
from rl_retest.planners.bounded import solve_bounded, solve_bounded_warm
from rl_retest.evaluation import evaluate


def invoke(problem, method, budget, epsilon):
    start, cpu = perf_counter(), process_time()
    seed = None
    if method == 'warm':
        plan, seed = solve_bounded_warm(problem, budget, epsilon)
    elif method == 'cold':
        plan = solve_bounded(problem, budget, epsilon)
    elif method == 'myopic':
        plan = solve_lookahead(problem, 1)
    else:
        plan = solve_blinkered(problem)
    return plan, seed, perf_counter()-start, process_time()-cpu


def run(out):
    protocol = json.loads(Path('configs/planning.json').read_text())
    config = json.loads(Path('configs/study.json').read_text())
    assert protocol['seed'] not in config['seeds'].values()
    config['seeds']['formal'] = protocol['seed']
    config['suites']['large']['tasks'] = protocol['tasks_per_setting']
    tasks = [replace(t, task_id=f"planning-{protocol['seed']}-{t.task_id}")
             for t in create_tasks(ExperimentSpec(config, 'large'))]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol':protocol, 'config':config, 'source_sha256':_source_digest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'platform':platform.platform(), 'cpu_count':os.cpu_count(),
                'load':os.getloadavg(), 'timing_scope':'single worker, one numeric thread; external load not controlled'}
    path = out/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('protocol','config','source_sha256','script_sha256','commit'):
            assert old[key] == manifest[key], key
    else:
        path.write_text(json.dumps(manifest,indent=2))
        (out/'tasks.jsonl').write_text(''.join(json.dumps(t.to_data())+'\n' for t in tasks))
    records = out/'records';records.mkdir(exist_ok=True)
    for i,task in enumerate(tasks):
        path = records/f'{task.task_id}.json'
        if path.exists():continue
        start = perf_counter()
        problem, truth = contexts(task, config)
        preparation = perf_counter()-start
        rows = []
        methods = [('myopic',None,None),('blinkered',None,None)]
        methods += [(m,b,e) for m in ('cold','warm') for b in protocol['budgets'] for e in protocol['epsilons']]
        for method,budget,epsilon in methods:
            plan,seed,wall,cpu = invoke(problem,method,budget,epsilon)
            start = perf_counter();value = evaluate(plan.policy,truth);evaluation = perf_counter()-start
            assert abs(value.expected_utility-plan.model_value)<1e-10
            if budget is not None:
                assert plan.evaluations<=budget
                assert plan.lower-1e-10<=value.expected_utility<=plan.upper+1e-10
            rows.append({'task_id':task.task_id,'parameters':task.parameters,'method':method,
                         'budget':budget,'epsilon':epsilon,'policy':policy_to_data(plan.policy),
                         'value':value.expected_utility,'lower':plan.lower,'upper':plan.upper,
                         'evaluations':plan.evaluations,'status':plan.status,'initialization':seed,
                         'planning_wall':wall,'planning_cpu':cpu,'preparation_wall':preparation,
                         'evaluation_wall':evaluation,'end_to_end_wall':preparation+wall+evaluation})
        myopic = rows[0]['value']
        for row in rows:
            if row['method']=='warm' and row['initialization']['status']=='complete':
                assert row['value']>=myopic-1e-10
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(rows));tmp.replace(path)
        if (i+1)%20==0:print(f'Completed {i+1}/{len(tasks)} tasks',flush=True)
    # Timing is a separate sequential phase, without batch workers or shared caches.
    timing = out/'timing.jsonl'
    if not timing.exists():
        rows=[];rng=random.Random(protocol['timing']['order_seed'])
        for task in tasks:
            if task.parameters['index'] not in protocol['timing']['indices']:continue
            start=perf_counter();problem,truth=contexts(task,config);preparation=perf_counter()-start
            for repeat in range(protocol['timing']['repeats']):
                order=['cold','warm','myopic','blinkered'];rng.shuffle(order)
                for position,method in enumerate(order):
                    plan,seed,wall,cpu=invoke(problem,method,10000,0)
                    rows.append({'task_id':task.task_id,'parameters':task.parameters,'method':method,
                                 'repeat':repeat,'position':position,'wall':wall,'cpu':cpu,
                                 'preparation_wall':preparation,'value':plan.model_value,
                                 'evaluations':plan.evaluations,'status':plan.status})
        timing.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('Confirmation and timing complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('data/planning'))
    run(parser.parse_args().out)
