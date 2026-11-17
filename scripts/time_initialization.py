"""Supplement missing phase timing on eight fixed existing tasks, serially."""
import argparse
from pathlib import Path
from time import perf_counter, process_time
from rl_retest.experiments.records import read_json,read_jsonl,dump_json,policy_to_data
from rl_retest.experiments.tasks import Task,contexts
from rl_retest.planners.bounded import solve_bounded_warm
from rl_retest.evaluation import evaluate


def main(source,out):
    protocol=read_json(Path('configs/research.json'))['timing']
    config=read_json(source/'planning/manifest.json')['config']
    rows=[]
    for raw in read_jsonl(source/'planning/tasks.jsonl'):
        if raw['parameters']['index'] not in protocol['planning_indices']:continue
        task=Task.from_data(raw);problem,truth=contexts(task,config)
        old=next(r for r in read_json(source/'planning/records'/f'{task.task_id}.json') if r['method']=='warm' and r['budget']==protocol['budget'] and r['epsilon']==protocol['epsilon'])
        for repeat in range(protocol['repeats']):
            start=perf_counter();cpu=process_time()
            plan,init=solve_bounded_warm(problem,protocol['budget'],protocol['epsilon'])
            wall=perf_counter()-start;cpu=process_time()-cpu
            start=perf_counter();ev=evaluate(plan.policy,truth);evaluation=perf_counter()-start
            assert policy_to_data(plan.policy)==old['policy'] and plan.evaluations==old['evaluations']
            assert abs(ev.expected_utility-old['value'])<1e-10
            rows.append(dict(task_id=task.task_id,**task.parameters,repeat=repeat,**init,planning_wall=wall,planning_cpu=cpu,evaluation_wall=evaluation,
                initialization_management_seconds=init['seconds']-init['myopic_seconds'],
                total_evaluations=plan.evaluations,search_evaluations=plan.evaluations-init['evaluations']))
    assert len(rows)==24
    dump_json(out,rows);print('24 serial phase timings; policies and counts identical to frozen cohort')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data'));p.add_argument('--out',type=Path,default=Path('data/research/phase_timing.json'))
    a=p.parse_args();main(a.source,a.out)
