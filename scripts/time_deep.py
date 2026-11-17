"""Matched serial timing of the existing 24 mechanism tasks; no new tasks."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import random
import subprocess

from rl_retest.experiments.records import read_json, read_jsonl, dump_json
from scripts.analyze_mechanisms import problem_from_data
from scripts.run_research import record

METHODS = ('exact', 'myopic', 'blinkered', 'cold', 'warm')


def main(source, out):
    jobs = [j for j in read_jsonl(source/'research/jobs.jsonl') if j['group']=='deep']
    assert len(jobs)==24
    protocol = {'repeats':3, 'order_seed':2026111703, 'budget':10000, 'epsilon':0,
                'methods':METHODS, 'task_ids':[j['id'] for j in jobs],
                'scope':'same 24 fixed tasks; serial, fresh method caches; planning excludes context construction and evaluation'}
    # Persist the protocol before measuring; output is separate from original runs.
    if out.exists():
        data=read_json(out)
        assert data['protocol']=={**protocol,'methods':list(METHODS)}
    else:
        data={'protocol':protocol,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'platform':platform.platform(),'load':os.getloadavg(),
              'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
              'records':[]}
        dump_json(out,data)
    done={(r['id'],r['repeat'],r['method']) for r in data['records']}
    rng=random.Random(protocol['order_seed'])
    for job in jobs:
        original=read_json(source/'research/records'/f'{job["id"]}.json')
        problem=problem_from_data(original['problem'])
        saved={r['method']:r for r in original['records']}
        for repeat in range(protocol['repeats']):
            order=list(METHODS);rng.shuffle(order)
            for position,method in enumerate(order):
                if (job['id'],repeat,method) in done:continue
                r=record(problem,method,budget=protocol['budget'],epsilon=protocol['epsilon'])
                old=saved[method]
                assert r['policy']==old['policy'] and r['evaluations']==old['evaluations']
                assert r['model_value']==old['model_value'] and r['upper']==old['upper']
                data['records'].append({'id':job['id'],'repeat':repeat,'position':position,'method':method,
                    'planning_seconds':r['planning_seconds'],'evaluation_seconds':r['evaluation_seconds'],
                    'value':r['expected_utility'],'evaluations':r['evaluations'],
                    'policy_matches_frozen':True,'upper':r['upper'],'lower':r['lower']})
            dump_json(out,data)
    assert len(data['records'])==360
    print('360 matched timings; all policies, values, bounds and update counts match frozen records')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,default=Path('data/research/deep_timing.json'))
    a=p.parse_args();main(a.source,a.out)
