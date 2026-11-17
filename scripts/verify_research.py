"""Re-evaluate all new saved policies; independently enumerate fixed small cases."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
from pathlib import Path
import numpy as np
from rl_retest.model import EvaluationContext, ModelSpec, PlanningProblem
from rl_retest.experiments.records import read_json, read_jsonl, dump_json, policy_from_data
from rl_retest.experiments.tasks import Task,contexts
from rl_retest.experiments.calibration import calibrate
from rl_retest.evaluation import evaluate
from tests.oracle import solve, joint, fixed_policy
from scripts.analyze_mechanisms import problem_from_data
from scripts.analyze_study import write_csv


def summarize_timing(phase):
    """Raw means add across phases; independently computed medians do not."""
    assert len(phase)==24
    timing=[]
    for n in (0,3):
        rr=[r for r in phase if r['n']==n]
        for r in rr:
            assert abs(r['total_seconds']-r['root_setup_seconds']-r['seconds']-r['search_seconds'])<1e-10
        for field in ('root_setup_seconds','myopic_seconds','initialization_management_seconds','search_seconds','total_seconds','planning_wall','planning_cpu','evaluation_wall'):
            per_task=[np.median([r[field] for r in rr if r['task_id']==tid]) for tid in sorted({r['task_id'] for r in rr})]
            timing.append(dict(n=n,phase=field,tasks=len(per_task),repeats=3,
                raw_mean=float(np.mean([r[field] for r in rr])),
                task_median_mean=float(np.mean(per_task)),median=float(np.median(per_task)),p90=float(np.quantile(per_task,.9))))
    return timing


def main(source,out):
    root=source/'research';manifest=read_json(root/'manifest.json');protocol=manifest['protocol']
    for name in ('jobs','tasks'):
        assert hashlib.sha256((root/f'{name}.jsonl').read_bytes()).hexdigest()==manifest[f'{name}_sha256']
    jobs=read_jsonl(root/'jobs.jsonl');tasks={t['task_id']:Task.from_data(t) for t in read_jsonl(root/'tasks.jsonl')}
    config=read_json(source/'main/manifest.json')['config'];true=read_json(root/'truth.json')
    contexts_by={tid:contexts(t,config)[1] for tid,t in tasks.items()}
    assert len(jobs)==1473 and len(tasks)==36
    maximum=0.;checks=0;independent=[];statuses=Counter()
    for job in jobs:
        data=read_json(root/'records'/f'{job["id"]}.json')
        assert data['job']==job
        records=data['records']
        if job['group']=='calibration':
            cal=data['calibration']
            assert cal==calibrate(protocol['calibration']['seed'],job['size'],job['repeat'])
            diag=job['size'] in (10,300)
            assert len(records)==(76 if diag else 72)
            assert len({(r['task_id'],r['method'],r['component']) for r in records})==len(records)
            for r in records:
                ctx=contexts_by[r['task_id']]
                ev=evaluate(policy_from_data(r['policy']),ctx)
                model=problem_from_data({'belief':r['planning_belief'],'spec':r['planning_spec']})
                mv=evaluate(policy_from_data(r['policy']),EvaluationContext(model.belief,model.spec))
                err=max(abs(ev.expected_utility-r['expected_utility']),abs(mv.expected_utility-r['model_value']),
                        abs(ev.expected_tests-r['expected_tests']))
                assert err<1e-10
                assert r['expected_utility']<=true[r['task_id']]['expected_utility']+1e-10
                maximum=max(maximum,err);checks+=1;statuses[r['status']]+=1
            # All 36 plug-in DP policies in the fixed first m10 repeat, including actual history.
            if job['size']==10 and job['repeat']==0:
                for r in records:
                    if r['method']!='exact' or r['component']!='both':continue
                    task=tasks[r['task_id']];ctx=contexts_by[task.task_id]
                    dist=joint(task.priors,ctx.spec.historical,task.history.successes,task.history.failures)
                    v=float(fixed_policy(r['policy'],dist,ctx.spec.current,ctx.spec.fallback,ctx.spec.reward,ctx.spec.cost))
                    error=abs(v-r['expected_utility']);assert error<1e-10
                    independent.append(dict(job=job['id'],task_id=task.task_id,kind='calibrated_fixed_policy',error=error))
        else:
            problem=problem_from_data(data['problem']);ctx=EvaluationContext(problem.belief,problem.spec)
            exact=records[0]['model_value']
            assert len(records)==(2 if job['group']=='grid' else 5)
            for r in records:
                ev=evaluate(policy_from_data(r['policy']),ctx)
                err=max(abs(ev.expected_utility-r['expected_utility']),abs(ev.expected_utility-r['model_value']),abs(ev.expected_tests-r['expected_tests']))
                assert err<1e-10 and r['expected_utility']<=exact+1e-10
                if r['upper'] is not None:assert r['lower']-1e-10<=exact<=r['upper']+1e-10
                if r['method'] in ('warm','cold'):assert r['evaluations']<=10000
                maximum=max(maximum,err);checks+=1;statuses[r['status']]+=1
            # Four predetermined exact joint-oracle checks spanning K/B; no outcome selection.
            if job['group']=='deep' and job['name']=='rare_low_cost':
                qs=[F(str(q)) for q in job['q'][:job['k']]]
                priors=tuple((0,q,1-q) for q in qs)
                v=float(solve(priors,(0,F(4,5),F(1,20)),F(1,2),1,F(1,200),job['horizon']))
                error=abs(v-exact);assert error<1e-10
                independent.append(dict(job=job['id'],kind='deep_joint_optimum',error=error))
                dist=joint(priors)
                for r in records:
                    v=float(fixed_policy(r['policy'],dist,(0,.8,.05),.5,1,.005))
                    error=abs(v-r['expected_utility']);assert error<1e-10
                    independent.append(dict(job=job['id'],kind='deep_fixed_'+r['method'],error=error))
    for tid,r in true.items():
        ev=evaluate(policy_from_data(r['policy']),contexts_by[tid]);assert abs(ev.expected_utility-r['expected_utility'])<1e-10
    phase=read_json(root/'phase_timing.json')
    write_csv(out/'phase_timing.csv',summarize_timing(phase))
    dump_json(out/'verification.json',{'valid':True,'jobs':len(jobs),'policy_evaluations':checks,'truth_policies':len(true),
        'maximum_value_or_test_error':maximum,'statuses':dict(statuses),'independent_checks':independent,
        'timing_records':len(phase),'tolerance':1e-10})
    dump_json(out/'input_sha256.json',{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file()})
    print(f'Verified {checks} saved policies, {len(independent)} independent rational checks')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data'));p.add_argument('--out',type=Path,default=Path('results/research'))
    a=p.parse_args();main(a.source,a.out)
