from pathlib import Path
import hashlib,json,zipfile
from tests.oracle import joint,fixed_policy
root=Path.cwd();old=root/'artifacts/handoff-20261119';out=root/'artifacts/research-v2-20261117/technical'
checks={}
expected=(old.with_suffix('.zip.sha256')).read_text().split()[0]
checks['archive_sha256']=hashlib.sha256(old.with_suffix('.zip').read_bytes()).hexdigest()
assert checks['archive_sha256']==expected
lines=(old/'SHA256SUMS').read_text().splitlines()
for line in lines:
    digest,name=line.split('  ',1)
    assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest,name
checks['manifest_files_verified']=len(lines)
with zipfile.ZipFile(old.with_suffix('.zip')) as z:
    checks['zip_files']=len(z.namelist());assert z.testzip() is None
checks['zip_crc_valid']=True
suite=root/'artifacts/final-20261117/main';manifest=json.loads((suite/'manifest.json').read_text());config=manifest['config'];m=config['model']
tasks=[json.loads(x) for x in (suite/'tasks.jsonl').read_text().splitlines()]
selected={t['task_id']:t for t in tasks if t['parameters']['k']==3 and t['parameters']['index']==0 and t['parameters']['horizon']==3}
results=[]
for file in sorted((suite/'records').glob('*.jsonl')):
    for r in [json.loads(x) for x in file.read_text().splitlines()]:
        if r['task_id'] not in selected or 'alias_of' in r or r['method'] not in ('exact','bounded'):continue
        t=selected[r['task_id']];p=t['parameters'];oldtable=(m['p_i'],m['p_c_old'],m['p_a']);curr=(m['p_i'],p['p_c_current'],m['p_a'])
        dist=joint(t['priors'],[oldtable],t['successes'],t['failures'])
        value=float(fixed_policy(r['policy'],dist,curr,m['fallback'],m['reward'],p['cost']))
        error=abs(value-r['expected_utility']);assert error<1e-10
        results.append(dict(task_id=t['task_id'],job_id=r['job_id'],method=r['method'],memory=r['memory'],rational_value=value,saved_value=r['expected_utility'],error=error))
checks['independent_policy_evaluations']=len(results);checks['independent_tasks']=len(selected);checks['max_value_error']=max(r['error'] for r in results)
checks['reference_method']='rational joint posterior + fixed-type policy evaluation; no production posterior or evaluate'
(out/'checks.json').write_text(json.dumps(checks,indent=2));(out/'independent_policy_values.json').write_text(json.dumps(results,indent=2))
print(json.dumps(checks,indent=2))
# Count actual Bayesian transition calculations independently of planner counters.
from rl_retest.model import ModelSpec,PlanningProblem
from rl_retest.planners import common
from rl_retest.planners.dynamic import solve_exact,solve_lookahead,solve_iterative
from rl_retest.planners.blinkered import solve_blinkered
from rl_retest.planners.bounded import solve_bounded
problem=PlanningProblem(((.2,.6,.2),(.1,.2,.7),(.6,.3,.1)),ModelSpec(1,.005,3,.5,(0,.5,.9),()))
original=common.update
counts=[]
for name,solver in [('exact',solve_exact),('myopic',lambda p:solve_lookahead(p,1)),('depth2',lambda p:solve_lookahead(p,2)),('blinkered',solve_blinkered),('iterative100',lambda p:solve_iterative(p,100)),('bounded100',lambda p:solve_bounded(p,100,0))]:
    observed=[0]
    def counted(*a,**k):
        observed[0]+=1
        return original(*a,**k)
    common.update=counted
    try:result=solver(problem)
    finally:common.update=original
    assert observed[0]==result.evaluations
    counts.append(dict(method=name,observed_updates=observed[0],reported_updates=result.evaluations,cache_hits=result.cache_hits,status=result.status))
(out/'transition_accounting.json').write_text(json.dumps(counts,indent=2))
print(json.dumps(counts,indent=2))
