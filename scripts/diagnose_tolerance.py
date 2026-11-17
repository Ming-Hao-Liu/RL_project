"""All nonoptimal main-search outputs: fixed epsilon-zero mechanism diagnostic."""
from pathlib import Path
import json,hashlib,time
from rl_retest.analysis import _rows
from rl_retest.experiments.records import read_jsonl,policy_to_data
from rl_retest.experiments.tasks import Task,contexts
from rl_retest.planners.bounded import solve_bounded
from rl_retest.evaluation import evaluate
from scripts.build_handoff import write_csv

root=Path('artifacts/final-20261117/main');out=Path('artifacts/research-v2-20261117/search');out.mkdir(exist_ok=True,parents=True)
manifest,rows=_rows(root)
refs={r['task_id']:r for r in rows if r['method']=='exact' and r['memory']=='correct'}
selected=[r for r in rows if r['method']=='bounded' and refs[r['task_id']]['model_value']-r['model_value']>1e-10]
tasks={r['task_id']:Task.from_data(r) for r in read_jsonl(root/'tasks.jsonl')}
protocol={'classification':'exploratory paired intervention on existing tasks, no new population inference','question':'Are observed small-task search losses explained by positive stopping tolerance?', 'selection':'ALL main bounded records with exact-minus-search >1e-10','count':len(selected),'epsilon_old':.01,'epsilon_new':0,'budget_unchanged':10000,'no_additional_samples':True,'task_ids':[r['task_id'] for r in selected],'source_sha256':manifest['source_sha256']}
(out/'tolerance_protocol.json').write_text(json.dumps(protocol,indent=2))
results=[];policies=[]
for old in selected:
    problem,truth=contexts(tasks[old['task_id']],manifest['config']);start=time.perf_counter();plan=solve_bounded(problem,10000,0);elapsed=time.perf_counter()-start
    value=evaluate(plan.policy,truth).expected_utility;optimum=refs[old['task_id']]['model_value']
    assert abs(value-plan.lower)<=1e-10 and plan.lower<=optimum+1e-10 and optimum<=plan.upper+1e-10 and plan.evaluations<=10000
    results.append(dict(task_id=old['task_id'],horizon=old['parameters']['horizon'],old_status=old['status'],new_status=plan.status,old_value=old['model_value'],new_value=value,exact_value=optimum,old_evaluations=old['evaluations'],new_evaluations=plan.evaluations,old_width=old['upper']-old['lower'],new_width=plan.upper-plan.lower,diagnostic_seconds=elapsed))
    policies.append(dict(task_id=old['task_id'],policy=policy_to_data(plan.policy)))
write_csv(out/'tolerance_diagnostic.csv',results)
(out/'tolerance_policies.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in policies))
print(json.dumps({'tasks':len(results),'attain_exact':sum(abs(r['exact_value']-r['new_value'])<=1e-10 for r in results),'old_all_tolerance':all(r['old_status']=='tolerance' for r in results),'new_statuses':sorted(set(r['new_status'] for r in results))}))
