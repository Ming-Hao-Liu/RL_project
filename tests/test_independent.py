from dataclasses import replace
from fractions import Fraction
import random
import pytest
from tests.oracle import joint, fixed_policy
from rl_retest.model import History, ModelSpec, PlanningProblem, EvaluationContext, posterior
from rl_retest.evaluation import evaluate, evaluate_resampling
from rl_retest.planners.dynamic import solve_exact
from rl_retest.planners.bounded import solve_bounded
from rl_retest.experiments.records import policy_to_data, read_jsonl, dump_jsonl
from rl_retest.experiments.runner import prepare, run, verify


def test_joint_history_resampling_and_combined_bound():
    rng=random.Random(20261108)
    options=((.2,.6,.2),(.6,.3,.1),(.1,.2,.7))
    for _ in range(30):
        priors=tuple(rng.choice(options) for i in range(2))
        old=((0,.1,.9),(0,.8,.9))
        history=History(((1,0),(0,1)),((0,1),(1,0)))
        true=posterior(priors,history,old)
        dist=joint(priors,old,history.successes,history.failures)
        for i in range(2):
            assert true[i]==pytest.approx([float(sum(w for z,w in dist.items() if z[i]==j)) for j in range(3)],abs=1e-10)
        spec=ModelSpec(1,.005,3,.5,(0,.5,.9),old)
        plan=solve_bounded(PlanningProblem(priors,spec),7,0)
        actual=evaluate(plan.policy,EvaluationContext(true,spec)).expected_utility
        assert actual==pytest.approx(float(fixed_policy(policy_to_data(plan.policy),dist,spec.current,.5,1,.005)),abs=1e-10)
        mixed={z:Fraction(0) for z in dist}
        for i in range(2):
            for z,w in dist.items():
                for j in range(3):
                    zz=list(z);zz[i]=j
                    mixed[tuple(zz)]+=w*Fraction(str(priors[i][j]))/2
        res=evaluate_resampling(plan.policy,EvaluationContext(true,spec),priors)
        assert res.expected_utility==pytest.approx(float(fixed_policy(policy_to_data(plan.policy),mixed,spec.current,.5,1,.005)),abs=1e-10)
        assumed=joint(priors)
        tv=float(sum(abs(dist[z]-assumed[z]) for z in dist)/2)
        optimum=solve_exact(PlanningProblem(true,spec)).model_value
        assert optimum-actual <= 2*(1+3*.005)*tv+plan.upper-plan.lower+1e-10


def test_verify_rejects_false_certificate(tmp_path):
    import json
    from rl_retest.experiments.records import read_json
    config=read_json('configs/study.json')
    config['search']['main_budget']=0
    file=tmp_path/'config.json';file.write_text(json.dumps(config))
    root=tmp_path/'run'
    prepare(file,'dev',root);run(root)
    records=[(f,read_jsonl(f)) for f in (root/'records').glob('*.jsonl')]
    exact={r['task_id']:r['model_value'] for _,rows in records for r in rows if r['method']=='exact' and r['memory']=='correct'}
    changed=False
    for file,rows in records:
        r=rows[0]
        if r['method']=='bounded' and exact[r['task_id']]>r['lower']+1e-8:
            # Still above the saved policy value, but below the true optimum.
            r['upper']=(r['lower']+exact[r['task_id']])/2
            dump_jsonl(file,rows);changed=True;break
    assert changed
    result=verify(root)
    assert not result['valid']
    assert any('exact value outside certificate' in x for x in result['errors'])
