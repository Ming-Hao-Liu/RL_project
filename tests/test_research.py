"""Independent checks for retest thresholds and finite-calibration inputs."""
from dataclasses import replace
from fractions import Fraction as F
import pytest
from rl_retest.model import ModelSpec, PlanningProblem, EvaluationContext, History, posterior
from rl_retest.policy import Stop, Test as PolicyTest
from rl_retest.planners.dynamic import solve_exact
from rl_retest.evaluation import evaluate
from rl_retest.experiments.calibration import calibrate, plugin_problem
from rl_retest.experiments.tasks import Task
from rl_retest.experiments.records import policy_to_data
from tests.oracle import solve, joint, fixed_policy


def gross(mu,p,q0,r=1):
    # Unnormalized failure contribution remains defined at mu=1.
    return r*(mu*p+max((1-mu)*q0,mu*(1-p))-max(q0,mu))


@pytest.mark.parametrize('n,action', [(0,'execute'),(1,'test'),(7,'fallback')])
def test_failure_chain(n,action):
    q=F(9,10)*F(1,2)**n/(F(1,10)+F(9,10)*F(1,2)**n)
    mu=F(4,5)*q
    gain=gross(mu,F(4,5),F(1,2))
    expected=max(F(1,2),mu)+max(0,gain-F(1,50))
    b=posterior(((.1,.9,0.),),History(((0,),),((n,),)),((0.,.5,0.),))
    assert b[0][1]==pytest.approx(float(q),abs=1e-14)
    problem=PlanningProblem(b,ModelSpec(1,.02,1,.5,(0,.8,0),((0,.5,0),)))
    plan=solve_exact(problem)
    assert plan.model_value==pytest.approx(float(expected),abs=1e-14)
    ev=evaluate(plan.policy,EvaluationContext(b,problem.spec))
    assert ev.expected_utility==pytest.approx(float(expected),abs=1e-14)
    assert ('test' if isinstance(plan.policy,PolicyTest) else 'fallback' if plan.policy.path==-1 else 'execute')==action
    distribution=joint(((.1,.9,0.),),((0,.5,0),),((0,),),((n,),))
    assert float(fixed_policy(policy_to_data(plan.policy),distribution,(0,.8,0),.5,1,.02))==pytest.approx(float(expected),abs=1e-14)


def test_threshold_and_boundaries():
    # Rational derivation checked against an independent joint Bellman enumerator.
    for p,q0 in ((F(4,5),F(1,2)),(F(1),F(1,2))):
        peak=q0*(p-q0)
        for cost in (F(0),peak/4,peak,peak*2):
            for q in (F(0),F(1,20),F(1,2),q0/p,F(9,10),F(1)):
                mu=p*q
                value=max(q0,mu)+max(0,gross(mu,p,q0)-cost)
                assert solve(((1-q,q,0),),(0,p,0),q0,1,cost,1)==value
                problem=PlanningProblem(((float(1-q),float(q),0),),ModelSpec(1,float(cost),1,float(q0),(0,float(p),0),()))
                plan=solve_exact(problem)
                assert plan.model_value==pytest.approx(float(value),abs=1e-13)
                assert evaluate(plan.policy,EvaluationContext(problem.belief,problem.spec)).expected_utility==pytest.approx(float(value),abs=1e-13)
                if 0<cost<peak:
                    assert (gross(mu,p,q0)>cost)==(cost/(p-q0)<mu<(q0-cost)/(1-p+q0))
                no_budget=solve_exact(replace(problem,spec=replace(problem.spec,horizon=0)))
                assert isinstance(no_budget.policy,Stop)
    # Exactly representable tie: stop priority; deterministic branches covered above.
    tied=PlanningProblem(((.5,.5,0),),ModelSpec(1,.25,1,.5,(0,1,0),()))
    assert solve_exact(tied).policy==Stop(-1)


def test_calibration_reproducible_and_no_truth_leakage():
    a=calibrate(2026110801,10,0)
    assert a==calibrate(2026110801,10,0)
    assert a!=calibrate(2026110801,10,1)
    assert a['estimates']['a']==(a['counts']['a_old']+a['counts']['a_current']+1)/22
    t=Task('test',{'p_c_current':.5},((.2,.6,.2),),History(((0,),),((3,),)),0)
    spec=ModelSpec(1,.02,3,.5,(0,.5,.9),((0,.1,.9),))
    p=plugin_problem(t,spec,a['estimates'])
    assert p.spec.current[0]==0
    assert p.spec.current[2]==p.spec.historical[0][2]
    assert p.belief==posterior(t.priors,t.history,p.spec.historical)
    plan=solve_exact(p)
    true=posterior(t.priors,t.history,spec.historical)
    independent=fixed_policy(policy_to_data(plan.policy),joint(t.priors,spec.historical,t.history.successes,t.history.failures),spec.current,.5,1,.02)
    assert evaluate(plan.policy,EvaluationContext(true,spec)).expected_utility==pytest.approx(float(independent),abs=1e-13)
    # Parameter changes cannot make evaluation select a new terminal path.
    assert evaluate(Stop(-1),EvaluationContext(true,spec)).expected_utility==.5


def test_cluster_interval_and_divergence():
    from scripts.analyze_calibration import cluster_interval
    from scripts.analyze_mechanisms import first_divergence
    from rl_retest.planners.dynamic import solve_lookahead
    ci=cluster_interval([.02]*30)
    assert ci['clusters']==30 and ci['mean']==pytest.approx(.02)
    assert ci['ci_low']==pytest.approx(.02) and ci['ci_high']==pytest.approx(.02)
    p=PlanningProblem(((0.,.05,.95),),ModelSpec(1,.005,2,.5,(0,.8,.05),()))
    a,b=solve_exact(p),solve_lookahead(p,1)
    d=first_divergence(policy_to_data(a.policy),policy_to_data(b.policy),p)
    assert d['history']==[] and d['local_gain']==pytest.approx(.00309375)
    assert sum(x['probability'] for x in d['exact_branches'])==pytest.approx(1.)
