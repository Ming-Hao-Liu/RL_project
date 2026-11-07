from dataclasses import replace
from fractions import Fraction
from itertools import product
import math
import random
import pytest
from rl_retest.model import ModelSpec, History, PlanningProblem, EvaluationContext, posterior, update
from rl_retest.policy import Stop, Test as PolicyTest
from rl_retest.bounds import perfect_information, initial_upper
from rl_retest.evaluation import evaluate, evaluate_resampling
from rl_retest.planners.dynamic import solve_exact, solve_lookahead, solve_iterative, solve_iterative_trace
from rl_retest.planners.blinkered import solve_blinkered
from rl_retest.planners.bounded import solve_bounded, solve_bounded_trace
from tests.cases import sequential
from tests.oracle import solve as oracle


def test_sequential_example():
    problem = sequential()
    one = solve_exact(PlanningProblem(problem.belief, replace(problem.spec, horizon=1)))
    two = solve_exact(problem)
    assert one.model_value == pytest.approx(.5)
    assert two.model_value == pytest.approx(16099 / 32000)
    assert two.policy == PolicyTest(0, Stop(-1), PolicyTest(0, Stop(-1), Stop(0)))
    assert evaluate(two.policy, EvaluationContext(problem.belief, problem.spec)).expected_utility == pytest.approx(two.model_value)
    assert solve_blinkered(problem).model_value == pytest.approx(two.model_value)


def test_historical_example():
    prior = ((.2,.6,.2),)
    spec = ModelSpec(1, .11, 1, .5, (0,1,1), ((0,0,1),))
    totals = dict(correct=0., reset=0., misread=0.)
    for outcome in (0,1):
        history = History(((outcome,),), ((1-outcome,),))
        true = posterior(prior, history, spec.historical)
        weight = .2 if outcome else .8
        for name, b in [('correct',true), ('reset',prior), ('misread',posterior(prior,history,(spec.current,)))]:
            policy = solve_exact(PlanningProblem(b,spec)).policy
            totals[name] += weight * evaluate(policy,EvaluationContext(true,spec)).expected_utility
    assert totals == pytest.approx({'correct':.812,'reset':.8,'misread':.6})


def test_dp_against_independent_joint_oracle():
    rng = random.Random(20261108)
    for k in (1,2,3):
        for h in range(4):
            for _ in range(3):
                # Decimal fractions sum exactly for the rational oracle.
                priors = tuple(rng.choice(((.2,.6,.2),(.6,.3,.1),(.1,.2,.7))) for _ in range(k))
                p = (0.,.5,.9)
                spec = ModelSpec(1,.02,h,.5,p,())
                problem = PlanningProblem(priors,spec)
                exact = solve_exact(problem)
                assert exact.model_value == pytest.approx(float(oracle(priors,p,.5,1,.02,h)), abs=1e-10)
                assert evaluate(exact.policy, EvaluationContext(priors,spec)).expected_utility == pytest.approx(exact.model_value)
                assert solve_lookahead(problem, max(1,h)).model_value == pytest.approx(exact.model_value)


def test_bounds_certificates_and_budget():
    rng = random.Random(44)
    for k in (1,2,3):
        for _ in range(8):
            belief = tuple(rng.choice(((.2,.6,.2),(.6,.3,.1),(.1,.2,.7))) for _ in range(k))
            spec = ModelSpec(1,.02,3,.5,(0,.5,.9),())
            problem = PlanningProblem(belief,spec)
            optimum = solve_exact(problem).model_value
            brute = sum(math.prod(belief[i][z] for i,z in enumerate(zs)) * max(spec.fallback, *(spec.current[z] for z in zs)) for zs in product(range(3),repeat=k))
            assert perfect_information(belief,spec) == pytest.approx(brute)
            assert optimum <= initial_upper(belief,spec,3) + 1e-10
            for m in (0,1,2,5,20,1000):
                result = solve_bounded(problem,m,0)
                assert result.evaluations <= m
                assert result.lower <= optimum + 1e-10 <= result.upper + 1e-10
                assert evaluate(result.policy,EvaluationContext(belief,spec)).expected_utility == pytest.approx(result.lower, abs=1e-10)
                deep = solve_iterative(problem,m)
                assert deep.evaluations <= m
                assert evaluate(deep.policy,EvaluationContext(belief,spec)).expected_utility == pytest.approx(deep.model_value)


def test_fixed_stop_and_impossible_branches():
    b = ((0.,1.,0.),)
    spec = ModelSpec(1,.01,1,.5,(0,1,0),())
    assert evaluate(Stop(-1), EvaluationContext(b,spec)).expected_utility == .5
    assert update(b,0,0,spec.current) is None
    assert posterior(((0.,1.,0.),), History(((0,),),((1,),)), ((0.,1.,1.),)) is None
    plan = solve_bounded(PlanningProblem(b,spec),10,0)
    assert plan.policy == Stop(0)


def test_resampling_negative_control():
    problem = sequential()
    policy = solve_exact(problem).policy
    ctx = EvaluationContext(problem.belief,problem.spec)
    assert evaluate_resampling(policy,ctx,problem.belief) == evaluate(policy,ctx)


def test_bounded_checkpoints_match_independent_runs():
    spec = ModelSpec(1,.005,3,.5,(0,.5,.9),())
    problem = PlanningProblem(((.2,.6,.2),(.1,.2,.7),(.6,.3,.1)),spec)
    budgets = [0,1,2,5,20,100]
    epsilons = [0,.01,.05]
    snapshots = solve_bounded_trace(problem,budgets,epsilons)
    optimum = solve_exact(problem).model_value
    for budget in budgets:
        for epsilon in epsilons:
            result, elapsed = snapshots[budget,epsilon]
            independent = solve_bounded(problem,budget,epsilon)
            assert (result.policy,result.lower,result.upper,result.evaluations) == (
                independent.policy,independent.lower,independent.upper,independent.evaluations)
            assert result.lower <= optimum + 1e-10 <= result.upper + 1e-10
            assert elapsed >= 0
    for epsilon in epsilons:
        for left,right in zip(budgets,budgets[1:]):
            assert snapshots[left,epsilon][0].lower <= snapshots[right,epsilon][0].lower + 1e-10
            assert snapshots[left,epsilon][0].upper + 1e-10 >= snapshots[right,epsilon][0].upper


def test_iterative_checkpoints_match_independent_runs():
    spec = ModelSpec(1,.005,3,.5,(0,.5,.9),())
    problem = PlanningProblem(((.2,.6,.2),(.1,.2,.7),(.6,.3,.1)),spec)
    budgets = [0,1,2,5,20,100,1000]
    snapshots = solve_iterative_trace(problem,budgets)
    for budget in budgets:
        result, elapsed = snapshots[budget]
        independent = solve_iterative(problem,budget)
        assert (result.policy,result.model_value,result.evaluations,result.status) == (
            independent.policy,independent.model_value,independent.evaluations,independent.status)
        assert elapsed >= 0
    for left,right in zip(budgets,budgets[1:]):
        assert snapshots[left][0].model_value <= snapshots[right][0].model_value + 1e-10


def test_sufficient_counts_and_memory_invariants():
    prior = ((.2,.6,.2),)
    tables = ((0,.1,.9),(0,.8,.9))
    history = History(((1,),(0,)),((0,),(1,)))
    counted = posterior(prior,history,tables)
    sequential_update = update(update(prior,0,1,tables[0]),0,0,tables[1])
    assert counted[0] == pytest.approx(sequential_update[0])
    reverse_update = update(update(prior,0,0,tables[1]),0,1,tables[0])
    assert counted[0] == pytest.approx(reverse_update[0])
    no_history = History(((0,),),((0,),))
    assert posterior(prior,no_history,(tables[0],))[0] == pytest.approx(prior[0])
    assert posterior(prior,history,tables)[0] != pytest.approx(posterior(prior,history,(tables[1],tables[1]))[0])


def test_initial_belief_tv_bound():
    rng = random.Random(20)
    options = ((.2,.6,.2),(.6,.3,.1),(.1,.2,.7))
    for k in (1,2,3):
        for _ in range(10):
            true = tuple(rng.choice(options) for _ in range(k))
            assumed = tuple(rng.choice(options) for _ in range(k))
            spec = ModelSpec(1,.02,2,.5,(0,.5,.9),())
            optimal = solve_exact(PlanningProblem(true,spec)).model_value
            assumed_policy = solve_exact(PlanningProblem(assumed,spec)).policy
            value = evaluate(assumed_policy,EvaluationContext(true,spec)).expected_utility
            delta = sum(abs(math.prod(true[i][z] for i,z in enumerate(zs)) -
                            math.prod(assumed[i][z] for i,z in enumerate(zs)))
                        for zs in product(range(3),repeat=k))/2
            assert -1e-10 <= optimal-value <= 2*(1+2*.02)*delta + 1e-10


def test_model_impossible_branch_remains_executable_under_truth():
    policy = PolicyTest(0,Stop(-1),Stop(0))
    true = ((0.,1.,0.),)
    spec = ModelSpec(1,.01,1,.5,(0,1,0),())
    assert evaluate(policy,EvaluationContext(true,spec)).expected_utility == pytest.approx(.99)


def test_evaluator_rejects_invalid_paths_and_excess_tests():
    problem = sequential()
    context = EvaluationContext(problem.belief, problem.spec)
    with pytest.raises(ValueError):
        evaluate(Stop(-2), context)
    with pytest.raises(ValueError):
        evaluate(PolicyTest(1, Stop(-1), Stop(-1)), context)
    too_deep = PolicyTest(0, PolicyTest(0, PolicyTest(0,Stop(-1),Stop(-1)),Stop(-1)),Stop(-1))
    with pytest.raises(ValueError):
        evaluate(too_deep, context)


def test_stop_bound_and_zero_information_edge_cases():
    belief = ((.2,.6,.2),(.1,.2,.7))
    expensive = ModelSpec(1,1.,3,.5,(0,.5,.9),())
    result = solve_exact(PlanningProblem(belief,expensive))
    assert isinstance(result.policy, Stop)
    assert initial_upper(belief,expensive,3) == pytest.approx(result.model_value)
    no_information = ModelSpec(1,0.,3,.5,(.3,.3,.3),())
    result = solve_exact(PlanningProblem(belief,no_information))
    assert result.policy == Stop(-1)
    assert solve_bounded(PlanningProblem(belief,no_information),0,0).policy == Stop(-1)


def test_pruning_does_not_change_search_result():
    problem = PlanningProblem(((.2,.6,.2),(.1,.2,.7),(.6,.3,.1)),
                              ModelSpec(1,.005,3,.5,(0,.5,.9),()))
    for budget in (0,2,10,100):
        pruned = solve_bounded(problem,budget,0,prune=True)
        scanned = solve_bounded(problem,budget,0,prune=False)
        assert (pruned.policy,pruned.lower,pruned.upper,pruned.evaluations) == (
            scanned.policy,scanned.lower,scanned.upper,scanned.evaluations)
