from rl_retest.model import ModelSpec, PlanningProblem


def sequential():
    spec = ModelSpec(1, .005, 2, .5, (0, .8, .05), ())
    return PlanningProblem(((0, .05, .95),), spec)
