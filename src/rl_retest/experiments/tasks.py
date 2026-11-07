"""Task generation and explicit model/true-condition construction."""
import hashlib
import itertools
import json
from dataclasses import dataclass
import numpy as np
from ..model import EvaluationContext, History, ModelSpec, PlanningProblem, posterior


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    config: dict
    suite: str


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    parameters: dict
    priors: tuple
    history: History
    seed: int

    def to_data(self):
        return {'task_id': self.task_id, 'parameters': self.parameters, 'priors': self.priors,
                'successes': self.history.successes, 'failures': self.history.failures, 'seed': self.seed}

    @classmethod
    def from_data(cls, data):
        return cls(data['task_id'], data['parameters'], tuple(map(tuple, data['priors'])),
                   History(tuple(map(tuple, data['successes'])), tuple(map(tuple, data['failures']))), data['seed'])


def _seed(base, tag):
    payload = json.dumps([base, tag], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def create_tasks(experiment: ExperimentSpec):
    config, suite = experiment.config, experiment.suite
    definition = config['suites'][suite]
    source = definition.get('source', suite)
    base = config['seeds']['dev' if source == 'dev' else 'formal']
    priors = config['priors']
    m = config['model']
    tasks = []
    for k, horizon, n, pc, cost, index in itertools.product(
            definition['k'], definition['horizon'], definition['history'], definition['p_c_current'],
            definition['cost'], range(definition['tasks'])):
        params = {'k': k, 'horizon': horizon, 'n': n, 'p_c_current': pc, 'cost': cost, 'index': index}
        task_id = hashlib.sha256(json.dumps([source, params], sort_keys=True).encode()).hexdigest()[:20]
        # Large n=0 and n=3 share prior draws, then history uses a separate generator.
        prior_tag = [source, k, horizon, pc, cost, index] if source == 'large' else [source, params]
        seed = _seed(base, prior_tag)
        rng_prior = np.random.default_rng(seed)
        selected = tuple(tuple(priors[int(i)]) for i in rng_prior.integers(0, len(priors), size=k))
        rng_history = np.random.default_rng(_seed(base, [source, params, 'history']))
        successes = []
        for prior in selected:
            latent = int(rng_history.choice(3, p=prior))
            probability = (m['p_i'], m['p_c_old'], m['p_a'])[latent]
            successes.append(int(rng_history.binomial(n, probability)))
        history = History((tuple(successes),), (tuple(n-x for x in successes),))
        tasks.append(Task(task_id, params, selected, history, seed))
    return tasks


def contexts(task: Task, config: dict, memory='correct', assumed_current=None):
    m, p = config['model'], task.parameters
    current = (m['p_i'], p['p_c_current'], m['p_a'])
    old = (m['p_i'], m['p_c_old'], m['p_a'])
    spec = ModelSpec(m['reward'], p['cost'], p['horizon'], m['fallback'], current, (old,))
    true = posterior(task.priors, task.history, spec.historical)
    if true is None:
        raise ValueError('generated history impossible under true model')
    assumed = current if assumed_current is None else (m['p_i'], assumed_current, m['p_a'])
    planning_spec = ModelSpec(spec.reward, spec.cost, spec.horizon, spec.fallback, assumed, spec.historical)
    if memory == 'correct':
        belief = true
    elif memory == 'reset':
        belief = task.priors
    elif memory == 'misread':
        belief = posterior(task.priors, task.history, (assumed,))
    else:
        raise ValueError('unknown memory condition')
    return (None if belief is None else PlanningProblem(belief, planning_spec), EvaluationContext(true, spec))


def jobs_for(task: Task, config: dict, suite: str):
    search = config['search']
    jobs = []
    def add(method, memory='correct', **options):
        data = {'task_id': task.task_id, 'method': method, 'memory': memory, **options}
        data['job_id'] = hashlib.sha256(json.dumps([suite, data], sort_keys=True).encode()).hexdigest()[:24]
        jobs.append(data)
    if suite in ('main', 'dev'):
        p = task.parameters
        aliases = []
        if p['horizon'] <= 1:
            aliases.append({'method':'myopic','memory':'correct'})
        if p['horizon'] <= 2:
            aliases.append({'method':'depth2','memory':'correct'})
        if p['horizon'] <= 3:
            aliases.append({'method':'depth3','memory':'correct'})
        if p['k'] == 1:
            aliases.append({'method':'blinkered','memory':'correct'})
        if p['n'] == 0:
            aliases.extend([{'method':'exact','memory':'reset'}, {'method':'exact','memory':'misread'}])
        elif p['p_c_current'] == config['model']['p_c_old']:
            aliases.append({'method':'exact','memory':'misread'})
        add('exact', aliases=aliases)
        alias_keys = {(a['method'],a['memory']) for a in aliases}
        for memory in ('reset','misread'):
            if ('exact',memory) not in alias_keys:
                add('exact',memory)
        for method in ('stop','myopic','depth2','depth3','blinkered'):
            if (method,'correct') not in alias_keys:
                add(method)
        add('iterative', budget=search['main_budget'])
        add('bounded', budget=search['main_budget'], epsilon=search['main_epsilon'])
    elif suite == 'large':
        for method in ('stop','myopic','blinkered'):
            add(method)
        add('iterative_trace',budgets=search['budgets'])
        add('bounded_trace',budgets=search['budgets'],epsilons=search['epsilons'])
    elif suite == 'stress':
        for assumed in config['suites']['stress']['assumed_current']:
            add('exact', assumed_current=assumed, scenario='probability_shift')
            add('bounded', assumed_current=assumed, scenario='probability_shift', budget=search['main_budget'],epsilon=search['main_epsilon'])
        add('exact', scenario='type_resampling')
        add('exact','reset',scenario='type_resampling')
        add('bounded',scenario='type_resampling',budget=search['main_budget'],epsilon=search['main_epsilon'])
        add('exact', scenario='nominal')
    elif suite == 'pruning':
        for prune in (False,True):
            add('bounded', budget=10000, epsilon=.01, prune=prune)
    return jobs
