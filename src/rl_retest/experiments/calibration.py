"""Synthetic labelled calibration and fixed plug-in planning inputs."""
from dataclasses import replace
import numpy as np
from ..model import PlanningProblem, posterior
from .tasks import _seed


def calibrate(seed: int, size: int, repeat: int, currents=(0.1, 0.5, 0.9)):
    """One independent cluster; A shares its probability across capabilities."""
    rng = np.random.default_rng(_seed(seed, ['calibration', size, repeat]))
    counts = {'c_old': int(rng.binomial(size, .1))}
    counts.update({f'c_{p}': int(rng.binomial(size, p)) for p in currents})
    counts['a_old'] = int(rng.binomial(size, .9))
    counts['a_current'] = int(rng.binomial(size, .9))
    estimates = {'c_old': (counts['c_old']+1)/(size+2),
                 'a': (counts['a_old']+counts['a_current']+1)/(2*size+2)}
    estimates.update({f'c_{p}': (counts[f'c_{p}']+1)/(size+2) for p in currents})
    return {'size': size, 'repeat': repeat, 'counts': counts, 'estimates': estimates,
            'effective_n': {'c_old': size, 'each_c_current': size, 'a': 2*size}}


def plugin_problem(task, true_spec, estimates, component='both'):
    """No true latent types or true values enter the planner.

    component is a prespecified diagnostic: history-only or future-only error.
    True probabilities are used only in the explicitly unperturbed component.
    """
    old = (0., estimates['c_old'], estimates['a'])
    current = (0., estimates[f"c_{task.parameters['p_c_current']}"], estimates['a'])
    if component == 'history':
        current = true_spec.current
    elif component == 'future':
        old = true_spec.historical[0]
    elif component != 'both':
        raise ValueError(component)
    spec = replace(true_spec, current=current, historical=(old,))
    belief = posterior(task.priors, task.history, spec.historical)
    if belief is None:
        raise ValueError('calibration history incompatible with structural prior')
    return PlanningProblem(belief, spec)
