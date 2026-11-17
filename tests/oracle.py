"""Independent exact joint-belief oracle for small instances."""
from fractions import Fraction
from itertools import product
from functools import lru_cache


def solve(priors, probabilities, fallback, reward, cost, horizon):
    states = tuple(product(range(3), repeat=len(priors)))
    p = tuple(Fraction(str(x)) for x in probabilities)
    q, r, c = map(lambda x: Fraction(str(x)), (fallback, reward, cost))
    initial = tuple(__import__('math').prod(Fraction(str(priors[i][z])) for i, z in enumerate(state)) for state in states)

    @lru_cache(None)
    def rec(weights, h):
        values = [q] + [sum(w * p[state[i]] for w, state in zip(weights, states)) for i in range(len(priors))]
        best = r * max(values)
        if h:
            for i in range(len(priors)):
                val = -c
                for o in (0, 1):
                    ws = tuple(w * (p[z[i]] if o else 1 - p[z[i]]) for w, z in zip(weights, states))
                    prob = sum(ws)
                    if prob:
                        val += prob * rec(tuple(w / prob for w in ws), h - 1)
                best = max(best, val)
        return best
    return rec(initial, horizon)


def joint(priors, tables=(), successes=(), failures=()):
    """Independent rational joint posterior, including observed history."""
    from math import prod
    f = lambda x: Fraction(str(x))
    weights = {}
    for zs in product(range(3), repeat=len(priors)):
        w = prod(f(priors[i][z]) for i,z in enumerate(zs))
        for table, ss, fs in zip(tables, successes, failures):
            w *= prod(f(table[z])**ss[i] * (1-f(table[z]))**fs[i] for i,z in enumerate(zs))
        weights[zs] = w
    total = sum(weights.values())
    return {z:w/total for z,w in weights.items()}


def fixed_policy(policy, distribution, probabilities, fallback, reward, cost):
    """Policy uses plain JSON; integrate types without Bayesian updates."""
    p = tuple(Fraction(str(x)) for x in probabilities)
    q,r,c = (Fraction(str(x)) for x in (fallback,reward,cost))
    def visit(node, zs):
        i = node['path']
        if node['action']=='stop':
            return r*(q if i==-1 else p[zs[i]])
        a = p[zs[i]]
        return -c + (1-a)*visit(node['failure'],zs)+a*visit(node['success'],zs)
    return sum(w*visit(policy,z) for z,w in distribution.items())
