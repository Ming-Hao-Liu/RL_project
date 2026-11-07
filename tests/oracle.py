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
