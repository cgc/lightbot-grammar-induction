from . import envs
import random
import numpy as np

def generate_program(*, a, b, p_normal, p_end, p_subprocess_end=None, subroutine_limit=4, rng=random):
    if p_subprocess_end is None:
        p_subprocess_end = p_end
    log_p = 0

    def flip(p):
        return choice([0, 1], [1 - p, p])

    def choice(pop, weights):
        nonlocal log_p
        rv = rng.choices(pop, weights=weights, k=1)[0]
        log_p += np.log(weights[pop.index(rv)] / sum(weights))
        return rv

    srs = [''] * subroutine_limit
    srs_ct = np.array([0] * subroutine_limit)
    m = 0

    def I():
        if flip(p_normal):
            return choice('ABCDE', weights=[1]*5)
        else:
            idx, is_new = bounded_crp()
            if is_new:
                assert not srs[idx]
                srs[idx] = SR()
            return str(idx + 1) # subroutine instructions are 1-indexed

    def bounded_crp():
        '''
        This is a CRP with an upper bound on the number of tables -- in this case, 4.
        '''
        nonlocal m
        w = (srs_ct[:m] - a).tolist() + [b + m * a]
        lim = min(subroutine_limit, len(w))

        idx = choice(range(m+1)[:lim], weights=w[:lim])
        assert idx < subroutine_limit
        srs_ct[idx] += 1

        is_new = idx == m
        if is_new:
            m += 1

        return idx, is_new

    def Is():
        a = I()
        if not flip(p_end):
            a += Is()
        return a

    def SR():
        a = I()
        if not flip(p_subprocess_end):
            a += SR()
        return a

    p = envs.Program(Is(), tuple(srs))
    assert m <= subroutine_limit
    return p, log_p
