import numpy as np
import collections

# Subroutine names are numbers, 1-indexed.
SR_NAME_TO_IDX = {str(i+1): i for i in range(4)}

def prior(subroutines, *, alpha, p_normal, p_end):
    p_action_uniform = 1/5
    p_call = 1 - p_normal

    # Counters for CRP
    call_counts = [0] * 4
    all_call_counts = 0

    lp = 0

    # This enumerates all subroutines -- this is incorrect if you don't guarantee that declared subroutines are called.
    for sr in subroutines:
        # Skip empty subroutines, which we treat as undeclared.
        if not sr:
            continue
        for idx, inst in enumerate(sr):
            # First, compute contribution of p(end). We skip first index since we assume subroutines are non-empty.
            if idx != 0:
                lp += np.log(1 - p_end)

            if inst in SR_NAME_TO_IDX:
                idx = SR_NAME_TO_IDX[inst]
                p_sr = (
                    # Undefined subroutine is new -- sample proportional to alpha
                    alpha if call_counts[idx] == 0
                    # Defined subroutine is sampled proportional to use
                    else call_counts[idx]
                ) / (all_call_counts + alpha)
                lp += np.log(p_call) + np.log(p_sr)
                # Increment counters for all calls, and this subroutine.
                call_counts[idx] += 1
                all_call_counts += 1
            else:
                # The simpler case for non-subroutine.
                lp += np.log(1 - p_call) + np.log(p_action_uniform)

        # Now add probability that we terminate.
        lp += np.log(p_end)
    return lp

import random
def generate_program(*, alpha, p_normal, p_end, subroutine_limit=4, rng=random):
    log_p = 0

    # Stochastic primitives, which update log_p
    def flip(p):
        return choice([0, 1], [1 - p, p])
    def choice(pop, weights):
        nonlocal log_p
        rv = rng.choices(pop, weights=weights, k=1)[0]
        log_p += np.log(weights[pop.index(rv)] / sum(weights))
        return rv

    # Counters for CRP
    srs = [''] * subroutine_limit
    srs_ct = np.array([0] * subroutine_limit)
    m = 0 # Index of first undefined subroutine.

    # Grammar non-terminals
    def SR():
        a = I()
        if not flip(p_end):
            a += SR()
        return a

    def I():
        if flip(p_normal):
            return choice('ABCDE', weights=[1]*5)
        else:
            idx, is_new = bounded_crp()
            if is_new:
                assert not srs[idx]
                srs[idx] = SR()
            return str(idx + 1) # subroutine names are 1-indexed

    def bounded_crp():
        '''
        This is a CRP with an upper bound on the number of tables -- by default, subroutine_limit=4.
        '''
        nonlocal m
        w = srs_ct[:m].tolist() + [alpha]
        lim = min(subroutine_limit, len(w))

        idx = choice(range(m+1)[:lim], weights=w[:lim])
        assert idx < subroutine_limit
        srs_ct[idx] += 1

        is_new = idx == m
        if is_new:
            m += 1

        return idx, is_new

    main = SR()
    assert m <= subroutine_limit
    return (main,)+tuple(srs), m, log_p

def main():
    rng = random.Random(42)
    ct = collections.Counter()

    for _ in range(1000):
        kw = dict(alpha=8, p_normal=1/2, p_end=1/5)
        p, subroutine_count, gen_logp = generate_program(rng=rng, **kw)
        logp = prior(p, **kw)
        match = 'same' if np.isclose(logp, gen_logp) else 'different'
        ct[subroutine_count, match] += 1

    print('Note: differences with maximum number of SRs b/c prior is unbounded, but generative prior is not')
    for key, count in sorted(ct.items()):
        sr_ct, match = key
        print(f'With subroutine count {sr_ct} and match={match}, count={count}')

if __name__ == '__main__':
    main()
