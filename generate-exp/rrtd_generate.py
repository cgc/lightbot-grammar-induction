import numpy as np
import math, random, itertools, functools, collections, collections.abc

def shuffled(seq, random=random):
    '''
    This function returns a shuffled copy of the supplied sequence.
    '''
    seq = list(seq) # make a copy
    random.shuffle(seq) # shuffle inplace
    return seq # return the shuffled copy

def block_rand(population, k, random=random):
    '''
    This implements block randomization for k samples.
    It is a blocked version of `random.sample`, so that we sample
    from the population without replacement, then replenish.
    We repeat this until we've returned the desired number of samples.

    >>> r = random.Random(42)
    >>> expected = [4] * 4 + [5] * 6
    >>> for _ in range(5):
    ...     assert sorted(collections.Counter(block_rand(range(10), 46, random=r)).values()) == expected
    '''
    quot, rem = divmod(k, len(population))
    return (
        # We shuffle as many times as k divides the population
        sum([shuffled(population, random=random) for _ in range(quot)], []) +
        # And sample the remainder
        random.sample(population, rem))

def _validate_and_count_experiment(experiment, assignment_spec):
    '''
    This is a helper function we use in experiment generation to validate
    a factored representation of an experiment and return the counts needed
    at each site where an assignment to a factor must happen.
    - Validate that types are correct (factors are lists; non-factor parents of factors are not lists)
    - Validate that counts are the same across many branches (child factors with same name should all have same number of levels.)

    >>> _validate_and_count_experiment(dict(f=[dict(g=[3, 4, 5])]*7, h=[dict(j=3)]), ['f', 'f.g'])
    {'f': 7, 'f.g': 3}
    >>> _validate_and_count_experiment(dict(f=[dict(g=[3, 4, 5])]*7), ['f.g'])
    Traceback (most recent call last):
        ...
    AssertionError: No support for configuring factorial assignment inside sequences. At "f".
    >>> _validate_and_count_experiment(dict(), ['f'])
    Traceback (most recent call last):
        ...
    AssertionError: Missing key "f".
    >>> _validate_and_count_experiment(dict(f=3), ['f'])
    Traceback (most recent call last):
        ...
    AssertionError: Expected a sequence at "f".
    >>> _validate_and_count_experiment(dict(f=[{'g':[0]}, {'g':[1, 2]}]), ['f', 'f.g'])
    Traceback (most recent call last):
        ...
    AssertionError: Found 2 options at "f.g", but expected 1.
    '''
    counts = {}

    def _recur(exp, spec, *, prefix=''):
        # first gather the keys by their current level of key
        subconf = {}
        for keys in spec:
            subconf.setdefault(keys[0], []).append(keys[1:])

        for curr_key, subkeys in subconf.items():
            full_curr_key = f'{prefix}{curr_key}'
            errkey = f'"{full_curr_key}"'
            assert curr_key in exp, f'Missing key {errkey}.'

            isseq = isinstance(exp[curr_key], collections.abc.Sequence)

            # First, we see if this key is a factor.
            if () in subkeys:
                # If it is, we remove it & assert on type.
                subkeys.remove(())
                assert isseq, f'Expected a sequence at {errkey}.'

                curr_len = len(exp[curr_key])
                if full_curr_key not in counts:
                    # If not yet encountered, we record the number of values for this key.
                    counts[full_curr_key] = curr_len
                # We ensure the count is the same at sibling keys.
                assert curr_len == counts[full_curr_key], f'Found {curr_len} options at {errkey}, but expected {counts[full_curr_key]}.'

                # Finally, we recurse for each possible value this key might take.
                for value in exp[curr_key]:
                    _recur(value, subkeys, prefix=prefix+curr_key+'.')
            else:
                # If this key isn't factor but we have made it here, that means we
                # have children that are factors. We recurse.
                assert not isseq, f'No support for configuring factorial assignment inside sequences. At {errkey}.'
                _recur(exp[curr_key], subkeys, prefix=prefix+curr_key+'.')

    _recur(experiment, [tuple(key.split('.')) for key in assignment_spec])
    return counts


def sample_factor_assignment(factored_config, *, counterbalance=[], sample=[], count, random=random):
    '''
    >>> list(sorted(sample_factor_assignment(dict(f=range(10)), counterbalance=['f'], count=10, random=random.Random(42))['f'])) # all items enumerated
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> a = sample_factor_assignment(dict(f=range(2), g=range(2)), counterbalance=['f', 'g'], count=4, random=random.Random(42)) # we enumerate the cross product
    >>> list(sorted(zip(a['f'], a['g'])))
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> list(sorted(sample_factor_assignment(dict(f=range(10)), sample=['f'], count=10, random=random.Random(42))['f'])) # just random
    [0, 0, 1, 2, 3, 3, 8, 8, 8, 9]
    '''
    assert not (set(counterbalance) & set(sample)), 'should have disjoint set of things for counterbalancing & sampling'
    counts = _validate_and_count_experiment(factored_config, counterbalance+sample)

    # We do column-based storage to avoid repeating keys
    res = {k: [] for k in counterbalance+sample}

    # We count every combination of the values for the counterbalanced keys
    cb_ct = {key: 0 for key in itertools.product(*[range(counts[cb]) for cb in counterbalance])}
    # This should equal the product of the counts of the keys, by the definition of itertools.product()
    assert len(cb_ct) == functools.reduce(lambda a, b: a*b, [counts[cb] for cb in counterbalance], 1)

    # We generate all of our samples.
    for _ in range(count):
        # First, we counterbalance.

        # 1. Find minimum value based on current counts
        minval = min(cb_ct.values())
        # 2. Sample from items with minimum value
        sampled = random.choice([key for key, ct in cb_ct.items() if ct == minval])
        # 3. Add to our count to factor into future counterbalancing
        cb_ct[sampled] += 1

        # 4. Assign values into column-wise storage.
        for key, sampled_idx in zip(counterbalance, sampled):
            assert 0 <= sampled_idx < counts[key], 'This should only happen if the application is incorrectly matching indices and keys'
            res[key].append(sampled_idx)

        # For other keys, we just sample!
        for key in sample:
            res[key].append(random.choice(range(counts[key])))

    # Now, some simple checks.
    empirical_ct = collections.Counter([tuple([res[key][idx] for key in counterbalance]) for idx in range(count)])
    # When we count again, ensure the values match. This is a sanity check for the application logic.
    for key, ct in cb_ct.items():
        assert empirical_ct[key] == ct

    # We ensure the counterbalancing has worked.
    minval = min(cb_ct.values())
    maxval = max(cb_ct.values())
    assert maxval - minval in (0, 1), 'Counterbalancing max and min values should be equal or off by one.'
    assert minval == math.floor(count / len(cb_ct)), 'Counterbalancing min should be floor of expectation.'
    assert maxval == math.ceil(count / len(cb_ct)), 'Counterbalancing max should be ceiling of expectation.'

    return res

def config_for_condition(allconfig, condition):
    '''
    >>> c = dict(f=dict(g=[dict(h=[0, 1]), dict(h=[2, 3])]), conditionToFactors={'f.g': [None, 0, None, 1], 'f.g.h': [None, 1, None, 0]})
    >>> assert config_for_condition(c, 1) == dict(f=dict(g=dict(h=1)))
    >>> assert config_for_condition(c, 3) == dict(f=dict(g=dict(h=2)))
    >>> corig = dict(f=dict(g=[dict(h=[0, 1]), dict(h=[2, 3])]), conditionToFactors={'f.g': [None, 0, None, 1], 'f.g.h': [None, 1, None, 0]})
    >>> assert c == corig, 'making sure there is no mutation of config'
    '''
    allconfig = dict(allconfig)

    keyidx = []
    for factor, values in allconfig['conditionToFactors'].items():
        idx = values[condition]
        keyidx.append((factor.split('.'), idx))

    # We sort to ensure that shorter keys appear first, so that their
    # conditions are applied before any that are more nested.
    keyidx = sorted(keyidx, key=lambda pair: len(pair[0]))

    for keys, idx in keyidx:
        # We walk the configuration to reach the parent of the current key.
        c = allconfig
        for key in keys[:-1]:
          c[key] = dict(c[key])
          c = c[key]

        # We assign the appropriate value to replace the array of potential values.
        key = keys[-1]
        c[key] = c[key][idx]

    del allconfig['conditionToFactors']
    return allconfig
