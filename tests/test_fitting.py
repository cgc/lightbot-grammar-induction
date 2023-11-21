from typing import Any
import lb
import numpy as np
import warnings
import pytest
from scipy.special import softmax

def test_compute_scores_and_partition_and_crossent():
    ts = lb.TraceSet(None, [], None)
    # NOTE: specifying programs this way, instead of traces via constructor.
    ts.programs = [
        lb.tools.mkprog('SS'),
        lb.tools.mkprog('RRR'),
        lb.tools.mkprog('LLLL'),
    ]
    a = lb.fitting.Args.new({}, default_values=dict(mdl_beta=1.))
    bs = lb.BehaviorSummary(None, {
        lb.tools.mkprog('W'): 3,
        lb.tools.mkprog('SS'): 2,
        lb.tools.mkprog('RRR'): 1,
    })

    # Testing compute_scores_and_partition()
    r = ts.compute_scores_and_partition(bs, a, batch_score=lb.scoring.batch_score_mdl)
    assert np.allclose(r['data_scores'], [-1, -2, -3])
    assert np.allclose(r['enum_scores'], [-2, -3, -4])
    expected = np.log(np.exp([-1, -2, -3, -4]).sum())
    assert np.isclose(r['log_partition'], expected), 'log partition should be computed based on unique programs across both bs and ts'

    # Testing crossent()
    expected = -sum(
        (-lb.hier_len(prog) - r['log_partition']) * ct
        for prog, ct in bs.program_counts.items()
    )
    assert np.isclose(ts.crossent(bs, a, batch_score=lb.scoring.batch_score_mdl), expected)

    # Testing how fit() handles nuisance parameters
    dc = lb.Dataset([lb.fitting.DatasetRow(None, None, bs, ts, {})], {}, {})
    # Kind of a hack: We don't fit any parameters, just evaluate the function by integrating out nuisance params.
    mdl_beta_values = [1, 2, 3]
    res = dc.fit(lb.scoring.batch_score_mdl, free_parameters=dict(), nuisance_params=dict(mdl_beta=mdl_beta_values))
    scores = np.array([-1, -2, -3, -4])
    counts = np.array([3, 2, 1, 0])
    log_probs_for_mdl_betas = np.array([
        counts @ np.log(softmax(w * scores))
        for w in mdl_beta_values
    ])
    expected_log_prob = np.log(np.mean(np.exp(log_probs_for_mdl_betas)))
    assert np.isclose(res.result['result'].fun, -expected_log_prob)

    # This is a more in-depth test of various incorrect ways of computing the above, showing some inequalities between them.
    args = [lb.fitting.Args.new({}), dict(mdl_beta=mdl_beta_values)]
    geometric_mean = lb.fitting.integrate_nuisance(lambda a: -dc.crossent(a, batch_score=lb.scoring.batch_score_mdl), *args)
    assert np.isclose(geometric_mean, np.mean(log_probs_for_mdl_betas))
    # Note that we are negating the crossent, which already is the negative LL. So there's a double negative here.
    arithmetic_mean_of_negative = lb.fitting.integrate_nuisance_for_negative_log_fn(lambda a: -dc.crossent(a, batch_score=lb.scoring.batch_score_mdl), *args)
    assert np.isclose(arithmetic_mean_of_negative, -lb.fitting.logmeanexp(-log_probs_for_mdl_betas))
    # Asserting general relationship between these
    assert arithmetic_mean_of_negative < geometric_mean < expected_log_prob
    # Asserting specific values, just copy/pasted from output, so only meant to make this example concrete, not a true test of behavior.
    assert np.isclose(expected_log_prob, -7.634431578229864)
    assert np.isclose(geometric_mean, -9.2726613484669)
    assert np.isclose(arithmetic_mean_of_negative, -11.242805218321593)
    assert np.allclose(log_probs_for_mdl_betas, [-6.64113819, -8.87046763, -12.30637822])

def test_args():
    # args = lb.fitting.Args.new(free_parameters=x, default_values=x, rng=x, minimize_param=x)
    class DummyRNG:
        def __init__(self, outputs):
            self.outputs = outputs
        def __getattr__(self, name):
            def fn(*a, **k):
                return self.outputs.pop(0)
            return fn
    for init_kws in [
        # Normal constructor
        lambda: dict(free_parameters=dict(a=.5), default_values=dict(b=2)),
        # Called during model fitting, extra param for current fit
        lambda: dict(free_parameters=dict(a=.2), default_values=dict(b=2), minimize_param=[.5]),
        # Called to sample other initializations
        lambda: dict(free_parameters=['a'], default_values=dict(b=2), rng=DummyRNG([0.5])),
    ]:
        args = lb.fitting.Args.new(**init_kws())

        # test reconstruct inputs
        assert args.reconstruct_inputs() == (dict(a=.5), dict(b=2))

        # Recreate by using m_param
        m_param = args.to_minimize_param()
        args2 = lb.fitting.Args.new(**dict(init_kws(), minimize_param=m_param))
        assert args == args2

        # Changing via m_param
        m_param[0] = .9
        args3 = lb.fitting.Args.new(**dict(init_kws(), minimize_param=m_param))
        assert args3.reconstruct_inputs() == (dict(a=.9), dict(b=2))

        # Updating
        args4 = args.updated(b=3)
        # First, make sure we didn't modify original
        assert args.reconstruct_inputs() == (dict(a=.5), dict(b=2))
        # Make sure we've updated
        assert args4.reconstruct_inputs() == (dict(a=.5), dict(b=3))


def test_logmeanexp():
    def overunderflowing_lme(arr):
        return np.log(np.mean(np.exp(arr)))

    # a basic case
    values = np.log([.1, .8, .9])
    assert np.isclose(lb.fitting.logmeanexp(values), np.log(.6))
    assert np.isclose(lb.fitting.logmeanexp(values), overunderflowing_lme(values))

    # Correctly avoids underflow for very small values (from the log)
    scale = 1e8
    values = scale * np.log(np.array([.1, .8, .9]))
    # At a sufficiently high scale, the largest contribution comes from the largest element
    assert np.isclose(lb.fitting.logmeanexp(values), scale * np.log(.9))
    # But at this high a scale, an implementation without a log-sum-exp trick will underflow
    assert overunderflowing_lme(values) == -np.inf

    # Correctly avoids overflow for very large values (from the exp)
    scale = 1e5
    values = scale * np.array([1, 2, 3])
    # At a sufficiently high scale, the largest contribution comes from the largest element
    assert np.isclose(lb.fitting.logmeanexp(values), scale * 3)
    # But at this high a scale, an implementation without a log-sum-exp trick will overflow
    with warnings.catch_warnings(record=True) as ws:
        assert overunderflowing_lme(values) == np.inf
    assert len(ws) == 1
    assert 'overflow encountered in exp' in str(ws[0])


def test_integrate_nuisance():
    def fn(a):
        return a.a ** 2
    values = [.1, .5, .9]
    assert lb.fitting.integrate_nuisance(fn, lb.fitting.Args.new({}), dict(a=values)) == np.mean(
        [v*v for v in values])

    # Now, for log(fn)
    log_fn = lambda a: -np.log(fn(a))
    assert lb.fitting.integrate_nuisance_for_negative_log_fn(log_fn, lb.fitting.Args.new({}), dict(a=values)) == -np.log(np.mean(
        [v*v for v in values]))

def test_fit():
    # Very simple case for finding a global minima
    def cost(a):
        return (a.b - 3)**2
    result = lb.fitting.fit(cost, free_parameters=dict(b=1))
    for r in result.results:
        assert r['result'].success
    assert np.isclose(result.result['final_args'].b, 3)
