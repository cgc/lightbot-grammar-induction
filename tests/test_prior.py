import numpy as np
import random

import lb
from lb import generative_prior, prior
from .tools import mkprog

pyp_process_prior = prior.pyp_process_prior

def _pyp_process_prior(p, a, b, p_normal, p_end, p_subprocess_end):
    logp = pyp_process_prior(p, a, b, p_normal, p_end, p_subprocess_end)
    batch_logp = lb.scoring.batch_pyp_process_prior(lb.scoring.get_program_count_matrix([p], None), a, b, p_normal, p_end, p_subprocess_end)[0]

    assert np.isclose(logp, batch_logp), 'difference between batch and non-batch prior' + str((p, (a, b, p_normal, p_end, p_subprocess_end), logp, batch_logp))
    return logp


def test_pyp_prior():
    p_normal = 1/3
    assert np.isclose(
        pyp_process_prior(lb.Program('AA', ('',)), 0, 0, p_normal=p_normal, use_deprecated=True),
        np.log(
            # primitives
            1/5 * p_normal *
            1/5 * p_normal))

    b = 1.75
    a = .1
    assert np.isclose(
        pyp_process_prior(lb.Program('1', ('A',)), a, b, p_normal=p_normal, use_deprecated=True),
        np.log(
            (1-p_normal) * 1 *
            # primitive
            p_normal * 1/5))

    assert np.isclose(
        pyp_process_prior(lb.Program('11', ('A',)), a, b, p_normal=p_normal, use_deprecated=True),
        np.log(
            (1-p_normal) * 1 *
            # p(reusing process)
            (1-p_normal) * ((1-a)/(b+1)) *
            # primitive
            p_normal * 1/5))

    assert np.isclose(
        pyp_process_prior(lb.Program('112', ('A', 'B')), a, b, p_normal=p_normal, use_deprecated=True),
        np.log(
            (1-p_normal) * 1 *
            # p(reusing process)
            (1-p_normal) * ((1-a)/(b+1)) *
            # p(new process)
            (1-p_normal) * ((b+a)/(b+2)) *
            # primitives
            p_normal * 1/5 *
            p_normal * 1/5))

    assert np.isclose(
        pyp_process_prior(lb.Program('11213', ('A', 'B', 'C')), a, b, p_normal=p_normal, use_deprecated=True),
        np.log(
            (1-p_normal) * 1 *
            # p(reusing process)
            (1-p_normal) * ((1-a)/(b+1)) *
            # p(new process)
            (1-p_normal) * ((b+1*a)/(b+2)) *
            # p(reusing process)
            (1-p_normal) * ((2-a)/(b+3)) *
            # p(new process)
            (1-p_normal) * ((b+2*a)/(b+4)) *
            # primitives
            p_normal * 1/5 *
            p_normal * 1/5 *
            p_normal * 1/5))


def test_pyp_prior_p_end():
    p_normal = 1/3
    p_end = 1/7
    assert np.isclose(
        pyp_process_prior(mkprog('AAA'), 0, 0, p_normal=p_normal, p_end=p_end, use_deprecated=True),
        np.log(
            # primitives
            1/5 * p_normal * (1-p_end) *
            1/5 * p_normal * (1-p_end) *
            1/5 * p_normal * p_end))

    b = 1.75
    a = .1

    assert np.isclose(
        pyp_process_prior(mkprog('112', 'AA', 'B'), a, b, p_normal=p_normal, p_end=p_end, use_deprecated=True),
        np.log(
            (1-p_normal) * 1 * (1-p_end) *
            # p(reusing process)
            (1-p_normal) * ((1-a)/(b+1)) * (1-p_end) *
            # p(new process)
            (1-p_normal) * ((b+a)/(b+2)) * (1-p_end) *
            # primitives
            1/5 * p_normal * (1-p_end) *
            1/5 * p_normal * (1-p_end) *
            1/5 * p_normal * p_end))


def test_pyp_prior_p_end_p_subprocess_end():
    p_normal = 1/3
    p_end = 1/7
    p_subprocess_end = 1/11
    assert np.isclose(
        _pyp_process_prior(mkprog('AAA', 'AB'), 0, 0, p_normal=p_normal, p_end=p_end, p_subprocess_end=p_subprocess_end),
        np.log(
            # primitives
            1/5 * p_normal * (1-p_end) *
            1/5 * p_normal * (1-p_end) *
            1/5 * p_normal * p_end *
            # subroutine
            1/5 * p_normal * (1-p_subprocess_end) *
            1/5 * p_normal * p_subprocess_end))

    b = 1.75
    a = .1

    assert np.isclose(
        _pyp_process_prior(mkprog('11D2', 'AA', 'B'), a, b, p_normal=p_normal, p_end=p_end, p_subprocess_end=p_subprocess_end),
        np.log(
            (1-p_normal) * 1 * (1-p_end) *
            # p(reusing process)
            (1-p_normal) * ((1-a)/(b+1)) * (1-p_end) *
            # primitive
            p_normal * 1/5 * (1-p_end) *
            # p(new process)
            (1-p_normal) * ((b+a)/(b+2)) * p_end *

            # subroutine 1
            1/5 * p_normal * (1-p_subprocess_end) *
            1/5 * p_normal * p_subprocess_end *
            1/5 * p_normal * p_subprocess_end))


def test_prior_against_generative():
    for kw in [
        # This is the full model, where subprocesses might have a separate probability of ending.
        dict(p_normal=2/3, p_end=1/7, a=.1, b=.2, p_subprocess_end=1/4),
        # This is the standard model we work with, where both the main routine and subroutines have a shared probability.
        # TODO add this?
        # dict(p_normal=2/3, p_end=1/7, a=.1, b=.2, p_subprocess_end=1/7),
    ]:
        rng = random.Random(39402893)
        # TODO: Use a realistic subroutine limit of 4?
        generated = [
            generative_prior.generate_program(**kw, subroutine_limit=10, rng=rng)
            for i in range(100)
        ]
        batch_logps = lb.scoring.batch_pyp_process_prior(
            lb.scoring.get_program_count_matrix([p for p, _ in generated], None),
            **kw)
        for i, (p, logp) in enumerate(generated):
            # This assertion will make things seed-dependent
            assert sum(1 for sr in p.subroutines if sr) <= 4, 'prior assumes only up to 4 subroutines, so we cannot generate longer programs'
            logp2 = pyp_process_prior(p, **kw)
            assert np.isclose(logp, logp2)

            batch_logp = batch_logps[i]
            assert np.isclose(logp, batch_logp)
