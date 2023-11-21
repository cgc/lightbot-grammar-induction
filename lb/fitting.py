from dataclasses import dataclass
from typing import Any, List
from . import program_analysis
from . import envs
from . import search
from . import scoring
from . import program_generation
from . import heuristics
from . import plotting
from . import exp
import functools
import warnings
import pathlib
import joblib

import time
import numpy as np
import scipy.optimize
from scipy.special import logsumexp
import collections
import contextlib
import heapq
import pickle
import lzma
import enum



@contextlib.contextmanager
def timed(msg=''):
    st = time.time()
    try:
        print(msg, end='')
        yield
    finally:
        print(f' -- elapsed {time.time()-st:.6f}s')


def minimize(obj, *args, progress=10, **kwargs):
    i = 0
    def fn(x):
        nonlocal i
        i += 1
        if progress is not None and i % progress == 0:
            print(f'iter={i}, cost={obj(x):.2f}, x={x}')
    return scipy.optimize.minimize(obj, *args, callback=fn, **kwargs)


# This was previously 1e-8, but we changed it to avoid cases
# of slow convergence to a boundary of 0, particularly with p_end.
eps = 1e-4

class BoundTypes(enum.Enum):
    # w0 -> includes 0
    # w1 -> includes 1
    # w01 -> includes 0 and 1

    pos = (eps, None)
    pos_w0 = (0, None)
    neg = (None, -eps)
    neg_w0 = (None, 0)
    prob = (eps, 1-eps)
    prob_w0 = (0, 1-eps)
    prob_w1 = (eps, 1)
    prob_w01 = (0, 1)
    unbounded = (None, None)

    inverse_temperature = (eps, 20)

    @classmethod
    def sample(cls, rng: np.random.RandomState, b):
        exp_scale = 2
        if b in (cls.pos, cls.pos_w0):
            return rng.exponential(scale=exp_scale)
        elif b in (cls.neg, cls.neg_w0):
            return -rng.exponential(scale=exp_scale)
        elif b in (cls.prob, cls.prob_w0, cls.prob_w1, cls.prob_w01):
            return rng.uniform()
        elif b in (cls.inverse_temperature,):
            low, high = b.value
            return np.clip(rng.exponential(scale=exp_scale), low, high)
        elif b in (cls.unbounded,):
            return rng.standard_t(df=exp_scale) # Kind of arbitrary choice to ensure wide tails?
        assert False, b

    @classmethod
    def in_bounds(cls, b, value):
        low, high = b.value
        return (low is None or low <= value) and (high is None or value <= high)

ArgField = collections.namedtuple('ArgField', ['bounds', 'default_value'])

BaseArgs = collections.namedtuple('BaseArgs', [
    # This field has two types -- it either holds
    # a list of the parameters we sampled, or it holds a
    # dictionary that has the initial values we specified.
    'free_parameters',

    'step_count_coef',
    'noop_count_coef',
    'post_count_coef',

    'a',
    'b',
    'p_normal',
    'p_end',

    'mdl_beta',
    'trace_beta',
    'prior_beta',
])

class Args(BaseArgs):
    free_parameters_field = 'free_parameters'

    field_info = dict(
        step_count_coef=ArgField(BoundTypes.pos, np.nan),
        noop_count_coef=ArgField(BoundTypes.pos, np.nan),
        post_count_coef=ArgField(BoundTypes.pos_w0, np.nan),

        a=ArgField(BoundTypes.prob_w0, np.nan),
        b=ArgField(BoundTypes.pos, np.nan),
        p_normal=ArgField(BoundTypes.prob, np.nan),
        p_end=ArgField(BoundTypes.prob, np.nan),

        mdl_beta=ArgField(BoundTypes.pos, np.nan),
        trace_beta=ArgField(BoundTypes.pos, np.nan), # at present, unused
        prior_beta=ArgField(BoundTypes.pos, np.nan),
    )

    @classmethod
    def new(cls, free_parameters={}, *, rng=None, minimize_param=None, default_values={}):
        '''
        free_parameters: Either a list or dictionary. As a list, it specifies the parameters that should be sampled.
            As a dictionary, specifies default values for parameters that should be fit.
        default_values: Values for fields that should not be fit. Notably, this dictionary should have a distinct set
            of keys from free_parameters.
        '''
        if not isinstance(free_parameters, dict):
            assert rng, 'sampling requires rng'
        if isinstance(free_parameters, dict):
            assert not (free_parameters.keys() & default_values.keys()), f'expected no shared keys b/w default and free, but found {free_parameters.keys() & default_values.keys()}'
        kw = {
            f: (
                (
                    free_parameters[f]
                    if isinstance(free_parameters, dict) else
                    BoundTypes.sample(rng, fi.bounds)
                )
                if f in free_parameters else
                default_values.get(f, fi.default_value)
            )
            for f, fi in cls.field_info.items()
        }
        if minimize_param is not None:
            assert len(minimize_param) == len(free_parameters)
            for f, param in zip(free_parameters, minimize_param):
                kw[f] = param
        a = cls(
            free_parameters=free_parameters,
            **kw,
        )
        a.validate()
        return a

    def to_bounds_kwarg_for_minimize(self):
        return [
            __class__.field_info[f].bounds.value
            for f in self.free_parameters
        ]

    def to_minimize_param(self):
        return np.array([
            getattr(self, f)
            for f in self.free_parameters
        ])

    def validate(self):
        for fi, value in enumerate(self):
            f = __class__._fields[fi]
            if f == __class__.free_parameters_field or np.isnan(value):
                continue
            assert BoundTypes.in_bounds(__class__.field_info[f].bounds, value), f'Field {f} has value {value} that is out of bounds {Args.field_info[f].bounds}'

    def short_repr(self):
        kv = [
            f'{f}={getattr(self, f)}'
            for f in self.free_parameters
        ]
        return f'Args({", ".join(kv)})'

    def to_numba(self):
        return BaseArgs(**{
            field: (
                np.nan
                if field == __class__.free_parameters_field else
                value
            )
            for field, value in zip(__class__._fields, self)
        })

    def items(self):
        return zip(__class__._fields, self)

    def reconstruct_inputs(self):
        '''
        This function generates input arguments for new() that can be used to augment
        the existing arguments.
        '''
        curr_fp = getattr(self, __class__.free_parameters_field)
        new_fp = {}
        new_dv = {}
        for field, value in self.items():
            if field == __class__.free_parameters_field:
                continue
            fi = __class__.field_info[field]
            if field in curr_fp:
                # In this case, we have a free parameter.
                # We can't just pass our current FP, because we might have
                # sampled them during construction, or they could have been
                # set during minimization. So we pack them into a dictionary here.
                new_fp[field] = value
                # if isinstance(curr_fp, dict):
                    # assert value == curr_fp[field], 'sanity check -- however things might have been changed by a minimize param'
            elif value == fi.default_value or (np.isnan(value) and np.isnan(fi.default_value)):
                # We leave implicit the case where value == fi.default_value, since
                # that value will be picked up from fi.
                pass
            else:
                # In this case, a default value was originally passed in
                new_dv[field] = value
        return new_fp, new_dv

    def updated(self, **kwargs):
        # First step is we reconstruct free_parameters, and default values.
        new_fp, new_dv = self.reconstruct_inputs()
        # Now that we have reconstructed parameters, we add in our updated kwargs.
        assert kwargs.keys().isdisjoint(new_fp.keys()), (kwargs.keys(), new_fp.keys())
        new_dv = new_dv | kwargs
        # NOTE Important to avoid passing an RNG, to prevent resampling.
        return __class__.new(free_parameters=new_fp, default_values=new_dv, rng=None)

    def display_pandas(self):
        import pandas as pd
        return pd.DataFrame([
            dict(
                field=f,
                was_fit='optimized' if f in self.free_parameters else 'fixed',
                value=(
                    f if f == __class__.free_parameters_field
                    else '{:.2f}'.format(getattr(self, f))),
            )
            for f in self._fields
        ])


def get_canon_programs(trace):
    rv = []
    counts = program_generation.all_substring_counts(trace, avoid_overlapping=True)
    for program in program_generation.gen_programs(trace, counts):
        # Subroutines are generated in order of length, so we canonicalize their ordering here.
        rv.append(program_analysis.ReorderSubroutineVisitor.reorder_subroutines(program))
    return rv


class TraceSet(object):
    def __init__(self, mdp, traces, astar_results, *, joblib_worker=None):
        self.mdp = mdp
        self.traces = set(traces)
        self.astar_results = astar_results
        self.joblib_worker = joblib_worker

    @functools.cached_property
    def programs(self):
        if self.joblib_worker is None:
            prog_lists = [
                get_canon_programs(trace)
                for trace in self.traces
            ]
        else:
            prog_lists = self.joblib_worker(
                joblib.delayed(get_canon_programs)(trace)
                for trace in self.traces
            )
        ps = {
            program
            for programs in prog_lists
            for program in programs
        }
        # HACK: We convert this to a tuple to avoid any possibility of order issues after deserializing, since sets don't have a guaranteed order.
        return tuple(ps)

    @functools.cached_property
    def program_step_counters(self):
        return {
            p: program_analysis.ProgramStepCounter.count(self.mdp, p)
            for p in self.programs
        }

    @functools.cached_property
    def program_count_matrix(self):
        return scoring.get_program_count_matrix(self.programs, self.mdp)

    @functools.cached_property
    def program_set(self):
        return frozenset(self.programs)

    @classmethod
    def from_astar(cls, base_mdp, *, topk=None, f_upper_bound=None, include_equal_score=None, joblib_worker=None):
        h_ = heuristics.make_heuristic_cost_navigation_to_mst(base_mdp)
        h = lambda mdp, s, _: h_(mdp.mdp, s.state, light_target=s.light_target)
        mdp = envs.LightbotTrace(base_mdp)

        kw = {}
        if topk is not None: kw['topk'] = topk
        if f_upper_bound is not None: kw['f_upper_bound'] = f_upper_bound
        if include_equal_score is not None: kw['include_equal_score'] = include_equal_score
        r = search.astar(mdp, h, **kw)
        assert r.non_monotonic_counter == 0, f'Heuristic is not monotonic: # of times {r.non_monotonic_counter}'

        return cls(
            base_mdp,
            [row.state.trace for row in r.results],
            r._replace(results=None), # removing results to make things smaller, mostly keeping this to see iteration counts
            joblib_worker=joblib_worker,
        )

    def __repr__(self):
        if not self.traces:
            return f'TraceSet #{len(self.traces)}'
        lens = [len(t) for t in self.traces]
        return f'TraceSet #{len(self.traces)} #programs={len(self.programs)} trace range {min(lens)}-{max(lens)} iters={self.astar_results.iteration_count} non-monotonic={self.astar_results.non_monotonic_counter}'

    @functools.lru_cache
    def _data_not_in_enum(self, data_programs):
        data_not_in_enum = np.array([
            program not in self.program_set
            for program in data_programs
        ], dtype=bool)
        return data_not_in_enum

    def compute_scores_and_partition(self, data: 'BehaviorSummary', a: Args, *, batch_score=None):
        orig_batch_score = batch_score
        def batch_score(programs, a, program_count_matrix, **kw):
            return orig_batch_score(program_count_matrix, a.to_numba())

        enum_scores = batch_score(
            self.programs,
            a,
            base_mdp=self.mdp,
            program_count_matrix=self.program_count_matrix,
        )

        data_scores = batch_score(
            data.programs,
            a,
            base_mdp=self.mdp,
            program_count_matrix=data.program_count_matrix,
        )

        data_not_in_enum = self._data_not_in_enum(data.programs)
        assert len(data_scores) == len(data_not_in_enum)

        unique_scores = np.concatenate([enum_scores, data_scores[data_not_in_enum]])
        assert (len(self.program_set | data.program_set),) == unique_scores.shape, unique_scores.shape

        log_partition = logsumexp(unique_scores)
        normalized_sum = np.exp(unique_scores - log_partition).sum()
        assert np.isclose(1, normalized_sum), f'sanity check for no numerical issues, but normalized sum was {normalized_sum}'

        return dict(
            enum_scores=enum_scores,
            data_scores=data_scores,
            log_partition=log_partition,
        )


    def crossent(self, data: 'BehaviorSummary', a: Args, *, batch_score=None, return_normalized_scores=False):
        s = self.compute_scores_and_partition(data, a, batch_score=batch_score)
        normalized_scores = s['data_scores'] - s['log_partition']
        if return_normalized_scores:
            return normalized_scores
        crossent = 0
        for pi, p in enumerate(data.programs):
            ct = data.program_counts[p]
            crossent += -ct * normalized_scores[pi]

        assert not np.isinf(crossent)
        assert not np.isnan(crossent)
        return crossent

    def discrepancy_report(self, data: 'BehaviorSummary', *, show=False, brief=False):
        def _log(prefix, full_cts, full, *, print_missing=False):
            for name, cts in [
                ('full', full_cts),
                ('by 2+', {k: ct for k, ct in full_cts.items() if ct >= 2}),
            ]:
                total = sum(cts.values())
                missing = cts.keys() - full
                missing_total = sum(cts[k] for k in missing)
                msg = (
                    f'{prefix} ({name}) missing '
                    f'total={missing_total}/{total} ({100*missing_total/total:.2f}%) '
                    f'unique={len(missing)}/{len(cts)} ({100*len(missing)/len(cts):.2f}%) '
                )
                if missing:
                    msg += f'avg count per unique={missing_total}/{len(missing)}={missing_total/len(missing):.2f}'
                args = [msg]
                if print_missing:
                    args.append(missing)
                print(*args)

        _log('trace', data.trace_counts, self.traces, print_missing=True)
        _log('program', data.program_counts, self.programs, print_missing=False)
        # missing_traces = data.trace_counts.keys() - self.traces
        # msg = f'missing traces count={len(missing_traces)}'
        # if len(missing_traces):
        #     msg += f' avg count per missing trace {sum(data.trace_counts[t] for t in missing_traces)/len(missing_traces)}'
        # print(msg, missing_traces)

        # missing_programs = data.program_counts.keys() - self.programs
        # msg = f'missing programs count={len(missing_programs)}'
        # if len(missing_programs):
        #     msg += f' avg count per missing program {sum(data.program_counts[p] for p in missing_programs)/len(missing_programs)}'
        # print(msg)

        max_len = max(len(t) for t in self.traces)
        behavior_lens = np.array([
            len(t) for t, ct in data.trace_counts.items()
            for _ in range(ct)])
        p_within = (behavior_lens <= max_len).sum() / len(behavior_lens)
        print(f'Participant % with trace within max length {100*p_within:.2f}% (assuming astar with include_equal_score)')

        print()

        if brief: return

        missing_programs = data.program_counts.keys() - self.programs
        for program in missing_programs:
            print(plotting.repr_program(program))
            rr, rp = envs.interpret_program_record_path(self.mdp, program)
            t = program_analysis.canonicalized_trace(self.mdp, program)
            true_t = ''.join([a for s, a in rp if a])
            print('true.', true_t, 'reward', rr.reward)
            print('canon', t)
            print(
                f'{program_analysis.ProgramStepCounter.count(self.mdp, program)}',
                f'ct={data.program_counts[program]}',
                f'recur={envs.is_recursive(program)}',
                f'has_canon_trace={t in self.traces}',
                f'has_true_trace={true_t in self.traces}')
            if show:
                plotting.plot_program(self.mdp, program)
                import matplotlib.pyplot as plt
                plt.show()
            print()

    def results_table(self, data: 'BehaviorSummary', a: Args, *, topk=30, batch_score=None, raw_program=False):
        float_fmt = '{:,.2f}'.format

        if batch_score is None:
            batch_score = scoring.batch_score_mdl
            a = Args.new({}, default_values=dict(mdl_beta=1))
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import display, HTML

        r = self.compute_scores_and_partition(data, a, batch_score=batch_score)
        data_scores = r['data_scores']
        enum_scores = r['enum_scores']
        logZ = r['log_partition']
        assert np.isclose(1, np.exp(enum_scores - logZ).sum()), 'sanity check for no numerical issues'

        best = heapq.nsmallest(topk, [
            (-score, prog)
            for scores, progs in [
                (data_scores, data.programs),
                (enum_scores, self.programs),
            ]
            for score, prog in zip(scores, progs)
        ])
        # We include the topk programs, and also make sure to include all participant programs
        progs_to_show = {p: -neg_score for neg_score, p in best} | {p: data_scores[pi] for pi, p in enumerate(data.programs)}

        pp = sorted([
            (-score, p) for p, score in progs_to_show.items()
        ])

        grp = {}
        path_to_idx = {}
        program_to_trace = {}
        program_to_cost = {}
        for _, p in pp:
            path = program_analysis.canonicalized_trace(self.mdp, p)
            program_to_trace[p] = path
            grp.setdefault(path, 0)
            grp[path] += data.program_counts.get(p, 0)
            if path not in path_to_idx:
                path_to_idx[path] = len(path_to_idx)

            ct = program_analysis.ProgramStepCounter.count(self.mdp, p)
            # HACK: copy/pasted
            post = ct.step_post + ct.step_post_noop
            cost = a.step_count_coef * ct.step + a.noop_count_coef * ct.step_noop + a.post_count_coef * post
            program_to_cost[p] = cost

        size = 1.5
        aspect = self.mdp.map_h.shape[0] / self.mdp.map_h.shape[1]
        desired_width_in = 12
        width = int(desired_width_in / (size * aspect))
        height = int(np.ceil(len(path_to_idx) / width))
        f, axes = plt.subplots(height, width, figsize=(size*aspect*width, size*height))
        axes = axes.flatten().tolist()
        for path, path_idx in path_to_idx.items():
            ax = axes.pop(0)
            plotting.plot_program(self.mdp, envs.Program(main=path, subroutines=()), frames=False, ax=ax, scale=1/2)
            ax.set_title(f'trace {path_idx}')
        for ax in axes:
            ax.axis('off')
        plt.show()

        df = []
        for negscore, p in pp:
            df.append(dict(
                Main=plotting.repr_instructions(p.main),
                P1=plotting.repr_instructions(p.subroutines[0]),
                P2=plotting.repr_instructions(p.subroutines[1]),
                P3=plotting.repr_instructions(p.subroutines[2]),
                P4=plotting.repr_instructions(p.subroutines[3]),
                score=-negscore,
                log_posterior=-negscore - logZ,
                posterior_percent=100*np.exp(-negscore - logZ),
                log_prior=envs.pyp_process_prior(p, a.a, a.b, p_normal=a.p_normal, p_end=a.p_end, p_subprocess_end=a.p_end),
                traceID=path_to_idx[program_to_trace[p]],
                cost=program_to_cost[p],
                trace_len=len(program_to_trace[p]),
                hier_len=len(p.main) + sum(len(sr) for sr in p.subroutines),
                **{
                    'match\\nbehavior\\n(program)': data.program_counts[p],
                    'match\\nbehavior\\n(trace)': data.trace_counts[program_to_trace[p]]
                },
                note='(from behavior / not in search)' if p not in self.programs else '',
            ))
            if raw_program:
                df[-1]['raw_program'] = f'{p.main}|{"|".join(p.subroutines)}'
        df = pd.DataFrame(df)
        # remove empty subroutines
        for col in df.columns:
            if not df[col].any():
                del df[col]

        # HACK: display.float_format not working below
        for col, dtype in list(df.dtypes.items()):
            if dtype == np.double:
                df[col] = df[col].apply(float_fmt)

        with pd.option_context('display.max_rows', df.shape[0], 'display.float_format', float_fmt):
            # HACK: only replacing commented-out newlines to avoid adding weird extra space
            display(HTML(
                df.style.set_properties(**{
                    'border-left': '1px solid #bdbdbd !important',
                    'text-align': 'left',
                })
                .to_html().replace("\\n","<br>")))



class BehaviorSummary(object):
    def __init__(self, mdp, program_counts):
        self.mdp = mdp
        self.program_counts = program_counts

    @property
    def programs(self):
        # We return a tuple, to avoid any possibility of order issues after deserializing
        return tuple(self.program_counts.keys())

    @functools.cached_property
    def program_step_counters(self):
        return {
            p: program_analysis.ProgramStepCounter.count(self.mdp, p)
            for p in self.programs
        }

    @functools.cached_property
    def trace_counts(self):
        counter = collections.Counter()
        for p, ct in self.program_counts.items():
            t = program_analysis.canonicalized_trace(self.mdp, p)
            counter[t] += ct
        return counter

    @functools.cached_property
    def program_count_matrix(self):
        return scoring.get_program_count_matrix(self.programs, self.mdp)

    @functools.cached_property
    def program_set(self):
        return frozenset(self.programs)

    def __repr__(self):
        tlens = [len(t) for t in self.trace_counts.keys()]
        p50, p75, p95, p99 = np.quantile(
            [len(t) for t, ct in self.trace_counts.items() for _ in range(ct)],
            [.5, .75, .95, .99])
        return f'Behavior #participants={sum(self.program_counts.values())} trace range={min(tlens)}-{max(tlens)} perc(50={p50:.2f}, 75={p75:.2f}, 95={p95:.2f}, 99={p99:.2f})'


@dataclass
class FitResult:
    seed: float
    x0s: List[Args]
    results: List[Any]
    dof: int
    observation_count: int

    @property
    def result(self):
        return min(self.results, key=lambda r: r['result'].fun)

    @property
    def log_likelihood(self):
        # Note: this assumes the function's result is a negative LL
        ll = -self.result['result'].fun
        return ll

    @property
    def aic(self):
        return 2 * self.dof - 2 * self.log_likelihood

    @property
    def bic(self):
        assert self.observation_count is not None
        return self.dof * np.log(self.observation_count) - 2 * self.log_likelihood

    def likelihood_ratio_test(self, null: 'FitResult'):
        # https://stackoverflow.com/a/38249020
        from scipy.stats.distributions import chi2
        def likelihood_ratio(llmin, llmax):
            assert llmin <= llmax, (llmin, llmax)
            return(2*(llmax-llmin))

        # We convert both to lists, since in some cases free_parameters is a dictionary, in others a list
        self_params = set(self.result['final_args'].free_parameters)
        null_params = set(null.result['final_args'].free_parameters)
        assert null_params.issubset(self_params), f'Expected null params ({null_params}) to be a subset of self params ({self_params})'

        LR = likelihood_ratio(null.log_likelihood, self.log_likelihood)
        p = chi2.sf(LR, self.dof - null.dof)
        return p

    def __repr__(self):
        funs = [r['result'].fun for r in self.results]
        succ = sum(r['result'].success for r in self.results)
        p = self.result['final_args'].short_repr()
        elapsed = [r['elapsed_seconds'] for r in self.results]
        return f'FitResult(range=[{min(funs):.02f}, {max(funs):.02f}], success={succ}/{len(self.results)}, best_params={p}, elapsed_sec_range=[{min(elapsed):.2f}, {max(elapsed):.2f}])'


def integrate_nuisance(fn, arg, nuisance_params):
    assert len(nuisance_params) == 1
    nuisance_key, nuisance_values = list(nuisance_params.items())[0]
    assert np.isnan(getattr(arg, nuisance_key)), 'Nuisance parameter should not be set'
    return sum(
        fn(arg.updated(**{nuisance_key: value}))
        for value in nuisance_values
    ) / len(nuisance_values)

def logmeanexp(arr):
    '''
    Copying the implementation in TensorFlow probability.
    Uses logsumexp to avoid over/underflow issues, then subtracts the log-length to compute the log-mean.
    https://github.com/tensorflow/probability/blob/v0.22.0/tensorflow_probability/python/math/generic.py#L221
    '''
    return logsumexp(arr) - np.log(len(arr))

def integrate_nuisance_for_negative_log_fn(negative_log_fn, arg, nuisance_params):
    assert len(nuisance_params) == 1
    nuisance_key, nuisance_values = list(nuisance_params.items())[0]
    assert np.isnan(getattr(arg, nuisance_key)), 'Nuisance parameter should not be set'
    return -logmeanexp([
        -negative_log_fn(arg.updated(**{nuisance_key: value}))
        for value in nuisance_values
    ])

def fit(
        cost,
        *,
        progress=None,
        nsampled=4,
        seed=None,
        default_values={},
        free_parameters=None,
        debug=False,
        extra_dof=0,
        observation_count=None,
    ):
    if seed is None:
        seed = np.random.randint(0, 2**30)
        if debug: print('seed', seed)
    if debug:
        progress = 10
    rng = np.random.RandomState(seed)

    assert free_parameters is not None

    # A special case for null models
    if len(free_parameters) == 0:
        assert extra_dof == 0, 'not handled'

        args = Args.new(free_parameters, default_values=default_values)

        start = time.time()
        c = cost(args)
        elapsed = time.time() - start

        result = dict(
            result=scipy.optimize.OptimizeResult(
                fun=c,
                success=True,
                no_parameters=True,
            ),
            final_args=args,
            elapsed_seconds=elapsed,
        )

        return FitResult(
            seed=None,
            x0s=None,
            results=[result],
            dof=0,
            observation_count=observation_count,
        )

    x00 = Args.new(free_parameters, default_values=default_values)

    if debug:
        with timed("timing of cost()"):
            print(x00, cost(x00))

    results = []

    args = [x00] + [
        Args.new(list(free_parameters.keys()), rng=rng, default_values=default_values) for _ in range(nsampled)
    ]
    for arg in args:
        if debug: print('x0', arg)

        start = time.time()
        r = minimize(
            lambda x: cost(Args.new(free_parameters, minimize_param=x, default_values=default_values)),
            arg.to_minimize_param(),
            progress=progress,
            bounds=arg.to_bounds_kwarg_for_minimize(),
        )
        elapsed = time.time() - start

        args = Args.new(free_parameters, minimize_param=r.x, default_values=default_values)
        assert cost(args) == r.fun, ('Difference between rerun cost and cost from fitting: rerun', cost(args), 'prev', r.fun, 'args', args)

        if debug:
            print(f'success={r.success} message={r.message} fun={r.fun} nfev={r.nfev} nit={r.nit} elapsed={elapsed:.2f}s')
            print(f'final args={args}')
            print()

        results.append(dict(
            result=r,
            final_args=args,
            elapsed_seconds=elapsed,
        ))

    return FitResult(
        seed=seed,
        x0s=args,
        results=results,
        dof=len(free_parameters) + extra_dof,
        observation_count=observation_count,
    )


CACHE_DIR = pathlib.Path(__file__).parent / '.cache'

@dataclass
class DatasetRow:
    mdp_name: tuple[str, int]
    base_mdp: ...
    bs: BehaviorSummary
    ts: TraceSet
    elapsed: dict[str, float]


class CachedFile:
    def __init__(self, name):
        self.name = name
        self.fn = CACHE_DIR / self.name
        self.obj = None

        self.load()

    def save(self, obj):
        '''
        Note: We lzma a pickled version of this class. Previously tried joblib.dump, but that seems to struggle
        with serializing programs, a couple orders of magnitude slower at serializing a list of programs
        than JSON or pickle.
        '''
        CACHE_DIR.mkdir(exist_ok=True)
        with lzma.open(self.fn, 'wb') as f:
            pickle.dump(obj, f)

        # Now that it's saved, we add as attribute.
        self.obj = obj

    def load(self):
        if not self.fn.exists():
            return

        import datetime
        mtime = self.fn.stat().st_mtime
        tz = datetime.timezone.utc
        mtime = datetime.datetime.fromtimestamp(mtime, tz=tz)
        now = datetime.datetime.now(tz=tz)

        start = time.time()
        try:
            with lzma.open(self.fn) as f:
                obj = pickle.load(f)
        except Exception as e:
            print('Skipping existing file due to error:', str(e))
            return

        elapsed = time.time() - start
        print(f'Loading from cached file took {elapsed:.2f} seconds. Last modified {mtime.isoformat()} ({now - mtime} ago)')

        self.obj = obj

    def __call__(self, func):
        import functools

        @functools.wraps(func)
        def wrapped():
            if self.obj is not None:
                return self.obj
            obj = func()
            self.save(obj)
            return obj

        return wrapped

class Dataset:
    def __init__(self, d: list[DatasetRow], search_params, overrides):
        self.d = d
        self.search_params = search_params
        self.overrides = overrides

    @classmethod
    def from_data(cls, data, search_params, *, overrides={}, parallel=False, load_from_cache=True):
        if 'quantile' in search_params:
            search_params.setdefault('topk', 10000)
        else:
            search_params.setdefault('topk', 250)

        # Now that argument parsing is complete, we look for any cached files
        if load_from_cache:
            fn = __class__._fn(search_params, overrides, ext='.bin.lzma')
            cached_file = CachedFile(fn)
            obj = cached_file.obj
            if obj is not None:
                obj.validate()
                return obj

        worker = None
        if parallel:
            # can optionally be kws
            parallel_kws = {} if parallel is True else parallel
            worker = joblib.Parallel(
                n_jobs=parallel_kws.get('n_jobs', 8),
                verbose=parallel_kws.get('verbose', 2))

        d = []
        for mdp_name, full_cts in data.programs_by_count().items():
            params = dict(search_params, **overrides.get(mdp_name, {}))

            base_mdp = exp.mdp_from_name(mdp_name)
            base_mdp.noop_reward = float('-inf')

            bs = BehaviorSummary(base_mdp, full_cts)

            f_upper_bound = None
            if 'quantile' in params:
                trace_len_cts = [
                    len(t)
                    for t, ct in bs.trace_counts.items()
                    for _ in range(ct)] # frequency weighted
                quantile_thresh = np.quantile(trace_len_cts, params['quantile'], method='higher')
                # We add one, because it is not inclusive.
                f_upper_bound = quantile_thresh + 1

            elapsed = {}

            # First, we find traces.
            start = time.time()
            ts = TraceSet.from_astar(
                base_mdp,
                topk=params['topk'],
                include_equal_score=True,
                f_upper_bound=f_upper_bound,
                joblib_worker=worker,
            )
            elapsed['enum:trace'] = time.time() - start

            # Then, we expand them into possible programs.
            start = time.time()
            ts.programs
            elapsed['enum:program'] = time.time() - start

            # Finally, we generate count matrices.
            start = time.time()
            ts.program_count_matrix
            bs.program_count_matrix
            elapsed['enum:program_count_matrix'] = time.time() - start

            # HACK: We need the worker in there for enumeration, but have to remove before saving to avoid errors.
            res = DatasetRow(mdp_name, base_mdp, bs, ts, elapsed)
            res.ts.joblib_worker = None
            d.append(res)

        obj = cls(d, search_params, overrides)
        if load_from_cache:
            cached_file.save(obj)
        return obj

    @staticmethod
    def _strdict(d):
        '''
        Serializes a dictionary into a format safe for filenames, assuming keys and value are safe for filenames. Ensures
        serialization is sorted, to avoid issues with key ordering.
        >>> Dataset._strdict(dict(f=3, g=4, a=2))
        'a=2_f=3_g=4'
        '''
        return '_'.join(f'{k}={v}' for k, v in sorted(d.items()))

    @staticmethod
    def _fn(search_params, overrides, *, ext=''):
        fn = __class__._strdict(search_params)
        if overrides:
            fn += '_overrides_' + __class__._strdict({
                '-'.join(map(str, k)): __class__._strdict(v)
                for k, v in sorted(overrides.items())
            })
        return fn + ext

    def show(self, *, plot=False):
        SEP = '-'*20

        cts = collections.Counter()
        cts_2plus = collections.Counter()

        for res in self.d:
            print(SEP)
            if plot:
                import matplotlib.pyplot as plt
                plotting.plot(res.base_mdp)
                plt.show()
            print(res.mdp_name, len(res.ts.programs), 'elapsed', {k: f'{v:.3f}s' for k, v in res.elapsed.items()})
            print(res.bs)
            print(res.ts)
            res.ts.discrepancy_report(res.bs, brief=True)

            # Duplicating logic from elsewhere
            def _inc(cts, behavior_cts):
                missing_programs = behavior_cts.keys() - res.ts.programs
                cts['program_ct'] += sum(behavior_cts.values())
                cts['missing_program_ct'] += sum(behavior_cts[p] for p in missing_programs)
            _inc(cts, res.bs.program_counts)
            _inc(cts_2plus, {k: ct for k, ct in res.bs.program_counts.items() if ct >= 2})
        print(SEP)

        for name, c in [
            ('All programs', cts),
            ('Programs by 2+', cts_2plus),
        ]:
            program_ct = c['program_ct']
            missing_program_ct = c['missing_program_ct']
            print(f'{name} {program_ct=} {missing_program_ct=} missing={100*missing_program_ct/program_ct:.2f}%')

    def crossent(self, args, *, batch_score=None):
        return sum([
            res.ts.crossent(res.bs, args, batch_score=batch_score)
            for res in self.d
        ])

    def observation_count(self):
        return sum([
            ct
            for res in self.d
            for prog, ct in res.bs.program_counts.items()
        ])

    def fit(self, batch_score, *, debug=False, free_parameters=None, default_values={}, extra_dof=0, nuisance_params=None):
        def original_loss_fn(a):
            return self.crossent(a, batch_score=batch_score)

        if nuisance_params is None:
            loss_fn = original_loss_fn
        else:
            # Special case for nuisance parameters
            def loss_fn(a):
                return integrate_nuisance_for_negative_log_fn(original_loss_fn, a, nuisance_params)

        return fit(
            loss_fn,
            default_values=default_values,
            free_parameters=free_parameters,
            debug=debug,
            extra_dof=extra_dof,
            observation_count=self.observation_count(),
        )

    def validate(self):
        '''
        At some point, serialization / deserialization could scramble program order, because of the
        use of a set of programs, which don't have guaranteed ordering. Now, we ensure storage
        is ordered. This validation step is a check to make sure ordering is correct.
        '''
        num_rows_to_check = 100
        for res in self.d:
            for x in [res.ts, res.bs]:
                ps = list(x.programs)[:num_rows_to_check]
                existing = x.program_count_matrix[:num_rows_to_check]
                recomputed = scoring.get_program_count_matrix(ps, x.mdp, tqdm_disable=True)
                assert (existing == recomputed).all()

def default_program_enumeration(D: exp.Data):
    return Dataset.from_data(D, dict(topk=1000), load_from_cache=True)

def analysis(dc: Dataset):
    fn = dc._fn(dc.search_params, dc.overrides, ext='-progll.bin.lzma')

    @CachedFile(fn)
    def run():
        default_values = dict(
            a=0,
        )

        nuisance_params = dict(
            p_end=np.linspace(0.1, 0.9, 9),
        )
        print('nuisance_params', nuisance_params)

        reward_free_parameters = dict(step_count_coef=1)

        models = {}

        models['score_reuse_and_reward'] = dc.fit(
            scoring.batch_score_reuse_and_reward,
            free_parameters=dict(b=1, prior_beta=1, p_normal=1/2, **reward_free_parameters),
            default_values=default_values,
            nuisance_params=nuisance_params,
            debug=True)
        print(models['score_reuse_and_reward'])

        models['score_reuse'] = dc.fit(
            scoring.batch_score_reuse,
            free_parameters=dict(b=1, prior_beta=1, p_normal=1/2),
            default_values=default_values,
            nuisance_params=nuisance_params,
            debug=True)
        print(models['score_reuse'])

        models['score_mdl_and_reward'] = dc.fit(
            scoring.batch_score_mdl_and_reward,
            free_parameters=dict(mdl_beta=1, **reward_free_parameters))
        print(models['score_mdl_and_reward'])

        models['score_mdl'] = dc.fit(
            scoring.batch_score_mdl,
            free_parameters=dict(mdl_beta=1))
        print(models['score_mdl'])

        models['score_reward'] = dc.fit(
            scoring.batch_score_reward,
            free_parameters=reward_free_parameters)
        print(models['score_reward'])

        models['score_null'] = dc.fit(
            scoring.batch_score_null,
            free_parameters=dict())
        print(models['score_null'])

        return models

    models = run()
    for k, v in models.items():
        print(k, v)

    return models

def plot_analysis(models):
    import journal.figs as figs
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.barplot(
        y='Model',
        x='Log Likelihood',
        data=pd.DataFrame([
            {
                'Model': m.name_label,
                'Log Likelihood': -models[m.score_key].result['result'].fun,
            }
            for m in figs.models.all
        ]),
    )
    plt.ylabel('')
