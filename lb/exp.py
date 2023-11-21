from . import envs, plotting, program_analysis, tools
import pandas as pd
import os
import json
import numpy as np
import scipy.stats
import collections
import functools
from tqdm.auto import tqdm
import pathlib
import warnings

codedir = pathlib.Path(__file__).absolute().parent
expdir = codedir / '../../cocosci-lightbot'
datadir = expdir / 'data/human_raw'
mapdir = expdir / 'static/lightbot/json'
cachedir = codedir / '../experiment'

def _cached_file(fn):
    '''
    This function copies relevant files over from the experiment codebase.
    '''
    fn: pathlib.Path = pathlib.Path(fn)
    cached_fn = cachedir / fn.relative_to(expdir)
    if not cached_fn.exists():
        cached_fn.parent.mkdir(parents=True, exist_ok=True)
        cached_fn.write_bytes(fn.read_bytes())
    return cached_fn

import sys
sys.path.append(str(codedir / '../generate-exp'))


class ProgramSerDe(object):
    @staticmethod
    def deserialize(p):
        return tools.mkprog(*p.split('|'))
    @staticmethod
    def serialize(p):
        return p.main + '|' + '|'.join(p.subroutines)

def _load_order_task_timeline_from_json(data):
    return [
        {
            **t,
            'program': ProgramSerDe.deserialize(t['program']),
        }
        for t in data
    ]

def _load_order_task_timeline_from_config(data):
    return [
        {
            **data['tasks'][mdp],
            'program': ProgramSerDe.deserialize(data['tasks'][mdp]['program']),
        }
        for mdp in data['mdp_order']
    ]

map_sources = {
    "cgcMaps": envs.EnvLoader._load_maps(_cached_file(os.path.join(mapdir, 'cgc-maps.json'))),
    "maps": envs.EnvLoader._load_maps(_cached_file(os.path.join(mapdir, 'maps.json'))),
}

def mdp_from_name(mdp_name):
    source, idx = mdp_name
    return map_sources[source][idx]


def clip(val, min_, max_):
    '''
    >>> assert clip(1, 0, 10) == 1
    >>> assert clip(0, 0, 10) == 0
    >>> assert clip(-1, 0, 10) == 0
    >>> assert clip(10, 0, 10) == 10
    >>> assert clip(11, 0, 10) == 10
    '''
    assert min_ <= max_
    return max(min_, (min(max_, val)))


def hier_len(p):
    return len(p.main) + sum(len(sr) for sr in p.subroutines)

def subroutine_count(p):
    return sum(1 for sr in p.subroutines if sr)

def df_data_filt(df, pred):
    return df[df.data.apply(pred)]


class OrderTaskRow(object):
    def __init__(self, row, config):
        self.row = row
        self.config = config
        assert self.config['addToData'] == self.row.data['trialConfig']

        # Not sure this is a meaningful check, but...
        is_flat_prog = self.program.endswith('|'*4)
        is_flat_conf = self.split_type[1] == 'flat'
        assert is_flat_prog == is_flat_conf, (is_flat_prog, is_flat_conf)

    @property
    def elapsed_minutes(self):
        d = self.row.data
        return (d['end'] - d['start']) / (1000 * 60)
    @property
    def mdp_name(self):
        d = self.row.data
        return tuple(d['trialConfig']['source'])
    @property
    def mdp(self):
        return mdp_from_name(self.mdp_name)
    @property
    def timeline_entry(self):
        return self.config
    @classmethod
    def run_for_light_order(cls, mdp, program):
        _, path = envs.interpret_program_record_path(mdp, program)
        return [
            list(s.position) # to have same data types as JSON input
            for s, a in path
            if a == envs.INSTRUCTIONS.LIGHT
        ]
    def is_correct(self):
        true_boxes = OrderTaskRow.run_for_light_order(self.mdp, self.timeline_entry['program'])
        return self.row.data['boxes'] == true_boxes
    @property
    def program(self):
        return self.row.data['trialConfig']['program']
    @property
    def parsed_program(self):
        return ProgramSerDe.deserialize(self.program)
    @property
    def type(self):
        return self.row.data['trialConfig']['type']
    @property
    def split_type(self):
        return self.type.split('-')


class TaskRow(object):
    def __init__(self, row):
        self.row = row

    @property
    def elapsed_minutes(self):
        d = self.row.data
        return (d['end'] - d['start']) / (1000 * 60)

    @property
    def program(self):
        last_complete = self.row.data['complete'][-1]
        assert last_complete['success']
        p = last_complete['program']
        return type(self)._json_prog_to_py(p)

    @classmethod
    def _json_prog_to_py(cls, p):
        return envs.Program(
            main=p['main'],
            subroutines=tuple(p[f'process{i+1}'] for i in range(4)),
        )

    @property
    def canonical_program(self):
        return program_analysis.canonicalize_program(self.program, mdp=self.mdp)

    @property
    def mdp_name(self):
        return tuple(self.row.data['trialConfig']['source'])

    @property
    def mdp(self):
        return mdp_from_name(self.mdp_name)

    @functools.cached_property
    def skipped(self):
        s = self.row.data.get('skip')
        if s:
            return s[-1].get('skipped', False)
        else:
            return False

    def plot(self):
        plotting.plot_program(self.mdp, self.program)

    def bonus(self, max_bonus, min_solution, instruction_cost):
        if self.skipped:
            return np.nan
        return clip(
            max_bonus - instruction_cost * (hier_len(self.program) - min_solution),
            0,
            max_bonus,
        )

    def editor_events(self):
        canon_label = {
            'Correr': 'Run',
            'Reiniciar': 'Reset',
            'Deténgase': 'Stop',
            'Carrera rápida⚡️': 'Quick Run⚡️',
        }
        events = self.row.data['clicks'] + self.row.data['programUpdates']
        events = sorted(events, key=lambda row: row['time'])
        # copy events
        events = [dict(e) for e in events]
        # fix labeling issue
        for e in events:
            if e.get('label') in canon_label:
                # This fixes an issue where someone used translation software
                e['label'] = canon_label[e['label']]
        return events

    def editor_runs(self):
        rv = []
        events = {'Run', 'Quick Run⚡️'}
        for e in self.editor_events():
            if e.get('label') in events:
                rv.append(e)
        return rv


class Participant(object):
    def __init__(self, worker, rows: pd.DataFrame, questions, config):
        self.worker = worker
        self.rows = rows
        self.questions = questions
        self.config = config
        self._add_elapsed()

    def _add_elapsed(self, *, difference_threshold_ms=30):
        r = self.rows.copy()

        te = r.data.apply(lambda d: d['time_elapsed'])
        r['duration_ms'] = np.diff(te, prepend=0)

        for row in r.itertuples():
            d = row.data
            if 'start' in d:
                start_end_dur = d['end'] - d['start']
                assert start_end_dur <= row.duration_ms, f'Expected total duration {row.duration_ms} to be smaller than end-start difference {start_end_dur}.'
                diff = abs(row.duration_ms - start_end_dur)
                big_difference = diff >= difference_threshold_ms
                if big_difference:
                    print(f'Large difference {diff}ms between duration {row.duration_ms}ms and end-start difference {start_end_dur}ms for pid={row.worker} trial={row.trial}')

        self.rows = r

    @property
    def condition(self):
        return Data.condition_from_question(self.questions)

    def filtered_rows(self, pred=lambda d: True, **kwargs):
        return df_data_filt(self.rows, lambda d:
            pred(d) and
            all(d[key] == value for key, value in kwargs.items())
        )

    def elapsed_minutes(self):
        t = self.rows.time.values
        return (np.max(t) - np.min(t)) / (1000 * 60)

    def programming_experience(self):
        row = next(self.filtered_rows(trial_type='survey-multi-choice').itertuples())
        q_to_prompt = {
            'Q0': "How much experience do you have with computer programming?",
            'Q1': "Have you played Lightbot or another similar programming game before?",
        }
        q_to_prompt = {
            'Q0': "programming-exp",
            'Q1': "programming-game-exp",
        }
        return {
            q_to_prompt[key]: val
            for key, val in json.loads(row.data['responses']).items()
        }

    @functools.cache
    def task_rows(self):
        return [
            TaskRow(row)
            for row in self.filtered_rows(lambda d: d['trial_type'] == 'LightbotTask' and not d['trialConfig']['practice']).itertuples()
        ]

    @property
    def order_task_timeline(self):
        return self.config['light_order_timeline']

    def order_task_rows(self):
        rows = list(self.filtered_rows(trial_type='LightbotLightOrderTask').itertuples())
        start_trial = min([row.trial for row in rows])
        return [
            OrderTaskRow(row, self.order_task_timeline[row.trial - start_trial])
            for row in rows
        ]

    def trial_type_sequence(self):
        return tuple([
            row.data['trial_type']
            for row in self.rows.itertuples()])

    def find_compatible_subsequence(self, seq):
        trial_seq = self.trial_type_sequence()
        assert trial_seq != seq, 'Participant only has this sequence.'
        assert len(seq) < len(trial_seq), 'Participant has too few trials.'
        for sl in [
            slice(None, len(seq)),
            slice(-len(seq), None),
        ]:
            assert len(trial_seq[sl]) == len(seq), 'sanity'
            if trial_seq[sl] == seq:
                break
        else:
            raise Exception('Trials did not start or end with subsequence.')
        return Participant(self.worker, self.rows.iloc[sl], self.questions, self.config)



class Data(object):
    def __init__(self, version, trial, questions):
        self.version = version
        self.trial = trial
        self.questions = questions

        self.questions_by_worker = {
            worker: {row.key: row.value for row in rows.itertuples()}
            for worker, rows in self.questions.groupby('worker')
        }
        self.participants = [
            Participant(
                worker,
                rows,
                self.questions_by_worker[worker],
                self._config_for_condition(Data.condition_from_question(self.questions_by_worker[worker])),
            )
            for worker, rows in self.trial.groupby('worker')
        ]

    @classmethod
    def condition_from_question(cls, q):
        return int(q['condition'])

    def _version_exec(self, fns, *args, **kwargs):
        defn = lambda *a, **k: None
        fn = fns.get(self.version) or fns.get('default') or defn
        return fn(*args, **kwargs)

    @tools.method_cache
    def _config_for_condition(self, condition):
        assert isinstance(condition, int)
        return self._version_exec({
            '0.5': lambda: dict(
                light_order_timeline=_load_order_task_timeline_from_config(
                    rrtd_generate.config_for_condition(
                        self._load_from_git(f'static/lightbot/json/light-order-configuration.json'),
                        condition,
                    )
                )
            ),
            '0.3': lambda: dict(
                # HACK: because of a bug with conditioning, all were in condition=0
                light_order_timeline=_load_order_task_timeline_from_json(self._load_from_git(
                    f'static/lightbot/json/light-order-timeline.random0.json')),
            ),
            'default': lambda: dict(
                light_order_timeline=_load_order_task_timeline_from_json(json.loads(
                    _cached_file(pathlib.Path(mapdir) / 'light-order-timeline.json').read_text()))
            ),
        })

    def _load_from_git(self, fn):
        import subprocess
        data = subprocess.check_output([
            'git',
            '--git-dir', expdir+'/.git',
            'show',
            f'v{self.version}:{fn}',
        ])
        if fn.endswith('.json'):
            return json.loads(data)
        elif fn.endswith('.js'):
            return json.loads(data[len('export default '):-len(';')])
        else:
            return data

    @classmethod
    def load(cls, version):
        trial = pd.read_csv(_cached_file(os.path.join(datadir, version, 'trialdata.csv')), names=['worker', 'trial', 'time', 'data'])
        trial['data'] = trial['data'].apply(lambda d: json.loads(d))
        questions = pd.read_csv(_cached_file(os.path.join(datadir, version, 'questiondata.csv')), names=['worker', 'key', 'value'])
        return cls(version, trial, questions)

    def can_pay(self, *, include_as_valid=[]):
        def serialize(seq):
            return ','.join(seq)

        modal_seq = serialize(self.modal_trial_sequence())

        valid = set(include_as_valid) | {
            p.worker
            for p in self.participants
            if modal_seq in serialize(p.trial_type_sequence())
        }

        have_survey_text = {
            p.worker
            for p in self.participants
            if 'survey-text' in p.trial_type_sequence()
        }
        if have_survey_text - valid:
            print(f'warning: found participants with survey-text, but modal sequence is not a substring: {have_survey_text - valid}')

        return type(self)(
            self.version,
            self.trial[self.trial.worker.isin(valid)],
            self.questions[self.questions.worker.isin(valid)],
        )

    def modal_trial_sequence(self):
        ct = collections.Counter([
            tuple([row.data['trial_type'] for row in p.rows.itertuples()])
            for p in self.participants
        ])
        return max(ct.keys(), key=lambda key: ct[key])

    def workers_task_completed(self):
        modal_seq = self.modal_trial_sequence()
        modal_count = len(modal_seq)
        return {
            p.worker
            for p in self.participants
            if len(p.rows) == modal_count and tuple([row.data['trial_type'] for row in p.rows.itertuples()]) == modal_seq
        }

    def workers_programming_experience(self, *, include=None):
        if include is None:
            include = {
                # Complete list of options:
                'None',
                'Between 1 and 3 college courses (or equivalent)',
                # 'More than 3 college courses (or equivalent)',
            }

        completed = self.workers_task_completed()

        return {
            p.worker
            for p in self.participants
            # This is redundant with below call, but makes it possible to filter
            if p.worker in completed
            if p.programming_experience()['programming-exp']  in include
        }

    def qualified(self, *, debug=True):
        modal_seq = self.modal_trial_sequence()
        modal_count = len(modal_seq)
        modal_workers = self.workers_task_completed()

        if debug:
            print(f'Modal row count {modal_count}. # matching {len(modal_workers)} / {len(self.participants)}')

        no_programming = self.workers_programming_experience()

        if debug:
            print(f'# with no programming {len(no_programming)} / {len(self.participants)}')

        valid = modal_workers & no_programming

        if debug:
            print(f'# valid workers {len(valid)} / {len(self.participants)}')

        return self.filtered_exp(valid)

    def filtered_exp(self, valid):
        return type(self)(
            self.version,
            self.trial[self.trial.worker.isin(valid)],
            self.questions[self.questions.worker.isin(valid)],
        )

    def report_programming_experience(self):
        Dcompleted = self.filtered_exp(self.workers_task_completed())
        n_participants = len(Dcompleted.participants)
        if n_participants == len(self.participants):
            print('WARN: Current data does not have any participants with a partially-completed assignment. Ensure the correct data is being used.')

        print()
        print(f'{n_participants} workers completed task')
        print()

        rows = {
            "programming-game-exp": dict(
                prompt="Have you played Lightbot or another similar programming game before?",
                ct=collections.Counter(),
            ),
            "programming-exp": dict(
                prompt="How much experience do you have with computer programming?",
                ct=collections.Counter(),
            ),
        }

        for p in Dcompleted.participants:
            pe = p.programming_experience()
            for k, v in pe.items():
                rows[k]['ct'][v] += 1

        for row in rows.values():
            print(row['prompt'])
            for k, v in sorted(row['ct'].items()):
                print(k, '|', f'{v}/{n_participants} {100*v/n_participants:.0f}%')
            print()

    def bonus(self, max_bonus, cost=None, *, min_lens=None):
        if min_lens is None:
            hier_lens = collections.defaultdict(list)
            for p in self.participants:
                for tr in p.task_rows():
                    if tr.skipped:
                        continue
                    hier_lens[tr.mdp_name].append(hier_len(tr.program))
            min_lens = {name: min(lens) for name, lens in hier_lens.items()}
            # HACK: add assertion for # of them???

        # HACK: partly memoizing here to make optimization faster
        if cost is None:
            return functools.partial(self.bonus, max_bonus, min_lens=min_lens)

        return {
            p.worker: {
                tr.mdp_name: tr.bonus(max_bonus, min_lens[tr.mdp_name], cost)
                for tr in p.task_rows()
            }
            for p in self.participants
        }

    def find_best_cost_for_bonus(self, max_bonus, *, debug=True):
        '''
        We give people a bonus based on solution length. Here we optimize for a bonus coefficient
        that achieves our stated average bonus (half of the max amount).
        '''
        bonus_fn = self.bonus(max_bonus)
        ntasks = len(self.participants[0].task_rows())
        def mean_bonus(cost):
            r = bonus_fn(cost)
            mdps = r[list(r.keys())[0]].keys()
            tot = {mdp: [] for mdp in mdps}
            for worker, mdp_to_bonus in r.items():
                assert len(mdp_to_bonus) == len(mdps) and mdp_to_bonus.keys() == mdps, 'making sure we are not carrying extra rows here'
                for mdp, bonus in mdp_to_bonus.items():
                    if np.isnan(bonus):
                        continue
                    tot[mdp].append(bonus)
            return sum(
                # We compute the average on a per-problem basis. This is notably different
                # from optimizing that the per-problem average has some value, because we
                # optimize based on the sum across problems.
                # We average on a per-problem basis because that gives us a simple way to
                # exclude people that skipped the problem, so should neither receive a bonus
                # nor influence fitting for this problem.
                np.mean(bonuses)
                for mdp, bonuses in tot.items())
        def obj(cost):
            expected_bonus = ntasks * max_bonus / 2
            return (mean_bonus(cost) - expected_bonus) ** 2

        xs = np.linspace(0, 1, 1001)
        ys = [obj(x) for x in tqdm(xs)]
        cost = xs[np.argmin(ys)]
        if debug:
            import matplotlib.pyplot as plt
            plt.plot(xs, ys)
            print(f'Min solution c={cost}. Average bonus {mean_bonus(cost):.04f}. Error {obj(cost)}')
        return cost

    def programs_by_count(self, *, canonical=True):
        ct = collections.defaultdict(collections.Counter)
        for p in self.participants:
            for tr in p.task_rows():
                if not tr.skipped:
                    if canonical:
                        prog = tr.canonical_program
                    else:
                        prog = tr.program
                    ct[tr.mdp_name][prog] += 1
        return ct

    def show_feedback(self):
        q_to_text = {
            'Q0': 'What strategy did you use?',
            'Q1': 'Was anything confusing or hard to understand?',
            'Q2': 'Do you have any suggestions on how we can improve the instructions or interface?',
            'Q3': 'Describe any technical difficulties you might have encountered.',
            'Q4': 'Any other comments?',
        }

        commentdf = []
        for p in self.participants:
            #form = ped.filtered_rows(lambda d: d['trial_type'] == 'HTMLForm')[0]
            surveyrow = next(p.filtered_rows(lambda d: d['trial_type'] == 'survey-text').itertuples())
            responses = json.loads(surveyrow.data['responses'])
            commentdf.append({
                **{q_to_text[k]: v for k, v in responses.items()},
                'pid': p.worker[:10]+'...',
                'elapsed': f'{p.elapsed_minutes():.01f}min',
            })

        from IPython.display import display
        with pd.option_context("display.max_rows", None, 'display.max_colwidth', None):
            display(pd.DataFrame(commentdf).sort_values('elapsed', key=lambda v: v.apply(lambda x: float(x[:-len('min')]))))
