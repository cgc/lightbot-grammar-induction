import pathlib
import os
import pandas as pd
import collections
import dataclasses
import seaborn as sns
import numpy as np
import scipy.stats

import sys
sys.path.append('..')
import lb

FILE_DIR = pathlib.Path(__file__).absolute().parent

FIGURE_DIR = FILE_DIR / 'figures'

def path(name):
    return FIGURE_DIR / name

def save(fn, *args, **kwargs):
    FIGURE_DIR.mkdir(exist_ok=True)

    EXTS = ['.png', '.pdf']

    fn, ext = os.path.splitext(fn)
    assert ext in ['.*', ''] + EXTS

    for ext in EXTS:
        import matplotlib.pyplot as plt
        plt.savefig(
            path(f'{fn}{ext}'),
            *args,
            bbox_inches='tight',
            dpi=300,
            facecolor=plt.gcf().get_facecolor(),
            **kwargs,
        )

def pd_to_latex(df: pd.DataFrame, *, kwargs=dict(), midrule_hack=False):
    if midrule_hack:
        df = latex_table_midrule_hack(df)
    kwargs.setdefault('index_names', False)
    kwargs.setdefault('index', False)
    kwargs.setdefault('escape', False)
    with pd.option_context('display.max_colwidth', 999999):
        return df.to_latex(**kwargs)
        # lines = mcdf.to_latex(escape=False, index_names=False).split('\n')
        # print('\n'.join([
        #     f'\\rule{{0pt}}{{3em}}{line}[2em]\n\\hline' if r'\\' in line else line
        #     for line in lines
        # ]))

def latex_table_midrule_hack(df):
    '''
    This is a slightly hacky way to add lines between every table row.
    '''
    assert isinstance(df, pd.DataFrame)
    df = df.copy()
    c = df.columns[0]
    df[c] = [
        # Skip first row
        v if idx == 0 else
        # Otherwise, prepend midrule
        rf'\midrule {v}'
        for idx, v in enumerate(df[c])
    ]
    return df

def set_mpl_style():
    # Look at $PIP_ENV/lib/python3.9/site-packages/matplotlib/mpl-data/stylelib/classic.mplstyle
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    # This is the default for jupyter figures -- but not in classic.mplstyle, so not sure where it comes from?
    plt.rcParams['font.size'] = 10

def _apa_pvalue(x, *, lower_bound = 0.001):
    '''
    >>> assert apa_pval(0.5) == "= .5"
    >>> assert apa_pval(0.51) == "= .51"
    >>> assert apa_pval(0.49) == "= .49"
    >>> assert apa_pval(0.051) == "= .051"
    >>> assert apa_pval(0.049) == "= .049"
    >>> assert apa_pval(0.0051) == "= .005"
    >>> assert apa_pval(0.0049) == "= .005"
    >>> assert apa_pval(0.0011) == "= .001"
    >>> assert apa_pval(0.0010) == "= .001"
    >>> assert apa_pval(0.0009) == "< .001"
    >>> assert apa_pval(0.00051) == "< .001"
    >>> assert apa_pval(0.00049) == "< .001"
    '''
    assert 0 <= x <= 1
    if x < lower_bound:
        op = '<'
        num = lower_bound
    else:
        op = '='
        num = x
    num_str = f'{num:.3f}'.lstrip('0').rstrip('0')
    return f'{op} {num_str}'

def pvalue(pv):
    return f'p {_apa_pvalue(pv)}'

@dataclasses.dataclass
class Model:
    key: str
    name: str
    color: tuple[float, float, float]

    def simple_fn(self, mdp, p):
        if self.key == 'reward':
            sc = lb.program_analysis.ProgramStepCounter.count(mdp, p)
            assert (
                sc.step_noop == 0 and
                sc.step_post == 0 and
                sc.step_post_noop == 0
            ), 'sanity check, programs should have only have step costs, no others'
            return sc.step
        elif self.key == 'mdl':
            return lb.hier_len(p)
        raise ValueError(f'Cannot call simple_fn for {self.key}')

    @property
    def score_key(self):
        return f'score_{self.key}'

    @property
    def batch_score_fn(self):
        return getattr(lb.scoring, f'batch_{self.score_key}')

    @property
    def name_label(self):
        '''
        This is a bit of a hack. Trying it out for some plots.
        '''
        return '\n+ '.join(self.name.split(' + '))

cmap_paired = list(sns.color_palette("Paired"))
cmap_tab10 = list(sns.color_palette('tab10'))
cmap_bright = list(sns.color_palette('bright'))
color_idx_gi = 0
color_idx_mdl = 6

_models = [
    Model(key='null', name='random choice', color=cmap_tab10[7]),
    Model(key='mdl', name='MDL', color=cmap_paired[color_idx_mdl+1]),
    Model(key='reuse', name='grammar induction', color=cmap_paired[color_idx_gi+1]),
    Model(key='reward', name='step cost', color=cmap_bright[7]),
    Model(key='mdl_and_reward', name='MDL + step cost', color=cmap_paired[color_idx_mdl]),
    Model(key='reuse_and_reward', name='grammar induction + step cost', color=cmap_paired[color_idx_gi]),
]

class Pile(dict):
    # https://stackoverflow.com/a/6738291
    def __getattr__(self, key):
        if key in self:
            return self[key]
        # raise AttributeError for missing key here to fulfill API
        raise AttributeError(key)

models = Pile()
for m in _models:
    models[m.key] = m
models['all'] = _models

mdp_names = (
    ('cgcMaps', 14),
    ('cgcMaps', 3),
    ('maps', 8),
    ('cgcMaps', 15),
    ('maps', 7),
    ('cgcMaps', 8),
    ('cgcMaps', 9),
    ('cgcMaps', 12),
    ('cgcMaps', 11),
    ('maps', 6),
)

mdp_labels_and_names = [
    (chr(idx+ord('a')), mdp_name)
    for idx, mdp_name in enumerate(mdp_names)
]

mdp_name_to_label = {
    mdp_name: label
    for label, mdp_name in mdp_labels_and_names
}

mdps = tuple(
    lb.exp.mdp_from_name(mdp_name)
    for mdp_name in mdp_names
)

LOT = collections.namedtuple('LOG', ['key', 'name'])
_lot_types = [
    LOT('behavior-flat', 'Behavior, Flat'),
    LOT('behavior-hier', 'Behavior, Hier'),
    LOT('gen-flat', 'Enumerated, Flat'),
    LOT('gen-hier', 'Enumerated, Hier'),
]
lot_types = Pile()
for lott in _lot_types:
    lot_types[lott.key] = lott
lot_types['all'] = _lot_types

def runr(rcode, **kw):
    '''
    This takes some R code and variables used in the R code (as kwargs)
    and runs it by dynamically making an appropriate function. Assumes
    that the variables you are handling are R objects.
    Previously called `r_with_vars`
    '''
    import rpy2.robjects as ro
    if kw:
        keys, values = zip(*kw.items())
    else:
        keys, values = (), ()
    return ro.r('function('+','.join(keys)+') {'+rcode+'}')(*values)

def wrap_runr(str, *, convert=False, scalar=False):
    def fn(**kw):
        rv = runr(str, **kw)
        if convert:
            rv = convert_rdf_to_df(rv)
        if scalar:
            assert len(rv) == 1
            rv = rv[0]
        return rv
    return fn

def convert_df_to_rdf(df):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)

def convert_rdf_to_df(df):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(df)

def anova(*, model, null):
    anova = wrap_runr('as.data.frame(anova(m, null))', convert=True)
    anova_df = anova(m=model, null=null)
    anova_row = anova_df.iloc[1]
    df = anova_df.npar[1] - anova_df.npar[0]
    assert df > 0, (df, anova_df.npar)
    print('anova', rf'$\chi^2({int(df)})={anova_row.Chisq:.2f}$, ${pvalue(anova_row["Pr(>Chisq)"])}$')

def makecell(s, *, align=''):
    if isinstance(s, str):
        s = s.split('\n')
    assert isinstance(s, (list, tuple))
    s = r' \\ '.join(s)
    if align:
        align = f'[{align}]'
    return fr'\makecell{align}{{{s}}}'

def wrap_model_name(name):
    '''
    We generally only need to wrap this model name.
    '''
    return name.replace('grammar induction +', 'grammar induction\n+')

def set_ax_size(w,h,ax):
    # NOTE: Doesn't seem to work well
    """ w, h: width, height in inches """
    # https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    # if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def meansd(arr, *, percent=False, range=False):
    M = np.mean(arr)
    SD = np.std(arr)
    if percent:
        assert not range
        print(f'$M={100*M:.0f}\\%$')
    else:
        r = f', range: {np.min(arr):.2f}--{np.max(arr):.2f}' if range else ''
        print(f'${M=:.2f}$ ${SD=:.2f}$'+r)

def spearmanr_permutation_test(x, y, *, n_resamples=100_000, report=False):
    # from pearsonr example in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    def statistic(x, y):
        return scipy.stats.spearmanr(x, y).correlation
    res = scipy.stats.permutation_test(
        (x, y),
        statistic,
        vectorized=False,
        permutation_type='pairings',
        alternative='two-sided',
        n_resamples=n_resamples,
    )
    if report:
        report_spearmanr(len(x), res)
    return res

def report_spearmanr(n_obs, test_result):
    print(rf'$\rho={test_result.statistic:.2f}$, ${pvalue(test_result.pvalue)}$, $N={n_obs}$')
    # or https://www.statisticssolutions.com/reporting-statistics-in-apa-format/?
    '''
    (r(112) = .60, p = .012)
    (rs(112) = .53, p < .001)
    '''
    # (\rho = .69, p < .001, N = 31)
    # https://guides.library.lincoln.ac.uk/mash/statstest/spearman_rho_correlation
