import argparse
import joblib

import sys
sys.path.append('..')
import lb

def main(args):
    Dall = lb.exp.Data.load('0.4')

    with joblib.Parallel(n_jobs=args.workers, verbose=2) as w:
        fk = dict(nsampled=args.nsampled, joblib_worker=w)
        if 'default' in args.analysis:
            D = Dall.qualified()
            dc = lb.fitting.Analyses.default(D)
            lb.fitting.analysis(dc, fit_kwargs=fk)
        if 'confound_no_canon' in args.analysis:
            D = Dall.qualified()
            dc = lb.fitting.Analyses.confound_no_canon(D)
            lb.fitting.analysis(dc, fit_kwargs=fk)
        if 'confound_no_prog_exp' in args.analysis:
            Dstrict = Dall.qualified(strict=True)
            dc = lb.fitting.Analyses.confound_no_prog_exp(Dstrict)
            lb.fitting.analysis(dc, fit_kwargs=fk)

if __name__ == '__main__':
    '''
    Official analyses will be the following, for each analysis type
    do_analyze.py --workers=12 --nsampled=50 --analysis=$A

    Local tests are probably best to do with this
    do_analyze.py --workers=2 --nsampled=1 --analysis=default,confound_no_canon,confound_no_prog_exp
    '''
    parser = argparse.ArgumentParser(description='run model fitting')
    # parser.add_argument('method', type=str, default='main', help='number of workers')
    parser.add_argument('--workers', type=int, help='number of workers')
    parser.add_argument('--nsampled', type=int, help='number of fits from a random sample')
    parser.add_argument('--analysis', type=str, help='analysis names (can be: default, confound_no_canon, confound_no_prog_exp)')
    args = parser.parse_args()
    args.analysis = set(args.analysis.split(','))

    assert args.analysis < {'default', 'confound_no_canon', 'confound_no_prog_exp'}
    # assert args.method in ('main',)

    main(args)
