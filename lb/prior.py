from .envs import NORMAL_INST, INSTRUCTIONS, PROCESS_INST
import warnings
import math

# This implementation isn't used in model fitting, now instead using the implementation in scoring.py

def pyp_process_prior(
        prog, a, b,
        p_normal=1/2,
        p_end=None,
        p_subprocess_end=None,
        *,
        use_deprecated=False,
        split_probability=False,
    ):
    if not use_deprecated:
        assert p_end is not None, 'To use legacy version, must supply use_deprecated=True'
        assert p_subprocess_end is not None, 'To use legacy version, must supply use_deprecated=True'
    def lgamma(n):
        return 0 if n == 0 else math.lgamma(n)
    def gen_gamma(integer, offset):
        return lgamma(integer+offset) - lgamma(offset)

    # HACK: copying since there isn't another way to have a numba xxxxx
    PROCESS_TO_IDX = {
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3}

    p_normal_inst = 1/len(NORMAL_INST)

    log_p_normal = math.log(p_normal)
    log_p_not_normal = math.log(1-p_normal)
    log_p_normal_inst = math.log(p_normal_inst)
    if p_end is not None:
        log_p_end = math.log(p_end)
        log_p_not_end = math.log(1-p_end)
    if p_subprocess_end is not None:
        log_p_subprocess_end = math.log(p_subprocess_end)
        log_p_not_subprocess_end = math.log(1-p_subprocess_end)

    ct = {inst: 0 for inst in INSTRUCTIONS}
    for inst in prog.main:
        ct[inst] += 1
    for sr in prog.subroutines:
        for inst in sr:
            ct[inst] += 1

    n = 0
    k = 0

    log_p = 0
    pyp_log_p = 0
    for inst in NORMAL_INST:
        if not ct[inst]:
            continue
        log_p += ct[inst] * (log_p_normal + log_p_normal_inst)

    #for inst_enum in INSTRUCTIONS:
    for inst in PROCESS_INST:
        if not ct[inst]:
            continue
        #inst = inst_enum.value
        #if inst not in PROCESS_TO_IDX:
        #    log_p += ct[inst] * (log_p_normal + log_p_normal_inst)
        #    continue
        n += ct[inst]
        # Because we increment k before we use it below, it is in [1, m]
        k += 1

        log_p += ct[inst] * log_p_not_normal
        # Computing logp for the PYP, trying to mirror names from Eq. 3 of Johnson et al. 2007
        pyp_log_p += math.log(a * (k - 1) + b)
        #crp_log_p += sum([math.log(j - a) for j in range(1, ct[inst])])
        pyp_log_p += gen_gamma(ct[inst]-1, 1-a)
    #crp_log_p -= sum([math.log(i + b) for i in range(n)])
    pyp_log_p -= gen_gamma(n, b)

    if p_end is not None:
        if p_subprocess_end is not None:
            # Then we penalize subprocesses separately
            log_p += (len(prog.main) - 1) * log_p_not_end + log_p_end
            for idx, _ in enumerate(PROCESS_INST):
                len_proc = len(prog.subroutines[idx])
                if not len_proc:
                    continue
                log_p += (len_proc - 1) * log_p_not_subprocess_end + log_p_subprocess_end
        else:
            warnings.warn("Prior without p(subproc end) is deprecated", DeprecationWarning)
            # Now adding weight for the non-terminals, gives preference to shorter sequences.
            all_tokens = sum(list(ct.values()))
            log_p += (all_tokens-1) * log_p_not_end + log_p_end
    else:
        warnings.warn("Prior without p(end) is deprecated", DeprecationWarning)

    total = log_p + pyp_log_p

    if split_probability:
        return dict(
            total=total,
            grammar=log_p,
            pyp=pyp_log_p,
        )

    return total
