'''
These are scoring functions for model fitting that assume an instruction count matrix.

The instruction count matrix has counts by instruction code, lengths of routines, and step counts when executed.
'''

import numpy as np
import numba
from . import envs as main
import math
from tqdm.auto import tqdm

@numba.njit
def batch(program_count_matrix, a, fn):
    return np.array([
        fn(program_count_matrix[i], a)
        for i in numba.prange(program_count_matrix.shape[0])
    ])

@numba.njit
def score_reward(row, a):
    steps = row[STEP_CT_INDEX] + row[STEP_NOOP_CT_INDEX]
    # We ignore STEP_POST_CT_INDEX and STEP_POST_NOOP_CT_INDEX
    cost = a.step_count_coef * steps
    return -cost

@numba.njit
def score_mdl(row, a):
    length = row[MAIN_LEN_INDEX]
    for i in PROCESS_LEN_INDEX:
        length += row[i]
    return a.mdl_beta * -length

@numba.njit
def batch_score_null(program_count_matrix, a):
    return batch(program_count_matrix, a, lambda row, args: 0)

@numba.njit
def batch_score_reward(program_count_matrix, a):
    return batch(program_count_matrix, a, score_reward)

@numba.njit
def batch_score_mdl(program_count_matrix, a):
    return batch(program_count_matrix, a, score_mdl)

@numba.njit
def batch_score_reuse(program_count_matrix, a):
    return a.prior_beta * batch_pyp_process_prior(
        program_count_matrix, a.a, a.b, p_normal=a.p_normal, p_end=a.p_end, p_subprocess_end=a.p_end)

@numba.njit
def batch_score_mdl_and_reward(program_count_matrix, a):
    return (
        batch_score_reward(program_count_matrix, a) +
        batch_score_mdl(program_count_matrix, a)
    )

@numba.njit
def batch_score_reuse_and_reward(program_count_matrix, a):
    return (
        batch_score_reward(program_count_matrix, a) +
        batch_score_reuse(program_count_matrix, a)
    )

'''
The following code specifies the data structure for the program count matrix.
'''


INSTRUCTION_NAME_TO_INDEX = main.InstructionMap(*range(len(main.INSTRUCTIONS)))

INSTRUCTION_CODE_TO_INDEX = {
    getattr(main.INSTRUCTIONS, inst): getattr(INSTRUCTION_NAME_TO_INDEX, inst)
    for inst in main.INSTRUCTIONS._fields
}

NORMAL_INST_INDEX = tuple(INSTRUCTION_CODE_TO_INDEX[inst] for inst in main.NORMAL_INST)
PROCESS_INST_INDEX = tuple(INSTRUCTION_CODE_TO_INDEX[inst] for inst in main.PROCESS_INST)

count_mat_len = len(main.INSTRUCTIONS)

def new_count_mat_field():
    global count_mat_len
    v = count_mat_len
    count_mat_len += 1
    return v

MAIN_LEN_INDEX = new_count_mat_field()
PROCESS_LEN_INDEX = tuple(new_count_mat_field() for _ in range(4))
STEP_CT_INDEX = new_count_mat_field()
STEP_NOOP_CT_INDEX = new_count_mat_field()
STEP_POST_CT_INDEX = new_count_mat_field()
STEP_POST_NOOP_CT_INDEX = new_count_mat_field()

def get_program_count_matrix(programs, base_mdp, *, tqdm_disable=None):
    '''
    This matrix is a kitchen sink, holding various counts related to programs that are
    helpful when fitting models. They are precomputed into matrix form for the efficiency
    of model fitting. In this matrix, we store:
    - use counts for each action and process
    - program lengths
    - and execution statistics (step counts, etc)
    '''
    mat = np.zeros((len(programs), count_mat_len), dtype=int)
    for pi, prog in enumerate(tqdm(programs, disable=tqdm_disable)):
        def _inc(inst):
            mat[pi, INSTRUCTION_CODE_TO_INDEX[inst]] += 1

        # Count instructions
        for inst in prog.main:
            _inc(inst)
        for sr in prog.subroutines:
            for inst in sr:
                _inc(inst)

        # Add lengths
        mat[pi, MAIN_LEN_INDEX] = len(prog.main)
        for proc_idx, idx in enumerate(PROCESS_LEN_INDEX):
            mat[pi, idx] = len(prog.subroutines[proc_idx])

        # Add step counters (optionally)
        if base_mdp is not None:
            from . import program_analysis
            ct = program_analysis.ProgramStepCounter.count(base_mdp, prog)
            mat[pi, STEP_CT_INDEX] = ct.step
            mat[pi, STEP_NOOP_CT_INDEX] = ct.step_noop
            mat[pi, STEP_POST_CT_INDEX] = ct.step_post
            mat[pi, STEP_POST_NOOP_CT_INDEX] = ct.step_post_noop
    return mat

import os
assert os.getenv('NUMBA_DISABLE_JIT') != '1'

@numba.njit
def batch_pyp_process_prior(program_count_matrix, a, b, p_normal, p_end, p_subprocess_end):
    '''
    This function is intended to be an efficient batch implementation used for model fitting.

    It updates the original in envs.py by
    - Precomputing as much as possible, passing it around in the instruction count matrix.
    - Being written in numba.
    - Being batched, so that function entry overhead is avoided. There may be a marginal benefit to precomputing some quantities.
    '''

    def lgamma(n):
        return 0 if n == 0 else math.lgamma(n)
    def gen_gamma(integer, offset):
        return lgamma(integer+offset) - lgamma(offset)

    p_normal_inst = 1/len(main.NORMAL_INST)
    log_p_normal_inst = math.log(p_normal_inst)

    log_p_normal = math.log(p_normal)
    log_p_not_normal = math.log(1-p_normal)

    log_p_end = math.log(p_end)
    log_p_not_end = math.log(1-p_end)

    log_p_subprocess_end = math.log(p_subprocess_end)
    log_p_not_subprocess_end = math.log(1-p_subprocess_end)

    n_programs = program_count_matrix.shape[0]
    log_ps = np.zeros(n_programs)

    for pi in numba.prange(n_programs):
        ct_mat = program_count_matrix[pi]
        log_p = 0
        n = 0
        k = 0

        for inst_idx in NORMAL_INST_INDEX:
            c = ct_mat[inst_idx]
            if c == 0:
                continue
            log_p += c * (log_p_normal + log_p_normal_inst)

        for inst_idx in PROCESS_INST_INDEX:
            c = ct_mat[inst_idx]
            if c == 0:
                continue
            n += c
            # Because we increment k before we use it below, it is in [1, m]
            k += 1

            log_p += c * log_p_not_normal
            log_p += math.log(a * (k - 1) + b)
            log_p += gen_gamma(c-1, 1-a)
        log_p -= gen_gamma(n, b)

        log_p += (ct_mat[MAIN_LEN_INDEX] - 1) * log_p_not_end + log_p_end
        for idx in PROCESS_LEN_INDEX:
            len_proc = ct_mat[idx]
            if len_proc == 0:
                continue
            log_p += (len_proc - 1) * log_p_not_subprocess_end + log_p_subprocess_end

        log_ps[pi] = log_p

    return log_ps
