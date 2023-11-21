import lb

def test_program_count_matrix():
    mdp = lb.exp.mdp_from_name(('maps', 8))
    # Added extra instructions to being/end to break symmetries and add noops. Added dummy subroutines.
    prog = lb.tools.mkprog('JJJ1LJL1RJR1WLL', 'WWWS', 'LL', 'L')
    m = lb.scoring.get_program_count_matrix([prog], mdp)[0]

    I = lb.INSTRUCTIONS
    C2I = lb.scoring.INSTRUCTION_CODE_TO_INDEX
    assert m[C2I[I.WALK]] == 4
    assert m[C2I[I.LEFT]] == 7
    assert m[C2I[I.RIGHT]] == 2
    assert m[C2I[I.JUMP]] == 5
    assert m[C2I[I.LIGHT]] == 1
    assert m[C2I[I.PROCESS1]] == 3
    assert m[C2I[I.PROCESS2]] == 0
    assert m[C2I[I.PROCESS3]] == 0
    assert m[C2I[I.PROCESS4]] == 0

    assert m[lb.scoring.MAIN_LEN_INDEX] == 15
    assert m[lb.scoring.PROCESS_LEN_INDEX[0]] == 4
    assert m[lb.scoring.PROCESS_LEN_INDEX[1]] == 2
    assert m[lb.scoring.PROCESS_LEN_INDEX[2]] == 1
    assert m[lb.scoring.PROCESS_LEN_INDEX[3]] == 0

    assert m[lb.scoring.STEP_CT_INDEX] == 18
    assert m[lb.scoring.STEP_NOOP_CT_INDEX] == 3 # initial jump
    assert m[lb.scoring.STEP_POST_CT_INDEX] == 2 # trailing left
    assert m[lb.scoring.STEP_POST_NOOP_CT_INDEX] == 1 # trailing walk

def test_scoring():
    mdp = lb.exp.mdp_from_name(('maps', 8))
    # Copied from above
    prog = lb.tools.mkprog('JJJ1LJL1RJR1WLL', 'WWWS', 'LL', 'L')
    mat = lb.scoring.get_program_count_matrix([prog], mdp)
    a = lb.fitting.Args.new(free_parameters={}, default_values=dict(step_count_coef=3, mdl_beta=2)).to_numba()
    assert lb.scoring.batch_score_reward(mat, a) == -3 * (18 + 3)
    assert lb.scoring.batch_score_mdl(mat, a) == -2 * lb.hier_len(prog) == -2 * 22
    assert lb.scoring.batch_score_mdl_and_reward(mat, a) == lb.scoring.batch_score_mdl(mat, a) + lb.scoring.batch_score_reward(mat, a)
