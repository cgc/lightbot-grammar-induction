import pytest
import lb
from .tools import I, mkinst, mkprog


def test_light_order():
    assert lb.light_order(mkinst('WRSWS')) == mkinst('WSRWS')
    assert lb.light_order(mkinst('RLRS')) == mkinst('SRLR')

def test_inline_subroutine_in_instructions():
    assert lb.inline_subroutine_in_instructions(mkinst('1R1'), I.PROCESS1, mkinst('SS')) == mkinst('SSRSS')
    # return if instruction not present
    s = mkinst('1R1')
    assert lb.inline_subroutine_in_instructions(s, I.WALK, mkinst('SS')) is s

def test_inline_subroutine_in_program():
    assert lb.inline_subroutine_in_program(mkprog('1', 'WR', '1'), I.PROCESS1) == mkprog('WR', '', 'WR')
    with pytest.raises(AssertionError):
        lb.inline_subroutine_in_program(mkprog('1', 'AB1'), I.PROCESS1)

def test_inline_single_use_and_dead_code():
    assert lb.inline_single_ref_and_dead_code(mkprog('W', 'W')) == mkprog('W', '')
    # even if it's recursive, but not used
    assert lb.inline_single_ref_and_dead_code(mkprog('W', '1')) == mkprog('W', '')
    assert lb.inline_single_ref_and_dead_code(mkprog('W', '2', '1')) == mkprog('W', '')

    assert lb.inline_single_ref_and_dead_code(mkprog('1', 'A2D1', 'BC')) == mkprog('1', 'ABCD1')
    # HACK: we make sort of large choices in simplifying mutually recursive routines...
    assert lb.inline_single_ref_and_dead_code(mkprog('1', 'A2', 'B1')) == mkprog('1', 'AB1')
    assert lb.inline_single_ref_and_dead_code(mkprog('W1S', 'R')) == mkprog('WRS')
    # These have some indirection
    assert lb.inline_single_ref_and_dead_code(mkprog('W1S', '2', 'R')) == mkprog('WRS')
    assert lb.inline_single_ref_and_dead_code(mkprog('W2S', '3', '1', '4', 'R')) == mkprog('WRS')

    # Called twice
    assert lb.inline_single_ref_and_dead_code(mkprog('11', '2', 'R')) == mkprog('11', 'R')
    # Called twice at different levels
    assert lb.inline_single_ref_and_dead_code(mkprog('1133', '2', 'R', '1')) == mkprog('1133', 'R', '', '1')

def test_inline_len1_subroutine():
    assert lb.inline_len1_subroutine(mkprog('342', '34WR', 'L', 'S', 'R')) == mkprog('SRL', 'SRWR')

def test_canonicalize_program():
    assert lb.canonicalize_program(mkprog('12', '2', 'WR')) == mkprog('11', 'WR')

    # light ordering
    assert lb.canonicalize_program(mkprog('RS1212', 'SL', 'LS')) == mkprog('SR1212', 'SL', 'SL')

    # how do we handle single-use inlining & dead code removal
    assert lb.canonicalize_program(mkprog('1', 'W')) == mkprog('W')
    assert lb.canonicalize_program(mkprog('1', 'W', '1')) == mkprog('W')
    assert lb.canonicalize_program(mkprog('1', 'W1')) == mkprog('1', 'W1')
    assert lb.canonicalize_program(mkprog('1', 'W2', 'J1')) == mkprog('1', 'WJ1')

    # noop seq
    for noop, repl in lb.NO_OP_MAP.items():
        assert lb.canonicalize_program(mkprog(noop)) == mkprog(repl)

    base_mdp = lb.EnvLoader.maps[0]
    p = mkprog('WWWRJSW')
    assert lb.canonicalize_program(p) == p
    canon_p = lb.canonicalize_program(p, base_mdp)
    assert canon_p == mkprog('WWWSR')
    # Our first pass makes the turn a post-terminal action, but because of
    # analysis order, we don't get rid of it. HACK instead we need a second pass.
    assert lb.canonicalize_program(canon_p, base_mdp) == mkprog('WWWS')

def test_canonicalized_trace():
    mdp = lb.EnvLoader.grid(rows=1)
    print(mdp.map_light, mdp.initial_state())
    print(mdp.next_state(mdp.initial_state(), lb.INSTRUCTIONS.LIGHT))

    assert lb.canonicalized_trace(mdp, mkprog('JWR')) == mkinst('WR')
    assert lb.canonicalized_trace(mdp, mkprog('L1', 'R')) == ''
    assert lb.canonicalized_trace(mdp, mkprog('LJR')) == ''

    # Light order handled
    assert lb.canonicalized_trace(mdp, mkprog('LS')) == mkinst('SL')
    assert lb.canonicalized_trace(mdp, mkprog('RS')) == mkinst('SR')
    assert lb.canonicalized_trace(mdp, mkprog('RRS')) == mkinst('SRR')

    # Canonicalize turns
    assert lb.canonicalized_trace(mdp, mkprog('RRR')) == mkinst('L')
    assert lb.canonicalized_trace(mdp, mkprog('LLL')) == mkinst('R')
    assert lb.canonicalized_trace(mdp, mkprog('LR')) == mkinst('')
    assert lb.canonicalized_trace(mdp, mkprog('RL')) == mkinst('')

    # Light order, and noop
    assert lb.canonicalized_trace(mdp, mkprog('RJRS')) == mkinst('SRR')

    # Skip post-terminal actions
    simple_mdp = lb.EnvLoader.maps[0]
    assert lb.canonicalized_trace(simple_mdp, mkprog('WWWSW')) == mkinst('WWWS')

    # A weird case from the data: A useless turn just before lighting.
    # Can't deal with this by preprocessing to change light order b/c of walk, which is removed by "no effect" filtering
    # Since it seems this can only impact the end of programs, I'm handling with a special case
    assert lb.canonicalized_trace(simple_mdp, mkprog('WWWLJS')) == mkinst('WWWS')

    # These are cases that require our old rule that replaces LL with RR
    orig_map = dict(lb.NO_OP_MAP)
    lb.NO_OP_MAP.clear()
    lb.NO_OP_MAP.update(lb.FULL_NO_OP_MAP)
    try:
        assert lb.canonicalized_trace(mdp, mkprog('LL')) == mkinst('RR')

        # A complex case: no-op jump, with double left that needs to be canonicalized and reordered with a light
        assert lb.canonicalized_trace(mdp, mkprog('LJLS')) == mkinst('SRR')
    finally:
        lb.NO_OP_MAP.clear()
        lb.NO_OP_MAP.update(orig_map)

def test_is_recursive():
    assert not lb.DetectRecursionProgramVisitor.is_recursive(mkprog('1', 'A'))
    assert lb.DetectRecursionProgramVisitor.is_recursive(mkprog('1', '1'))
    assert lb.DetectRecursionProgramVisitor.is_recursive(mkprog('1', '2', '1'))

def test_reorder_subroutines():
    assert lb.ReorderSubroutineVisitor.reorder_subroutines(mkprog('314', 'A', 'B', 'C', 'D')) == mkprog('123', 'C', 'A', 'D')
    assert lb.ReorderSubroutineVisitor.reorder_subroutines(mkprog('3W2', 'JJ', 'SR', '41', '1W')) == mkprog('1W4', '23', '3W', 'JJ', 'SR')
    assert lb.ReorderSubroutineVisitor.reorder_subroutines(mkprog('2', 'W', '2')) == mkprog('1', '1')
    assert lb.ReorderSubroutineVisitor.reorder_subroutines(mkprog('2', '2', '1')) == mkprog('1', '2', '1')

def test_rename_subroutines():
    p = mkprog('AB1C2', 'DC', '1EB')
    # Basic
    assert lb.rename_subroutines(
        p, {'1': '2', '2': '1'}) == mkprog('AB2C1', '2EB', 'DC')
    assert lb.rename_subroutines(
        p, {'1': '3', '2': '1'}) == mkprog('AB3C1', '3EB', '', 'DC')
    # Unmapped, but used, subroutines raise errors
    with pytest.raises(KeyError):
        lb.rename_subroutines(p, {'2': '1'})
    # Unmapped and unused subroutines are dropped
    assert lb.rename_subroutines(mkprog('2', 'A', 'B'), {'2': '1'}) == mkprog('1', 'B')

def test_flatten():
    assert lb.FlattenProgram.flatten(mkprog('1', '22', '33', '44', 'W')) == 'C' * 8
    assert lb.FlattenProgram.flatten(mkprog('1', 'WJ1')) == mkinst('WJ')
    assert lb.FlattenProgram.flatten(mkprog('3W2', 'JJ', 'SR', '41', '1W')) == mkinst('JJWJJWSR')


def test_without_dead_and_noop_instructions():
    base_mdp = lb.EnvLoader.maps[0]

    # We remove post-terminal instructions that have no effect.
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('CCCAD')) == mkprog('CCCA')
    # We can even do this with instructions tucked into subroutines.
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('1', 'W2J', 'W3J', 'WSJ')) == mkprog('1', 'W2', 'W3', 'WS')
    # We remove final, extraneous, instruction calls
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('11111', 'CA')) == mkprog('111', 'CA')

    # The light instruction in this subroutine is used at least once, in final call.
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('111', 'CA')) == mkprog('111', 'CA')
    # Eliminate the light instruction in the subroutine, since it has no impact when not called for final step.
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('11CA', 'CA')) == mkprog('11CA', 'C')

    # Unreachable instruction after unconditional recursion.
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('1', 'CA1A')) == mkprog('1', 'CA1')
    # Recursive call is never reached.
    assert lb.without_dead_and_noop_instructions(base_mdp, mkprog('CCC1', 'A1')) == mkprog('CCC1', 'A')

def test_program_step_counter():
    assert lb.ProgramStepCounter.count(lb.EnvLoader.maps[0], lb.Program('111E', ('CA',))) == (4, 2, 1, 0)
    assert lb.ProgramStepCounter.count(lb.EnvLoader.maps[0], lb.Program('1', ('CA1',))) == (4, 2, 0, 0)
    # This is a weird case... left/right have an effect, so they're technically not a noop, from the one-step perspective.
    assert lb.ProgramStepCounter.count(lb.EnvLoader.maps[0], lb.Program('1', ('CADE1',))) == (8, 2, 2, 0)
    assert lb.ProgramStepCounter.count(lb.EnvLoader.maps[0], lb.Program('1', ('CABB1',))) == (4, 6, 0, 2)

def test_process_state_tracker():
    base_mdp = lb.EnvLoader.maps[8]

    # Simple test case
    inst = lb.ProcessStateTracker(base_mdp)
    inst.visit(mkprog('1LJL1RJR1', 'WWWS'))
    assert base_mdp.is_terminal(inst.state)
    p2s = inst.proc_to_state
    assert len(p2s) == 1
    assert [s.position for s in p2s['1']] == [(1, 4), (2, 1), (3, 4)]

    # Now trying a weirder variant, where we have a nested subroutine
    inst = lb.ProcessStateTracker(base_mdp)
    inst.visit(mkprog('1LJL1RJR1', '22WS', 'W'))
    assert base_mdp.is_terminal(inst.state)
    p2s = inst.proc_to_state
    assert len(p2s) == 2
    assert [s.position for s in p2s['1']] == [(1, 4), (2, 1), (3, 4)]
    assert [s.position for s in p2s['2']] == [(1, 4), (1, 3), (2, 1), (2, 2), (3, 4), (3, 3)]

def test_ensure_max_subroutine_use():
    p = mkprog('1', '2', '3', '4', 'C')
    assert lb.ChildFirstSort.sorted(p) == ['4', '3', '2', '1']

    p = mkprog('4', 'C', '1', '2', '3')
    assert lb.ChildFirstSort.sorted(p) == ['1', '2', '3', '4']

    # called multiple times at different depths
    p = mkprog('123', 'A2', 'B', 'C')
    assert lb.ChildFirstSort.sorted(p) == ['2', '1', '3']

    # recursion
    p = mkprog('1', 'A21', 'B')
    assert lb.ChildFirstSort.sorted(p) == ['2', '1']

    # HACK testing a quirk in recursive cases; will not rewrite programs to maximize recursive use,
    # since we only do one level of substitution.
    p = mkprog('ABCABC1', 'ABC1')
    assert lb.ensure_max_subroutine_use(p) == mkprog('ABC1', 'ABC1')
    assert lb.ensure_max_subroutine_use(lb.ensure_max_subroutine_use(p)) == mkprog('1', 'ABC1')

    p = mkprog('ABCDECD', 'AB2', 'CD')
    assert lb.ChildFirstSort.sorted(p) == ['2', '1']
    assert lb.ensure_max_subroutine_use(p) == mkprog('1E2', 'AB2', 'CD')

    # Trying fully expanded program, in two orders
    p = mkprog('ABCDECD', 'ABCD', 'CD')
    assert lb.ensure_max_subroutine_use(p) == mkprog('1E2', 'AB2', 'CD')
    p = mkprog('ABCDECD', 'CD', 'ABCD')
    assert lb.ensure_max_subroutine_use(p) == mkprog('2E1', 'CD', 'AB1')

    # Case to test that sorting based on length does not work
    p = mkprog('ABCDEBCD', 'A2', 'BCD')
    assert lb.ChildFirstSort.sorted(p) == ['2', '1']
    assert lb.ensure_max_subroutine_use(p) == mkprog('1E2', 'A2', 'BCD')

def test_result_is_fixed_point():
    def _is_fixed_point(fns, p):
        ps = [p]
        for _ in range(2):
            p = ps[-1]
            for fn in fns:
                p = fn(p)
            ps.append(p)
        print(ps)
        return ps[-2] == ps[-1]

    assert not _is_fixed_point([
        lb.ensure_max_subroutine_use,
        lb.inline_single_ref_and_dead_code,
    ], mkprog('AB1A2', 'AB', 'BC'))
