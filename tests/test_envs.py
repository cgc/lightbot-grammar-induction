import numpy as np
import lb

def test_interpret_instruction():
    mdp = lb.LightbotMap(
        np.array([
            [0, 0],
            [0, 1],
        ], dtype=int),
        np.array([
            [-1, -1],
            [0, -1],
        ], dtype=int),
        (0, 1),
        0,
    )
    s = mdp.initial_state()
    for i, expected in [
        (lb.INSTRUCTIONS.WALK, s._replace(position=(0, 0))),
        (lb.INSTRUCTIONS.RIGHT, s._replace(direction=3)),
        (lb.INSTRUCTIONS.LEFT, s._replace(direction=1)),
        (lb.INSTRUCTIONS.LIGHT, s),
        (lb.INSTRUCTIONS.JUMP, s),
    ]:
        ns = lb.interpret_instruction(mdp, s, i)
        assert ns == expected, (i, ns, expected)

    s = mdp.next_state(mdp.initial_state(), lb.INSTRUCTIONS.LEFT)
    for i, expected in [
        (lb.INSTRUCTIONS.WALK, s),
        (lb.INSTRUCTIONS.LIGHT, s),
        (lb.INSTRUCTIONS.JUMP, s._replace(position=(1, 1))),
    ]:
        ns = lb.interpret_instruction(mdp, s, i)
        assert ns == expected, (i, ns, expected)

    s = mdp.initial_state()._replace(position=(1, 0))
    for i, expected in [
        (lb.INSTRUCTIONS.LIGHT, s._replace(map_lit=lb.CONST_MAP_LIT_TRUE)),
    ]:
        ns = lb.interpret_instruction(mdp, s, i)
        assert ns == expected, (i, ns, expected)


def test_interpret_instruction_height():
    mdp = lb.LightbotMap(
        np.array([[0, 1, 0]], dtype=int),
        np.array([[-1, -1, -1]], dtype=int),
        (0, 0),
        2,
    )

    s = mdp.initial_state()
    s_mid = s._replace(position=(0, 1))
    s_end = s._replace(position=(0, 2))
    for start, is_, expected in [
        (s, [lb.INSTRUCTIONS.WALK], s),
        (s, [lb.INSTRUCTIONS.JUMP], s_mid),
        (s, [lb.INSTRUCTIONS.JUMP]*2, s_end),
        (s_mid, [lb.INSTRUCTIONS.WALK], s_mid),
        (s_mid, [lb.INSTRUCTIONS.JUMP], s_end),
    ]:
        ns = start
        for i in is_:
            ns = lb.interpret_instruction(mdp, ns, i)
        assert ns == expected, (i, ns, expected)

    # Now change height
    mdp.map_h[s_mid.position] = 2

    for start, is_, expected in [
        (s, [lb.INSTRUCTIONS.WALK], s),
        (s, [lb.INSTRUCTIONS.JUMP], s), # This one changes!
        (s_mid, [lb.INSTRUCTIONS.WALK], s_mid),
        (s_mid, [lb.INSTRUCTIONS.JUMP], s_end), # But you can jump down
    ]:
        ns = start
        for i in is_:
            ns = lb.interpret_instruction(mdp, ns, i)
        assert ns == expected, (i, ns, expected)


def test_interpret_program():
    # testing recursion

    program = lb.Program('1', ('AC1', '', '', ''))

    (state, cost, _), path = lb.interpret_program_record_path(
        lb.EnvLoader.grid(rows=1),
        program)
    assert cost == -5
    assert ''.join(a for s, a in path if a) == 'ACACA'

    (state, cost, _), path = lb.interpret_program_record_path(
        lb.EnvLoader.maps[0],
        program)
    assert cost == -10
    assert ''.join(a for s, a in path if a) == 'ACACACA'


def test_interpret_program_halt_when_terminal():
    mdp = lb.LightbotMap(
        np.array([[0, 0, 0]], dtype=int),
        np.array([[-1, 0, -1]], dtype=int),
        (0, 0),
        2,
    )
    s = mdp.initial_state()

    for is_, expected, halt in [
        (
            [lb.INSTRUCTIONS.WALK, lb.INSTRUCTIONS.LIGHT, lb.INSTRUCTIONS.WALK],
            s._replace(position=(0, 1), map_lit=lb.CONST_MAP_LIT_TRUE), True),
        (
            [lb.INSTRUCTIONS.WALK, lb.INSTRUCTIONS.LIGHT, lb.INSTRUCTIONS.WALK],
            s._replace(position=(0, 2), map_lit=lb.CONST_MAP_LIT_TRUE), False),
    ]:
        for p in [
            lb.Program(main='1', subroutines=(''.join(is_), '')),
            lb.Program(main=''.join(is_), subroutines=('', '')),
        ]:
            state, _, _ = lb.interpret_program(mdp, p, state=s, halt_when_terminal=halt)
            assert state == expected



def test_is_recursive():
    assert lb.is_recursive(lb.Program(main='1', subroutines=('A1', '')))
    assert lb.is_recursive(lb.Program(main='1', subroutines=('A2', 'B1')))
    assert not lb.is_recursive(lb.Program(main='11', subroutines=('22', '33', '44', 'A')))

def test_instruction_use_count():
    assert (
        lb.envs.instruction_use_count(lb.tools.mkprog('1AB2', 'C', 'B1', 'E')) ==
        {'A': 1, 'B': 2, 'C': 1, 'D': 0, 'E': 1, '1': 2, '2': 1, '3': 0, '4': 0})

def test_sr_use_count():
    assert lb.envs.sr_use_count(lb.tools.mkprog('A')) == 0
    assert lb.envs.sr_use_count(lb.tools.mkprog('A', 'B', 'C', 'D', 'E')) == 0
    assert lb.envs.sr_use_count(lb.tools.mkprog('111A', '2')) == 4

def test_trace():
    base_mdp = lb.EnvLoader.maps[0]

    # Hierarchical one
    mdp = lb.LightbotTrace(base_mdp)
    s = mdp.initial_state()
    assert s.light_target is None
    s = mdp.next_state(s, (3, 2))
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'C')
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'C')
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'C')
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'A')
    assert s.light_target is None
    assert mdp.is_terminal(s)
    assert s.trace == 'CCCA'

    # Simple one
    mdp = lb.SimpleLightbotTrace(base_mdp)
    s = mdp.initial_state()
    s, _ = mdp.next_state_and_reward(s, 'C')
    s, _ = mdp.next_state_and_reward(s, 'C')
    s, _ = mdp.next_state_and_reward(s, 'C')
    s, _ = mdp.next_state_and_reward(s, 'A')
    assert mdp.is_terminal(s)
    assert s.trace == 'CCCA'

    # testing no-ops -- focusing on left,right and right,left
    s0 = mdp.initial_state()
    s, _ = mdp.next_state_and_reward(s0, 'D')
    assert mdp.actions(s) == ['B', 'C', 'D']
    s, _ = mdp.next_state_and_reward(s0, 'E')
    assert mdp.actions(s) == ['B', 'C', 'E']
