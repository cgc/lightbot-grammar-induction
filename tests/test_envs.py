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
