import json
import math
import typing
import collections
import pathlib
import warnings
import functools

import numpy as np

code_dir = pathlib.Path(__file__).absolute().parent

NO_LIGHT = -1
# Constants for representation of whether a map is lit or not.
CONST_MAP_LIT_TRUE = '1'
CONST_MAP_LIT_FALSE = '0'


DIRECTIONS = (
    (0, -1), # SE
    (+1, 0), # NE
    (0, +1), # NW
    (-1, 0), # SW
)


def str_replace(s, idx, value):
    return s[:idx] + value + s[idx+1:]

def is_unlit_light(map_lit, idx):
    return idx != NO_LIGHT and map_lit[idx] == CONST_MAP_LIT_FALSE

def get_light_positions(mdp):
    lights = [None] * mdp.total_lights
    for light_position in zip(*np.where(mdp.map_light != NO_LIGHT)):
        lights[mdp.map_light[light_position]] = light_position
    return lights

def lit_count(map_lit):
    ct = 0
    for light in map_lit:
        if light == CONST_MAP_LIT_TRUE:
            ct += 1
    return ct

def interpret_instruction(mdp, state, instruction):
    '''
    This function interprets one instruction, given a map and current state (position, direction, map_lit).
    It directly modifies `map_lit`, the array of light states, and returns `position` and `direction`.
    '''
    def inbounds(position):
        return (
            0 <= position[0] < mdp.map_h.shape[0] and
            0 <= position[1] < mdp.map_h.shape[1]
        )

    position, direction, map_lit = state

    if instruction == INSTRUCTIONS.LEFT:
        direction = (direction + 1) % 4
    elif instruction == INSTRUCTIONS.RIGHT:
        direction = (direction - 1) % 4
    elif instruction == INSTRUCTIONS.LIGHT:
        # Lights can only be lit if they're unlit.
        if not is_unlit_light(state.map_lit, mdp.map_light[position]):
            return state
        map_lit = str_replace(map_lit, mdp.map_light[position], CONST_MAP_LIT_TRUE)
    elif instruction in (INSTRUCTIONS.WALK, INSTRUCTIONS.JUMP):
        d = DIRECTIONS[direction]
        next_position = (position[0] + d[0], position[1] + d[1])
        if not inbounds(next_position):
            return state

        curr_h = mdp.map_h[position]
        next_h = mdp.map_h[next_position]
        valid_walk = instruction == INSTRUCTIONS.WALK and curr_h == next_h
        valid_jump = instruction == INSTRUCTIONS.JUMP and (curr_h + 1 == next_h or curr_h > next_h)
        if not (valid_walk or valid_jump):
            return state
        position = next_position
    else:
        raise ValueError('Invalid instruction')

    return State(position, direction, map_lit)


Program = collections.namedtuple('Program', ['main', 'subroutines'])

InstructionMap = collections.namedtuple(
    'InstructionMap', [
        'LIGHT',
        'JUMP',
        'WALK',
        'RIGHT',
        'LEFT',
        'PROCESS1',
        'PROCESS2',
        'PROCESS3',
        'PROCESS4',
    ],
)

INSTRUCTIONS = InstructionMap(
    LIGHT = 'A',
    JUMP = 'B',
    WALK = 'C',
    RIGHT = 'D',
    LEFT = 'E',
    PROCESS1 = '1',
    PROCESS2 = '2',
    PROCESS3 = '3',
    PROCESS4 = '4',
)

PROCESS_TO_IDX = {}
IDX_TO_PROCESS = []
for i in range(4):
    key = f'{i+1}'
    PROCESS_TO_IDX[key] = i
    IDX_TO_PROCESS.append(key)

NORMAL_INST = tuple(
    inst for inst in INSTRUCTIONS
    if inst not in PROCESS_TO_IDX
)
NORMAL_INST_NO_LIGHT = tuple(
    inst for inst in INSTRUCTIONS
    if inst not in PROCESS_TO_IDX and inst != INSTRUCTIONS.LIGHT
)
PROCESS_INST = tuple(
    inst for inst in INSTRUCTIONS
    if inst in PROCESS_TO_IDX
)

InterpretProgramResult = collections.namedtuple('InterpretProgramResult', ['state', 'reward', 'limit'])

def interpret_program(mdp, program, state=None, current_function=None, limit=1000, reward=0, record_path=None, halt_when_terminal=True):
    PROCESS_TO_IDX = {
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3}
    if current_function is None:
        current_function = program.main

    if state is None:
        state = mdp.initial_state()

    for instruction in current_function:
        limit -= 1
        if limit <= 0:
            break
        if halt_when_terminal and mdp.is_terminal(state):
            break

        if instruction in PROCESS_TO_IDX:
            state, reward, limit = interpret_program(
                mdp,
                program,
                state=state,
                current_function=program.subroutines[PROCESS_TO_IDX[instruction]],
                limit=limit,
                reward=reward,
                record_path=record_path,
                halt_when_terminal=halt_when_terminal,
            )
        else:
            if record_path is not None:
                record_path.append((state, instruction))
            # HACK: consider changing this to mdp.next_state?
            next_state = interpret_instruction(mdp, state, instruction)
            reward += mdp.reward(state, instruction, next_state)
            state = next_state

    if record_path is not None:
        if current_function == program.main:
            record_path.append((state, None))

    return InterpretProgramResult(state, reward, limit)


def interpret_program_record_path(mdp, program, **kwargs):
    rp = []
    res = interpret_program(mdp, program, record_path=rp, **kwargs)
    return res, rp


def is_recursive(program, curr=None, stack=None):
    PROCESS_TO_IDX = {
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3}

    if stack is None:
        stack = ['__ROOT_PLACEHOLDER__'] # placeholder for type of empty list
    if curr is None:
        curr = program.main

    for instruction in curr:
        if instruction not in PROCESS_TO_IDX:
            continue

        if instruction in stack:
            return True

        stack.append(instruction)
        r = is_recursive(
            program,
            program.subroutines[PROCESS_TO_IDX[instruction]],
            stack,
        )
        assert stack.pop() == instruction
        if r:
            return True

    return False

def hier_len(p):
    return len(p.main) + sum(len(sr) for sr in p.subroutines)


State = collections.namedtuple('State', [
    'position',
    'direction',
    'map_lit',
])

class LightbotMap:
    map_h: np.ndarray
    map_light: np.ndarray
    direction0: int
    position0: typing.Tuple[int, int]
    total_lights: int
    all_lit: str
    noop_reward: int

    def __init__(self, map_h, map_light, position0, direction0, noop_reward=-2):
        self.map_h = map_h
        self.map_light = map_light
        self.position0 = position0
        self.direction0 = direction0
        self.noop_reward = noop_reward

        self.total_lights = np.sum(map_light != NO_LIGHT)
        self.all_lit = CONST_MAP_LIT_TRUE * self.total_lights

    def initial_state(self):
        '''
        Representing with strings for two reasons:
        - Hashable (rules out list and numpy array)
        - Can have dynamic size (rules out tuple)
        '''
        map_lit = CONST_MAP_LIT_FALSE * self.total_lights
        return self.newState(self.position0, self.direction0, map_lit)
    def newState(self, p, d, m):
        return State(p, d, str(m))
    def actions(self, s):
        if is_unlit_light(s.map_lit, self.map_light[s.position]):
            return [INSTRUCTIONS.LIGHT]
        else:
            return NORMAL_INST_NO_LIGHT
    def next_state_and_reward(self, s, a):
        ns = self.next_state(s, a)
        return ns, self.reward(s, a, ns)
    def next_state(self, s, instruction):
        return interpret_instruction(
            self, s, instruction
        )
    def reward(self, s, a, ns):
        if s == ns:
            return self.noop_reward
        return -1

    def is_terminal(self, s):
        return s.map_lit == self.all_lit

    def lit_count(self, state):
        return lit_count(state.map_lit)

class LightbotMapWithGoalPosition:
    '''
    Used for testing
    '''
    mdp: LightbotMap
    goal_position: typing.Tuple[int, int]

    def __init__(self, mdp, goal_position):
        self.mdp = mdp
        self.goal_position = goal_position

    def is_terminal(self, s):
        return s.position == self.goal_position
    def actions(self, s):
        return NORMAL_INST_NO_LIGHT

    # All call into mdp
    def next_state_and_reward(self, s, a):
        return self.mdp.next_state_and_reward(s, a)
    def next_state(self, s, a):
        return self.mdp.next_state(s, a)
    def initial_state(self):
        return self.mdp.initial_state()
    def reward(self, s, a, ns):
        return self.mdp.reward(s, a, ns)

StateAndProgram = collections.namedtuple('StateAndProgram', ['state', 'program'])

FULL_NO_OP_MAP = {
    ''.join(k): ''.join(v)
    for k, v in {
        # Avoid actions that undo each other
        (INSTRUCTIONS.LIGHT, INSTRUCTIONS.LIGHT): (INSTRUCTIONS.LIGHT,),
        (INSTRUCTIONS.RIGHT, INSTRUCTIONS.LEFT): (),
        (INSTRUCTIONS.LEFT, INSTRUCTIONS.RIGHT): (),
        # We should only consider RIGHT RIGHT
        (INSTRUCTIONS.LEFT, INSTRUCTIONS.LEFT): (INSTRUCTIONS.RIGHT, INSTRUCTIONS.RIGHT),
        # Prefer LIGHT before a turn
        (INSTRUCTIONS.LEFT, INSTRUCTIONS.LIGHT): (INSTRUCTIONS.LIGHT, INSTRUCTIONS.LEFT),
        (INSTRUCTIONS.RIGHT, INSTRUCTIONS.LIGHT): (INSTRUCTIONS.LIGHT, INSTRUCTIONS.RIGHT),
        # y'all know this one
        (INSTRUCTIONS.RIGHT, INSTRUCTIONS.RIGHT, INSTRUCTIONS.RIGHT): (INSTRUCTIONS.LEFT,),
        (INSTRUCTIONS.LEFT, INSTRUCTIONS.LEFT, INSTRUCTIONS.LEFT): (INSTRUCTIONS.RIGHT,),
    }.items()
}

NO_OP_MAP = dict(FULL_NO_OP_MAP)
# HACK Removing for better coverage of participant programs
del NO_OP_MAP[INSTRUCTIONS.LEFT*2]

NO_OP_SEQUENCES = tuple(NO_OP_MAP.keys())


class EnvLoader:
    @classmethod
    @property
    def maps(cls):
        return [LightbotMap(*cls._parse_map(m)) for m in cls._load_flat_maps()]

    @classmethod
    def grid(cls, *, rows=3):
        map_h, map_light, position, direction = cls._parse_map(cls._load_flat_maps()[5])
        map_light = cls._map_light_int_to_bool(map_light)
        map_light[:, :(8-rows)] = False
        return LightbotMap(map_h, cls._map_light_bool_to_int(map_light), position, direction)

    @classmethod
    @property
    def bleacher(cls):
        return cls.maps[3]

    @classmethod
    def single_bleacher(cls):
        map_h, map_light, position, direction = cls._parse_map(cls._load_flat_maps()[3])
        map_light = cls._map_light_int_to_bool(map_light)
        map_light[:, :4] = False
        return LightbotMap(map_h, cls._map_light_bool_to_int(map_light), position, direction)

    @classmethod
    def _map_light_int_to_bool(cls, m):
        return m != NO_LIGHT

    @classmethod
    def _map_light_bool_to_int(cls, m):
        map_light = np.full(m.shape, NO_LIGHT, dtype=int)
        ct = 0
        for x in range(map_light.shape[0]):
            for y in range(map_light.shape[1]):
                if m[x, y]:
                    map_light[x, y] = ct
                    ct += 1
        return map_light

    @classmethod
    def _load_flat_maps(cls):
        # could work as well? maps/lightbot_experiment-maps.txt
        with open(code_dir / '../maps/lightbot_experiment-flat-maps.txt') as f:
            return json.load(f)

    @classmethod
    def _load_maps(cls, fn):
        with open(fn) as f:
            ms = json.load(f)
        return [LightbotMap(*cls._parse_map(m)) for m in ms]

    @classmethod
    def _parse_map(cls, m):
        '''
        Parses a map from a maps.txt JSON file.

        We rearrange maps so their coordinates are x, y instead of the y, x format in maps.txt

        Returns a 2D array of map heights, a 2D array of tiles that are lights, and starting position/direction.

        Lights are returned in a slightly peculiar format:
        All light locations have a unique number, starting from 0. Non-light
        locations are assigned -1. This representation is intended to make it
        simple to track light state in a dense array representation, with the
        unique number of each light corresponding to an array index.
        '''
        map_h = np.zeros((len(m['map'][0]), len(m['map'])), dtype=int)
        map_light = np.full((len(m['map'][0]), len(m['map'])), NO_LIGHT, dtype=int)
        light_index = 0
        for yidx, c in enumerate(m['map']):
            for xidx, r in enumerate(c):
                map_h[xidx, yidx] = r['h']
                is_light = r['t'] == 'l'
                if is_light:
                    map_light[xidx, yidx] = light_index
                    light_index += 1

        # HACK Per js/lightbot.model.map.js
        map_h = map_h[:, ::-1]
        map_light = map_light[:, ::-1]

        position = (m['position']['x'], m['position']['y'])
        direction = m['direction']
        return map_h, map_light, position, direction

TraceState = collections.namedtuple('TraceState', ['trace', 'state', 'light_target'])

class LightbotTrace(object):
    '''
    This class is a hierarchical lightbot instance, where each light being lit is a possible subgoal.
    The state also contains a trace of actions instead of only the current state.

    The current subgoal/light is stored in the state's `light_target` field. When the field is unset,
    actions correspond to choices among lights. Once the light is lit, the field is again unset.
    '''
    def __init__(self, mdp):
        self.mdp = mdp
        self.lights = get_light_positions(mdp)

        @functools.lru_cache(maxsize=None)
        def mdp_next_state(s, a):
            return self.mdp.next_state(s, a)
        self._cached_mdp_next_state = mdp_next_state

    def initial_state(self):
        return TraceState('', self.mdp.initial_state(), None)
    def is_terminal(self, s):
        return self.mdp.is_terminal(s.state)
    def actions(self, s):
        if s.light_target is None:
            return [
                l for l in self.lights
                if s.state.map_lit[self.mdp.map_light[l]] == CONST_MAP_LIT_FALSE]
        else:
            return [
                a
                for a in self.mdp.actions(s.state)
                if not (
                    (s.trace + a)[-3:] in NO_OP_SEQUENCES or
                    (s.trace + a)[-2:] in NO_OP_SEQUENCES
                )
            ]

    def next_state_and_reward(self, s, a):
        if s.light_target is None:
            return TraceState(s.trace, s.state, a), 0
        ns = self._cached_mdp_next_state(s.state, a)
        r = self.mdp.reward(s.state, a, ns)
        l = (
            None if s.state.position == s.light_target and a == INSTRUCTIONS.LIGHT else
            s.light_target
        )
        return TraceState(s.trace + a, ns, l), r
    def next_state(self, s, a):
        ns, r = self.next_state_and_reward(s, a)
        return ns
    def reward(self, s, a, ns):
        ns, r = self.next_state_and_reward(s, a)
        return r
