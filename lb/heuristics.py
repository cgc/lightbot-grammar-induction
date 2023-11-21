from operator import is_
import collections
import heapq
import numba
import numpy as np
from .envs import DIRECTIONS, is_unlit_light, get_light_positions, CONST_MAP_LIT_FALSE, CONST_MAP_LIT_TRUE, lit_count
import functools

# @numba.njit
def heuristic_navigation_cost_to_target(position, direction, goal_position):
    return_direction = True # fixing b/c of numba
    diff = (
        goal_position[0] - position[0],
        goal_position[1] - position[1],
    )
    diff_heading = (
        +1 if diff[0] > 0 else 0 if diff[0] == 0 else -1,
        +1 if diff[1] > 0 else 0 if diff[1] == 0 else -1,
    )

    if diff == (0, 0):
        cost = 0
        return (cost, direction) if return_direction else cost

    heading = DIRECTIONS[direction]
    heading_and_diff_match = (
        #(heading[0] > 0 and diff[0] > 0) or (heading[0] < 0 and diff[0] < 0),
        #(heading[1] > 0 and diff[1] > 0) or (heading[1] < 0 and diff[1] < 0),
        heading[0] != 0 and heading[0] == diff_heading[0],
        heading[1] != 0 and heading[1] == diff_heading[1],
    )

    # This case means we only have to move in one direction.
    if diff[0] == 0 or diff[1] == 0:
        idx = 1 if diff[0] == 0 else 0
        if heading_and_diff_match[idx]: # Moving in our heading.
            turn_count = 0
        elif heading[idx] == 0: # Movement is perpendicular to our direction
            turn_count = 1
        else: # We have to move opposite our heading, so we do a u-turn
            turn_count = 2
        # Total cost is number of turns, and movement distance.
        cost = turn_count + abs(diff[idx])
        return (cost, DIRECTIONS.index(diff_heading)) if return_direction else cost

    # We always have to turn when changing directions.
    turn_count = 1

    # Does the heading match either direction?
    if not (heading_and_diff_match[0] or heading_and_diff_match[1]):
        # If it doesn't match either, then we'll have to turn
        turn_count += 1

    # Total cost is number of turns, and movement distance.
    cost = abs(diff[0]) + abs(diff[1]) + turn_count
    return (
        cost,
        DIRECTIONS.index(
            # If your oriented correctly for one direction, then you'll be facing in the other at the end.
            # Otherwise, if you're facing the opposite direction of the heading you need, then
            # you'll turn once to go perpendicular, then complete your move by turning to face the goal.
            (diff_heading[0], 0) if heading_and_diff_match[1] or diff_heading[0] == -heading[0] else
            (0, diff_heading[1])
        ),
    ) if return_direction else cost


# @numba.njit
def simple_heuristic_navigation_cost_to_target(position, goal_position):
    diff = (
        goal_position[0] - position[0],
        goal_position[1] - position[1],
    )
    if diff == (0, 0):
        return 0

    turn_cost = 1
    if diff[0] == 0 or diff[1] == 0:
        turn_cost = 0

    return turn_cost + abs(diff[0]) + abs(diff[1])


# @numba.njit
def heuristic_cost_lights_and_nearest_distance(mdp, s):
    min_dist = np.inf

    unlit_count = 0

    for x in range(mdp.map_light.shape[0]):
        for y in range(mdp.map_light.shape[1]):
            light_idx = mdp.map_light[x, y]
            if not is_unlit_light(s.map_lit, light_idx):
                continue
            unlit_count += 1
            dist, _ = heuristic_navigation_cost_to_target(s.position, s.direction, (x, y))
            if dist < min_dist:
                min_dist = dist

    if unlit_count == 0:
        return 0

    # Takes at least 2 instructions to light: a movement & lighting (imagine a line of lights)
    # To make sure we're admissible, we sum
    # - ideal/perfect navigation to the next light (dist)
    # - a light instruction (+1)
    # - and at least 2 instructions for each remaining light (2 * (unlit_count - 1))
    return min_dist + 1 + 2 * (unlit_count - 1)


# @numba.njit
def heuristic_cost_to_go_goal_distance(mdp, state, cost_so_far):
    gp = mdp.goal_position
    p = state.position
    return (
        abs(gp[0] - p[0]) +
        abs(gp[1] - p[1])
    )


# @numba.njit
def heuristic_cost_to_go_light_progress(mdp, state):
    lights_remaining = mdp.total_lights - mdp.lit_count(state)
    light_idx = mdp.map_light[state.position]
    unlit_light = is_unlit_light(state.map_lit, light_idx)
    # Takes at least 2 instructions to light: a movement & lighting (imagine a line of lights)
    # To make sure we're admissible, we assume the next light will only take 1
    # instruction (LIGHT) if we're at an unlit light.
    h = 2 * lights_remaining + (
        -1 if unlit_light else 0)
    return h

MSTResult = collections.namedtuple('MSTResult', ['total_cost', 'C', 'E'])

# @numba.njit
def prims_algorithm(mdp):
    '''
    We're using prim's for our MST b/c it is better than kruskal when you have large # of edges (as we do)
    Following wiki very closely: https://en.wikipedia.org/wiki/Prim%27s_algorithm#Description
    '''
    Q = {s: True for s in mdp.state_list} # Value is arbitrary
    queue = [(np.inf, mdp.state_list[0])] # by doing this, we assume graph is connected
    #C = collections.defaultdict(lambda: np.inf)
    C = {}
    E = {}

    while Q:
        _, v = heapq.heappop(queue)

        # Doing this to handle duplicate entries in priority queue
        if v not in Q:
            continue

        Q.pop(v)

        for vw in mdp.actions(v):
            w = mdp.next_state(v, vw)
            vw_cost = -float(mdp.reward(v, vw, w))
            #if w in Q and vw_cost < C[w]:
            if w in Q and vw_cost < C.get(w, np.inf):
                C[w] = vw_cost
                E[w] = v
                heapq.heappush(queue, (C[w], w))

    return MSTResult(
        #total_cost=sum(C.values()), # this implicitly avoids the value for s0
        total_cost=sum([v for v in C.values()]), # this implicitly avoids the value for s0
        C=C,
        E=E,
    )

class _LightGraph(object):
    def __init__(self, mdp, map_lit, all_lights, navigation_cost):
        self.mdp = mdp
        self.navigation_cost = navigation_cost

        self.lights = [
            pos
            for idx, pos in enumerate(all_lights)
            if map_lit[idx] == CONST_MAP_LIT_FALSE
        ]

    @property
    def state_list(self):
        return self.lights
    def actions(self, s):
        return self.lights
    def next_state(self, s, a):
        return a
    def reward(self, s, a, ns):
        if self.navigation_cost is None:
            c = simple_heuristic_navigation_cost_to_target(s, ns)
        else:
            c = self.navigation_cost(s, ns)
        return -(c + 1) # add one for lighting


def make_heuristic_cost_navigation_to_mst(mdp):
    all_lights = get_light_positions(mdp)

    @functools.lru_cache(maxsize=None)
    def simple_navigation_cost(s, ns):
        return simple_heuristic_navigation_cost_to_target(s, ns)

    @functools.lru_cache(maxsize=None)
    def navigation_cost(position, direction, goal_position):
        # Can't use this on TSP b/c it is a directed cost b/c it's orientation-dependent --
        # pretty sure that taking the best case over orientations gives us simple_navigation_cost above
        # separately caching this b/c it is invariant to map_lit? not sure if this matters on small problems...
        c, _ = heuristic_navigation_cost_to_target(position, direction, goal_position)
        return c

    @functools.lru_cache(maxsize=None)
    def tsp_heuristic_from_mst(map_lit):
        if lit_count(map_lit) == mdp.total_lights - 1:
            return 0
        lg = _LightGraph(mdp, map_lit, all_lights, simple_navigation_cost)
        r = prims_algorithm(lg)
        return r.total_cost

    @functools.lru_cache(maxsize=None)
    def closest_light_cost(s):
        min_cost = float('inf')
        for light_idx, light_position in enumerate(all_lights):
            if s.map_lit[light_idx] == CONST_MAP_LIT_TRUE:
                continue
            #c = main.heuristic_navigation_cost_to_target(s.position, s.direction, light_position)
            c = navigation_cost(s.position, s.direction, light_position)
            if c < min_cost:
                min_cost = c
        return min_cost

    def heuristic_cost_navigation_to_mst(mdp, s, *, light_target=None):
        '''
        We use MST as an admissible heuristic for the TSP part of things
        I think we can't use the more complex cost I've written for this
        Since it will mean the graph is directed
        Then we add a cost for the closest light (but here we _can_ use the more complex cost)
        '''
        assert mdp is mdp
        if mdp.all_lit == s.map_lit:
            return 0

        if light_target is not None:
            closest = navigation_cost(s.position, s.direction, light_target)
        else:
            closest = closest_light_cost(s)

        # add one for lighting the first light (since the n lights are connected with n-1
        # edges&costs which means there's one last light left)
        return tsp_heuristic_from_mst(s.map_lit) + closest + 1

    return heuristic_cost_navigation_to_mst

