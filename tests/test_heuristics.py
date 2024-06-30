import numpy as np
import collections
import heapq
import numba

import lb

def test_simple_heuristic_navigation_cost_to_target():
    assert lb.simple_heuristic_navigation_cost_to_target((0, 0), (3, 4)) == 7+1
    assert lb.simple_heuristic_navigation_cost_to_target((0, 4), (3, 4)) == 3
    assert lb.simple_heuristic_navigation_cost_to_target((3, 0), (3, 4)) == 4
    assert lb.simple_heuristic_navigation_cost_to_target((3, 4), (3, 4)) == 0

def test_heuristic_navigation_cost_to_target():
    map_h, map_light = np.zeros((7, 9), dtype=int), -np.ones((7, 9), dtype=int)

    g = (3, 4)
    x_pos = (0, 4)
    x_neg = (6, 4)
    y_pos = (3, 0)
    y_neg = (3, 8)
    all_pos = (0, 0)
    all_neg = (6, 8)

    dir_y_neg = lb.DIRECTIONS.index((0, -1))
    dir_x_pos = lb.DIRECTIONS.index((+1, 0))
    dir_y_pos = lb.DIRECTIONS.index((0, +1))
    dir_x_neg = lb.DIRECTIONS.index((-1, 0))

    for (pos, dir, expected_cost, expected_dir) in [
        (g, 0, 0, 0),
        (g, 1, 0, 1),
        (g, 2, 0, 2),
        (g, 3, 0, 3),

        (x_pos, 0, 4, dir_x_pos),
        (x_pos, 1, 3, dir_x_pos),
        (x_pos, 2, 4, dir_x_pos),
        (x_pos, 3, 5, dir_x_pos),

        (x_neg, 0, 4, dir_x_neg),
        (x_neg, 1, 5, dir_x_neg),
        (x_neg, 2, 4, dir_x_neg),
        (x_neg, 3, 3, dir_x_neg),

        (y_pos, 0, 6, dir_y_pos),
        (y_pos, 1, 5, dir_y_pos),
        (y_pos, 2, 4, dir_y_pos),
        (y_pos, 3, 5, dir_y_pos),

        (y_neg, 0, 4, dir_y_neg),
        (y_neg, 1, 5, dir_y_neg),
        (y_neg, 2, 6, dir_y_neg),
        (y_neg, 3, 5, dir_y_neg),

        (all_pos, 0, 9, dir_y_pos),
        (all_pos, 1, 8, dir_y_pos),
        (all_pos, 2, 8, dir_x_pos),
        (all_pos, 3, 9, dir_x_pos),

        (all_neg, 0, 8, dir_x_neg),
        (all_neg, 1, 9, dir_x_neg),
        (all_neg, 2, 9, dir_y_neg),
        (all_neg, 3, 8, dir_y_neg),
    ]:
        assert lb.heuristic_navigation_cost_to_target(pos, dir, g) == (expected_cost, expected_dir)
        mdp = lb.LightbotMapWithGoalPosition(lb.LightbotMap(map_h, map_light, pos, dir), g)
        # Make sure the heuristic matches the cost from search
        assert lb.search.astar(mdp, lb.heuristic_cost_to_go_goal_distance).result.cost_so_far == expected_cost


def test_heuristic_cost_lights_and_nearest_distance():
    mdp = lb.EnvLoader.maps[1]
    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 1, '11')) == 0

    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 1, '01')) == 4 + 1
    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 1, '10')) == 5 + 1

    # either light
    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 0, '00')) == 5 + 1 + 2
    # upper light
    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 1, '00')) == 4 + 1 + 2
    # upper light
    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 2, '00')) == 5 + 1 + 2
    # either light
    assert lb.heuristic_cost_lights_and_nearest_distance(mdp, lb.State((1, 3), 3, '00')) == 6 + 1 + 2


def test_prims_algortithm():
    class Graph:
        def __init__(self, adj, initial):
            self.adj = adj
            self.initial = initial
        def initial_state(self):
            return self.initial
        def actions(self, s):
            return list(range(len(self.adj[s])))
        def next_state(self, s, a):
            return self.adj[s][a][0]
        def reward(self, s, a, ns):
            return -self.adj[s][a][1]
        @property
        def state_list(self):
            return list(self.adj.keys())

    adj = {
        'A': [
            ('B', 3), # HACK: changed this to get same as wiki
            ('D', 1),
        ],
        'B': [
            ('A', 3), # HACK: changed this to get same as wiki
            ('D', 2),
        ],
        'C': [
            ('D', 3),
        ],
        'D': [
            ('A', 1),
            ('B', 2),
            ('C', 3),
        ],
    }

    g = Graph(adj, 'A')

    r = lb.prims_algorithm(g)
    assert r.total_cost == 6
    assert r.E == dict(B='D', D='A', C='D')

def test_make_heuristic_cost_navigation_to_mst():
    base_mdp = lb.EnvLoader.maps[8]
    h = lb.make_heuristic_cost_navigation_to_mst(base_mdp)

    s = base_mdp.initial_state()
    assert h(base_mdp, s) == 3 + 6 + 3
    s = base_mdp.next_state(s, 'E')
    s = base_mdp.next_state(s, 'B')
    assert h(base_mdp, s) == 1 + 6 + 3
    s = base_mdp.next_state(s, 'A')
    assert h(base_mdp, s) == 6 + 3

def test_reachable_states():
    mdp = lb.LightbotMap(
        np.array([
            [0, 0, 0],
            [0, 1, 2],
            [0, 0, 0],
        ]),
        np.array([
            [-1, -1, -1],
            [-1, 0, -1],
            [-1, -1, -1],
        ]),
        (0, 0),
        0,
    )
    reachable = set(lb.heuristics.reachable_states(mdp))
    enumerated_states = {
        lb.State(
            (x, y),
            dir,
            ('{:0' + str(mdp.total_lights) + 'b}').format(map_lit))
        for x in range(mdp.map_h.shape[0])
        for y in range(mdp.map_h.shape[1])
        for dir in range(4)
        for map_lit in range(2**mdp.total_lights)
    }
    # Since you can only reach the height=2 block by passing the light, certain
    # states are not reachable.
    assert enumerated_states - reachable == {
        # Can only jump up to light from 3 directions, then you have to activate the light.
        # So, you can't be on an unlit light from 1 direction.
        lb.State((1, 1), 0, '0'),
        # Can only reach height=2 block from light, so it will never be unlit.
        *{
            lb.State((1, 2), dir, '0')
            for dir in range(4)
        },
    }

def test_shortest_path_length_to_any_goal_simple():
    import networkx as nx
    g = nx.Graph()
    for i in range(9):
        g.add_edge(i, i+1)
    assert len(g) == 10
    for impl in ['hidden', 'augment', 'loop-min', 'diy']:
        rv = lb.heuristics.shortest_path_length_to_any_goal(g, [0, 9], impl=impl)
        assert rv == {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 4,
            6: 3,
            7: 2,
            8: 1,
            9: 0,
        }

def test_shortest_path_length_to_any_goal_lightbot():

    mdp = lb.exp.mdp_from_name(('maps', 8))
    # Copy it, so we can change initial state
    mdp = lb.envs.LightbotMap(mdp.map_h, mdp.map_light, mdp.position0, mdp.direction0)
    mdp.noop_reward = float('-inf')

    reachable = list(lb.heuristics.reachable_states(mdp))

    # Make heuristic
    h_ = lb.heuristics.make_heuristic_cost_navigation_to_mst(mdp)
    wrapped_mdp = lb.envs.LightbotTrace(mdp)
    h = lambda wrapped_mdp, s, _: h_(wrapped_mdp.mdp, s.state, light_target=s.light_target)

    astar_search_path_lengths = {}
    # Get path lengths by running A* search from all starting states
    for s in reachable:
        if s.map_lit != '0'*mdp.total_lights:
            continue
        mdp.position0 = s.position
        mdp.direction0 = s.direction
        res = lb.search.astar(wrapped_mdp, h, topk=1, include_equal_score=False)
        astar_search_path_lengths[mdp.initial_state()] = res.result.score

    converted = lb.heuristics.convert_lightbot_to_networkx(mdp)
    first_spl = lb.heuristics.shortest_path_length_to_any_goal(*converted)
    for impl in ['hidden', 'augment', 'loop-min', 'diy']:
        spl = lb.heuristics.shortest_path_length_to_any_goal(*converted, impl=impl)
        # Check against A* search
        for key, pl in astar_search_path_lengths.items():
            assert pl == spl[key]
        # Check that all implementations match
        assert spl == first_spl
