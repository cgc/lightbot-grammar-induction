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
