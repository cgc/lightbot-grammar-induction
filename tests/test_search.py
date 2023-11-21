import numpy as np
import lb

class Graph(object):
    def __init__(self, adj, initial, goal, *, path=False, step_reward=-1):
        self.initial = initial
        self.goal = goal
        self.adj = adj
        self.step_reward = step_reward
        self.path = path
    def initial_state(self):
        return (self.initial,) if self.path else self.initial
    def actions(self, s):
        if self.path: s = s[-1]
        return range(len(self.adj[s]))
    def next_state_and_reward(self, s, a):
        arg_s = s
        if self.path: s = s[-1]
        ns = self.adj[s][a]
        if self.path:
            ns = arg_s + (ns,)
        return ns, self.reward(s, a, ns[-1] if self.path else ns)
    def reward(self, s, a, ns):
        if callable(self.step_reward):
            return self.step_reward(s, a, ns)
        return self.step_reward
    def is_terminal(self, s):
        if self.path: s = s[-1]
        return s == self.goal

def test_astar():
    bleacher = lb.EnvLoader.bleacher

    mdp = lb.LightbotMap(
        bleacher.map_h,
        bleacher.map_light,
        (1, 7),
        1,
    )

    # Basic test.
    v = lb.search.astar(lb.LightbotMapWithGoalPosition(mdp, (3, 7)), lb.heuristic_cost_to_go_goal_distance)
    assert v.result.cost_so_far == 2

    # Testing an unreachable goal.
    v = lb.search.astar(lb.LightbotMapWithGoalPosition(mdp, (8, 7)), lb.heuristic_cost_to_go_goal_distance)
    assert v.result is None
    # Every location * direction is visited
    assert v.iteration_count == np.prod(bleacher.map_h.shape) * len(lb.DIRECTIONS)

    # This is a more complex test of the topk kwarg of astar.
    g = Graph({
        0: [1, 3, 5, 7],
        1: [2],
        3: [4], 4: [2],
        5: [6], 6: [2],
        7: [8], 8: [9], 9: [2],
    }, 0, 2, path=True)
    paths = [
        (0, 1, 2),
        (0, 3, 4, 2),
        (0, 5, 6, 2),
        (0, 7, 8, 9, 2),
    ]
    no_heuristic = lambda *args: 0
    # Trying to get more than the possible paths
    assert {r.state for r in lb.search.astar(g, no_heuristic, topk=100).results} == set(paths)
    # Pretty sure paths[2] is expected, since we do LIFO
    assert {r.state for r in lb.search.astar(g, no_heuristic, topk=2).results} == {paths[0], paths[2]}
    assert {r.state for r in lb.search.astar(g, no_heuristic, topk=2, include_equal_score=True).results} == {paths[0], paths[1], paths[2]}
    assert {r.state for r in lb.search.astar(g, no_heuristic, topk=100, f_upper_bound=4).results} == {paths[0], paths[1], paths[2]}

    # This tests that we drop traces with infinite cost.
    g = Graph({
        0: [1, 2],
        2: [1],
    }, 0, 1, path=True, step_reward=lambda s, a, ns: float('-inf') if s == 2 and ns == 1 else -1)
    paths = [
        (0, 1),
        (0, 2, 1),
    ]
    assert {r.state for r in lb.search.astar(g, no_heuristic, topk=100, drop_infinite_cost=False).results} == set(paths)
    assert {r.state for r in lb.search.astar(g, no_heuristic, topk=100).results} == {paths[0]}

class RomaniaSubsetAIMA:
    '''
    This small weighted graph is from Figure 3.15 in Artificial Intelligence: A Modern Approach, 3rd edition.
    It's used to illustrate an important case for Uniform Cost Search (and A* with no heuristic), where
    a state can be subsequently encountered through a more efficient path.
    '''
    state_list = ('Sibiu', 'Fagaras', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest')
    costs = {
        frozenset({'Sibiu', 'Fagaras'}): 99,
        frozenset({'Sibiu', 'Rimnicu Vilcea'}): 80,
        frozenset({'Rimnicu Vilcea', 'Pitesti'}): 97,
        frozenset({'Pitesti', 'Bucharest'}): 101,
        frozenset({'Fagaras', 'Bucharest'}): 211,
    }
    optimal_path = ['Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest']
    def initial_state(self): return 'Sibiu'
    def is_terminal(self, s): return s == 'Bucharest'
    def actions(self, s):
        return [
            ns for ns in self.state_list
            if frozenset({s, ns}) in self.costs]
    def next_state_and_reward(self, s, a):
        return a, -self.costs[frozenset({s, a})]

def test_astar_costs():
    def no_heuristic(*args): return 0
    v = lb.search.astar(RomaniaSubsetAIMA(), no_heuristic, return_path=True)
    assert v.result.path == RomaniaSubsetAIMA.optimal_path
