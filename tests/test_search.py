import numpy as np
import lb
from journal.figs import mdp_names

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

def test_astar_topk():
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

def _exhaustive_enumeration(mdp, *, limit=None):
    assert isinstance(mdp, lb.envs.SimpleLightbotTrace)
    curr = [mdp.initial_state()]
    results = []
    for _ in range(limit):
        next_ = []
        for s in curr:
            for a in mdp.actions(s):
                if a == lb.INSTRUCTIONS.JUMP: continue
                ns, r = mdp.next_state_and_reward(s, a)
                if r == mdp.mdp.noop_reward: continue
                if mdp.is_terminal(ns):
                    results.append(ns.trace)
                else:
                    next_.append(ns)
        curr = next_
    return results

def test_astar_topk_lightbot():
    no_heuristic = lambda *args: 0
    mdp = lb.EnvLoader.maps[0]
    mdp.noop_reward = float('-inf')

    # Incomplete enumeration of routes without trace-based MDP
    r = lb.search.astar(mdp, no_heuristic, topk=2, include_equal_score=True)
    assert [rr.score for rr in r.results] == [4, 9, 9]

    # We need trace-based MDP for full enumeration of routes
    r = lb.search.astar(lb.envs.SimpleLightbotTrace(mdp), no_heuristic, topk=2, include_equal_score=True)
    dev_l = 'LWR'
    dev_l_end = 'RWS'
    dev_r = 'RWL'
    dev_r_end = 'LWS'
    assert {
        rr.state.trace
        for rr in r.results
    } == {lb.tools.mkinst(t) for t in [
        'WWWS',
        f'{dev_l}WWW{dev_l_end}',
        f'{dev_r}WWW{dev_r_end}',
        f'W{dev_l}WW{dev_l_end}',
        f'W{dev_r}WW{dev_r_end}',
        f'WW{dev_l}W{dev_l_end}',
        f'WW{dev_r}W{dev_r_end}',
    ]}

    # Here's another test case.
    mdp = lb.LightbotMap(
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]),
        np.array([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, 0],
        ]),
        (0, 0),
        1,
    )
    mdp.noop_reward = float('-inf')

    wrapped_mdp = lb.envs.SimpleLightbotTrace(mdp)

    r = lb.search.astar(wrapped_mdp, no_heuristic, topk=4, include_equal_score=True)
    assert {
        rr.state.trace
        for rr in r.results
    } == {lb.tools.mkinst(t) for t in [
        'WWLWWS',
        'WLWWRWS',
        'LWWRWWS',
        'WLWRWLWS',
        'LWRWWLWS',
    ]} == set(_exhaustive_enumeration(wrapped_mdp, limit=8))


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

def test_heuristics_match():
    topk = 1000
    # NOTE: To test that the heuristics match for all maps, comment this out.
    # Keeping this reduced list for now to keep the test fast.
    mdp_names = [('maps', 8)]

    for mdp_name in mdp_names:
        print(f'\n{mdp_name}')
        mdp = lb.exp.mdp_from_name(mdp_name)
        mdp.noop_reward = float('-inf')

        # Search with new heuristic based on shortest paths

        with lb.fitting.timed('make heuristic'):
            heuristic_cost = lb.heuristics.make_shortest_path_heuristic(mdp)
            h = lambda mdp, state, _: heuristic_cost(mdp.mdp, state.state, _)

        with lb.fitting.timed('A* with shortest path heuristic'):
            wrapped_mdp = lb.envs.SimpleLightbotTrace(mdp)
            res = lb.search.astar(wrapped_mdp, h, topk=topk, include_equal_score=True)
        assert res.non_monotonic_counter == 0

        # Now using old TSP/MST heuristic

        mst_h = lb.heuristics.make_heuristic_cost_navigation_to_mst(mdp)
        h = lambda wrapped_mdp, s, _: mst_h(wrapped_mdp.mdp, s.state, light_target=s.light_target)

        with lb.fitting.timed('A* with TSP heuristic'):
            wrapped_mdp = lb.envs.LightbotTrace(mdp)
            res2 = lb.search.astar(wrapped_mdp, h, topk=topk, include_equal_score=True)
        assert res2.non_monotonic_counter == 0

        # Assert match
        assert len(res.results) == len(res2.results)
        assert (
            {(r.score, r.state.trace) for r in res.results} ==
            {(r.score, r.state.trace) for r in res2.results})
