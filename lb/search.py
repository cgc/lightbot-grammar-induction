import collections
import heapq

AStarSearchResult = collections.namedtuple('AStarSearchResult', ['score', 'cost_so_far', 'state', 'path'])
AStarSearchReturnValue = collections.namedtuple('AStarSearchReturnValue', ['results', 'result', 'iteration_count', 'frontier_counter', 'frontier_size', 'non_monotonic_counter'])

def astar(mdp, heuristic_cost_to_go, topk=1, f_upper_bound=None, include_equal_score=False, return_path=False, drop_infinite_cost=True):
    def extract_path(came_from, state):
        path = [state]
        while state != mdp.initial_state():
            state = came_from[state]
            path.append(state)
        return path[::-1]

    results = []
    iter_counter = 0
    non_monotonic_counter = 0

    queue_counter = 0
    queued = {}
    queue = []

    def _add_to_queue(s, g, f):
        nonlocal queue_counter
        node = (
            (
                f,
                -queue_counter,
                g,
            ),
            s,
        )
        heapq.heappush(queue, node)
        queue_counter += 1
        queued[s] = node

    s0 = mdp.initial_state()
    _add_to_queue(s0, 0, heuristic_cost_to_go(mdp, s0, 0))

    visited = set()
    if return_path: came_from = {}

    while queue:
        iter_counter += 1
        node = heapq.heappop(queue)
        (f, _, g), state = node

        if state in visited:
            assert state not in queued, 'Previously visited states should be removed from queued dictionary'
            continue
        else:
            assert queued[state] is node, 'Ensure queued dictionary always contains best node.'
            del queued[state]

        if mdp.is_terminal(state):
            # If we're over the bound, then we return.
            if f_upper_bound is not None and f >= f_upper_bound:
                break
            # If we're trying to include all with equal score, we check for termination condition.
            if include_equal_score and len(results) >= topk:
                if results[-1].score != f:
                    break
            # Only now do we append.
            results.append(AStarSearchResult(f, g, state, extract_path(came_from, state) if return_path else None))
            # If we've collected enough results, we return.
            if not include_equal_score and len(results) == topk:
                break
            continue

        visited.add(state)

        for action in mdp.actions(state):
            next_state, reward = mdp.next_state_and_reward(state, action)

            # Some simple tests for neighbors to skip
            if next_state in visited:
                continue
            if drop_infinite_cost and reward == float('-inf'):
                continue

            # Now, a more complicated test, based on whether we improve on a previously time it was queued
            next_g = g - reward
            if next_state in queued:
                (_, _, existing_g), _ = queued[next_state]
                # We skip when the cost is the same or greater than previous
                if next_g >= existing_g:
                    continue

            # Add to queue
            next_h = heuristic_cost_to_go(mdp, next_state, next_g)
            next_f = next_g + next_h
            _add_to_queue(next_state, next_g, next_f)
            if return_path: came_from[next_state] = state
            if f > next_f:
                non_monotonic_counter += 1

    return AStarSearchReturnValue(
        results,
        None if len(results) == 0 else results[0],
        iter_counter,
        queue_counter,
        len(queue),
        non_monotonic_counter=non_monotonic_counter,
    )
