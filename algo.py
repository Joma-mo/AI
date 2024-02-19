import heapq
import numpy as np
from state import next_state, solved_state
from location import next_location
from collections import OrderedDict


def dfs(init_state, visited, depth_limit, list_action, expand):
    if np.array_equal(init_state, solved_state()):
        return True
    if depth_limit > 0:
        expand[0] += 1
        for action in range(1, 13):
            list_action.append(action)
            state = next_state(init_state, action)
            visited.add(tuple(map(tuple, state)))
            if dfs(state, visited, depth_limit - 1, list_action, expand):
                return True
            else:
                list_action.pop()
    return False


def heuristic(current_location):
    small_cube_location = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    distance = 0
    for num in current_location.flatten():
        index = np.where(current_location == num)
        distance += (
                abs(index[0][0] - small_cube_location[num - 1][0]) + abs(index[1][0] - small_cube_location[num - 1][1])
                + abs(index[2][0] - small_cube_location[num - 1][2]))
    return distance / 4


def bfs(start_frontier, end_frontier, explored_start, explored_end, expand, end):
    start_vertex, start_actions = start_frontier.popitem(last=False)
    expand[0] += 1
    for action in range(1, 13):
        state = tuple(map(tuple, next_state(start_vertex, action)))
        list_actions = start_actions.copy()
        list_actions.append(action)
        start_frontier[state] = list_actions
        explored_start.add(state)
        if state in explored_end:
            list_all_actions = list_actions[::-1] if end else end_frontier[state][::-1]
            for i in range(len(list_all_actions)):
                if list_all_actions[i] <= 6:
                    list_all_actions[i] = list_all_actions[i] + 6
                else:
                    list_all_actions[i] = list_all_actions[i] - 6
            if end:
                list_all_actions = end_frontier[state] + list_all_actions
            else:
                list_all_actions = start_frontier[state] + list_all_actions
            print('Explored: ', len(explored_start) + len(explored_end))
            print('Expand: ', expand[0])
            print('Depth: ', len(list_all_actions))
            return True, list_all_actions
    return False, []


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.
 
    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.

    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cubes when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    if method == 'Random':
        return list(np.random.randint(1, 12 + 1, 10))

    elif method == 'IDS-DFS':
        visited = set()
        expand = [1]
        limit = 1
        while True:
            list_action = []
            if dfs(init_state, visited, limit, list_action, expand):
                print('Number of explored nodes: ', len(visited))
                print('Number of expanded nodes: ', expand[0])
                print('Depth: ', len(list_action))
                return list_action
            limit += 1

    elif method == 'A*':
        heap = []
        explored = set()
        expand = 0
        if np.array_equal(init_state, solved_state()):
            return []
        heapq.heappush(heap, (heuristic(init_location) + 0, [], init_state, init_location))
        explored.add(tuple(map(tuple, init_state)))
        while True:
            current_node = heapq.heappop(heap)
            expand += 1
            for action in range(1, 13):
                list_actions = current_node[1].copy()
                list_actions.append(action)
                state = next_state(current_node[2], action)
                loc = next_location(current_node[3], action)
                heuristic_value = heuristic(loc)
                if np.array_equal(state, solved_state()):
                    print('Expand: ', expand)
                    print('Explored: ', len(explored))
                    return list_actions
                if tuple(map(tuple, state)) not in explored:
                    heapq.heappush(heap, (heuristic_value + len(list_actions), list_actions, state, loc))
                    explored.add(tuple(map(tuple, state)))

    elif method == 'BiBFS':
        start_frontier = OrderedDict()
        end_frontier = OrderedDict()
        explored_start = set()
        explored_end = set()
        expand = [0]
        if np.array_equal(init_state, solved_state()):
            return []
        start_frontier[tuple(map(tuple, init_state))] = []
        explored_start.add(tuple(map(tuple, init_state)))
        end_frontier[tuple(map(tuple, solved_state()))] = []
        explored_end.add(tuple(map(tuple, solved_state())))

        while True:
            start_flag, start_actions = bfs(start_frontier, end_frontier, explored_start, explored_end, expand, False)
            if start_flag:
                return start_actions
            end_flag, end_actions = bfs(end_frontier, start_frontier, explored_end, explored_start, expand, True)
            if end_flag:
                return end_actions
    else:
        return []


