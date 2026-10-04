#!/usr/bin/env python3
import spot
spot.setup()
import itertools
from utils import get_words_from_conditions, check_acceptance, trace_len
import random


EXPERT_ORDER = [
    [0, 0, 1, 0],
    [0, 1, 2, 3, 0],
    [0, 1, 2, 3, 4, 5, 5],
    [0, 1, 2, 3, 1],
    [0, 1, 2, 3, 4, 4],
    [0, 0, 1, 1, 2, 2],
    [0, 1, 1, 2, 2],
    [0, 1, 2, 1],
    [0, 1, 2, 3, 3],
    [0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 2, 2],
    [0, 0],
    [0, 1, 1],
]



def traversal_expert(candidate, max_visits=3):

    traces = traversal_gradual(candidate, max_visits)
    rank = {str(value): i for i, value in enumerate(EXPERT_ORDER)}
    traces.sort(key=lambda trace: rank.get(str(trace[2]), -1), reverse=True)

    return traces

def traversal_by_length(candidate, max_visits=3):

    traces = traversal_gradual(candidate, max_visits)
    traces.sort(key=lambda trace: trace_len(trace[0]))

    return traces


def traversal_random(candidate, max_visits=3):

    traces = traversal_gradual(candidate, max_visits)
    random.shuffle(traces)
    return traces


def traversal_gradual(candidate, max_visits=3):

    aut = spot.translate(candidate, 'parity', 'sbacc', 'state-based', 'complete', 'colored', 'deterministic')

    traces = []
    for i in range(1, max_visits):
        traces.extend(generate_traces(aut, i))
    return traces



def largest_repeated_sublist(nums: list[int]) -> int:
    n = len(nums)

    for size in range(1, n + 1):
        if n % size == 0:
            pattern = nums[:size]
            if pattern * (n // size) == nums:
                return size

    return nums


def generate_traces(aut, max_visits, max_generate=30):

    traces = []
    paths = []

    for edge in aut.out(aut.get_init_state_number()):
        paths.append(([edge], {edge.src:1}))

    all_vars = aut.ap()

    while paths:

        path, visited = paths.pop(0)
        state = path[-1].dst

        if state in visited and visited[state] == max_visits:

            nodes = [edge.src for edge in path]
            index = nodes.index(state)
            sublist = largest_repeated_sublist(nodes[index:])

            path = path[:index+sublist]

            
            conditions = [spot.bdd_format_formula(aut.get_dict(), edge.cond) for edge in path]
            words = get_words_from_conditions(conditions, index)

            mapping = []
            nodes = nodes[:index+sublist] + [state]

            for node in nodes:
                if node not in mapping:
                    mapping.append(node)

            props = str([mapping.index(node) for node in nodes])

            if words:
                if aut.intersects(spot.parse_word(words[0])):
                    traces.extend([(word, True, props) for word in words])
                else:
                    traces.extend([(word, False, props) for word in words])

                if len(traces) >= max_generate:
                    break

        else:

            if state not in visited:
                visited[state] = 1
            else:
                visited[state] += 1

            for edge in aut.out(state):
                paths.append((path + [edge], visited.copy()))

    return traces
