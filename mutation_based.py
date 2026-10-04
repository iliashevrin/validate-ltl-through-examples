#!/usr/bin/env python3
import re
import spot
spot.setup()
from utils import get_words_from_conditions, check_acceptance, collect_aps, trace_len
from utils import Mutation

import random
from collections import defaultdict
import itertools

from utils import top400_context, top200_context, top100_context, top50_context, top25_context, all_contexts



def ap_formula(name: str):
    return spot.formula.ap(name)


def build_ap_swap_map(aps):
    """
    If 2 APs: a <-> b
    If >=3 APs: cyclic shift a->b, b->c, ..., last->first
    """
    if len(aps) < 2:
        return {}

    if len(aps) == 2:
        return {
            aps[0]: aps[1],
            aps[1]: aps[0],
        }

    return {
        aps[i]: aps[(i + 1) % len(aps)]
        for i in range(len(aps))
    }


def apply_ap_swap(f, swap_map):
    def mapper(node):
        if node._is(spot.op_ap):
            name = str(node)
            if name in swap_map:
                return ap_formula(swap_map[name])
        return node.map(mapper)

    return mapper(f)



def children_count(f):
    return f.size()


def replace_child(f, target_idx, new_child):
    idx = -1

    def mapper(child):
        nonlocal idx
        idx += 1
        return new_child if idx == target_idx else child

    return f.map(mapper)


def current_node_mutations(f, depth):

    muts = []

    # Add negation to any subformula
    if not f._is(spot.op_Not):
        muts.append((spot.formula.Not(f), (Mutation.ADD_NEGATION, depth)))

    # Add X to any subformula
    muts.append((spot.formula.X(f), (Mutation.ADD_X, depth)))

    # Add F to any subformula, except if it already starts with F
    if not f._is(spot.op_F):
        muts.append((spot.formula.F(f), (Mutation.ADD_F, depth)))

    # Add G to any subformula, except if it already starts with G
    if not f._is(spot.op_G):
        muts.append((spot.formula.G(f), (Mutation.ADD_G, depth)))



    if f._is(spot.op_Implies) or f._is(spot.op_Equiv) or \
        f._is(spot.op_And) or f._is(spot.op_Or) or \
        f._is(spot.op_U) or f._is(spot.op_W):

        if children_count(f) == 2:
            muts.append((f[0], (Mutation.REMOVE_RIGHT_SUBFORMULA, depth)))
            muts.append((f[1], (Mutation.REMOVE_LEFT_SUBFORMULA, depth)))


    # Remove negation from any negated subformula
    if f._is(spot.op_Not) and f.size() == 1:
        muts.append((f[0], (Mutation.REMOVE_NEGATION, depth)))

    # Remove X from any X-subformula
    if f._is(spot.op_X) and f.size() == 1:
        muts.append((f[0], (Mutation.REMOVE_X, depth)))

    # Swap F/G/X
    if f._is(spot.op_F):
        muts.append((spot.formula.G(f[0]), (Mutation.SWAP_F_WITH_G, depth)))
        muts.append((spot.formula.X(f[0]), (Mutation.SWAP_F_WITH_X, depth)))
        muts.append((f[0], (Mutation.REMOVE_F, depth)))

    if f._is(spot.op_G):
        muts.append((spot.formula.F(f[0]), (Mutation.SWAP_G_WITH_F, depth)))
        muts.append((spot.formula.X(f[0]), (Mutation.SWAP_G_WITH_X, depth)))
        muts.append((f[0], (Mutation.REMOVE_G, depth)))

    elif f._is(spot.op_X):
        muts.append((spot.formula.F(f[0]), (Mutation.SWAP_X_WITH_F, depth)))
        muts.append((spot.formula.G(f[0]), (Mutation.SWAP_X_WITH_G, depth)))

    # U/W mutations
    if f._is(spot.op_U):
        muts.append((spot.formula.W(f[0], f[1]), (Mutation.SWAP_U_WITH_W, depth)))
        muts.append((spot.formula.U(f[1], f[0]), (Mutation.SWAP_U_OPERANDS, depth)))

    elif f._is(spot.op_W):
        muts.append((spot.formula.U(f[0], f[1]), (Mutation.SWAP_W_WITH_U, depth)))
        muts.append((spot.formula.W(f[1], f[0]), (Mutation.SWAP_W_OPERANDS, depth)))

    # Propositional operator mutations:
    # and, or, implies, iff -> any other one
    if f._is(spot.op_And):
        children = [f[i] for i in range(children_count(f))]

        if len(children) == 2:
            muts.append((spot.formula.Implies(children[0], children[1]),
                         (Mutation.SWAP_AND_WITH_IMPLIES, depth)))
            muts.append((spot.formula.Equiv(children[0], children[1]),
                         (Mutation.SWAP_AND_WITH_EQUIV, depth)))

        muts.append((spot.formula.Or(children),
                     (Mutation.SWAP_AND_WITH_OR, depth)))

    elif f._is(spot.op_Or):
        children = [f[i] for i in range(children_count(f))]

        if len(children) == 2:
            muts.append((spot.formula.Implies(children[0], children[1]),
                         (Mutation.SWAP_OR_WITH_IMPLIES, depth)))
            muts.append((spot.formula.Equiv(children[0], children[1]),
                         (Mutation.SWAP_OR_WITH_EQUIV, depth)))

        muts.append((spot.formula.And(children),
                     (Mutation.SWAP_OR_WITH_AND, depth)))

    elif f._is(spot.op_Implies):
        muts.append((spot.formula.And([f[0], f[1]]),
                     (Mutation.SWAP_IMPLIES_WITH_AND, depth)))
        muts.append((spot.formula.Or([f[0], f[1]]),
                     (Mutation.SWAP_IMPLIES_WITH_OR, depth)))
        muts.append((spot.formula.Equiv(f[0], f[1]),
                     (Mutation.SWAP_IMPLIES_WITH_EQUIV, depth)))

    elif f._is(spot.op_Equiv):
        muts.append((spot.formula.And([f[0], f[1]]),
                     (Mutation.SWAP_EQUIV_WITH_AND, depth)))
        muts.append((spot.formula.Or([f[0], f[1]]),
                     (Mutation.SWAP_EQUIV_WITH_OR, depth)))
        muts.append((spot.formula.Implies(f[0], f[1]),
                     (Mutation.SWAP_EQUIV_WITH_IMPLIES, depth)))


    return muts



def generate_mutants_at_all_nodes(f, depth=1):
    """
    Generate formulas where exactly one mutation is applied somewhere in the AST.

    Yields:
        tuple[spot.formula, Mutation]
    """
    for mutated, mutation_type in current_node_mutations(f, depth):
        yield mutated, mutation_type

    for i in range(children_count(f)):
        child = f[i]

        for mutated_child, mutation_type in generate_mutants_at_all_nodes(child, depth+1):
            yield replace_child(f, i, mutated_child), mutation_type


def mutate_ltl_formula(formula_str):
    """
    Args:
        formula_str: LTL formula string

    Returns:
        list[tuple[str, Mutation]]:
            Each item is (mutated_formula_string, mutation_type).
    """
    original = spot.formula(formula_str)
    original_str = str(original)

    mutants = []
    seen = set()


    # AP swap/cycle mutation, applied once globally
    aps = collect_aps(original)
    swap_map = build_ap_swap_map(aps)

    if swap_map:
        ap_swapped = apply_ap_swap(original, swap_map)
        ap_swapped_str = str(ap_swapped)

        if ap_swapped_str != original_str:
            key = (ap_swapped_str, Mutation.SWAP_APS)
            seen.add(key)
            mutants.append((ap_swapped_str, (Mutation.SWAP_APS, 0)))
            

    for mutant, mutation_type in generate_mutants_at_all_nodes(original):
        mutant_str = str(mutant)
        key = (mutant_str, mutation_type)

        if mutant_str != original_str and key not in seen:
            seen.add(key)
            mutants.append((mutant_str, mutation_type))

    return mutants


def accepting_traces(mutant):

    d = spot.make_bdd_dict()

    # optional but stabilizing: parse first only to get AP names
    f = spot.formula(mutant)
    aps = sorted(str(ap) for ap in spot.atomic_prop_collect(f))

    anchor = spot.make_twa_graph(d)
    for ap in aps:
        anchor.register_ap(ap)

    trans = spot.translator(d)
    aut = trans.run(mutant)

    run = aut.accepting_run()

    if run is None:
        return []

    return [str(spot.twa_word(run))]

    # conditions = []

    # # Prefix part
    # for edge in run.prefix:
    #     conditions.append(spot.bdd_format_formula(aut.get_dict(), edge.label))

    # cycle_start = len(conditions)

    # # Cycle part
    # for edge in run.cycle:
    #     conditions.append(spot.bdd_format_formula(aut.get_dict(), edge.label))

    # return get_words_from_conditions(conditions, cycle_start)



def rejecting_traces(mutant):
    return accepting_traces(f"!({mutant})")




def generate_traces(formula, strategy):
    mutants = mutate_ltl_formula(formula)

    traces = []
    used_traces = set()

    d = spot.make_bdd_dict()
    trans = spot.translator(d)
    original_aut = trans.run(formula)

    for mutant, mutation_type in mutants:

        # Generate candidate traces from mutant and negated mutant
        candidates = []
        candidates.extend(accepting_traces(mutant))
        candidates.extend(rejecting_traces(mutant))

        for trace in candidates:

            if trace is None or trace in used_traces:
                continue

            used_traces.add(trace)

            candidate_acceptance = check_acceptance(original_aut, trace)
            traces.append((trace, candidate_acceptance, mutation_type))

    traces = [(t, ac, f'{str(mut[0])}_{mut[1]}_{ac}_{trace_len(t)}') for (t, ac, mut) in traces if globals()[strategy](mut, ac, trace_len(t))]
    return traces
