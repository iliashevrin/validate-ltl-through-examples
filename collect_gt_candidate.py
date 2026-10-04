#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import re

import random

import pandas as pd

from openai import OpenAI

import spot
spot.setup()

import config
import json

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY



INSTRUCTION = """
Return EXACTLY one LTL formula as a single line.
The output must be directly parseable as an LTL formula using the Spot API.
"""

BASIC = """
You are a Linear Temporal Logic ( LTL ) Parser. Your task is to convert a given natural language statement to an LTL formula, using the provided mapping of natural language phrases to atomic propositions.

LTL Symbols :
    - AND : &
    - OR : |
    - NOT : !
    - IMPLIES : ->
    - BIIMPLICATION : <->
    - NEXT : X
    - EVENTUALLY : F
    - ALWAYS : G
    - UNTIL : U

Natural Language statement : {requirement}
Atomic Propositions : {atomic_proposition}
"""

# Direct TL variant"
ARTEMIS = """
You are an expert in translating natural language to linear temporal logic (LTL). Your job is to translate natural language to LTL. You must only use LTL operators and atomic propositions (NO NUMERICAL COMPARISON OPERATORS ALLOWED).
Recall that in LTL, G = globally, F = eventually, V = releases, X = next, U = until. You may use boolean operators (e.g., !, &, |, ->, <->) and can only use atomic propositions (NO NUMERICAL COMPARISON OPERATORS ALLOWED).

Inputs consist of:
1. unstructured natural language (string)
2. atomic proposition dictionary
    
The Outputs consist of:
1. output_LTL
    
Provide a list of the top 1 most likely translations (ordered by most likely first to least likely last) in the above output format for the following:

input_natural_language:{requirement}
atomic_propositions:{atomic_proposition}

"""

ADARULE = """
Task:
Translate natural language (NL) sentences into Linear Temporal Logic (LTL) formulas accurately.
Your answers always need to follow the output format.

Rules:
The converted formula should only contain atomic statements and operators.
Use standard LTL syntax and operators: G (globally), F (eventually), X (next), U (until), R (release), ! (negation), & (conjunction), | (disjunction), -> (implication), <-> (equivalence).
G means "globally": G a indicates that a is true in all future states.
F means "finally": F a indicates that a will eventually be true in some future state.
X means "next": X a is true if a is true in the next state.
U means "until": a U b is true if a remains true until b becomes true.
R means "release": a R b means b must be true until the moment when a is true and b is true. Once a is true, b can no longer be true. If a is never true, then b must always remain true.
Remember especially that the brackets match, we stipulate that each atomic formula is followed by a space. 
Do not change atomic propositions in NL.


Guidelines:
When translate "never", use  G!;
When translate "every time", "always", "all the time", use  G;
When translate "at certain moment", "eventually", "in the futhure" , "sooner or later", use  F;
When expressing "both A and B" or A and B will happen together at some moment, use A ∧ B;
When translate  "A or B holds"  or at some moment at least one of  A , B will be true , use A | B;
When the sentence is a discription of the system state, it means the state is always satisfied, so use G;
When the sentence is a discription of a and b happens, then finally,c and d will happen, you should considered the situation that the state "a and b" never satisfied , (e.g. G(!(a & b)) );
When translate "it is going to happen that a after b", use b -> F (a);
When translate  "Never (a) after (b)",use G((b) -> G(!(a)));
When translate  "Whenever (a), then (b)", use G((b) -> (a));
When translate  "A never happens", use G(!A);
For "After A, B happens"， Weak interpretation (B may happen even if A doesn’t)， use G(A -> F(B)),  Strong interpretation (A must happen first, then B), use !A | F(A & F(B));
For sentences like "Whenever A or B, then eventually C or D", use G((A | B) -> F(C | D));
For sentences like "At some point (A), and later (B)", use F(A & F(B));
For "First A and B, then eventually C or D" , use !(A & B) | F((A & B) & F(C | D));
For "After A and B, eventually C", use G((A & B) -> F(C));

Natural Language: {requirement}
Atomic Propositions: {atomic_proposition}

Please response in plain text format. DO NOT use markdown, latex or any other formats.
Please response in the following format, and replace the '[LTL formula]' with the LTL formula translated from the natural language sentences:
[LTL formula]
"""

def normalize_formula(text: str) -> str:
    """Strip common formatting so the result is parser-friendly."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Keep only the first non-empty line if the model accidentally adds extra text.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def ask_chatgpt(client: OpenAI, prompt: str, requirement: str, atomic_proposition: str) -> str:

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": globals()[prompt].format(
                    requirement=requirement,
                    atomic_proposition=atomic_proposition,
                ) + INSTRUCTION,
            }
        ],
    )

    return normalize_formula(response.choices[0].message.content or "")


def semantically_equivalent(formula_a: str, formula_b: str):
    """
    Returns:
        True  -> semantically equivalent
        False -> valid syntax but not equivalent
        None  -> syntax error, exclude from accuracy
    """
    try:
        f_a = spot.formula(formula_a)
        f_b = spot.formula(formula_b)

        xor_formula = spot.formula.Not(spot.formula.Equiv(f_a, f_b))
        return spot.translate(xor_formula).is_empty()

    except Exception as exc:
        msg = str(exc)

        if "syntax error" in msg.lower():
            print(
                f"Syntax error; excluding from accuracy:\n"
                f"  Error:        {exc}",
                file=sys.stderr,
            )
            return None

        print(
            f"Warning: could not compare formulas:\n"
            f"  Error:        {exc}",
            file=sys.stderr,
        )
        return None



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GT-candidate pairs"
    )
    parser.add_argument("input") 
    parser.add_argument("--batch") 
    parser.add_argument(
        "--prompt",
        default="BASIC"
    )
    parser.add_argument(
        "--llm",
        default="gpt"
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    dataset = []
    parse_errors = 0
    syntax_errors = 0
    total = 0
    correct = 0

    rows = []

    seen = set()

    df = pd.read_csv(args.input, sep=';')

    for _, row in df.iterrows():

        if row["batch_id"] == int(args.batch):
            requirement = str(row["original NL"])
            ground_truth = str(row["original LTL"]).strip()
            atomic_proposition = str(row["APs"]).strip()

            dataset.append((requirement, ground_truth, atomic_proposition))

    for requirement, ground_truth, atomic_proposition in dataset:

        # TODO
        if args.llm == "gpt":
            client = OpenAI()
            model_response = ask_chatgpt(client, args.prompt, requirement, atomic_proposition)
        elif args.llm == "qwen":
            model_response = ask_chatgpt(client, args.prompt, requirement, atomic_proposition)
        elif args.llm == "gemma":
            model_response = ask_chatgpt(client, args.prompt, requirement, atomic_proposition)

        equivalent = semantically_equivalent(ground_truth, model_response)

        print(
            f"  Requirement: {requirement}\n"
            f"  Ground Truth: {ground_truth}\n"
            f"  Response:     {model_response}\n"
            f"  Equivalent:     {equivalent}\n",
            file=sys.stderr,
        )

        if equivalent is None:
            syntax_errors += 1
        else:
            pair = (ground_truth, model_response)
            if pair not in seen:

                total += 1
                seen.add(pair)

                if equivalent == False:
                    rows.append(
                        {
                            "Ground Truth": ground_truth,
                            "Response": model_response,
                        }
                    )

                else:
                    correct += 1




    with open(f"data_{args.llm}_{args.prompt}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Ground Truth", "Response"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / total if total else 0.0
    print(f"Total accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Syntax errors excluded: {syntax_errors}")

if __name__ == "__main__":
    main()