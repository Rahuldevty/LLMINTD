#!/usr/bin/env python3
"""Interactive terminal chatbot with per-module trace output."""

import json

import requests

PLANNER_URL = "http://localhost:8001/planner/analyze"
RESEARCHER_URL = "http://localhost:8002/researcher/rewrite"
GENERATOR_URL = "http://localhost:8003/generator/respond"
VERIFIER_URL = "http://localhost:8004/verifier/check"
API_URL = "http://localhost:8000/guardrail_graph"


def post_json(url, payload, timeout=600):
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def print_section(title, data):
    print(f"\n[{title}]")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def trace_pipeline(prompt):
    trace = {"prompt": prompt}
    planner_result = post_json(PLANNER_URL, {"prompt": prompt})
    trace["planner"] = planner_result
    print_section("Planner", planner_result)

    decision = planner_result.get("decision", "BLOCK")
    if decision == "BLOCK":
        print("\n[Researcher]")
        print("Skipped because planner returned BLOCK.")
        print("\n[Generator]")
        print("Skipped because planner returned BLOCK.")
        print("\n[Verifier]")
        print("Skipped because planner returned BLOCK.")
        return trace

    final_prompt = prompt
    if decision == "REWRITE":
        researcher_result = post_json(RESEARCHER_URL, {"prompt": prompt})
        trace["researcher"] = researcher_result
        final_prompt = researcher_result.get("rewritten_prompt", prompt)
        print_section("Researcher", researcher_result)
    else:
        print("\n[Researcher]")
        print("Skipped because planner returned ALLOW.")

    generator_result = post_json(GENERATOR_URL, {"prompt": final_prompt})
    trace["generator"] = generator_result
    print_section("Generator", generator_result)

    verifier_payload = {
        "prompt": final_prompt,
        "response": generator_result.get("response", ""),
    }
    verifier_result = post_json(VERIFIER_URL, verifier_payload)
    trace["verifier"] = verifier_result
    print_section("Verifier", verifier_result)
    return trace

def main():
    print("LLMINTD Chatbot - Type your message or 'quit' to exit.")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            print(f"\nYou: {user_input}")
            print("\nTracing pipeline...")
            trace_pipeline(user_input)

            final_result = post_json(API_URL, {"prompt": user_input})
            print_section("Final API Result", final_result)
            print(f"\nBot: {final_result.get('response', 'No response generated.')}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
