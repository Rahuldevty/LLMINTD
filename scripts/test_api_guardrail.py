from _http_test_helper import post_json, read_prompt


if __name__ == "__main__":
    prompt = read_prompt("What is Artificial Intelligence?")
    raise SystemExit(
        post_json(
            "http://localhost:8000/guardrail_graph",
            {"prompt": prompt},
            timeout=600,
        )
    )
