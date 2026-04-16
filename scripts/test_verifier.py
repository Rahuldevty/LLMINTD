import json
import sys

from _http_test_helper import post_json


DEFAULT_PROMPT = "What is Artificial Intelligence?"
DEFAULT_RESPONSE = (
    "Artificial intelligence is the field of building computer systems that can perform "
    "tasks like learning, reasoning, and pattern recognition."
)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    response = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RESPONSE
    raise SystemExit(
        post_json(
            "http://localhost:8004/verifier/check",
            {"prompt": prompt, "response": response},
            timeout=120,
        )
    )
