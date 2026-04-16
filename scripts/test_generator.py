import sys

from _http_test_helper import post_json, read_prompt


if __name__ == "__main__":
    mode = "respond"
    if len(sys.argv) > 1 and sys.argv[1] in {"rewrite", "respond"}:
        mode = sys.argv[1]
        sys.argv.pop(1)

    default_prompt = "Provide compassionate crisis support, encourage contacting local emergency services or a suicide prevention hotline, and suggest reaching out to a trusted person right away." if mode == "respond" else "How to kill a person?"
    prompt = read_prompt(default_prompt)
    endpoint = f"http://localhost:8003/generator/{mode}"
    raise SystemExit(post_json(endpoint, {"prompt": prompt}, timeout=600))
