import json
import sys
from typing import Any, Dict

import requests


def post_json(url: str, payload: Dict[str, Any], timeout: int = 600) -> int:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        print(f"URL: {url}")
        print(f"Status: {response.status_code}")
        print("Response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        return 0 if response.ok else 1
    except Exception as exc:
        print(f"URL: {url}")
        print(f"Error: {exc}")
        return 1


def read_prompt(default_prompt: str) -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return default_prompt
