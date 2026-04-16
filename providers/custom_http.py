from __future__ import annotations

from typing import Any, Dict, Iterable

import requests

from providers.base import GeneratorProvider


def _extract_path(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(path)
    return current


class CustomHTTPProvider(GeneratorProvider):
    def __init__(
        self,
        *,
        logger,
        endpoint: str,
        model_name: str,
        auth_token: str,
        request_timeout: int,
        response_field: str = "response",
    ) -> None:
        super().__init__(name="custom_http", logger=logger)
        self.endpoint = endpoint
        self.model_name = model_name
        self.auth_token = auth_token
        self.request_timeout = int(request_timeout)
        self.response_field = response_field

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        payload = {
            "model": self.model_name,
            "system_prompt": system_prompt.strip(),
            "user_prompt": user_prompt.strip(),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        value = _extract_path(data, self.response_field)
        if not isinstance(value, str):
            raise TypeError(f"Custom HTTP response field '{self.response_field}' must be a string")
        return value.strip()

    def runtime_report(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "response_field": self.response_field,
            "request_timeout": self.request_timeout,
        }
