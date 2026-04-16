from __future__ import annotations

from typing import Any, Dict

import requests

from providers.base import GeneratorProvider


class OllamaProvider(GeneratorProvider):
    def __init__(
        self,
        *,
        logger,
        base_url: str,
        model_name: str,
        request_timeout: int,
    ) -> None:
        super().__init__(name="ollama", logger=logger)
        self.base_url = (base_url or "").rstrip("/")
        self.model_name = (model_name or "").strip()
        self.request_timeout = int(request_timeout)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("Provider 'ollama' has no base_url configured")
        if not self.model_name:
            raise RuntimeError("Provider 'ollama' has no model_name configured")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    def runtime_report(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "request_timeout": self.request_timeout,
        }
