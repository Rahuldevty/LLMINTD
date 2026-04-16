from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from providers.base import GeneratorProvider


class OpenAICompatibleProvider(GeneratorProvider):
    def __init__(
        self,
        *,
        name: str,
        logger,
        base_url: str,
        api_key: str,
        model_name: Optional[str],
        request_timeout: int,
        auto_discover_model: bool = False,
    ) -> None:
        super().__init__(name=name, logger=logger)
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model_name = (model_name or "").strip()
        self.request_timeout = int(request_timeout)
        self.auto_discover_model = auto_discover_model
        self._resolved_model_name: Optional[str] = None
        self._available_models: Optional[list[str]] = None

    def _normalize_base_url(self) -> str:
        if not self.base_url:
            return ""
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def list_models(self) -> list[str]:
        base_url = self._normalize_base_url()
        if not base_url:
            return []

        response = requests.get(
            f"{base_url}/v1/models",
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
        self._available_models = models
        return models

    def resolve_model_name(self) -> str:
        if self._resolved_model_name:
            return self._resolved_model_name

        preferred = self.model_name
        if not self.auto_discover_model:
            self._resolved_model_name = preferred
            return self._resolved_model_name

        try:
            available_models = self.list_models()
        except Exception as exc:
            self.logger.warning(
                "Model discovery failed for provider '%s' and preferred model '%s': %s",
                self.name,
                preferred,
                exc,
            )
            self._resolved_model_name = preferred
            return self._resolved_model_name

        if preferred and preferred in available_models:
            self._resolved_model_name = preferred
        elif available_models:
            self._resolved_model_name = available_models[0]
            self.logger.info(
                "Provider '%s' selected discovered model '%s' instead of configured '%s'.",
                self.name,
                self._resolved_model_name,
                preferred,
            )
        else:
            self._resolved_model_name = preferred

        return self._resolved_model_name

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        base_url = self._normalize_base_url()
        if not base_url:
            raise RuntimeError(f"Provider '{self.name}' has no base_url configured")

        model_name = self.resolve_model_name()
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": False,
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def runtime_report(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "resolved_model_name": self._resolved_model_name,
            "available_models": self._available_models,
            "request_timeout": self.request_timeout,
            "auto_discover_model": self.auto_discover_model,
        }
