from __future__ import annotations

import logging
from typing import Any, Dict

from providers.base import GeneratorProvider
from providers.custom_http import CustomHTTPProvider
from providers.huggingface_local import HuggingFaceLocalProvider
from providers.lm_studio import LMStudioProvider
from providers.ollama import OllamaProvider
from providers.openai_compatible import OpenAICompatibleProvider


def _build_provider_settings(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    inference = config.get("inference", {})
    provider = dict(config.get("generator_provider", {}))

    if provider:
        provider.setdefault("model_name", model_name)
        provider.setdefault("request_timeout", int(inference.get("timeout_seconds", 600)))
        provider.setdefault("api_key", inference.get("api_key", ""))
        provider.setdefault("base_url", inference.get("base_url", ""))
        return provider

    return {
        "type": "openai_compatible" if inference.get("backend") == "openai_compatible" else "huggingface_local",
        "model_name": inference.get("model_name") or model_name,
        "request_timeout": int(inference.get("timeout_seconds", 600)),
        "api_key": inference.get("api_key", ""),
        "base_url": inference.get("base_url", ""),
        "auto_discover_model": bool(inference.get("auto_discover_model", True)),
    }


SUPPORTED_GENERATOR_PROVIDER_TYPES = (
    "lm_studio",
    "openai_compatible",
    "ollama",
    "custom_http",
    "huggingface_local",
)


def create_generator_provider_from_settings(
    settings: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> GeneratorProvider:
    model_name = config["models"]["generator"]
    provider_type = settings.get("type", "openai_compatible")

    if provider_type == "lm_studio":
        return LMStudioProvider(
            logger=logger,
            base_url=settings.get("base_url", ""),
            api_key=settings.get("api_key", "lm-studio"),
            model_name=settings.get("model_name"),
            request_timeout=int(settings.get("request_timeout", 600)),
            auto_discover_model=bool(settings.get("auto_discover_model", True)),
        )
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            name="openai_compatible",
            logger=logger,
            base_url=settings.get("base_url", ""),
            api_key=settings.get("api_key", ""),
            model_name=settings.get("model_name"),
            request_timeout=int(settings.get("request_timeout", 600)),
            auto_discover_model=bool(settings.get("auto_discover_model", False)),
        )
    if provider_type == "ollama":
        return OllamaProvider(
            logger=logger,
            base_url=settings.get("base_url", ""),
            model_name=settings.get("model_name", ""),
            request_timeout=int(settings.get("request_timeout", 600)),
        )
    if provider_type == "custom_http":
        return CustomHTTPProvider(
            logger=logger,
            endpoint=settings.get("endpoint", ""),
            model_name=settings.get("model_name", ""),
            auth_token=settings.get("auth_token", ""),
            request_timeout=int(settings.get("request_timeout", 600)),
            response_field=settings.get("response_field", "response"),
        )
    if provider_type == "huggingface_local":
        return HuggingFaceLocalProvider(
            logger=logger,
            model_name=settings.get("model_name", model_name),
        )

    raise ValueError(f"Unsupported generator provider type: {provider_type}")


def create_generator_provider(config: Dict[str, Any], logger: logging.Logger) -> GeneratorProvider:
    model_name = config["models"]["generator"]
    settings = _build_provider_settings(config, model_name)
    return create_generator_provider_from_settings(settings, config, logger)
