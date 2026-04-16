from __future__ import annotations

from typing import Any, Dict, Optional

from shared.model_runtime import LocalTransformersRuntime
from providers.base import GeneratorProvider


class HuggingFaceLocalProvider(GeneratorProvider):
    def __init__(self, *, logger, model_name: str) -> None:
        super().__init__(name="huggingface_local", logger=logger)
        self.model_name = model_name
        self.runtime = LocalTransformersRuntime(model_name, logger)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        return self.runtime.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def runtime_report(self) -> Dict[str, Any]:
        report = {"provider": self.name, "model_name": self.model_name}
        report.update(self.runtime.runtime_report())
        return report
