import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class GeneratorProvider(ABC):
    def __init__(self, name: str, logger: logging.Logger) -> None:
        self.name = name
        self.logger = logger

    @abstractmethod
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        raise NotImplementedError

    def runtime_report(self) -> Dict[str, Any]:
        return {"provider": self.name}
