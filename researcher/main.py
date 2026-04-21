import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.model_runtime import load_config, setup_logger
from shared.semantic_guard import assess_risk
from shared.safety_text import build_safe_rewrite, sanitize_generated_text
from providers.lm_studio import LMStudioProvider
from providers.openai_compatible import OpenAICompatibleProvider


config = load_config()
logger = setup_logger("researcher")
app = FastAPI(title="Researcher Agent")

REWRITE_SYSTEM_PROMPT = """
You rewrite user prompts into safe prompts for a downstream assistant.

Rules:
- Preserve the user's benign goal and domain context when possible.
- Remove or neutralize harmful, illegal, abusive, or malicious intent.
- Convert risky requests into safe educational, preventive, or defensive framing.
- Output only the rewritten prompt text.
"""


class ResearcherRewriteModel:
    def __init__(self, settings, logger):
        self.settings = settings or {}
        self.logger = logger
        self._provider = None

    def load(self):
        if self._provider is not None:
            return self._provider

        provider_type = (self.settings.get("type") or "lm_studio").strip()
        if provider_type == "lm_studio":
            self._provider = LMStudioProvider(
                logger=self.logger,
                base_url=self.settings.get("base_url", "http://127.0.0.1:1234/v1"),
                api_key=self.settings.get("api_key", "lm-studio"),
                model_name=self.settings.get("model_name"),
                request_timeout=int(self.settings.get("request_timeout", 600)),
                auto_discover_model=bool(self.settings.get("auto_discover_model", True)),
            )
        elif provider_type == "openai_compatible":
            self._provider = OpenAICompatibleProvider(
                name="researcher_openai_compatible",
                logger=self.logger,
                base_url=self.settings.get("base_url", ""),
                api_key=self.settings.get("api_key", ""),
                model_name=self.settings.get("model_name"),
                request_timeout=int(self.settings.get("request_timeout", 600)),
                auto_discover_model=bool(self.settings.get("auto_discover_model", False)),
            )
        else:
            raise RuntimeError(
                f"Unsupported researcher rewrite model type: {provider_type}. "
                "Use 'lm_studio' or 'openai_compatible'."
            )

        self.logger.info(
            "Researcher rewrite model connected via %s with settings=%s",
            provider_type,
            self.settings,
        )
        return self._provider

    def rewrite(self, prompt: str, fallback_rewrite: str, categories: list[str] | None = None) -> str:
        provider = self.load()
        category_text = ", ".join(categories or []) or "none"
        user_prompt = (
            "Please safely rewrite this prompt for a downstream assistant.\n\n"
            f"Original prompt:\n{prompt}\n\n"
            f"Risk categories from planner: {category_text}\n\n"
            "Return only the safely rewritten prompt in a single sentence."
        )

        candidate = provider.generate_text(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=float(self.settings.get("temperature", 0.0)),
            max_new_tokens=int(self.settings.get("max_tokens", 96)),
        )
        return sanitize_generated_text(candidate, fallback_rewrite)


rewrite_model = ResearcherRewriteModel(config.get("researcher_rewrite_model", {}), logger)


class RewriteRequest(BaseModel):
    prompt: str
    planner_decision: str | None = None
    categories: list[str] | None = None


@app.post("/researcher/rewrite")
async def rewrite(request: RewriteRequest):
    prompt = request.prompt
    fallback_rewrite = build_safe_rewrite(prompt)
    rewritten = fallback_rewrite
    rewrite_source = "fallback"

    try:
        candidate = rewrite_model.rewrite(prompt, fallback_rewrite, request.categories)
        if assess_risk(candidate)["decision"] == "ALLOW":
            rewritten = candidate
            rewrite_source = "abliterated_llama"
        else:
            logger.warning(
                "Researcher model candidate was not ALLOW and fallback was used. prompt='%s' candidate='%s'",
                prompt,
                candidate,
            )
    except Exception as exc:
        logger.exception("Researcher model rewrite failed: %s", exc)

    logger.info(
        "Researcher rewrite: prompt='%s' planner_decision='%s' categories=%s source=%s -> rewritten='%s'",
        prompt,
        request.planner_decision,
        request.categories,
        rewrite_source,
        rewritten,
    )
    print(f"researcher output: rewritten_prompt={rewritten}")
    return {"rewritten_prompt": rewritten, "rewrite_source": rewrite_source}
