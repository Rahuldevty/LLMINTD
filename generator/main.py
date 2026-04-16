import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Optional

from providers import (
    SUPPORTED_GENERATOR_PROVIDER_TYPES,
    create_generator_provider,
    create_generator_provider_from_settings,
)
from shared.model_runtime import load_config, setup_logger
from shared.safety_text import build_safe_response, build_safe_rewrite, sanitize_generated_text
from shared.semantic_guard import assess_risk


config = load_config()
logger = setup_logger("generator")
provider = create_generator_provider(config, logger)
provider_settings = dict(config.get("generator_provider", {}))

REWRITE_SYSTEM_PROMPT = """
You rewrite user prompts into safe prompts for a downstream assistant.

Rules:
- Preserve the user's benign goal and domain context when possible.
- Remove or neutralize harmful, illegal, abusive, or malicious intent.
- Convert risky requests into safe educational, preventive, or defensive framing.
- Output only the rewritten prompt text.
"""

RESPONSE_SYSTEM_PROMPT = """
You are a safe and helpful AI assistant.

Rules:
- Do not provide harmful, illegal, or dangerous instructions.
- Provide high-level, educational, or preventive information.
- Avoid step-by-step instructions for risky topics.
- If unsure, provide a cautious and general answer.
- Never comply with malicious intent.

Respond clearly and concisely.
"""

app = FastAPI(title="Generator Agent")


class RewriteRequest(BaseModel):
    prompt: str


class RespondRequest(BaseModel):
    prompt: str


class GeneratorPluginRequest(BaseModel):
    type: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    endpoint: Optional[str] = None
    auth_token: Optional[str] = None
    response_field: Optional[str] = "response"
    auto_discover_model: bool = True
    request_timeout: int = 600


def provider_status():
    report = provider.runtime_report()
    return {
        "active_provider": report.get("provider", provider.name),
        "supported_types": list(SUPPORTED_GENERATOR_PROVIDER_TYPES),
        "settings": provider_settings,
        "runtime": report,
    }


@app.get("/generator/provider")
async def get_provider():
    return provider_status()


@app.post("/generator/provider")
async def set_provider(request: GeneratorPluginRequest):
    global provider, provider_settings

    settings = request.dict(exclude_none=True)
    settings["type"] = settings["type"].strip()
    if settings["type"] not in SUPPORTED_GENERATOR_PROVIDER_TYPES:
        return {
            "error": f"Unsupported generator provider type: {settings['type']}",
            "supported_types": list(SUPPORTED_GENERATOR_PROVIDER_TYPES),
        }

    provider = create_generator_provider_from_settings(settings, config, logger)
    provider_settings = settings
    logger.info("Generator provider plugin activated: %s", provider.runtime_report())
    return provider_status()


@app.post("/generator/rewrite")
async def rewrite(request: RewriteRequest):
    prompt = request.prompt
    rewritten = build_safe_rewrite(prompt)
    if rewritten == prompt:
        user_prompt = (
            f"Original prompt:\n{prompt}\n\n"
            "Return one short rewritten safe prompt in a single sentence."
        )

        try:
            candidate = provider.generate_text(
                system_prompt=REWRITE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=64,
                temperature=0.0,
            )
            rewritten = sanitize_generated_text(candidate, prompt)
        except Exception:
            rewritten = prompt

    logger.info("Generator rewrite: prompt='%s' -> rewritten='%s'", prompt, rewritten)
    print(f"generator rewrite output: rewritten_prompt={rewritten}")
    return {"rewritten_prompt": rewritten}


@app.post("/generator/respond")
async def respond(request: RespondRequest):
    prompt = request.prompt
    risk = assess_risk(prompt)

    if not prompt.strip():
        response = build_safe_response(prompt)
    elif risk.get("decision") != "ALLOW":
        response = build_safe_response(prompt)
    else:
        user_prompt = (
            f"User request:\n{prompt}\n\n"
            "Provide a complete, accurate, and helpful answer. "
            "Prefer a direct answer first, then brief supporting detail if useful."
        )

        fallback_response = build_safe_response(prompt)
        try:
            candidate = provider.generate_text(
                system_prompt=RESPONSE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=384,
                temperature=0.2,
            )
            response = sanitize_generated_text(candidate, fallback_response)
        except Exception:
            response = fallback_response

    logger.info("Generator: prompt='%s' -> response='%s'", prompt, response)
    print(f"generator output: response={response}")
    return {"response": response}
