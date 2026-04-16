import os
import sys

import requests
from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.model_runtime import load_config, setup_logger
from shared.semantic_guard import assess_risk
from shared.safety_text import build_safe_rewrite, sanitize_generated_text


config = load_config()
logger = setup_logger("researcher")
app = FastAPI(title="Researcher Agent")


class RewriteRequest(BaseModel):
    prompt: str


@app.post("/researcher/rewrite")
async def rewrite(request: RewriteRequest):
    prompt = request.prompt
    generator_url = f"http://localhost:{config['ports']['generator']}/generator/rewrite"
    fallback_rewrite = build_safe_rewrite(prompt)
    rewritten = fallback_rewrite

    try:
        response = requests.post(generator_url, json={"prompt": prompt}, timeout=600)
        response.raise_for_status()
        candidate = response.json().get("rewritten_prompt", "")
        candidate = sanitize_generated_text(candidate, fallback_rewrite)
        if assess_risk(candidate)["decision"] == "ALLOW":
            rewritten = candidate
    except Exception as exc:
        logger.exception("Researcher proxy failed: %s", exc)

    logger.info("Researcher proxy: prompt='%s' -> rewritten='%s'", prompt, rewritten)
    print(f"researcher output: rewritten_prompt={rewritten}")
    return {"rewritten_prompt": rewritten}
