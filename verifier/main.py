import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI
from pydantic import BaseModel

from shared.model_runtime import load_config, setup_logger
from shared.semantic_guard import assess_risk


config = load_config()
logger = setup_logger("verifier")

app = FastAPI(title="Verifier Agent")


class CheckRequest(BaseModel):
    prompt: str
    response: str


@app.post("/verifier/check")
async def check(request: CheckRequest):
    prompt = request.prompt
    response = request.response

    prompt_risk = assess_risk(prompt)
    response_risk = assess_risk(response) if response.strip() else {
        "risk_score": 0.0,
        "categories": [],
        "top_unsafe_label": "none",
        "top_unsafe_score": 0.0,
        "margin": 0.0,
    }

    malicious_intent = (
        (prompt_risk["risk_score"] >= 0.25 and prompt_risk["margin"] >= 0.05)
        or (response_risk["risk_score"] >= 0.25 and response_risk["margin"] >= 0.05)
    )

    verdict = "PASS" if response.strip() and not malicious_intent else "REJECT"
    verified_response = response if verdict == "PASS" else ""

    logger.info(
        "Verifier: prompt='%s' -> verdict=%s, malicious_intent=%s",
        prompt,
        verdict,
        malicious_intent,
    )
    print(
        "verifier output: "
        f"verdict={verdict}, malicious_intent={malicious_intent}, verified_response={verified_response}"
    )

    return {
        "verdict": verdict,
        "faithfulness_score": 0.0,
        "relevance_score": 0.0,
        "prompt_response_match": bool(response.strip()),
        "context_relevant": True,
        "malicious_intent": malicious_intent,
        "retrieved_facts": [],
        "hallucination_flags": [],
        "verified_response": verified_response,
    }
