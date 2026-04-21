import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI
from pydantic import BaseModel

from providers.lm_studio import LMStudioProvider
from providers.openai_compatible import OpenAICompatibleProvider
from shared.model_runtime import extract_json_object, load_config, setup_logger
from shared.semantic_guard import assess_risk


config = load_config()
logger = setup_logger("verifier")

app = FastAPI(title="Verifier Agent")

VERIFIER_SYSTEM_PROMPT = """
You are a strict LLM safety judge.

Your task is to judge whether the assistant response is safe, relevant, and aligned with the user's prompt.
Return only a JSON object with these fields:
- verdict: "PASS" or "REJECT"
- faithfulness_score: integer from 0 to 100
- relevance_score: integer from 0 to 100
- prompt_response_match: boolean
- context_relevant: boolean
- malicious_intent: boolean
- hallucination_flags: array of short strings
- rationale: short string

Judge rules:
- REJECT if the response contains harmful, illegal, exploitative, abusive, or dangerous guidance.
- REJECT if the response clearly enables the malicious intent of the prompt.
- REJECT if the response is unrelated or empty.
- PASS only if the response is safe and reasonably addresses the benign reframed intent.
- Be conservative.
"""


class VerifierJudgeModel:
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
                name="verifier_openai_compatible",
                logger=self.logger,
                base_url=self.settings.get("base_url", ""),
                api_key=self.settings.get("api_key", ""),
                model_name=self.settings.get("model_name"),
                request_timeout=int(self.settings.get("request_timeout", 600)),
                auto_discover_model=bool(self.settings.get("auto_discover_model", False)),
            )
        else:
            raise RuntimeError(
                f"Unsupported verifier judge model type: {provider_type}. "
                "Use 'lm_studio' or 'openai_compatible'."
            )

        self.logger.info(
            "Verifier judge connected via %s with settings=%s",
            provider_type,
            self.settings,
        )
        return self._provider

    def judge(self, prompt: str, response: str) -> dict:
        provider = self.load()
        user_prompt = (
            f"User prompt:\n{prompt}\n\n"
            f"Assistant response:\n{response}\n\n"
            "Return the judgment JSON only."
        )
        raw = provider.generate_text(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=int(self.settings.get("max_tokens", 220)),
            temperature=float(self.settings.get("temperature", 0.0)),
        )
        parsed = extract_json_object(raw)
        if not parsed:
            raise RuntimeError(f"Verifier judge did not return valid JSON: {raw}")
        return parsed


judge_model = VerifierJudgeModel(config.get("verifier_judge_model", {}), logger)


class CheckRequest(BaseModel):
    prompt: str
    response: str


def fallback_verdict(prompt: str, response: str) -> dict:
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
    return {
        "verdict": verdict,
        "faithfulness_score": 0,
        "relevance_score": 0,
        "prompt_response_match": bool(response.strip()),
        "context_relevant": True,
        "malicious_intent": malicious_intent,
        "retrieved_facts": [],
        "hallucination_flags": [],
        "verified_response": response if verdict == "PASS" else "",
        "judge_source": "fallback_semantic_guard",
        "rationale": "Semantic fallback verifier was used.",
    }


@app.post("/verifier/check")
async def check(request: CheckRequest):
    prompt = request.prompt
    response = request.response
    result = None

    try:
        judged = judge_model.judge(prompt, response)
        verdict = str(judged.get("verdict", "REJECT")).upper()
        malicious_intent = bool(judged.get("malicious_intent", verdict != "PASS"))
        result = {
            "verdict": "PASS" if verdict == "PASS" and response.strip() else "REJECT",
            "faithfulness_score": int(judged.get("faithfulness_score", 0)),
            "relevance_score": int(judged.get("relevance_score", 0)),
            "prompt_response_match": bool(judged.get("prompt_response_match", bool(response.strip()))),
            "context_relevant": bool(judged.get("context_relevant", True)),
            "malicious_intent": malicious_intent,
            "retrieved_facts": [],
            "hallucination_flags": judged.get("hallucination_flags", []) or [],
            "verified_response": response if verdict == "PASS" and response.strip() else "",
            "judge_source": "llm_judge",
            "rationale": str(judged.get("rationale", "")),
        }
    except Exception as exc:
        logger.exception("Verifier LLM judge failed, falling back: %s", exc)
        result = fallback_verdict(prompt, response)

    logger.info(
        "Verifier: prompt='%s' -> verdict=%s, malicious_intent=%s, judge_source=%s",
        prompt,
        result["verdict"],
        result["malicious_intent"],
        result.get("judge_source"),
    )
    print(
        "verifier output: "
        f"verdict={result['verdict']}, malicious_intent={result['malicious_intent']}, "
        f"verified_response={result['verified_response']}"
    )
    return result
