import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.model_runtime import setup_logger
from shared.semantic_guard import assess_risk


logger = setup_logger("planner")
app = FastAPI(title="Planner Agent")


class AnalyzeRequest(BaseModel):
    prompt: str


@app.post("/planner/analyze")
async def analyze(request: AnalyzeRequest):
    prompt = request.prompt
    result = assess_risk(prompt)

    reason = (
        f"semantic-risk top_unsafe={result['top_unsafe_label']}:{result['top_unsafe_score']:.2f}, "
        f"top_benign={result['top_benign_label']}:{result['top_benign_score']:.2f}, "
        f"margin={result['margin']:.2f}"
    )

    logger.info(
        "Planner: prompt='%s' -> decision=%s, risk_score=%.2f, categories=%s, reason='%s'",
        prompt,
        result["decision"],
        result["risk_score"],
        result["categories"],
        reason,
    )
    print(
        f"planner output: decision={result['decision']}, risk_score={result['risk_score']:.2f}, "
        f"categories={result['categories']}, reason={reason}"
    )
    return {
        "decision": result["decision"],
        "risk_score": result["risk_score"],
        "categories": result["categories"],
        "reason": reason,
    }
