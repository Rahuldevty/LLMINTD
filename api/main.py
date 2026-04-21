from pathlib import Path
from typing import Optional
import sys

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import requests
import yaml
import logging
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from requests import RequestException

try:
    from auth_store import AuthStore
except ImportError:  # pragma: no cover
    from api.auth_store import AuthStore

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.dpo_store import (
    load_generation,
    save_generation,
    save_planner_category_feedback,
    save_preference,
)
from shared.safety_text import build_safe_response

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Setup logging
log_file = config['logging']['file']
if not os.path.isabs(log_file):
    log_file = os.path.join(os.path.dirname(__file__), '..', log_file)
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(level=config['logging']['level'], filename=log_file, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Guardrail API")
WEBUI_DIR = Path(__file__).resolve().parent / "webui"
AUTH_CONFIG = config.get("auth", {})
UI_CONFIG = config.get("ui", {})
SESSION_COOKIE_NAME = AUTH_CONFIG.get("session_cookie_name", "llmintd_session")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID") or AUTH_CONFIG.get("google_client_id")
OPEN_WEBUI_URL = os.getenv("OPEN_WEBUI_URL") or UI_CONFIG.get("open_webui_url")
auth_db_path = AUTH_CONFIG.get("database", "logs/app_auth.db")
if not os.path.isabs(auth_db_path):
    auth_db_path = os.path.join(BASE_DIR, auth_db_path)
auth_store = AuthStore(auth_db_path)

if WEBUI_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(WEBUI_DIR)), name="ui-static")

class GuardrailRequest(BaseModel):
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


class SignupRequest(BaseModel):
    full_name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleLoginRequest(BaseModel):
    credential: str


class DPOPreferenceRequest(BaseModel):
    generation_id: str
    preference: str
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    reason: Optional[str] = None


class PlannerCategoryFeedbackRequest(BaseModel):
    prompt: str
    planner_decision: str
    planner_categories: list[str] = []
    is_correct: bool
    corrected_categories: list[str] = []


def build_session_response(response: Response, user, token: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=14 * 24 * 60 * 60,
    )
    return {"user": user.to_public_dict()}


def current_user_from_request(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return auth_store.get_user_by_session(token)


def verify_google_credential(credential: str):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google login is not configured")
    try:
        response = requests.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": credential},
            timeout=15,
        )
        response.raise_for_status()
    except RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Unable to verify Google credential: {exc}") from exc

    data = response.json()
    if data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Google token audience mismatch")
    if data.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status_code=401, detail="Invalid Google token issuer")
    return data

def call_agent(url, data):
    try:
        response = requests.post(url, json=data, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error calling {url}: {e}")
        return None


def generator_provider_url():
    return f"http://localhost:{config['ports']['generator']}/generator/provider"


def collect_generator_provider_status():
    try:
        response = requests.get(generator_provider_url(), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def store_dpo_generation(
    *,
    original_prompt: str,
    final_prompt: str,
    response: str,
    trace: dict,
) -> dict:
    provider_status = collect_generator_provider_status()
    generation = save_generation(
        {
            "prompt": final_prompt,
            "original_prompt": original_prompt,
            "response": response,
            "final_status": trace.get("final_status"),
            "planner_decision": (trace.get("planner") or {}).get("decision"),
            "verifier_verdict": (trace.get("verifier") or {}).get("verdict"),
            "provider": provider_status,
        }
    )
    trace["dpo_generation"] = {
        "id": generation["id"],
        "prompt": final_prompt,
        "original_prompt": original_prompt,
        "response": response,
    }
    return generation


def run_sequential_pipeline(prompt: str):
    planner_url = f"http://localhost:{config['ports']['planner']}/planner/analyze"
    researcher_url = f"http://localhost:{config['ports']['researcher']}/researcher/rewrite"
    generator_url = f"http://localhost:{config['ports']['generator']}/generator/respond"
    verifier_url = f"http://localhost:{config['ports']['verifier']}/verifier/check"

    trace = {
        "prompt": prompt,
        "planner": None,
        "researcher": None,
        "generator": None,
        "verifier": None,
        "final_response": "",
        "final_status": "FAILED",
    }

    planner_result = call_agent(planner_url, {"prompt": prompt})
    trace["planner"] = planner_result
    if planner_result is None:
        trace["final_response"] = "I'm sorry, but I cannot process this request because the planner is unavailable."
        return trace

    decision = planner_result.get("decision", "BLOCK")
    logger.info("Trace planner decision: %s", decision)
    if decision == "BLOCK":
        trace["final_status"] = "BLOCK"
        trace["final_response"] = "I'm sorry, but I cannot assist with that request as it violates safety guidelines."
        return trace

    final_prompt = prompt
    if decision == "REWRITE":
        researcher_result = call_agent(
            researcher_url,
            {
                "prompt": prompt,
                "planner_decision": decision,
                "categories": planner_result.get("categories", []),
            },
        )
        trace["researcher"] = researcher_result
        if researcher_result is None:
            trace["final_response"] = "I'm sorry, but I cannot process this request because the researcher is unavailable."
            return trace
        final_prompt = researcher_result.get("rewritten_prompt", prompt)

    generator_result = call_agent(generator_url, {"prompt": final_prompt})
    trace["generator"] = generator_result
    if generator_result is None:
        trace["final_response"] = (
            "The generator is unavailable. If you are using gpt-oss-20b through LM Studio, "
            "start the Local Server in LM Studio and make sure it is serving on http://127.0.0.1:1234/v1."
        )
        return trace

    response = generator_result.get("response", "")
    verifier_result = call_agent(verifier_url, {"prompt": final_prompt, "response": response})
    trace["verifier"] = verifier_result
    if verifier_result is None:
        trace["final_response"] = "I'm sorry, I cannot provide a verified response at this time."
        return trace

    if verifier_result.get("verdict") == "PASS":
        trace["final_status"] = "PASS"
        trace["final_response"] = response
        store_dpo_generation(
            original_prompt=prompt,
            final_prompt=final_prompt,
            response=response,
            trace=trace,
        )
    else:
        trace["final_status"] = verifier_result.get("verdict", "REJECT")
        trace["final_response"] = "I'm sorry, I cannot provide a verified response at this time."

    return trace

@app.post("/guardrail")
async def guardrail(request: GuardrailRequest):
    prompt = request.prompt
    logger.info(f"New request: prompt='{prompt}'")
    trace = run_sequential_pipeline(prompt)
    logger.info("Final response status: %s", trace["final_status"])
    print(f"guardrail final response ({trace['final_status']}): {trace['final_response']}")
    return {
        "response": trace["final_response"],
        "generation_id": (trace.get("dpo_generation") or {}).get("id"),
        "final_status": trace["final_status"],
    }


@app.post("/guardrail_trace")
async def guardrail_trace(request: GuardrailRequest):
    prompt = request.prompt
    logger.info("Guardrail_trace: prompt='%s'", prompt)
    return run_sequential_pipeline(prompt)


@app.get("/generator/provider")
async def get_generator_provider():
    try:
        response = requests.get(generator_provider_url(), timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Generator service is unavailable: {exc}") from exc


@app.post("/generator/provider")
async def set_generator_provider(request: GeneratorPluginRequest):
    try:
        response = requests.post(generator_provider_url(), json=request.dict(), timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Generator service is unavailable: {exc}") from exc
    if data.get("error"):
        raise HTTPException(status_code=400, detail=data["error"])
    return data


@app.post("/dpo/preference")
async def record_dpo_preference(request: DPOPreferenceRequest, http_request: Request):
    generation = load_generation(request.generation_id)
    if generation is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    preference = request.preference.lower().strip()
    if preference not in {"chosen", "rejected"}:
        raise HTTPException(status_code=400, detail="Preference must be 'chosen' or 'rejected'")

    generated_response = generation.get("response", "")
    baseline_response = build_safe_response(generation.get("prompt", ""))
    chosen = request.chosen
    rejected = request.rejected

    if not chosen or not rejected:
        if preference == "chosen":
            chosen = generated_response
            rejected = baseline_response
        else:
            chosen = baseline_response
            rejected = generated_response

    user = current_user_from_request(http_request)
    record = save_preference(
        {
            "generation_id": request.generation_id,
            "prompt": generation.get("prompt", ""),
            "original_prompt": generation.get("original_prompt", ""),
            "chosen": chosen,
            "rejected": rejected,
            "preference": preference,
            "reason": request.reason,
            "source": "chat_feedback",
            "user_id": user.id if user else None,
            "provider": generation.get("provider"),
        }
    )
    return {"ok": True, "preference_id": record["id"]}


@app.post("/planner/category_feedback")
async def record_planner_category_feedback(
    request: PlannerCategoryFeedbackRequest,
    http_request: Request,
):
    normalized_corrected = [
        category.strip()
        for category in request.corrected_categories
        if category and category.strip()
    ]
    if not request.is_correct and not normalized_corrected:
        raise HTTPException(status_code=400, detail="Provide at least one corrected category")

    user = current_user_from_request(http_request)
    record = save_planner_category_feedback(
        {
            "prompt": request.prompt,
            "planner_decision": request.planner_decision,
            "planner_categories": request.planner_categories,
            "is_correct": request.is_correct,
            "corrected_categories": normalized_corrected,
            "user_id": user.id if user else None,
            "source": "test_llm_interface",
        }
    )
    return {"ok": True, "feedback_id": record["id"]}


# LangGraph orchestration endpoint
import sys
sys.path.append(os.path.dirname(__file__))
from langgraph_pipeline import run_pipeline

@app.post("/guardrail_graph")
async def guardrail_graph(request: GuardrailRequest):
    prompt = request.prompt
    logger.info(f"Guardrail_graph: prompt='{prompt}'")

    result = run_pipeline(prompt)
    if not result.get('verification'):
        return {"response": "I'm sorry, I cannot provide a verified response at this time."}

    # If verifier gave PASS, return final response.
    if result['verification'].get('verdict') == 'PASS':
        response = result.get('final_response', '')
        output = {"response": response, "verification": result['verification'], "retry_count": result.get('retry_count', 0)}
        logger.info(f"Guardrail_graph final response: {response}")
        print(f"guardrail_graph final response (PASS): {response}")
        return output
    
    # On retry/reject fallback
    response = "I'm sorry, I cannot provide a verified response at this time."
    output = {"response": response, "verification": result['verification'], "retry_count": result.get('retry_count', 0)}
    logger.info(f"Guardrail_graph final response: {response}")
    print(f"guardrail_graph final response (fail): {response}")
    return output


@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    index_path = WEBUI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI files not found.")
    return index_path.read_text(encoding="utf-8")


@app.get("/ui/test", response_class=HTMLResponse)
async def web_ui_test():
    page_path = WEBUI_DIR / "tester.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Tester UI files not found.")
    return page_path.read_text(encoding="utf-8")


@app.get("/ui/login", response_class=HTMLResponse)
async def web_ui_login():
    page_path = WEBUI_DIR / "login.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Login UI files not found.")
    return page_path.read_text(encoding="utf-8")


@app.get("/ui/choice", response_class=HTMLResponse)
async def web_ui_choice():
    page_path = WEBUI_DIR / "choice.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Choice UI files not found.")
    return page_path.read_text(encoding="utf-8")


@app.get("/ui/deploy", response_class=HTMLResponse)
async def web_ui_deploy():
    page_path = WEBUI_DIR / "deploy.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Deploy UI files not found.")
    return page_path.read_text(encoding="utf-8")


@app.get("/ui/deploy/config")
async def deploy_ui_config():
    return {"open_webui_url": OPEN_WEBUI_URL}


@app.get("/auth/google/config")
async def auth_google_config():
    return {"enabled": bool(GOOGLE_CLIENT_ID), "client_id": GOOGLE_CLIENT_ID}


@app.post("/auth/signup")
async def auth_signup(request: SignupRequest, response: Response):
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    try:
        user = auth_store.create_local_user(
            email=request.email,
            full_name=request.full_name,
            password=request.password,
        )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="An account with that email already exists") from exc
        raise
    token = auth_store.create_session(user.id)
    return build_session_response(response, user, token)


@app.post("/auth/login")
async def auth_login(request: LoginRequest, response: Response):
    user = auth_store.authenticate_local_user(email=request.email, password=request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth_store.create_session(user.id)
    return build_session_response(response, user, token)


@app.post("/auth/google")
async def auth_google(request: GoogleLoginRequest, response: Response):
    payload = verify_google_credential(request.credential)
    email = payload.get("email")
    sub = payload.get("sub")
    if not email or not sub:
        raise HTTPException(status_code=400, detail="Google account payload is missing email or subject")
    user = auth_store.upsert_google_user(
        email=email,
        full_name=payload.get("name") or email.split("@")[0],
        google_sub=sub,
        avatar_url=payload.get("picture"),
    )
    token = auth_store.create_session(user.id)
    return build_session_response(response, user, token)


@app.post("/auth/logout")
async def auth_logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        auth_store.delete_session(token)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/auth/session")
async def auth_session(request: Request):
    user = current_user_from_request(request)
    return {"user": user.to_public_dict() if user else None}
