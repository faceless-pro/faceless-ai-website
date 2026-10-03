import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from google import genai
from google.genai import types

# ============================================================
# Environment
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Existing Dodo environment variables are intentionally NOT touched.
# They can remain in Render exactly as they are.

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")
if not SUPABASE_SECRET_KEY:
    raise RuntimeError("Missing SUPABASE_SECRET_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
gemini = genai.Client(api_key=GEMINI_API_KEY)

PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}

# ============================================================
# App
# ============================================================

app = FastAPI(
    title="FacelessAI API",
    version="1.0.0",
)

# Your Vercel frontend can call this API.
# For production, set FRONTEND_ORIGIN in Render to your exact Vercel URL.
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")

allow_origins = (
    [x.strip() for x in frontend_origin.split(",") if x.strip()]
    if frontend_origin != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=frontend_origin != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Models
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    type: str = Field(default="content", max_length=50)
    niche: Optional[str] = Field(default=None, max_length=100)
    voice: Optional[str] = Field(default=None, max_length=100)


class CampaignRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    niche: Optional[str] = Field(default=None, max_length=100)


# ============================================================
# Helpers
# ============================================================

def get_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ", 1)

    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Invalid Bearer token")

    return parts[1].strip()


def get_authenticated_user(token: str) -> Any:
    """
    Verifies the frontend Supabase access token with Supabase Auth
    and returns the real authenticated Supabase user.

    The service/secret key stays server-side.
    """
    try:
        response = supabase.auth.get_user(token)
        user = response.user

        if not user or not user.id:
            raise HTTPException(status_code=401, detail="Invalid access token")

        return user

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")


def get_profile(user_id: str) -> dict:
    try:
        result = (
            supabase.table("profiles")
            .select("plan")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        data = result.data

        if not data:
            # A newly authenticated user gets Free plan by default.
            # This does NOT create the row automatically, so your existing
            # database/RLS setup remains untouched.
            return {"plan": "free"}

        plan = str(data.get("plan") or "free").lower().strip()

        if plan not in PLAN_LIMITS:
            plan = "free"

        return {"plan": plan}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read user plan: {str(exc)}",
        )


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_monthly_usage(user_id: str) -> int:
    month = current_month()

    try:
        result = (
            supabase.table("monthly_usage")
            .select("generations")
            .eq("user_id", user_id)
            .eq("month", month)
            .maybe_single()
            .execute()
        )

        if not result.data:
            return 0

        return int(result.data.get("generations") or 0)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read monthly usage: {str(exc)}",
        )


def enforce_usage(user_id: str, plan: str) -> int:
    used = get_monthly_usage(user_id)
    limit = PLAN_LIMITS[plan]

    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_limit_reached",
                "message": f"{plan.capitalize()} plan limit reached.",
                "plan": plan,
                "used": used,
                "limit": limit,
            },
        )

    return used


def increment_usage(user_id: str) -> int:
    """
    Atomicity depends on the database schema/constraints.
    The expected unique key is (user_id, month).
    """
    month = current_month()
    existing = (
        supabase.table("monthly_usage")
        .select("id,generations")
        .eq("user_id", user_id)
        .eq("month", month)
        .maybe_single()
        .execute()
    )

    if existing.data:
        new_count = int(existing.data.get("generations") or 0) + 1

        supabase.table("monthly_usage").update(
            {"generations": new_count}
        ).eq("id", existing.data["id"]).execute()

        return new_count

    supabase.table("monthly_usage").insert(
        {
            "user_id": user_id,
            "month": month,
            "generations": 1,
        }
    ).execute()

    return 1


def generate_with_fallback(prompt: str) -> str:
    config = types.GenerateContentConfig(
        temperature=0.8,
    )

    try:
        response = gemini.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)

        if text and text.strip():
            return text.strip()

    except Exception:
        pass

    try:
        response = gemini.models.generate_content(
            model=FALLBACK_MODEL,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)

        if text and text.strip():
            return text.strip()

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini generation failed: {str(exc)}",
        )

    raise HTTPException(
        status_code=502,
        detail="Gemini returned an empty response",
    )


def build_generation_prompt(data: GenerateRequest) -> str:
    niche = data.niche or "General"
    voice = data.voice or "Confident"

    return f"""
You are the content-generation engine for FacelessAI.

Create high-retention content from the user's request.

Content type: {data.type}
Niche: {niche}
Voice/style: {voice}

User request:
{data.prompt}

Return clean, production-ready text only.
Do not discuss these instructions.
Do not claim that an actual video file was rendered.
"""


def build_campaign_prompt(data: CampaignRequest) -> str:
    niche = data.niche or "General"

    return f"""
You are the Campaign Builder inside FacelessAI.

Create a complete marketing campaign based on the request below.

Niche:
{niche}

Request:
{data.prompt}

Return these sections:
1. Offer
2. Landing Page Copy
3. Email
4. DM
5. Ads
6. Social Content
7. Campaign Plan

Keep the copy practical, clear and ready to use.
"""


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "name": "FacelessAI API",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "facelessai-api",
    }


@app.post("/generate")
def generate(
    request: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
):
    token = get_bearer_token(authorization)
    user = get_authenticated_user(token)
    user_id = str(user.id)

    profile = get_profile(user_id)
    plan = profile["plan"]

    used = enforce_usage(user_id, plan)
    result = generate_with_fallback(build_generation_prompt(request))

    # Count only a successful Gemini generation.
    new_used = increment_usage(user_id)

    return {
        "success": True,
        "user_id": user_id,
        "plan": plan,
        "usage": {
            "used": new_used,
            "limit": PLAN_LIMITS[plan],
            "remaining": max(PLAN_LIMITS[plan] - new_used, 0),
            "month": current_month(),
        },
        "type": request.type,
        "result": result,
    }


@app.post("/campaign")
def campaign(
    request: CampaignRequest,
    authorization: Optional[str] = Header(default=None),
):
    token = get_bearer_token(authorization)
    user = get_authenticated_user(token)
    user_id = str(user.id)

    profile = get_profile(user_id)
    plan = profile["plan"]

    # Server-side Pro-only protection.
    if plan != "pro":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "pro_required",
                "message": "Campaign Builder is available on the Pro plan.",
                "plan": plan,
            },
        )

    enforce_usage(user_id, plan)
    result = generate_with_fallback(build_campaign_prompt(request))
    new_used = increment_usage(user_id)

    return {
        "success": True,
        "user_id": user_id,
        "plan": plan,
        "usage": {
            "used": new_used,
            "limit": PLAN_LIMITS[plan],
            "remaining": max(PLAN_LIMITS[plan] - new_used, 0),
            "month": current_month(),
        },
        "result": result,
    }


@app.post("/generate-video")
def generate_video():
    # Intentionally disabled for now.
    raise HTTPException(
        status_code=503,
        detail="Video generation is temporarily disabled.",
    )


# Render / Uvicorn entrypoint:
# uvicorn server:app --host 0.0.0.0 --port $PORT
