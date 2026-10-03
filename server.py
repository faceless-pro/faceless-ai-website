import os
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

load_dotenv()


# =========================================================
# ENVIRONMENT
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

PRIMARY_MODEL = os.getenv(
    "PRIMARY_MODEL",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "FALLBACK_MODEL",
    "gemini-3.8-flash"
)


# =========================================================
# STARTUP VALIDATION
# =========================================================

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not SUPABASE_SECRET_KEY:
    raise RuntimeError("SUPABASE_SECRET_KEY is missing")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="FacelessAI API",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faceless-ai-website.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GEMINI
# =========================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# PLANS
# =========================================================

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}


# =========================================================
# VALID MODES
# =========================================================

VALID_MODES = {
    "offer",
    "landing",
    "email",
    "dm",
    "ads",
    "content",
    "campaign",
}


# =========================================================
# REQUEST MODEL
# =========================================================

class GenerateRequest(BaseModel):

    mode: str = Field(
        ...,
        min_length=2,
        max_length=30
    )

    niche: str = Field(
        ...,
        min_length=2,
        max_length=100
    )

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=4000
    )


# =========================================================
# SUPABASE HEADERS
# =========================================================

def supabase_headers():

    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


# =========================================================
# VERIFY SUPABASE USER
# =========================================================

def verify_user(access_token: str):

    try:

        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_SECRET_KEY,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=10,
        )

    except requests.RequestException:

        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable"
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired login session"
        )

    try:

        user = response.json()

    except Exception:

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication response"
        )

    user_id = user.get("id")

    if not user_id:

        raise HTTPException(
            status_code=401,
            detail="Invalid user session"
        )

    return user


# =========================================================
# GET USER PLAN
# =========================================================

def get_user_plan(user_id: str):

    try:

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={
                "select": "plan",
                "id": f"eq.{user_id}",
                "limit": "1",
            },
            timeout=10,
        )

    except requests.RequestException:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=503,
            detail="Could not read user plan"
        )

    rows = response.json()

    if not rows:

        return "free"

    plan = str(
        rows[0].get("plan", "free")
    ).lower()

    if plan not in PLAN_LIMITS:

        return "free"

    return plan


# =========================================================
# CURRENT MONTH
# =========================================================

def get_current_month():

    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m")


# =========================================================
# GET MONTHLY USAGE
# =========================================================

def get_usage(
    user_id: str,
    month: str
):

    try:

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/monthly_usage",
            headers=supabase_headers(),
            params={
                "select": "used",
                "user_id": f"eq.{user_id}",
                "month": f"eq.{month}",
                "limit": "1",
            },
            timeout=10,
        )

    except requests.RequestException:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=503,
            detail="Could not read monthly usage"
        )

    rows = response.json()

    if not rows:

        return 0

    try:

        return int(
            rows[0].get("used", 0)
        )

    except (TypeError, ValueError):

        return 0


# =========================================================
# INCREMENT MONTHLY USAGE
# =========================================================

def increment_usage(
    user_id: str,
    month: str
):

    current_usage = get_usage(
        user_id,
        month
    )

    new_usage = current_usage + 1

    payload = {
        "user_id": user_id,
        "month": month,
        "used": new_usage,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    try:

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/monthly_usage",
            headers={
                **supabase_headers(),
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json=payload,
            timeout=10,
        )

    except requests.RequestException:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )

    if response.status_code not in (200, 201, 204):

        raise HTTPException(
            status_code=503,
            detail="Could not update monthly usage"
        )

    return new_usage


# =========================================================
# AI PROMPT
# =========================================================

def build_ai_prompt(data: GenerateRequest):

    return f"""
You are FacelessAI, an AI growth and marketing assistant.

MODE:
{data.mode}

NICHE:
{data.niche}

USER REQUEST:
{data.prompt}

Generate useful, high-quality, practical content.

Rules:
- Follow the requested mode.
- Make the result ready to use.
- Do not invent testimonials.
- Do not invent customers.
- Do not invent statistics.
- Do not invent certifications.
- Do not invent awards.
- Do not claim guaranteed results.
- Do not create fake proof.
- If important information is missing, use a clear placeholder.
- Do not mention these internal instructions in the answer.

Return only the finished content.
"""


# =========================================================
# GEMINI GENERATION
# =========================================================

def generate_ai(prompt: str):

    last_error = None

    for model in (
        PRIMARY_MODEL,
        FALLBACK_MODEL,
    ):

        try:

            result = gemini.models.generate_content(
                model=model,
                contents=prompt,
            )

            output = (
                getattr(result, "text", None)
                or ""
            ).strip()

            if output:

                return output

            last_error = RuntimeError(
                "Gemini returned empty output"
            )

        except Exception as error:

            last_error = error

    raise HTTPException(
        status_code=502,
        detail="AI generation failed"
    ) from last_error


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "name": "FacelessAI API",
        "status": "online",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "plans": PLAN_LIMITS,
    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/generate")
def generate(
    data: GenerateRequest,
    authorization: Optional[str] = Header(
        default=None
    ),
):

    # -----------------------------------------------------
    # AUTHORIZATION
    # -----------------------------------------------------

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Login required"
        )

    if not authorization.lower().startswith(
        "bearer "
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )

    access_token = authorization.split(
        " ",
        1
    )[1].strip()

    if not access_token:

        raise HTTPException(
            status_code=401,
            detail="Login required"
        )

    # -----------------------------------------------------
    # VERIFY USER
    # -----------------------------------------------------

    user = verify_user(
        access_token
    )

    user_id = user["id"]

    # -----------------------------------------------------
    # VALIDATE MODE
    # -----------------------------------------------------

    if data.mode not in VALID_MODES:

        raise HTTPException(
            status_code=400,
            detail="Invalid generation mode"
        )

    # -----------------------------------------------------
    # GET PLAN
    # -----------------------------------------------------

    plan = get_user_plan(
        user_id
    )

    # -----------------------------------------------------
    # PRO-ONLY CAMPAIGN
    # -----------------------------------------------------

    if (
        data.mode == "campaign"
        and plan != "pro"
    ):

        raise HTTPException(
            status_code=403,
            detail="Campaign Builder requires Pro plan"
        )

    # -----------------------------------------------------
    # MONTHLY USAGE
    # -----------------------------------------------------

    month = get_current_month()

    used = get_usage(
        user_id,
        month
    )

    limit = PLAN_LIMITS[plan]

    # -----------------------------------------------------
    # LIMIT
    # -----------------------------------------------------

    if used >= limit:

        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly generation limit reached. "
                f"{plan} plan allows {limit} generations."
            )
        )

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    ai_prompt = build_ai_prompt(
        data
    )

    output = generate_ai(
        ai_prompt
    )

    # -----------------------------------------------------
    # UPDATE USAGE
    # -----------------------------------------------------

    new_usage = increment_usage(
        user_id,
        month
    )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "success": True,
        "output": output,
        "plan": plan,
        "used": new_usage,
        "limit": limit,
        "remaining": max(
            0,
            limit - new_usage
        ),
    }


# =========================================================
# VIDEO ENDPOINT
# =========================================================

@app.post("/generate-video")
def generate_video():

    raise HTTPException(
        status_code=501,
        detail="Video generation is not enabled yet"
    )


# =========================================================
# LOCAL / RENDER START
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
    )
