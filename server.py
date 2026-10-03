import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
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


if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not SUPABASE_SECRET_KEY:
    raise RuntimeError("SUPABASE_SECRET_KEY is missing")


# =========================================================
# APP
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
# RATE LIMIT
# =========================================================

RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 60

rate_store = defaultdict(deque)


def get_client_ip(request: Request):

    forwarded = request.headers.get("x-forwarded-for")

    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request):

    ip = get_client_ip(request)

    now = time.time()

    timestamps = rate_store[ip]

    while timestamps and (
        now - timestamps[0] > RATE_LIMIT_WINDOW
    ):
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:

        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

    timestamps.append(now)


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
# SUPABASE AUTH
# =========================================================

def verify_user(access_token: str):

    import requests

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

    if not user.get("id"):

        raise HTTPException(
            status_code=401,
            detail="Invalid user session"
        )

    return user


# =========================================================
# SUPABASE REST HEADERS
# =========================================================

def supabase_headers():

    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


# =========================================================
# GET USER PLAN
# =========================================================

def get_user_plan(user_id: str):

    import requests

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
# MONTH
# =========================================================

def current_month():

    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m")


# =========================================================
# GET USAGE
# =========================================================

def get_usage(
    user_id: str,
    month: str
):

    import requests

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
            detail="Could not read usage"
        )

    rows = response.json()

    if not rows:
        return 0

    try:
        return int(rows[0].get("used", 0))
    except (TypeError, ValueError):
        return 0


# =========================================================
# INCREMENT USAGE
# =========================================================

def increment_usage(
    user_id: str,
    month: str
):

    import requests

    current = get_usage(
        user_id,
        month
    )

    new_value = current + 1

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/monthly_usage",
        headers={
            **supabase_headers(),
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json={
            "user_id": user_id,
            "month": month,
            "used": new_value,
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        timeout=10,
    )

    if response.status_code not in (200, 201, 204):

        raise HTTPException(
            status_code=503,
            detail="Could not update usage"
        )

    return new_value


# =========================================================
# AI PROMPT
# =========================================================

def build_prompt(data: GenerateRequest):

    return f"""
You are FacelessAI, an AI growth and marketing assistant.

Mode:
{data.mode}

Niche:
{data.niche}

User request:
{data.prompt}

Create high-quality, practical, ready-to-use content.

Rules:
- Follow the requested mode.
- Do not invent testimonials.
- Do not invent customers.
- Do not invent statistics.
- Do not invent certifications.
- Do not invent awards.
- Do not promise guaranteed results.
- Do not create fake proof.
- Use placeholders when required information is missing.

Return only the finished content.
"""


# =========================================================
# GEMINI
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
    request: Request,
    authorization: str | None = Header(default=None),
):

    # -------------------------
    # RATE LIMIT
    # -------------------------

    check_rate_limit(request)

    # -------------------------
    # AUTH
    # -------------------------

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Login required"
        )

    if not authorization.lower().startswith("bearer "):

        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )

    token = authorization.split(
        " ",
        1
    )[1].strip()

    if not token:

        raise HTTPException(
            status_code=401,
            detail="Login required"
        )

    # -------------------------
    # VERIFY USER
    # -------------------------

    user = verify_user(token)

    user_id = user["id"]

    # -------------------------
    # MODE
    # -------------------------

    if data.mode not in VALID_MODES:

        raise HTTPException(
            status_code=400,
            detail="Invalid generation mode"
        )

    # -------------------------
    # PLAN
    # -------------------------

    plan = get_user_plan(user_id)

    # -------------------------
    # CAMPAIGN PRO ONLY
    # -------------------------

    if (
        data.mode == "campaign"
        and plan != "pro"
    ):

        raise HTTPException(
            status_code=403,
            detail="Campaign Builder requires Pro"
        )

    # -------------------------
    # USAGE
    # -------------------------

    month = current_month()

    used = get_usage(
        user_id,
        month
    )

    limit = PLAN_LIMITS[plan]

    if used >= limit:

        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly limit reached. "
                f"{plan} plan allows {limit} generations."
            )
        )

    # -------------------------
    # AI
    # -------------------------

    output = generate_ai(
        build_prompt(data)
    )

    # -------------------------
    # COUNT
    # -------------------------

    new_usage = increment_usage(
        user_id,
        month
    )

    # -------------------------
    # RESPONSE
    # -------------------------

    return {
        "success": True,
        "result": output,
        "mode": data.mode,
        "plan": plan,
        "usage": {
            "used": new_usage,
            "limit": limit,
            "remaining": max(
                0,
                limit - new_usage
            ),
        },
    }


# =========================================================
# VIDEO
# =========================================================

@app.post("/generate-video")
def generate_video():

    raise HTTPException(
        status_code=410,
        detail="Video generation is not available yet"
    )


# =========================================================
# START
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
        port=port
    )
