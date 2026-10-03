import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client
from google import genai

load_dotenv()

# =========================
# ENVIRONMENT
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")

if not SUPABASE_SECRET_KEY:
    raise RuntimeError("Missing SUPABASE_SECRET_KEY")

# =========================
# CLIENTS
# =========================

ai = genai.Client(api_key=GEMINI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash"
)

# =========================
# APP
# =========================

app = FastAPI(title="FacelessAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faceless-ai-website.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# PLANS
# =========================

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500
}

VALID_MODES = {
    "offer",
    "landing",
    "email",
    "dm",
    "ads",
    "content",
    "campaign"
}

PRO_MODES = {
    "campaign"
}

# =========================
# REQUEST
# =========================

class GenerateRequest(BaseModel):
    mode: str = Field(
        ...,
        min_length=1,
        max_length=50
    )

    niche: str = Field(
        "Other",
        min_length=1,
        max_length=50
    )

    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000
    )

# =========================
# RATE LIMIT
# =========================

rate_hits = defaultdict(deque)

def rate_limit(key: str):
    now = time.time()
    queue = rate_hits[key]

    while queue and now - queue[0] > 60:
        queue.popleft()

    if len(queue) >= 15:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": "Too many requests. Try again in a minute."
            }
        )

    queue.append(now)

# =========================
# AUTH
# =========================

def get_user(request: Request):

    authorization = request.headers.get(
        "Authorization",
        ""
    )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_auth",
                "message": "Login is required."
            }
        )

    token = authorization[7:].strip()

    try:
        response = supabase.auth.get_user(token)
        user = response.user

    except Exception:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_auth",
                "message": "Invalid or expired login session."
            }
        )

    if not user:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_auth",
                "message": "Unable to verify account."
            }
        )

    return user

# =========================
# PLAN
# =========================

def get_plan(user_id: str):

    try:
        response = (
            supabase
            .table("profiles")
            .select("plan")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )

        plan = (
            (response.data or {})
            .get("plan", "free")
            .lower()
        )

        if plan not in PLAN_LIMITS:
            return "free"

        return plan

    except Exception:
        return "free"

# =========================
# MONTH
# =========================

def current_month():

    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m")

# =========================
# USAGE
# =========================

def get_usage(user_id: str, month: str):

    response = (
        supabase
        .table("monthly_usage")
        .select("used")
        .eq("user_id", user_id)
        .eq("month", month)
        .maybe_single()
        .execute()
    )

    return int(
        (response.data or {}).get(
            "used",
            0
        )
    )

def add_usage(user_id: str, month: str):

    current = get_usage(
        user_id,
        month
    )

    new_value = current + 1

    if current == 0:

        supabase.table(
            "monthly_usage"
        ).insert({
            "user_id": user_id,
            "month": month,
            "used": new_value
        }).execute()

    else:

        supabase.table(
            "monthly_usage"
        ).update({
            "used": new_value,
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat()
        }).eq(
            "user_id",
            user_id
        ).eq(
            "month",
            month
        ).execute()

    return new_value

# =========================
# AI GENERATION
# =========================

def generate_content(
    mode: str,
    niche: str,
    prompt: str
):

    instructions = {

        "offer":
            "Create a strong realistic offer with positioning, benefits, deliverables and CTA.",

        "landing":
            "Create conversion-focused landing page copy with headline, benefits, objections, FAQ and CTA.",

        "email":
            "Create a natural professional cold email sequence with follow-ups and CTA.",

        "dm":
            "Create a natural non-spammy sales DM sequence.",

        "ads":
            "Create ad hooks, copy, headlines, CTA, audience and creative angles.",

        "content":
            "Create useful content ideas, hooks, CTAs and a practical content plan.",

        "campaign":
            "Create a complete coordinated marketing campaign including offer, landing page, email, DM, ads and content."
    }

    instructions_text = instructions[mode]

    prompt_text = f"""
You are FacelessAI Growth Copilot.

Create practical, useful marketing output.

Never invent:
- testimonials
- customers
- revenue
- reviews
- awards
- fake statistics
- guaranteed results

MODE:
{mode}

NICHE:
{niche}

TASK:
{instructions_text}

USER REQUEST:
{prompt}
"""

    for model in [
        MODEL,
        FALLBACK_MODEL
    ]:

        try:

            result = ai.models.generate_content(
                model=model,
                contents=prompt_text
            )

            if result.text:
                return result.text.strip()

        except Exception as error:

            print(
                f"Gemini error ({model}):",
                error
            )

            time.sleep(1)

    raise RuntimeError(
        "AI generation failed."
    )

# =========================
# ROUTES
# =========================

@app.get("/")
def root():

    return {
        "name": "FacelessAI API",
        "status": "online"
    }

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# =========================
# GENERATE
# =========================

@app.post("/generate")
def generate(
    data: GenerateRequest,
    request: Request
):

    # IP rate limit
    ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    rate_limit(
        "ip:" + ip
    )

    # Verify Supabase user
    user = get_user(request)

    user_id = str(user.id)

    # User rate limit
    rate_limit(
        "user:" + user_id
    )

    # Validate mode
    mode = data.mode.lower().strip()

    if mode not in VALID_MODES:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_mode",
                "message": "Invalid workflow."
            }
        )

    # Get server-side plan
    plan = get_plan(user_id)

    # Pro-only workflow
    if (
        mode in PRO_MODES
        and plan != "pro"
    ):

        raise HTTPException(
            status_code=403,
            detail={
                "error": "pro_required",
                "message": "Campaign Builder requires Pro.",
                "plan": plan
            }
        )

    # Usage
    month = current_month()

    used = get_usage(
        user_id,
        month
    )

    limit = PLAN_LIMITS[plan]

    if used >= limit:

        raise HTTPException(
            status_code=403,
            detail={
                "error": "monthly_limit_reached",
                "message": "Monthly generation limit reached.",
                "plan": plan,
                "used": used,
                "limit": limit
            }
        )

    # Generate
    try:

        result = generate_content(
            mode,
            data.niche.strip(),
            data.prompt.strip()
        )

    except Exception as error:

        print(
            "Generation failed:",
            error
        )

        raise HTTPException(
            status_code=502,
            detail={
                "error": "generation_failed",
                "message": "AI is temporarily unavailable."
            }
        )

    # Count successful generation
    new_usage = add_usage(
        user_id,
        month
    )

    return {
        "success": True,
        "result": result,
        "mode": mode,
        "plan": plan,
        "usage": {
            "used": new_usage,
            "limit": limit,
            "remaining": max(
                limit - new_usage,
                0
            )
        }
    }

# =========================
# VIDEO ENDPOINT
# =========================

@app.post("/generate-video")
def generate_video():

    raise HTTPException(
        status_code=410,
        detail={
            "error": "removed",
            "message": "Video generation is not available."
        }
    )

# =========================
# START
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )
