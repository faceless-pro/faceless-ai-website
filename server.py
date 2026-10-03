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
# ENV
# =========================

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing required environment variables.")

ai = genai.Client(api_key=GEMINI_KEY)
db = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")

app = FastAPI(title="FacelessAI API")

# =========================
# CORS
# =========================

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

LIMITS = {
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

PRO_MODES = {"campaign"}

# =========================
# REQUEST
# =========================

class GenerateRequest(BaseModel):
    mode: str = Field(..., min_length=1, max_length=50)
    niche: str = Field("Other", min_length=1, max_length=50)
    prompt: str = Field(..., min_length=10, max_length=2000)

# =========================
# RATE LIMIT
# =========================

hits = defaultdict(deque)

def rate_limit(key):
    now = time.time()
    q = hits[key]

    while q and now - q[0] > 60:
        q.popleft()

    if len(q) >= 15:
        raise HTTPException(
            429,
            detail={
                "error": "rate_limited",
                "message": "Too many requests. Try again in a minute."
            }
        )

    q.append(now)

# =========================
# AUTH
# =========================

def get_user(request: Request):

    auth = request.headers.get("Authorization", "")

    if not auth.startswith("Bearer "):
        raise HTTPException(
            401,
            detail={
                "error": "missing_auth",
                "message": "Login is required."
            }
        )

    token = auth[7:].strip()

    try:
        user = db.auth.get_user(token).user
    except Exception:
        raise HTTPException(
            401,
            detail={
                "error": "invalid_auth",
                "message": "Invalid or expired login session."
            }
        )

    if not user:
        raise HTTPException(
            401,
            detail={
                "error": "invalid_auth",
                "message": "Unable to verify account."
            }
        )

    return user

# =========================
# PLAN
# =========================

def get_plan(user_id):

    try:
        r = (
            db.table("profiles")
            .select("plan")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )

        plan = (r.data or {}).get("plan", "free").lower()

        return plan if plan in LIMITS else "free"

    except Exception:
        return "free"

# =========================
# USAGE
# =========================

def month_now():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_usage(user_id, month):

    r = (
        db.table("monthly_usage")
        .select("used")
        .eq("user_id", user_id)
        .eq("month", month)
        .maybe_single()
        .execute()
    )

    return int((r.data or {}).get("used", 0))


def add_usage(user_id, month):

    old = get_usage(user_id, month)

    if old == 0:
        db.table("monthly_usage").insert({
            "user_id": user_id,
            "month": month,
            "used": 1
        }).execute()

        return 1

    new = old + 1

    db.table("monthly_usage").update({
        "used": new,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq(
        "user_id", user_id
    ).eq(
        "month", month
    ).execute()

    return new

# =========================
# AI
# =========================

def generate(mode, niche, prompt):

    instructions = {
        "offer": "Create a strong realistic offer with positioning, benefits, deliverables and CTA.",
        "landing": "Create conversion-focused landing page copy with headline, benefits, objections, FAQ and CTA.",
        "email": "Create a natural professional cold email sequence with follow-ups and CTA.",
        "dm": "Create a natural non-spammy sales DM sequence.",
        "ads": "Create ad hooks, copy, headlines, CTA, audience and creative angles.",
        "content": "Create useful content ideas, hooks, CTAs and a practical content plan.",
        "campaign": "Create a complete coordinated marketing campaign with offer, landing page, email, DM, ads and content."
    }

    text = f"""
You are FacelessAI Growth Copilot.

Create practical marketing output.

Never invent testimonials, customers, revenue,
reviews, awards, statistics or guaranteed results.

MODE:
{mode}

NICHE:
{niche}

TASK:
{instructions[mode]}

USER:
{prompt}
"""

    for model in [MODEL, FALLBACK]:

        try:
            r = ai.models.generate_content(
                model=model,
                contents=text
            )

            if r.text:
                return r.text.strip()

        except Exception as e:
            print("AI error:", e)
            time.sleep(1)

    raise RuntimeError("AI generation failed.")

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
    return {"status": "ok"}


@app.post("/generate")
def generate_asset(
    data: GenerateRequest,
    request: Request
):

    ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    rate_limit("ip:" + ip)

    user = get_user(request)
    user_id = str(user.id)

    rate_limit("user:" + user_id)

    mode = data.mode.lower().strip()

    if mode not in VALID_MODES:
        raise HTTPException(
            400,
            detail={
                "error": "invalid_mode",
                "message": "Invalid workflow."
            }
        )

    plan = get_plan(user_id)

    if mode in PRO_MODES and plan != "pro":
        raise HTTPException(
            403,
            detail={
                "error": "pro_required",
                "message": "Campaign Builder requires Pro.",
                "plan": plan
            }
        )

    month = month_now()
    used = get_usage(user_id, month)
    limit = LIMITS[plan]

    if used >= limit:
        raise HTTPException(
            403,
            detail={
                "error": "monthly_limit_reached",
                "message": "Monthly generation limit reached.",
                "plan": plan,
                "used": used,
                "limit": limit
            }
        )

    try:
        result = generate(
            mode,
            data.niche.strip(),
            data.prompt.strip()
        )
    except Exception:
        raise HTTPException(
            502,
            detail={
                "error": "generation_failed",
                "message": "AI is temporarily unavailable."
            }
        )

    used += add_usage(user_id, month) - get_usage(user_id, month)

    # Re-read actual value after update
    used = get_usage(user_id, month)

    return {
        "success": True,
        "result": result,
        "mode": mode,
        "plan": plan,
        "usage": {
            "used": used,
            "limit": limit,
            "remaining": max(limit - used, 0)
        }
    }


@app.post("/generate-video")
def generate_video():
    raise HTTPException(
        410,
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
        port=int(os.getenv("PORT", 8000))
    )
