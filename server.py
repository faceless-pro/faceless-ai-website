import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Deque

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="FacelessAI Growth Copilot API",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faceless-ai-website.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PLAN LIMITS
# ============================================================

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):
    mode: str = Field(..., min_length=1, max_length=50)
    prompt: str = Field(..., min_length=10, max_length=2000)

    # IMPORTANT:
    # Current frontend doesn't send this field, so it defaults
    # to Free.
    #
    # This is ONLY a temporary testing mechanism.
    # Real paid-plan verification will later come from
    # Dodo + Supabase on the server.
    plan: str = Field(default="free", max_length=20)


# ============================================================
# VALID MODES
# ============================================================

VALID_MODES = {
    "offer",
    "landing",
    "email",
    "dm",
    "ads",
    "content",
    "campaign",
}


# ============================================================
# MODE INSTRUCTIONS
# ============================================================

MODE_INSTRUCTIONS = {

    "offer": """
Create a strong, realistic offer for the user's business or idea.

Include:
1. Offer name
2. Target customer
3. Core problem
4. Desired outcome
5. Value proposition
6. Deliverables
7. Pricing/positioning suggestion if enough information exists
8. Objection handling
9. Clear CTA

Do not invent customer numbers, revenue, testimonials,
case studies, logos, awards, or guaranteed results.
""",

    "landing": """
Create high-converting landing page copy.

Include:
1. Headline
2. Subheadline
3. Problem
4. Solution
5. Benefits
6. Features
7. How it works
8. Objection handling
9. CTA
10. FAQ

Keep claims realistic.
Never invent testimonials, customer counts, revenue,
logos, case studies, awards, or guaranteed outcomes.
""",

    "email": """
Create a professional cold outreach email.

Include:
1. Subject line
2. Opening
3. Personalized problem
4. Value proposition
5. Specific offer
6. Low-friction CTA
7. Follow-up suggestion

Keep it concise and natural.
Do not make unsupported claims.
Do not pretend to know personal facts that were not provided.
""",

    "dm": """
Create a natural sales DM suitable for Instagram, LinkedIn,
X, Facebook, or similar platforms.

Give:
1. First message
2. Follow-up message
3. Objection response
4. Closing CTA

Avoid spammy language.
Avoid fake urgency.
Avoid fake testimonials or results.
""",

    "ads": """
Create an advertising campaign concept.

Include:
1. Campaign angle
2. Target audience
3. Main hook
4. Primary ad copy
5. Short headline
6. CTA
7. Creative concept
8. Variations

Do not claim guaranteed results.
Do not invent statistics or customer outcomes.
""",

    "content": """
Create a content engine for the user's idea.

Include:
1. Content pillars
2. 10 content ideas
3. Hooks
4. Short-form video concepts
5. CTA ideas
6. Posting strategy
7. Repurposing ideas

Keep ideas practical and original.
Do not promise virality.
""",

    "campaign": """
Create a complete marketing campaign.

Include:
1. Campaign objective
2. Target audience
3. Offer
4. Positioning
5. Landing page angle
6. Cold email
7. Sales DM
8. Ad concepts
9. Organic content ideas
10. CTA
11. 7-day execution plan
12. Measurement framework

Do not invent performance data.
Do not claim guaranteed revenue, leads, customers, or virality.
""",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are FacelessAI Growth Copilot, a high-quality AI marketing
strategy and copywriting assistant.

Your job is to turn a user's idea into practical marketing assets
that can actually be used.

IMPORTANT RULES:

- Never fabricate testimonials.
- Never fabricate customer numbers.
- Never fabricate revenue.
- Never fabricate logos.
- Never fabricate case studies.
- Never fabricate awards.
- Never fabricate reviews.
- Never claim a result is guaranteed.
- Never guarantee virality.
- Never guarantee revenue.
- Never guarantee leads or sales.
- Never pretend a business has customers unless the user explicitly
  provided that information.
- If information is missing, make a clearly labeled assumption or
  keep the claim generic.
- Prefer specific, useful copy over generic motivational advice.
- Keep output organized with clear headings.
- Write naturally.
- Avoid excessive emojis.
- Avoid spammy language.
- Avoid fake scarcity.
- Avoid deceptive marketing.
- Respect the user's actual product and audience.
"""


# ============================================================
# ANTI-ABUSE RATE LIMIT
# ============================================================

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_REQUESTS = 15

request_log: Dict[str, Deque[float]] = defaultdict(deque)


def get_client_ip(request: Request) -> str:
    """
    Gets the client IP.

    Note:
    On Render, the actual client IP may be supplied through
    X-Forwarded-For.
    """

    forwarded_for = request.headers.get("x-forwarded-for")

    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def check_rate_limit(ip: str) -> None:
    now = time.time()

    timestamps = request_log[ip]

    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again."
        )

    timestamps.append(now)


# ============================================================
# MONTHLY USAGE
# ============================================================

# Temporary in-memory usage storage.
#
# IMPORTANT:
# This resets when Render restarts/redeploys.
#
# Production version should move this to Supabase.
#
# Structure:
# {
#   "ip": {
#       "month": "2026-09",
#       "used": 2
#   }
# }

usage_store = {}


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_usage(ip: str) -> int:

    month = current_month()

    record = usage_store.get(ip)

    if not record:
        usage_store[ip] = {
            "month": month,
            "used": 0,
        }

        return 0

    # Automatic monthly reset
    if record["month"] != month:
        usage_store[ip] = {
            "month": month,
            "used": 0,
        }

        return 0

    return int(record["used"])


def increment_usage(ip: str) -> int:

    month = current_month()

    record = usage_store.get(ip)

    if not record or record["month"] != month:
        usage_store[ip] = {
            "month": month,
            "used": 1,
        }

        return 1

    record["used"] += 1

    return record["used"]


def check_plan_limit(ip: str, plan: str) -> int:

    limit = PLAN_LIMITS[plan]
    used = get_usage(ip)

    if used >= limit:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "monthly_limit_reached",
                "message": (
                    f"You have used all {limit} generations "
                    f"available on the {plan.capitalize()} plan this month."
                ),
                "plan": plan,
                "used": used,
                "limit": limit,
                "remaining": 0,
            },
        )

    return used


# ============================================================
# GEMINI HELPERS
# ============================================================

def is_retryable_error(error: Exception) -> bool:

    message = str(error).lower()

    retryable_words = [
        "429",
        "rate limit",
        "resource exhausted",
        "quota",
        "timeout",
        "temporarily unavailable",
        "503",
        "500",
        "internal",
        "overloaded",
        "deadline",
    ]

    return any(word in message for word in retryable_words)


def generate_with_model(
    model_name: str,
    mode: str,
    user_prompt: str,
) -> str:

    mode_instruction = MODE_INSTRUCTIONS[mode]

    full_prompt = f"""
{SYSTEM_PROMPT}

TASK MODE:
{mode}

MODE-SPECIFIC INSTRUCTIONS:
{mode_instruction}

USER REQUEST:
{user_prompt}

Return a polished, useful answer that the user can directly use.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.75,
            max_output_tokens=5000,
        ),
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()


def generate_ai_result(
    mode: str,
    user_prompt: str,
) -> str:

    models_to_try = [
        PRIMARY_MODEL,
        FALLBACK_MODEL,
    ]

    last_error = None

    for model_name in models_to_try:

        # Avoid trying the same model twice
        if not model_name:
            continue

        for attempt in range(3):

            try:

                result = generate_with_model(
                    model_name=model_name,
                    mode=mode,
                    user_prompt=user_prompt,
                )

                if result:
                    return result

            except Exception as error:

                last_error = error

                print(
                    f"[Gemini Error] "
                    f"model={model_name} "
                    f"attempt={attempt + 1}/3 "
                    f"error={error}"
                )

                if not is_retryable_error(error):
                    break

                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"AI generation failed. Last error: {last_error}"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name": "FacelessAI Growth Copilot API",
        "status": "online",
        "version": "2.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "plans": PLAN_LIMITS,
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
async def generate(
    payload: GenerateRequest,
    request: Request,
):

    # --------------------------------------------------------
    # CLIENT
    # --------------------------------------------------------

    ip = get_client_ip(request)

    # --------------------------------------------------------
    # ANTI-ABUSE RATE LIMIT
    # --------------------------------------------------------

    check_rate_limit(ip)

    # --------------------------------------------------------
    # VALIDATE MODE
    # --------------------------------------------------------

    mode = payload.mode.strip().lower()

    if mode not in VALID_MODES:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_mode",
                "message": (
                    f"Invalid mode '{mode}'. "
                    f"Valid modes are: {', '.join(sorted(VALID_MODES))}"
                ),
            },
        )

    # --------------------------------------------------------
    # VALIDATE PROMPT
    # --------------------------------------------------------

    user_prompt = payload.prompt.strip()

    if len(user_prompt) < 10:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "prompt_too_short",
                "message": "Please enter at least 10 characters.",
            },
        )

    if len(user_prompt) > 2000:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "prompt_too_long",
                "message": "Prompt cannot exceed 2000 characters.",
            },
        )

    # --------------------------------------------------------
    # VALIDATE PLAN
    # --------------------------------------------------------

    plan = payload.plan.strip().lower()

    if plan not in PLAN_LIMITS:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_plan",
                "message": "Invalid plan.",
            },
        )

    # --------------------------------------------------------
    # CHECK MONTHLY LIMIT
    # --------------------------------------------------------

    used_before = check_plan_limit(
        ip=ip,
        plan=plan,
    )

    limit = PLAN_LIMITS[plan]

    remaining_before = max(
        limit - used_before,
        0,
    )

    # --------------------------------------------------------
    # GENERATE AI RESULT
    # --------------------------------------------------------

    try:

        result = generate_ai_result(
            mode=mode,
            user_prompt=user_prompt,
        )

    except Exception as error:

        print(f"[Generation Failed] {error}")

        raise HTTPException(
            status_code=502,
            detail={
                "error": "generation_failed",
                "message": (
                    "AI generation is temporarily unavailable. "
                    "Please try again in a moment."
                ),
            },
        )

    # --------------------------------------------------------
    # ONLY COUNT SUCCESSFUL GENERATIONS
    # --------------------------------------------------------

    used_after = increment_usage(ip)

    remaining_after = max(
        limit - used_after,
        0,
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,

        "result": result,

        "mode": mode,

        "plan": plan,

        "usage": {
            "used": used_after,
            "limit": limit,
            "remaining": remaining_after,
            "month": current_month(),
        },

        "message": (
            f"{remaining_after} generation(s) remaining "
            f"on your {plan.capitalize()} plan this month."
        ),
    }


# ============================================================
# OLD VIDEO ENDPOINT
# ============================================================

@app.post("/generate-video")
async def generate_video():

    raise HTTPException(
        status_code=410,
        detail={
            "error": "endpoint_removed",
            "message": (
                "Video generation is no longer part of "
                "FacelessAI Growth Copilot."
            ),
        },
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)
    print("FacelessAI Growth Copilot API")
    print("=" * 60)

    print(f"Primary model : {PRIMARY_MODEL}")
    print(f"Fallback model: {FALLBACK_MODEL}")

    print("Plan limits:")
    print(f"  Free    : {PLAN_LIMITS['free']}")
    print(f"  Starter : {PLAN_LIMITS['starter']}")
    print(f"  Pro     : {PLAN_LIMITS['pro']}")

    print("Rate limit:")
    print(f"  {RATE_LIMIT_REQUESTS} requests / {RATE_LIMIT_WINDOW} seconds")

    print("=" * 60)


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
)
