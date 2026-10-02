import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Deque

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# ENV
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash"
)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="FacelessAI Growth Copilot API",
    version="3.0.0"
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
# PLANS
# ============================================================

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}


# ============================================================
# REQUEST
# ============================================================

class GenerateRequest(BaseModel):

    mode: str = Field(
        ...,
        min_length=1,
        max_length=50
    )

    niche: str = Field(
        default="Other",
        min_length=1,
        max_length=50
    )

    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000
    )


# ============================================================
# MODES
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
Create a strong, realistic offer.

Include:
1. Offer name
2. Target customer
3. Core problem
4. Desired outcome
5. Value proposition
6. Deliverables
7. Pricing/positioning suggestion when enough information exists
8. Objection handling
9. CTA

Never invent testimonials, revenue, customer numbers,
logos, awards, reviews or guaranteed results.
""",

    "landing": """
Create structured landing-page copy.

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

Keep all claims realistic.
""",

    "email": """
Create a concise professional cold-email sequence.

Include:
1. Subject
2. Opening
3. Problem
4. Value proposition
5. Offer
6. CTA
7. Follow-up

Never pretend to know information that was not provided.
""",

    "dm": """
Create a natural sales DM sequence.

Include:
1. First message
2. Follow-up
3. Objection response
4. CTA

Avoid spammy language and fake urgency.
""",

    "ads": """
Create an advertising campaign concept.

Include:
1. Campaign angle
2. Target audience
3. Hook
4. Primary copy
5. Headline
6. CTA
7. Creative concept
8. Variations

Never invent statistics or guaranteed results.
""",

    "content": """
Create a practical content engine.

Include:
1. Content pillars
2. 10 content ideas
3. Hooks
4. Short-form concepts
5. CTA ideas
6. Posting strategy
7. Repurposing ideas

Do not promise virality.
""",

    "campaign": """
Create a complete marketing campaign.

Include:
1. Objective
2. Audience
3. Offer
4. Positioning
5. Landing-page angle
6. Cold email
7. Sales DM
8. Ad concepts
9. Organic content
10. CTA
11. 7-day execution plan
12. Measurement framework

Do not invent performance data.
""",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are FacelessAI Growth Copilot.

Turn the user's actual idea into practical marketing assets.

Rules:

- Never fabricate testimonials.
- Never fabricate customer numbers.
- Never fabricate revenue.
- Never fabricate logos.
- Never fabricate awards.
- Never fabricate reviews.
- Never guarantee revenue.
- Never guarantee leads.
- Never guarantee sales.
- Never guarantee virality.
- Never use fake scarcity.
- Never use deceptive marketing.
- If information is missing, clearly label assumptions.
- Prefer specific useful output over generic advice.
- Use clear headings.
- Write naturally.
"""


# ============================================================
# RATE LIMIT
# ============================================================

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_REQUESTS = 15

request_log: Dict[
    str,
    Deque[float]
] = defaultdict(deque)


def get_client_ip(
    request: Request
) -> str:

    forwarded_for = request.headers.get(
        "x-forwarded-for"
    )

    if forwarded_for:

        return forwarded_for.split(",")[0].strip()

    if request.client:

        return request.client.host

    return "unknown"


def check_rate_limit(
    ip: str
):

    now = time.time()

    timestamps = request_log[ip]

    while (
        timestamps
        and
        now - timestamps[0]
        > RATE_LIMIT_WINDOW
    ):

        timestamps.popleft()

    if (
        len(timestamps)
        >= RATE_LIMIT_REQUESTS
    ):

        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message":
                    "Too many requests. Please wait a minute."
            }
        )

    timestamps.append(now)


# ============================================================
# TEMPORARY USAGE
# ============================================================
#
# IMPORTANT:
# This is only the temporary Free-plan fallback.
# Production paid subscriptions must use Supabase.
#

usage_store = {}


def current_month():

    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m")


def get_free_usage(
    ip: str
):

    month = current_month()

    record = usage_store.get(ip)

    if not record:

        usage_store[ip] = {
            "month": month,
            "used": 0,
        }

        return 0

    if record["month"] != month:

        usage_store[ip] = {
            "month": month,
            "used": 0,
        }

        return 0

    return int(
        record["used"]
    )


def increment_free_usage(
    ip: str
):

    month = current_month()

    record = usage_store.get(ip)

    if (
        not record
        or
        record["month"] != month
    ):

        usage_store[ip] = {
            "month": month,
            "used": 1,
        }

        return 1

    record["used"] += 1

    return record["used"]


# ============================================================
# GEMINI
# ============================================================

def is_retryable_error(
    error: Exception
):

    message = str(error).lower()

    retryable = [
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

    return any(
        word in message
        for word in retryable
    )


def generate_with_model(
    model_name: str,
    mode: str,
    niche: str,
    user_prompt: str,
):

    full_prompt = f"""
{SYSTEM_PROMPT}

MODE:
{mode}

NICHE:
{niche}

MODE INSTRUCTIONS:
{MODE_INSTRUCTIONS[mode]}

USER REQUEST:
{user_prompt}

Return a polished answer the user can directly use.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.75,
            max_output_tokens=5000,
        ),
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


def generate_ai_result(
    mode: str,
    niche: str,
    user_prompt: str,
):

    models = [
        PRIMARY_MODEL,
        FALLBACK_MODEL,
    ]

    last_error = None

    for model_name in models:

        if not model_name:
            continue

        for attempt in range(3):

            try:

                result = generate_with_model(
                    model_name=model_name,
                    mode=mode,
                    niche=niche,
                    user_prompt=user_prompt,
                )

                if result:

                    return result

            except Exception as error:

                last_error = error

                print(
                    f"[Gemini Error] "
                    f"{model_name} "
                    f"attempt={attempt + 1}/3: "
                    f"{error}"
                )

                if not is_retryable_error(
                    error
                ):

                    break

                time.sleep(
                    1.5 * (attempt + 1)
                )

    raise RuntimeError(
        f"AI generation failed: {last_error}"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name":
            "FacelessAI Growth Copilot API",

        "status":
            "online",

        "version":
            "3.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status":
            "ok",

        "primary_model":
            PRIMARY_MODEL,

        "fallback_model":
            FALLBACK_MODEL,

        "plans":
            PLAN_LIMITS,
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
async def generate(
    payload: GenerateRequest,
    request: Request,
):

    ip = get_client_ip(request)

    check_rate_limit(ip)

    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------

    mode = payload.mode.strip().lower()

    if mode not in VALID_MODES:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "invalid_mode",

                "message":
                    f"Invalid mode '{mode}'."
            }
        )

    # --------------------------------------------------------
    # NICHE
    # --------------------------------------------------------

    niche = payload.niche.strip()

    if not niche:

        niche = "Other"

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    user_prompt = payload.prompt.strip()

    if len(user_prompt) < 10:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "prompt_too_short",

                "message":
                    "Please enter at least 10 characters."
            }
        )

    # --------------------------------------------------------
    # TEMPORARY FREE LIMIT
    # --------------------------------------------------------

    used_before = get_free_usage(ip)

    if used_before >= PLAN_LIMITS["free"]:

        raise HTTPException(
            status_code=403,
            detail={
                "error":
                    "monthly_limit_reached",

                "message":
                    "You have used all 3 Free generations this month.",

                "plan":
                    "free",

                "used":
                    used_before,

                "limit":
                    3,

                "remaining":
                    0,
            }
        )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    try:

        result = generate_ai_result(
            mode=mode,
            niche=niche,
            user_prompt=user_prompt,
        )

    except Exception as error:

        print(
            f"[Generation Failed] {error}"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "error":
                    "generation_failed",

                "message":
                    "AI generation is temporarily unavailable. Please try again."
            }
        )

    # --------------------------------------------------------
    # COUNT SUCCESS
    # --------------------------------------------------------

    used_after = increment_free_usage(ip)

    remaining = max(
        PLAN_LIMITS["free"]
        - used_after,
        0
    )

    return {

        "success":
            True,

        "result":
            result,

        "mode":
            mode,

        "plan":
            "free",

        "usage": {

            "used":
                used_after,

            "limit":
                PLAN_LIMITS["free"],

            "remaining":
                remaining,

            "month":
                current_month(),
        },

        "message":
            f"{remaining} generation(s) remaining "
            "on your Free plan this month.",
    }


# ============================================================
# OLD VIDEO ENDPOINT
# ============================================================

@app.post("/generate-video")
async def generate_video():

    raise HTTPException(
        status_code=410,
        detail={
            "error":
                "endpoint_removed",

            "message":
                "Video generation is no longer part of "
                "FacelessAI Growth Copilot."
        }
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)

    print(
        "FacelessAI Growth Copilot API"
    )

    print("=" * 60)

    print(
        f"Primary model : {PRIMARY_MODEL}"
    )

    print(
        f"Fallback model: {FALLBACK_MODEL}"
    )

    print(
        f"Free limit    : {PLAN_LIMITS['free']}"
    )

    print(
        f"Starter limit : {PLAN_LIMITS['starter']}"
    )

    print(
        f"Pro limit     : {PLAN_LIMITS['pro']}"
    )

    print(
        f"Rate limit    : "
        f"{RATE_LIMIT_REQUESTS}/"
        f"{RATE_LIMIT_WINDOW}s"
    )

    print("=" * 60)


# ============================================================
# LOCAL
# ============================================================

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
        ),
        reload=False,
        )
