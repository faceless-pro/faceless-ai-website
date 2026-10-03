import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types
from supabase import create_client, Client


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

APP_NAME = "FacelessAI Growth Copilot"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash"
)

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://faceless-ai-website.vercel.app"
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

MAX_PROMPT_LENGTH = 2000
MIN_PROMPT_LENGTH = 10

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_REQUESTS = 15

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(APP_NAME)


# ============================================================
# GEMINI CLIENT
# ============================================================

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is missing. "
        "Server will start, but generation will fail."
    )
    client = None
else:
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception:
        logger.exception(
            "Could not initialize Gemini client."
        )
        client = None


# ============================================================
# SUPABASE CLIENT
# ============================================================

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    logger.warning(
        "Supabase environment variables are missing. "
        "Server will start, but authenticated generation "
        "will not work until they are configured."
    )
    supabase = None
else:
    try:
        supabase: Optional[Client] = create_client(
            SUPABASE_URL,
            SUPABASE_SECRET_KEY
        )
    except Exception:
        logger.exception(
            "Could not initialize Supabase client."
        )
        supabase = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="AI-powered marketing growth copilot.",
    version="3.0.0"
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    FRONTEND_URL,
    "https://faceless-ai-website.vercel.app",
]

allowed_origins = list(dict.fromkeys(
    origin.strip()
    for origin in allowed_origins
    if origin and origin.strip()
))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# RATE LIMITER
# ============================================================

request_log = defaultdict(deque)


def get_client_identifier(
    request: Request
) -> str:

    forwarded = request.headers.get(
        "x-forwarded-for"
    )

    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def check_rate_limit(
    request: Request
):

    client_id = get_client_identifier(
        request
    )

    now = time.time()

    timestamps = request_log[client_id]

    while (
        timestamps
        and now - timestamps[0] > RATE_LIMIT_WINDOW
    ):
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many requests. "
                "Please wait a moment and try again."
            )
        )

    timestamps.append(now)


# ============================================================
# AUTHENTICATION
# ============================================================

def get_bearer_token(
    authorization: Optional[str]
) -> str:

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header."
        )

    parts = authorization.strip().split(
        " ",
        1
    )

    if (
        len(parts) != 2
        or parts[0].lower() != "bearer"
        or not parts[1].strip()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid Bearer token."
        )

    return parts[1].strip()


def get_authenticated_user(
    authorization: Optional[str]
) -> Any:

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase authentication is not configured "
                "on the server."
            )
        )

    token = get_bearer_token(
        authorization
    )

    try:
        response = supabase.auth.get_user(
            token
        )

        user = response.user

        if not user or not user.id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired access token."
            )

        return user

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Supabase token verification failed."
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token."
        )


# ============================================================
# PLAN
# ============================================================

def get_user_plan(
    user_id: str
) -> str:

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured."
        )

    try:

        result = (
            supabase
            .table("profiles")
            .select("plan")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        data = result.data

        if not data:
            return "free"

        plan = str(
            data.get("plan") or "free"
        ).strip().lower()

        if plan not in PLAN_LIMITS:
            logger.warning(
                "Unknown plan '%s' for user %s. "
                "Using free plan.",
                plan,
                user_id
            )

            return "free"

        return plan

    except Exception:

        logger.exception(
            "Could not read profile for user %s.",
            user_id
        )

        raise HTTPException(
            status_code=500,
            detail="Could not read your account plan."
        )


# ============================================================
# MONTH
# ============================================================

def current_month() -> str:

    return time.strftime(
        "%Y-%m",
        time.gmtime()
    )


# ============================================================
# MONTHLY USAGE
# ============================================================

def get_monthly_usage(
    user_id: str
) -> int:

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured."
        )

    month = current_month()

    try:

        result = (
            supabase
            .table("monthly_usage")
            .select("generations")
            .eq("user_id", user_id)
            .eq("month", month)
            .maybe_single()
            .execute()
        )

        if not result.data:
            return 0

        return max(
            int(
                result.data.get(
                    "generations",
                    0
                )
                or 0
            ),
            0
        )

    except Exception:

        logger.exception(
            "Could not read monthly usage for %s.",
            user_id
        )

        raise HTTPException(
            status_code=500,
            detail="Could not read your monthly usage."
        )


def check_plan_limit(
    user_id: str,
    plan: str
) -> int:

    used = get_monthly_usage(
        user_id
    )

    limit = PLAN_LIMITS[plan]

    if used >= limit:

        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_limit_reached",
                "message": (
                    f"{plan.capitalize()} plan "
                    f"monthly generation limit reached."
                ),
                "plan": plan,
                "used": used,
                "limit": limit,
                "month": current_month(),
            }
        )

    return used


def increment_monthly_usage(
    user_id: str
) -> int:

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured."
        )

    month = current_month()

    try:

        existing = (
            supabase
            .table("monthly_usage")
            .select("id,generations")
            .eq("user_id", user_id)
            .eq("month", month)
            .maybe_single()
            .execute()
        )

        if existing.data:

            row_id = existing.data["id"]

            current = int(
                existing.data.get(
                    "generations",
                    0
                )
                or 0
            )

            new_count = current + 1

            (
                supabase
                .table("monthly_usage")
                .update({
                    "generations": new_count
                })
                .eq("id", row_id)
                .execute()
            )

            return new_count

        (
            supabase
            .table("monthly_usage")
            .insert({
                "user_id": user_id,
                "month": month,
                "generations": 1,
            })
            .execute()
        )

        return 1

    except Exception:

        logger.exception(
            "Could not increment monthly usage for %s.",
            user_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Generation succeeded, but usage "
                "could not be recorded."
            )
        )


# ============================================================
# REQUEST MODEL
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


class GenerateRequest(BaseModel):

    mode: str = Field(
        ...,
        min_length=1,
        max_length=50
    )

    prompt: str = Field(
        ...,
        min_length=MIN_PROMPT_LENGTH,
        max_length=MAX_PROMPT_LENGTH
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(
        cls,
        value: str
    ) -> str:

        value = value.strip().lower()

        if value not in VALID_MODES:
            raise ValueError(
                "Invalid mode. Allowed modes: "
                + ", ".join(
                    sorted(VALID_MODES)
                )
            )

        return value

    @field_validator("prompt")
    @classmethod
    def validate_prompt(
        cls,
        value: str
    ) -> str:

        value = value.strip()

        if len(value) < MIN_PROMPT_LENGTH:
            raise ValueError(
                "Please provide a more detailed idea."
            )

        if len(value) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt cannot exceed "
                f"{MAX_PROMPT_LENGTH} characters."
            )

        return value


# ============================================================
# MODE INSTRUCTIONS
# ============================================================

MODE_INSTRUCTIONS = {

    "offer": """
Create a strong, clear and commercially useful offer.

Include:
1. Core offer
2. Target customer
3. Main problem
4. Desired outcome
5. Unique value proposition
6. Offer structure
7. Pricing suggestion if enough information exists
8. Risk reversal or guarantee idea only when appropriate
9. Strong CTA

Do not invent testimonials, customer numbers,
revenue figures or fake proof.
""",

    "landing": """
Create conversion-focused landing page copy.

Include:
1. Hero headline
2. Supporting subheadline
3. Problem
4. Solution
5. Benefits
6. How it works
7. Features
8. Objection handling
9. CTA sections
10. FAQ
11. Final CTA

Make the copy ready to paste into a real website.
Do not use fake testimonials or fake statistics.
""",

    "email": """
Create a professional cold outreach email sequence.

Include:
1. Subject line options
2. Opening
3. Personalization angle
4. Problem
5. Value proposition
6. CTA
7. Follow-up 1
8. Follow-up 2

Keep it concise, natural and non-spammy.
Do not make unsupported claims.
""",

    "dm": """
Create a high-quality sales DM sequence.

Include:
1. First message
2. Follow-up
3. Value message
4. Objection response
5. Soft close
6. Final follow-up

Make it conversational rather than robotic.
Do not use fake urgency or fake social proof.
""",

    "ads": """
Create an advertising campaign concept.

Include:
1. Campaign angle
2. Target audience
3. Core message
4. 5 headline variations
5. 3 primary-text variations
6. 3 CTA variations
7. Creative concepts
8. Testing ideas

Do not claim guaranteed results.
""",

    "content": """
Create a practical content engine.

Include:
1. Content strategy
2. 10 content ideas
3. Hooks
4. Short-form post/video concepts
5. CTA ideas
6. Repurposing strategy

Make ideas specific to the user's audience.
Avoid generic filler.
""",

    "campaign": """
Build a complete coordinated marketing campaign from one idea.

Generate:

1. Offer
2. Positioning
3. Landing page messaging
4. Cold email
5. Sales DM
6. Ad campaign
7. Content ideas
8. Campaign sequence
9. CTA
10. Execution checklist

All assets must be connected to the same positioning,
audience and offer.

Do not create fake testimonials,
fake customer counts,
fake revenue,
fake logos,
fake case studies,
or guaranteed-result claims.

Make the result practical enough that a founder,
creator, freelancer or agency could actually use it.
"""
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the senior marketing strategist and growth copywriter
inside FacelessAI Growth Copilot.

Your job is to turn a user's rough business idea into
specific, useful marketing assets.

CORE RULES:

- Write in clear professional English.
- Be specific rather than generic.
- Optimize for usefulness and execution.
- Understand the user's audience before writing.
- Never invent testimonials.
- Never invent customer counts.
- Never invent revenue numbers.
- Never invent case studies.
- Never invent logos or partnerships.
- Never claim guaranteed viral growth.
- Never guarantee revenue.
- Never fabricate data.
- Do not use unnecessary emojis.
- Avoid repetitive filler.
- Do not begin with generic AI disclaimers.
- Use clean Markdown headings.
- Make outputs easy to copy and use.
- If important information is missing, make a reasonable
  clearly-labeled assumption rather than inventing facts.
- Prefer short paragraphs and useful bullet points.
- Make every section actionable.

QUALITY STANDARD:

The output should feel like it was prepared by an experienced
growth marketer for a real business, not like generic AI text.
"""


# ============================================================
# GEMINI ERROR DETECTION
# ============================================================

def is_retryable_error(
    error: Exception
) -> bool:

    message = str(error).upper()

    retryable_terms = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "TIMEOUT",
        "DEADLINE",
        "INTERNAL",
        "SERVICE_UNAVAILABLE",
        "TEMPORARY",
    ]

    return any(
        term in message
        for term in retryable_terms
    )


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_model(
    model_name: str,
    mode: str,
    prompt: str
) -> str:

    if client is None:

        raise RuntimeError(
            "Gemini API is not configured. "
            "Please add GEMINI_API_KEY "
            "to Render environment variables."
        )

    mode_instruction = MODE_INSTRUCTIONS[
        mode
    ]

    full_prompt = f"""
USER'S BUSINESS IDEA / REQUEST:

{prompt}

WORKFLOW:

{mode_instruction}

Now produce the final marketing output.

Remember:
- Stay specific.
- Keep everything connected to the user's idea.
- Do not fabricate proof.
- Do not promise guaranteed results.
- Make the result directly usable.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.75,
            max_output_tokens=5000,
        ),
    )

    if not response:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini returned no usable text."
        )

    text = text.strip()

    if not text:
        raise RuntimeError(
            "Gemini returned an empty result."
        )

    return text


# ============================================================
# GENERATION WITH RETRIES + FALLBACK
# ============================================================

def generate_ai_output(
    mode: str,
    prompt: str
) -> str:

    models = []

    if PRIMARY_MODEL:
        models.append(
            PRIMARY_MODEL
        )

    if (
        FALLBACK_MODEL
        and FALLBACK_MODEL != PRIMARY_MODEL
    ):
        models.append(
            FALLBACK_MODEL
        )

    last_error: Optional[
        Exception
    ] = None

    for model_name in models:

        for attempt in range(3):

            try:

                logger.info(
                    "Generating | mode=%s | model=%s | attempt=%s",
                    mode,
                    model_name,
                    attempt + 1
                )

                result = generate_with_model(
                    model_name=model_name,
                    mode=mode,
                    prompt=prompt
                )

                logger.info(
                    "Generation successful | mode=%s | model=%s",
                    mode,
                    model_name
                )

                return result

            except Exception as error:

                last_error = error

                logger.warning(
                    "Generation failed | mode=%s | "
                    "model=%s | attempt=%s | error=%s",
                    mode,
                    model_name,
                    attempt + 1,
                    str(error)
                )

                if not is_retryable_error(
                    error
                ):
                    break

                                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

    raise HTTPException(
        status_code=502,
        detail="AI generation failed. Please try again."
    )


@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "status": "online",
        "message": "FacelessAI backend is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "supabase_configured": bool(
            SUPABASE_URL and SUPABASE_SECRET_KEY
        ),
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL
    }


@app.post("/generate")
def generate(
    request: GenerateRequest,
    authorization: Optional[str] = Header(default=None)
):
    user = get_authenticated_user(authorization)
    user_id = str(user.id)

    plan = get_user_plan(user_id)

    if request.mode == "campaign" and plan != "pro":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "campaign_requires_pro",
                "message": "Campaign Builder is available on Pro only."
            }
        )

    used = check_plan_limit(user_id, plan)

    result = generate_with_gemini(
        mode=request.mode,
        prompt=request.prompt
    )

    new_usage = increment_monthly_usage(user_id)

    return {
        "success": True,
        "user_id": user_id,
        "plan": plan,
        "result": result,
        "usage": {
            "used": new_usage,
            "limit": PLAN_LIMITS[plan],
            "remaining": max(
                PLAN_LIMITS[plan] - new_usage,
                0
            )
        }
    }


@app.post("/generate-video")
def generate_video():
    raise HTTPException(
        status_code=410,
        detail="Video generation is temporarily disabled."
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
