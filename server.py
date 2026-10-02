import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types


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

MAX_PROMPT_LENGTH = 2000
MIN_PROMPT_LENGTH = 10

# Basic anti-abuse protection.
# This is intentionally lightweight because Render instances
# can restart and memory is not persistent.
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_REQUESTS = 15


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
        "The server will start, but generation will fail."
    )
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="AI-powered marketing growth copilot.",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "https://faceless-ai-website.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# SIMPLE RATE LIMITER
# ============================================================

request_log = defaultdict(deque)


def get_client_identifier(request: Request) -> str:
    """
    Gets a basic client identifier.

    This is NOT a replacement for proper authentication.
    It is only an additional layer against accidental abuse.
    """

    forwarded = request.headers.get("x-forwarded-for")

    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def check_rate_limit(request: Request):
    client_id = get_client_identifier(request)
    now = time.time()

    timestamps = request_log[client_id]

    # Remove old requests
    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW:
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
    def validate_mode(cls, value: str) -> str:
        value = value.strip().lower()

        if value not in VALID_MODES:
            raise ValueError(
                f"Invalid mode. Allowed modes: "
                f"{', '.join(sorted(VALID_MODES))}"
            )

        return value

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()

        if len(value) < MIN_PROMPT_LENGTH:
            raise ValueError(
                "Please provide a more detailed idea."
            )

        if len(value) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt cannot exceed {MAX_PROMPT_LENGTH} characters."
            )

        return value


# ============================================================
# MODE DEFINITIONS
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

If the user's idea is weak or unclear, improve its positioning
without pretending unsupported facts are true.
"""


# ============================================================
# GEMINI ERROR DETECTION
# ============================================================

def is_retryable_error(error: Exception) -> bool:
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

    return any(term in message for term in retryable_terms)


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
            "Please add GEMINI_API_KEY to Render environment variables."
        )

    mode_instruction = MODE_INSTRUCTIONS[mode]

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
        raise RuntimeError("Gemini returned an empty response.")

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned no usable text.")

    text = text.strip()

    if not text:
        raise RuntimeError("Gemini returned an empty result.")

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
        models.append(PRIMARY_MODEL)

    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        models.append(FALLBACK_MODEL)

    last_error: Optional[Exception] = None

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
                    "Generation failed | mode=%s | model=%s | "
                    "attempt=%s | error=%s",
                    mode,
                    model_name,
                    attempt + 1,
                    str(error)
                )

                if not is_retryable_error(error):
                    break

                if attempt < 2:
                    # 1.5s -> 3s
                    wait_time = 1.5 * (attempt + 1)
                    time.sleep(wait_time)

    if last_error:
        raise last_error

    raise RuntimeError(
        "No Gemini model was available."
    )


# ============================================================
# ERROR MESSAGE CLEANER
# ============================================================

def friendly_error_message(error: Exception) -> str:

    message = str(error)

    upper = message.upper()

    if "429" in upper or "RESOURCE_EXHAUSTED" in upper:
        return (
            "AI generation is temporarily busy. "
            "Please wait a little and try again."
        )

    if any(
        code in upper
        for code in ["500", "502", "503", "504", "UNAVAILABLE"]
    ):
        return (
            "The AI service is temporarily unavailable. "
            "Please try again in a moment."
        )

    if "API_KEY" in upper or "AUTH" in upper:
        return (
            "AI service configuration needs attention. "
            "Please contact support."
        )

    if "TIMEOUT" in upper or "DEADLINE" in upper:
        return (
            "The AI request took too long. "
            "Please try again."
        )

    return (
        "Something went wrong while generating your result. "
        "Please try again."
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "name": APP_NAME,
        "status": "online",
        "version": "2.0.0",
        "service": "AI Marketing Generation API",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": "2.0.0",
        "gemini_configured": client is not None,
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
async def generate(
    request: Request,
    data: GenerateRequest
):

    check_rate_limit(request)

    start_time = time.time()

    logger.info(
        "Generation request | mode=%s | prompt_length=%s",
        data.mode,
        len(data.prompt)
    )

    try:

        result = generate_ai_output(
            mode=data.mode,
            prompt=data.prompt
        )

        elapsed = round(
            time.time() - start_time,
            2
        )

        logger.info(
            "Generation complete | mode=%s | time=%ss",
            data.mode,
            elapsed
        )

        return {
            "success": True,
            "mode": data.mode,
            "result": result,
            "meta": {
                "processing_time": elapsed,
                "model": PRIMARY_MODEL,
            }
        }

    except Exception as error:

        logger.exception(
            "Generation error | mode=%s",
            data.mode
        )

        raise HTTPException(
            status_code=503,
            detail=friendly_error_message(error)
        )


# ============================================================
# LEGACY VIDEO ENDPOINT
# ============================================================

@app.post("/generate-video")
async def generate_video():
    raise HTTPException(
        status_code=410,
        detail=(
            "Video generation is no longer part of the "
            "FacelessAI Growth Copilot API."
        )
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    logger.info("=" * 60)
    logger.info("%s starting...", APP_NAME)
    logger.info("Primary model: %s", PRIMARY_MODEL)
    logger.info("Fallback model: %s", FALLBACK_MODEL)
    logger.info("Frontend: %s", FRONTEND_URL)
    logger.info(
        "Gemini configured: %s",
        client is not None
    )
    logger.info("=" * 60)


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False
)
