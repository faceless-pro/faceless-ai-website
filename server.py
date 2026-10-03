import os
import time
import random
from collections import defaultdict, deque
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash"
).strip()

IP_RATE_LIMIT = int(os.getenv("IP_RATE_LIMIT", "15"))
IP_RATE_WINDOW = int(os.getenv("IP_RATE_WINDOW", "60"))

USER_RATE_LIMIT = int(os.getenv("USER_RATE_LIMIT", "8"))
USER_RATE_WINDOW = int(os.getenv("USER_RATE_WINDOW", "60"))


# ============================================================
# PLAN LIMITS
# ============================================================

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}


# ============================================================
# VALID WORKFLOWS
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
# REQUIRED ENVIRONMENT
# ============================================================

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY is missing")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is missing")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="FacelessAI Growth Copilot API",
    version="5.0.0"
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
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# RATE LIMIT STORAGE
# ============================================================

ip_requests = defaultdict(deque)
user_requests = defaultdict(deque)


def check_rate_limit(store, key, limit, window):
    now = time.time()
    bucket = store[key]

    while bucket and now - bucket[0] > window:
        bucket.popleft()

    if len(bucket) >= limit:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment and try again."
        )

    bucket.append(now)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")

    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


# ============================================================
# AUTHENTICATION
# ============================================================

def get_bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Login required. Please log in before generating."
        )

    token = authorization[7:].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token."
        )

    return token


def get_authenticated_user(request: Request) -> dict:
    token = get_bearer_token(request)

    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        print(
            f"AUTH SERVICE ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="Authentication service is temporarily unavailable."
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=401,
            detail="Your login session is invalid or expired. Please log in again."
        )

    try:
        user = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Invalid authentication response."
        ) from exc

    user_id = user.get("id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid user session."
        )

    return user


# ============================================================
# SUPABASE HELPERS
# ============================================================

def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def get_user_plan(user_id: str) -> str:
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={
                "id": f"eq.{user_id}",
                "select": "plan",
                "limit": "1",
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        print(
            f"PLAN LOOKUP ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="Account service is temporarily unavailable."
        ) from exc

    if response.status_code != 200:
        print(
            f"PLAN LOOKUP HTTP {response.status_code}: {response.text}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="Could not verify your account plan."
        )

    try:
        rows = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Invalid account response."
        ) from exc

    if not rows:
        return "free"

    plan = str(rows[0].get("plan", "free")).lower()

    if plan not in PLAN_LIMITS:
        return "free"

    return plan


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_monthly_usage(user_id: str) -> int:
    month = current_month()

    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/monthly_usage",
            headers=supabase_headers(),
            params={
                "user_id": f"eq.{user_id}",
                "month": f"eq.{month}",
                "select": "used",
                "limit": "1",
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Usage service is temporarily unavailable."
        ) from exc

    if response.status_code != 200:
        print(
            f"USAGE READ ERROR {response.status_code}: {response.text}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="Could not verify usage."
        )

    rows = response.json()

    if not rows:
        return 0

    return int(rows[0].get("used", 0))


def increment_monthly_usage(user_id: str) -> int:
    month = current_month()

    current_used = get_monthly_usage(user_id)
    new_used = current_used + 1

    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/monthly_usage",
            headers={
                **supabase_headers(),
                "Prefer": "resolution=merge-duplicates,return=representation",
            },
            json={
                "user_id": user_id,
                "month": month,
                "used": new_used,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Usage service is temporarily unavailable."
        ) from exc

    if response.status_code not in (200, 201):
        print(
            f"USAGE WRITE ERROR {response.status_code}: {response.text}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="Could not update usage."
        )

    return new_used


def check_and_consume_usage(user_id: str, plan: str):
    limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

    used = get_monthly_usage(user_id)

    if used >= limit:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Monthly {plan} limit reached: "
                f"{used}/{limit} generations used. "
                "Your allowance resets next month."
            ),
        )

    new_used = increment_monthly_usage(user_id)

    return new_used, limit


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):
    mode: str = Field(..., min_length=1, max_length=30)
    prompt: str = Field(..., min_length=10, max_length=2000)


# ============================================================
# WORKFLOW INSTRUCTIONS
# ============================================================

WORKFLOW_INSTRUCTIONS = {

    "offer": """
Create a strong, specific business offer.

Include:
- Clear offer name
- Target customer
- Core problem
- Desired outcome
- Main value proposition
- Deliverables
- Differentiation
- Pricing/packaging suggestion if useful
- Risk reversal or guarantee idea only if appropriate
- Strong CTA

Make it practical and easy to understand.
Do not invent fake testimonials, fake customers, fake statistics, or guaranteed results.
""",

    "landing": """
Create conversion-focused landing page copy.

Include:
- Hero headline
- Subheadline
- Problem
- Solution
- Benefits
- How it works
- Features where useful
- Objection handling
- Social-proof placeholders only when appropriate
- CTA
- FAQ ideas

Make the copy specific to the user's business.
Do not invent fake testimonials, fake statistics, or fake claims.
""",

    "email": """
Create a natural B2B cold email sequence.

Include:
- Subject line options
- Initial email
- Follow-up 1
- Follow-up 2
- Simple CTA

Keep it concise, human and non-spammy.
Avoid manipulative claims.
Personalization should be based only on information provided by the user.
""",

    "dm": """
Create a conversational sales DM sequence.

Include:
- Opening message
- Follow-up
- Value message
- Soft CTA
- Optional final follow-up

Make it natural and not pushy.
Avoid spammy language and unrealistic claims.
""",

    "ads": """
Create a practical paid-ad campaign concept.

Include:
- Campaign objective
- Target audience
- 3-5 hooks
- Primary text variations
- Headline variations
- CTA options
- Creative concepts
- Testing ideas

Do not claim guaranteed performance.
Do not invent performance statistics.
""",

    "content": """
Create a practical content pack.

Include:
- Content strategy
- Strong hooks
- Post/video ideas
- Short explanations
- CTA ideas
- A 7-day content outline

Make the content useful rather than generic.
Keep it relevant to the user's business and audience.
""",

    "campaign": """
Create a coordinated marketing campaign from the user's brief.

Include:
- Campaign positioning
- Offer
- Landing-page angle
- Cold-email angle
- Sales-DM angle
- Ad angles
- Content ideas
- CTA
- Suggested execution order

Make the assets consistent with one another.
Do not invent fake proof, statistics or guaranteed outcomes.
""",
}


# ============================================================
# GEMINI ERROR DETECTION
# ============================================================

def is_temporary_error(error: Exception) -> bool:
    text = str(error).upper()

    temporary_signals = [
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
    ]

    return any(signal in text for signal in temporary_signals)


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_marketing_asset(mode: str, user_prompt: str) -> str:

    workflow = WORKFLOW_INSTRUCTIONS.get(mode)

    if not workflow:
        raise HTTPException(
            status_code=400,
            detail="Unsupported workflow."
        )

    system_prompt = f"""
You are the AI engine for FacelessAI Growth Copilot.

FacelessAI helps creators, freelancers, agencies and businesses
turn a business idea into useful marketing assets.

Current workflow:
{mode}

Workflow requirements:
{workflow}

User brief:
{user_prompt}

General requirements:

- Be specific.
- Be practical.
- Write copy that can actually be used.
- Understand the user's business before writing.
- Avoid generic filler.
- Do not mention that you are an AI unless necessary.
- Do not invent testimonials.
- Do not invent customer counts.
- Do not invent revenue figures.
- Do not invent case studies.
- Do not guarantee sales, leads, virality or conversion rates.
- Use clear headings.
- Use concise sections.
- Use Markdown.
- Return only the finished marketing asset.
"""

    models = [
        PRIMARY_MODEL,
        FALLBACK_MODEL,
    ]

    errors = []

    for model in models:

        for attempt in range(3):

            try:
                print(
                    f"GENERATION: model={model} "
                    f"attempt={attempt + 1}/3 "
                    f"mode={mode}",
                    flush=True,
                )

                response = gemini_client.models.generate_content(
                    model=model,
                    contents=system_prompt,
                )

                result = (response.text or "").strip()

                if not result:
                    raise RuntimeError(
                        f"{model} returned an empty response."
                    )

                print(
                    f"GENERATION SUCCESS: model={model} "
                    f"characters={len(result)}",
                    flush=True,
                )

                return result

            except Exception as exc:

                print(
                    f"GENERATION ERROR: "
                    f"model={model} "
                    f"attempt={attempt + 1} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

                errors.append(
                    f"{model}: {type(exc).__name__}: {exc}"
                )

                if not is_temporary_error(exc):
                    break

                if attempt < 2:
                    delay = (2 ** attempt) + random.uniform(
                        0.3,
                        1.0
                    )

                    time.sleep(delay)

    raise RuntimeError(
        "Gemini could not generate the requested asset. "
        + " | ".join(errors[-4:])
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "service": "FacelessAI Growth Copilot API",
        "status": "running",
        "version": "5.0.0",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "plans": {
            "free": 3,
            "starter": 100,
            "pro": 500,
        },
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
def generate(
    request: Request,
    payload: GenerateRequest,
):

    print(
        f"REQUEST /generate mode={payload.mode}",
        flush=True,
    )

    # --------------------------------------------------------
    # IP RATE LIMIT
    # --------------------------------------------------------

    ip = get_client_ip(request)

    check_rate_limit(
        ip_requests,
        ip,
        IP_RATE_LIMIT,
        IP_RATE_WINDOW,
    )

    # --------------------------------------------------------
    # AUTH
    # --------------------------------------------------------

    user = get_authenticated_user(request)

    user_id = user["id"]

    # --------------------------------------------------------
    # USER RATE LIMIT
    # --------------------------------------------------------

    check_rate_limit(
        user_requests,
        user_id,
        USER_RATE_LIMIT,
        USER_RATE_WINDOW,
    )

    # --------------------------------------------------------
    # VALIDATE MODE
    # --------------------------------------------------------

    mode = payload.mode.strip().lower()

    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid workflow. Supported workflows: "
                + ", ".join(sorted(VALID_MODES))
            ),
        )

    # --------------------------------------------------------
    # VALIDATE PROMPT
    # --------------------------------------------------------

    prompt = payload.prompt.strip()

    if len(prompt) < 10:
        raise HTTPException(
            status_code=400,
            detail="Please provide a little more detail."
        )

    if len(prompt) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Prompt must be 2000 characters or less."
        )

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    plan = get_user_plan(user_id)

    print(
        f"AUTH OK user={user_id[:8]}... "
        f"plan={plan} mode={mode}",
        flush=True,
    )

    # --------------------------------------------------------
    # PRO CAMPAIGN BUILDER
    # --------------------------------------------------------

    if mode == "campaign" and plan != "pro":
        raise HTTPException(
            status_code=403,
            detail=(
                "Campaign Builder is available on the Pro plan. "
                "Upgrade to Pro to use it."
            ),
        )

    # --------------------------------------------------------
    # USAGE
    # --------------------------------------------------------

    used, limit = check_and_consume_usage(
        user_id,
        plan,
    )

    print(
        f"USAGE user={user_id[:8]}... "
        f"plan={plan} "
        f"used={used}/{limit}",
        flush=True,
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    try:

        result = generate_marketing_asset(
            mode,
            prompt,
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"FINAL GENERATION ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The AI could not generate the asset right now. "
                "Please try again."
            ),
        ) from exc

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "result": result,
        "mode": mode,
        "plan": plan,
        "usage": {
            "used": used,
            "limit": limit,
            "remaining": max(limit - used, 0),
        },
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 60)
    print("FacelessAI Growth Copilot API")
    print("=" * 60)
    print(f"Primary model : {PRIMARY_MODEL}")
    print(f"Fallback model: {FALLBACK_MODEL}")
    print("Plans         : Free 3 | Starter 100 | Pro 500")
    print("Endpoint      : POST /generate")
    print("Health        : GET /health")
    print("=" * 60)
