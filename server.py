import os
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dodopayments import DodoPayments


# =========================================================
# CONFIG
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash",
)

DODO_PAYMENT_KEY = os.getenv("DODO_PAYMENT_KEY")
DODO_WEBHOOK_SECRET = os.getenv("DODO_WEBHOOK_SECRET")

DODO_STARTER_PRODUCT_ID = os.getenv(
    "DODO_STARTER_PRODUCT_ID",
    "pdt_0NoQXaM9rTHblSiCaXFqb",
)

DODO_PRO_PRODUCT_ID = os.getenv(
    "DODO_PRO_PRODUCT_ID",
    "pdt_0Nor4IRUyNeuBzfuxt6wI",
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("facelessai")


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="FacelessAI Growth Copilot API",
    version="6.0.0",
)


# Keep this simple because the frontend is currently public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# =========================================================
# CLIENTS
# =========================================================

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


dodo_client = None

if DODO_PAYMENT_KEY:
    dodo_client = DodoPayments(
        bearer_token=DODO_PAYMENT_KEY,
        webhook_key=DODO_WEBHOOK_SECRET,
    )


# =========================================================
# PLANS
# =========================================================

PLAN_LIMITS = {
    "free": 3,
    "starter": 100,
    "pro": 500,
}


PLAN_NAMES = {
    "free": "Free",
    "starter": "Starter",
    "pro": "Pro",
}


PRODUCT_TO_PLAN = {
    DODO_STARTER_PRODUCT_ID: "starter",
    DODO_PRO_PRODUCT_ID: "pro",
}


# =========================================================
# WORKFLOWS
# =========================================================

WORKFLOW_NAMES = {
    "offer": "Offer Builder",
    "landing": "Landing Page",
    "email": "Cold Email",
    "dm": "Sales DM",
    "ads": "Ad Campaign",
    "content": "Content Pack",
    "campaign": "Campaign Builder",
}


WORKFLOW_INSTRUCTIONS = {

    "offer": """
Create a clear, specific, compelling offer.

Return:
1. Offer name
2. Core promise
3. Target customer
4. Main pain point
5. Desired outcome
6. What is included
7. Why this offer is different
8. Suggested CTA

Keep it practical and conversion-focused.
""",

    "landing": """
Create conversion-focused landing-page copy.

Return:
1. Hero headline
2. Supporting subheadline
3. Problem
4. Solution
5. Benefits
6. How it works
7. Offer
8. Social-proof placeholder
9. FAQ
10. Final CTA

Do not invent fake testimonials, customer counts,
revenue numbers, guarantees, or credentials.
""",

    "email": """
Create a natural B2B cold-email sequence.

Return:
1. Subject line
2. Initial email
3. Follow-up 1
4. Follow-up 2
5. Final follow-up

Keep it concise, human and non-spammy.
Avoid fake personalization.
""",

    "dm": """
Create a natural sales DM sequence.

Return:
1. Opening message
2. Value message
3. Qualification question
4. Offer/message
5. Follow-up

Keep it conversational and not aggressive.
""",

    "ads": """
Create an ad campaign concept.

Return:
1. Campaign angle
2. Target audience
3. 5 hooks
4. 3 primary texts
5. 3 headlines
6. 3 CTAs
7. Creative concepts
8. Testing ideas

Do not invent performance claims.
""",

    "content": """
Create a useful content pack.

Return:
1. 10 content ideas
2. Hook for each
3. Short outline
4. CTA
5. Suggested platform

Make the ideas specific to the user's business.
""",

    "campaign": """
Create a complete campaign plan.

Return:
1. Campaign objective
2. Target audience
3. Offer
4. Messaging angle
5. Channel strategy
6. Content plan
7. Outreach plan
8. Ad concepts
9. CTA
10. 7-day execution plan

This workflow is available only to Pro.
""",
}


# =========================================================
# REQUEST MODEL
# =========================================================

class GenerateRequest(BaseModel):
    mode: str = Field(min_length=1, max_length=50)
    prompt: str = Field(min_length=1, max_length=2000)


# =========================================================
# ANONYMOUS USAGE
# =========================================================
#
# IMPORTANT:
# This is process memory only.
# It resets if Render restarts/redeploys and is not a
# persistent per-customer billing database.
#
# Since the frontend currently has NO login/device identity,
# there is no secure way to permanently associate a browser
# with a Dodo customer here.
#

anonymous_usage: dict[str, int] = {}


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_anonymous_key(request: Request) -> str:
    """
    Temporary anonymous bucket.

    We deliberately do NOT use a device-ID system or
    IP-rate-limit system here, based on the current
    architecture.

    This fallback is intentionally process-local.
    """

    # A single public bucket for the current anonymous setup.
    return "anonymous"


def get_usage(key: str) -> int:
    return anonymous_usage.get(key, 0)


def increment_usage(key: str) -> int:
    anonymous_usage[key] = get_usage(key) + 1
    return anonymous_usage[key]


# =========================================================
# DODO CUSTOMER ENTITLEMENTS
# =========================================================
#
# Keyed by Dodo customer_id, NOT by a global current_plan.
#
# This prevents one customer's payment from changing the
# plan for every other customer.
#

customer_entitlements: dict[str, dict[str, Any]] = {}


def set_customer_plan(
    customer_id: str,
    plan: str,
    subscription_id: str | None = None,
) -> None:

    if plan not in PLAN_LIMITS:
        return

    customer_entitlements[customer_id] = {
        "plan": plan,
        "subscription_id": subscription_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Dodo entitlement updated | customer=%s | plan=%s",
        customer_id,
        plan,
    )


def get_customer_plan(customer_id: str) -> str:
    record = customer_entitlements.get(customer_id)

    if not record:
        return "free"

    return record.get("plan", "free")


# =========================================================
# HELPERS
# =========================================================

def product_id_from_event(event_data: Any) -> str | None:

    product_cart = getattr(
        event_data,
        "product_cart",
        None,
    )

    if not product_cart:
        return None

    try:
        first_item = product_cart[0]
        return getattr(first_item, "product_id", None)
    except Exception:
        return None


def customer_id_from_event(event_data: Any) -> str | None:

    customer = getattr(
        event_data,
        "customer",
        None,
    )

    if customer:
        return getattr(
            customer,
            "customer_id",
            None,
        )

    return getattr(
        event_data,
        "customer_id",
        None,
    )


def subscription_id_from_event(event_data: Any) -> str | None:

    return getattr(
        event_data,
        "subscription_id",
        None,
    )


def plan_from_product(product_id: str | None) -> str | None:

    if not product_id:
        return None

    return PRODUCT_TO_PLAN.get(product_id)


# =========================================================
# DODO WEBHOOK
# =========================================================

@app.post("/webhook/dodo")
async def dodo_webhook(request: Request):

    if not dodo_client:
        logger.error("Dodo client is not configured.")
        raise HTTPException(
            status_code=500,
            detail="Dodo webhook is not configured.",
        )

    # IMPORTANT:
    # Read RAW bytes. Do not request.json() before verification.
    raw_body = await request.body()

    headers = {
        "webhook-id": request.headers.get(
            "webhook-id",
            "",
        ),
        "webhook-signature": request.headers.get(
            "webhook-signature",
            "",
        ),
        "webhook-timestamp": request.headers.get(
            "webhook-timestamp",
            "",
        ),
    }

    if not all(headers.values()):
        logger.warning(
            "Dodo webhook rejected: missing signature headers."
        )

        raise HTTPException(
            status_code=400,
            detail="Missing webhook signature headers.",
        )

    try:

        event = dodo_client.webhooks.unwrap(
            raw_body.decode("utf-8"),
            headers=headers,
        )

    except Exception as error:

        logger.warning(
            "Dodo webhook signature verification failed: %s",
            str(error),
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature.",
        )

    event_type = getattr(
        event,
        "type",
        None,
    )

    event_data = getattr(
        event,
        "data",
        None,
    )

    logger.info(
        "Verified Dodo webhook | type=%s",
        event_type,
    )

    # -----------------------------------------------------
    # PAYMENT SUCCEEDED
    # -----------------------------------------------------

    if event_type == "payment.succeeded":

        customer_id = customer_id_from_event(
            event_data
        )

        product_id = product_id_from_event(
            event_data
        )

        plan = plan_from_product(
            product_id
        )

        status = getattr(
            event_data,
            "status",
            None,
        )

        if (
            customer_id
            and plan
            and status == "succeeded"
        ):

            set_customer_plan(
                customer_id=customer_id,
                plan=plan,
                subscription_id=subscription_id_from_event(
                    event_data
                ),
            )

            logger.info(
                "Payment succeeded | customer=%s | product=%s | plan=%s",
                customer_id,
                product_id,
                plan,
            )

    # -----------------------------------------------------
    # SUBSCRIPTION ACTIVE / RENEWED
    # -----------------------------------------------------

    elif event_type in {
        "subscription.active",
        "subscription.renewed",
        "subscription.updated",
        "subscription.plan_changed",
        "subscription.unpaused",
    }:

        customer_id = customer_id_from_event(
            event_data
        )

        product_id = getattr(
            event_data,
            "product_id",
            None,
        )

        plan = plan_from_product(
            product_id
        )

        if customer_id and plan:

            set_customer_plan(
                customer_id=customer_id,
                plan=plan,
                subscription_id=subscription_id_from_event(
                    event_data
                ),
            )

    # -----------------------------------------------------
    # SUBSCRIPTION CANCELLED / EXPIRED / FAILED / PAUSED
    # -----------------------------------------------------

    elif event_type in {
        "subscription.cancelled",
        "subscription.expired",
        "subscription.failed",
        "subscription.paused",
        "subscription.on_hold",
        "subscription.past_due",
    }:

        customer_id = customer_id_from_event(
            event_data
        )

        if customer_id:

            customer_entitlements.pop(
                customer_id,
                None,
            )

            logger.info(
                "Dodo entitlement removed | customer=%s | event=%s",
                customer_id,
                event_type,
            )

    # -----------------------------------------------------
    # REFUND
    # -----------------------------------------------------

    elif event_type in {
        "refund.succeeded",
        "refund.failed",
    }:

        logger.info(
            "Dodo refund event received | type=%s",
            event_type,
        )

    # -----------------------------------------------------
    # OTHER PAYMENT EVENTS
    # -----------------------------------------------------

    elif event_type in {
        "payment.failed",
        "payment.processing",
        "payment.cancelled",
    }:

        logger.info(
            "Dodo payment status event | type=%s",
            event_type,
        )

    return {
        "received": True,
        "type": event_type,
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
async def root():

    return {
        "service": "FacelessAI Growth Copilot API",
        "status": "running",
        "version": "6.0.0",
        "primary_model": GEMINI_MODEL,
        "fallback_model": GEMINI_FALLBACK_MODEL,
    }


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "dodo_configured": bool(DODO_PAYMENT_KEY),
        "webhook_configured": bool(DODO_WEBHOOK_SECRET),
    }


# =========================================================
# GEMINI GENERATION
# =========================================================

def build_ai_prompt(
    mode: str,
    user_prompt: str,
) -> str:

    workflow_name = WORKFLOW_NAMES[mode]
    instructions = WORKFLOW_INSTRUCTIONS[mode]

    return f"""
You are FacelessAI Growth Copilot.

Workflow:
{workflow_name}

User's business/request:
{user_prompt}

Instructions:
{instructions}

Rules:
- Give useful, specific output.
- Do not invent facts about the user's business.
- Do not invent testimonials, statistics, revenue numbers,
  customers, certifications, or guarantees.
- Use the information supplied by the user.
- Avoid unnecessary filler.
- Make the output easy to copy and use.
"""


def generate_with_model(
    model_name: str,
    prompt: str,
) -> str:

    if not gemini_client:
        raise RuntimeError(
            "Gemini API is not configured."
        )

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=5000,
        ),
    )

    text = getattr(
        response,
        "text",
        None,
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


def generate_ai_output(prompt: str) -> str:

    try:

        return generate_with_model(
            GEMINI_MODEL,
            prompt,
        )

    except Exception as primary_error:

        logger.warning(
            "Primary Gemini model failed | model=%s | error=%s",
            GEMINI_MODEL,
            str(primary_error),
        )

        if (
            GEMINI_FALLBACK_MODEL
            and GEMINI_FALLBACK_MODEL != GEMINI_MODEL
        ):

            return generate_with_model(
                GEMINI_FALLBACK_MODEL,
                prompt,
            )

        raise


# =========================================================
# GENERATE
# =========================================================

@app.post("/generate")
async def generate(
    payload: GenerateRequest,
    request: Request,
):

    mode = payload.mode.strip().lower()
    user_prompt = payload.prompt.strip()

    if mode not in WORKFLOW_NAMES:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_mode",
                "message": "Unknown workflow mode.",
            },
        )

    if not user_prompt:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_prompt",
                "message": "Prompt cannot be empty.",
            },
        )

    # Campaign Builder is Pro only.
    #
    # Because the current frontend has no authentication or
    # customer identity, this endpoint cannot securely know
    # that an anonymous browser belongs to a Dodo Pro customer.
    #
    # Do not trust a plan value sent from the browser.
    #
    if mode == "campaign":

        raise HTTPException(
            status_code=403,
            detail={
                "error": "pro_required",
                "message": (
                    "Campaign Builder is available "
                    "on the Pro plan."
                ),
                "plan": "free",
                "plan_name": "Free",
                "upgrade_plan": "pro",
                "used": 0,
                "limit": PLAN_LIMITS["free"],
                "remaining": PLAN_LIMITS["free"],
            },
        )

    anonymous_key = get_anonymous_key(request)

    used_before = get_usage(
        anonymous_key
    )

    limit = PLAN_LIMITS["free"]

    if used_before >= limit:

        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_limit_reached",
                "message": (
                    "You've reached your Free "
                    "monthly limit."
                ),
                "plan": "free",
                "plan_name": "Free",
                "used": used_before,
                "limit": limit,
                "remaining": 0,
                "upgrade_plan": "starter",
                "reset_month": current_month(),
            },
        )

    ai_prompt = build_ai_prompt(
        mode=mode,
        user_prompt=user_prompt,
    )

    try:

        result = generate_ai_output(
            ai_prompt
        )

    except Exception as error:

        logger.exception(
            "Generation failed | mode=%s | error=%s",
            mode,
            str(error),
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "ai_unavailable",
                "message": (
                    "AI generation is temporarily "
                    "unavailable. Please try again."
                ),
            },
        )

    used = increment_usage(
        anonymous_key
    )

    remaining = max(
        limit - used,
        0,
    )

    return {
        "success": True,
        "result": result,
        "mode": mode,
        "plan": "free",
        "usage": {
            "used": used,
            "limit": limit,
            "remaining": remaining,
        },
}
