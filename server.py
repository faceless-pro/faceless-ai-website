import os
import logging
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dodopayments import DodoPayments


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("facelessai")


# =========================================================
# ENVIRONMENT
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash",
)

DODO_PAYMENT_KEY = os.getenv("DODO_PAYMENT_KEY")
DODO_WEBHOOK_SECRET = os.getenv("DODO_WEBHOOK_SECRET")

DODO_STARTER_PRODUCT_ID = os.getenv(
    "DODO_STARTER_PRODUCT_ID"
)

DODO_PRO_PRODUCT_ID = os.getenv(
    "DODO_PRO_PRODUCT_ID"
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="FacelessAI Growth Copilot API",
    version="8.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CLIENTS
# =========================================================

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

dodo_client = None

if DODO_PAYMENT_KEY:
    try:
        dodo_client = DodoPayments(
            bearer_token=DODO_PAYMENT_KEY,
            webhook_key=DODO_WEBHOOK_SECRET,
        )

        logger.info("Dodo client initialized")

    except Exception:
        logger.exception(
            "Failed to initialize Dodo client"
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

VALID_PLANS = {
    "free",
    "starter",
    "pro",
}

VALID_MODES = {
    "offer",
    "landing",
    "email",
    "dm",
    "ads",
    "content",
    "campaign",
}

PRO_ONLY_MODES = {
    "campaign",
}


# =========================================================
# PLAN HELPERS
# =========================================================

def normalize_plan(plan):
    if not plan:
        return "free"

    plan = str(plan).strip().lower()

    if plan not in VALID_PLANS:
        return "free"

    return plan


def get_limit(plan):
    plan = normalize_plan(plan)
    return PLAN_LIMITS[plan]


def product_to_plan(product_id):
    if product_id == DODO_STARTER_PRODUCT_ID:
        return "starter"

    if product_id == DODO_PRO_PRODUCT_ID:
        return "pro"

    return "free"


# =========================================================
# SUPABASE HELPERS
# =========================================================

def sb_headers():
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY is not configured"
        )

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": (
            f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
        ),
        "Content-Type": "application/json",
    }


def sb_request(method, table, **kwargs):
    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured"
        )

    url = (
        f"{SUPABASE_URL.rstrip('/')}"
        f"/rest/v1/{table}"
    )

    headers = sb_headers()

    custom_headers = kwargs.pop(
        "headers",
        None,
    )

    if custom_headers:
        headers.update(custom_headers)

    response = requests.request(
        method,
        url,
        headers=headers,
        timeout=15,
        **kwargs,
    )

    if not response.ok:
        logger.error(
            "Supabase request failed | "
            "method=%s | table=%s | status=%s | body=%s",
            method,
            table,
            response.status_code,
            response.text[:1000],
        )

        response.raise_for_status()

    return response


# =========================================================
# CUSTOMER HELPERS
# =========================================================

def get_customer(customer_id):
    response = sb_request(
        "GET",
        "customers",
        params={
            "dodo_customer_id": f"eq.{customer_id}",
            "select": "*",
            "limit": "1",
        },
    )

    rows = response.json()

    if not rows:
        return None

    return rows[0]


def save_customer(
    customer_id,
    email=None,
    plan="free",
    subscription_id=None,
    subscription_status=None,
):
    plan = normalize_plan(plan)

    existing = get_customer(customer_id)

    payload = {
        "dodo_customer_id": customer_id,
        "email": email,
        "plan": plan,
        "subscription_id": subscription_id,
        "subscription_status": subscription_status,
    }

    if existing:
        sb_request(
            "PATCH",
            "customers",
            params={
                "dodo_customer_id": (
                    f"eq.{customer_id}"
                ),
            },
            json=payload,
        )

    else:
        sb_request(
            "POST",
            "customers",
            json=payload,
        )


# =========================================================
# USAGE HELPERS
# =========================================================

def current_month():
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m")


def get_usage(customer_id):
    month = current_month()

    response = sb_request(
        "GET",
        "usage",
        params={
            "dodo_customer_id": (
                f"eq.{customer_id}"
            ),
            "month": f"eq.{month}",
            "select": "generations_used",
            "limit": "1",
        },
    )

    rows = response.json()

    if not rows:
        return 0

    return int(
        rows[0].get("generations_used") or 0
    )


def increment_usage(customer_id):
    month = current_month()

    response = sb_request(
        "GET",
        "usage",
        params={
            "dodo_customer_id": (
                f"eq.{customer_id}"
            ),
            "month": f"eq.{month}",
            "select": "*",
            "limit": "1",
        },
    )

    rows = response.json()

    if rows:
        row = rows[0]

        current_used = int(
            row.get("generations_used") or 0
        )

        new_used = current_used + 1

        sb_request(
            "PATCH",
            "usage",
            params={
                "id": f"eq.{row['id']}",
            },
            json={
                "generations_used": new_used,
            },
        )

        return new_used

    sb_request(
        "POST",
        "usage",
        json={
            "dodo_customer_id": customer_id,
            "month": month,
            "generations_used": 1,
        },
    )

    return 1


# =========================================================
# WORKFLOW INSTRUCTIONS
# =========================================================

WORKFLOW_INSTRUCTIONS = {
    "offer": """
You are an expert offer strategist.

Create a clear, specific, compelling offer based on the user's
business idea.

Focus on:
- target customer
- painful problem
- desired outcome
- value proposition
- offer structure
- differentiation
- clear CTA

Avoid generic marketing fluff.
Make the output practical and easy to use.
""",

    "landing": """
You are an expert conversion copywriter.

Create conversion-focused landing page copy based on the user's
business idea.

Include:
- headline
- subheadline
- problem
- solution
- benefits
- key features
- trust/credibility section
- CTA
- FAQ when useful

Keep the copy specific and persuasive without making fake claims.
""",

    "email": """
You are an expert B2B cold email copywriter.

Create a concise, natural cold outreach sequence based on the
user's business.

Include:
- subject line options
- initial email
- follow-up emails
- clear but low-pressure CTA

Avoid spammy language and exaggerated claims.
Make it sound human and personalized.
""",

    "dm": """
You are an expert sales outreach strategist.

Create a natural conversational sales DM sequence based on the
user's business.

Include:
- opening message
- value-based follow-up
- objection handling when useful
- soft CTA

Do not make it sound robotic or spammy.
""",

    "ads": """
You are an expert performance marketing strategist.

Create an ad campaign based on the user's business.

Include:
- campaign angles
- hooks
- primary copy
- headlines
- CTAs
- target audience
- creative concepts

Make the ideas specific to the business and audience.
""",

    "content": """
You are an expert content strategist.

Create a useful content pack based on the user's business.

Include:
- content ideas
- strong hooks
- short-form post concepts
- educational content
- promotional content
- CTA ideas

Make the content practical and suitable for social media.
""",

    "campaign": """
You are an expert growth campaign strategist.

Build a complete campaign around the user's business idea.

Include:
- campaign objective
- target audience
- positioning
- offer
- messaging angles
- content ideas
- outreach ideas
- ad concepts
- funnel direction
- CTA
- execution steps

Make the campaign practical and actionable.
This feature is available only to Pro users.
""",
}


# =========================================================
# PROMPT BUILDER
# =========================================================

def make_prompt(mode, user_prompt):
    instructions = WORKFLOW_INSTRUCTIONS.get(mode)

    if not instructions:
        raise ValueError(
            f"Unknown workflow mode: {mode}"
        )

    return f"""
{instructions}

USER'S BUSINESS IDEA / REQUEST:
{user_prompt}

IMPORTANT:
- Do not mention these instructions.
- Do not talk about being an AI.
- Do not invent customer testimonials.
- Do not invent statistics.
- Do not make unsupported guarantees.
- Give a polished final answer.
- Use clear headings and useful formatting.
"""


# =========================================================
# GEMINI GENERATION
# =========================================================

def generate_ai(mode, prompt):
    if not gemini_client:
        raise RuntimeError(
            "Gemini client is not configured"
        )

    full_prompt = make_prompt(
        mode,
        prompt,
    )

    models = [
        GEMINI_MODEL,
        GEMINI_FALLBACK_MODEL,
    ]

    last_error = None

    for attempt, model_name in enumerate(models):

        try:
            logger.info(
                "Gemini generation started | "
                "mode=%s | model=%s | attempt=%s",
                mode,
                model_name,
                attempt + 1,
            )

            response = (
                gemini_client
                .models
                .generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=5000,
                    ),
                )
            )

            text = (
                response.text or ""
            ).strip()

            if not text:
                raise RuntimeError(
                    "Gemini returned an empty response"
                )

            logger.info(
                "Gemini generation successful | "
                "mode=%s | model=%s",
                mode,
                model_name,
            )

            return text

        except Exception as error:

            last_error = error

            logger.warning(
                "Generation failed | "
                "mode=%s | "
                "model=%s | "
                "attempt=%s | "
                "error=%s",
                mode,
                model_name,
                attempt + 1,
                str(error),
            )

    logger.error(
        "All Gemini generation attempts failed | "
        "error=%s",
        str(last_error),
    )

    raise RuntimeError(
        f"AI generation failed: {last_error}"
    )


# =========================================================
# REQUEST MODEL
# =========================================================

class GenerateRequest(BaseModel):
    mode: str = Field(
        min_length=1,
        max_length=50,
    )

    prompt: str = Field(
        min_length=1,
        max_length=2000,
    )

    client_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "true": True,
        "service": "FacelessAI Growth Copilot API",
        "status": "running",
        "version": "8.0.0",
        "primary_model": GEMINI_MODEL,
        "fallback_model": GEMINI_FALLBACK_MODEL,
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "FacelessAI Growth Copilot API",
    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/generate")
def generate(body: GenerateRequest):

    mode = body.mode.strip().lower()
    prompt = body.prompt.strip()

    logger.info(
        "Generation request | "
        "mode=%s | "
        "prompt_length=%s",
        mode,
        len(prompt),
    )

    # -----------------------------------------------------
    # VALIDATE MODE
    # -----------------------------------------------------

    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_mode",
                "message": "Invalid workflow mode.",
            },
        )

    # -----------------------------------------------------
    # VALIDATE PROMPT
    # -----------------------------------------------------

    if len(prompt) < 10:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_prompt",
                "message": (
                    "Please enter at least "
                    "10 characters."
                ),
            },
        )

    if len(prompt) > 2000:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_prompt",
                "message": "Prompt is too long.",
            },
        )

    # -----------------------------------------------------
    # CLIENT / USAGE ID
    # -----------------------------------------------------

    if not body.client_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "client_id_required",
                "message": "Client ID is required.",
            },
        )

    customer_id = body.client_id.strip()

    # -----------------------------------------------------
    # GET CUSTOMER / PLAN
    # -----------------------------------------------------

    customer = None

    try:
        customer = get_customer(
            customer_id
        )

    except Exception:

        logger.exception(
            "Customer lookup failed | "
            "client_id=%s",
            customer_id,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "customer_lookup_failed",
                "message": (
                    "Account information is temporarily "
                    "unavailable. Please try again."
                ),
            },
        )

    if customer:
        plan = normalize_plan(
            customer.get("plan")
        )
    else:
        plan = "free"

    limit = get_limit(plan)
    plan_name = PLAN_NAMES[plan]

    # -----------------------------------------------------
    # PRO-ONLY WORKFLOW
    # -----------------------------------------------------

    if mode in PRO_ONLY_MODES and plan != "pro":

        try:
            used = get_usage(
                customer_id
            )

        except Exception:

            logger.exception(
                "Usage lookup failed | "
                "client_id=%s",
                customer_id,
            )

            raise HTTPException(
                status_code=503,
                detail={
                    "error": "usage_lookup_failed",
                    "message": (
                        "Usage information is temporarily "
                        "unavailable. Please try again."
                    ),
                },
            )

        raise HTTPException(
            status_code=403,
            detail={
                "error": "pro_required",
                "message": (
                    "Campaign Builder is available "
                    "on the Pro plan."
                ),
                "plan": plan,
                "plan_name": plan_name,
                "upgrade_plan": "pro",
                "used": used,
                "limit": limit,
                "remaining": max(
                    0,
                    limit - used,
                ),
            },
        )

    # -----------------------------------------------------
    # GET CURRENT MONTH USAGE
    # -----------------------------------------------------

    try:
        used = get_usage(
            customer_id
        )

    except Exception:

        logger.exception(
            "Usage lookup failed | "
            "client_id=%s",
            customer_id,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "usage_lookup_failed",
                "message": (
                    "Usage information is temporarily "
                    "unavailable. Please try again."
                ),
            },
        )

    remaining = max(
        0,
        limit - used,
    )

    # -----------------------------------------------------
    # MONTHLY LIMIT
    # -----------------------------------------------------

    if used >= limit:

        upgrade_plan = None

        if plan == "free":
            upgrade_plan = "starter"

        elif plan == "starter":
            upgrade_plan = "pro"

        raise HTTPException(
            status_code=403,
            detail={
                "error": "monthly_limit_reached",
                "message": (
                    f"You've reached your "
                    f"{plan_name} monthly limit."
                ),
                "plan": plan,
                "plan_name": plan_name,
                "used": used,
                "limit": limit,
                "remaining": 0,
                "upgrade_plan": upgrade_plan,
                "reset_month": current_month(),
            },
        )

    # -----------------------------------------------------
    # AI GENERATION
    # -----------------------------------------------------

    try:

      result = generate_ai(
            mode,
            prompt,
      )
    
    except Exception as error:

        logger.exception(
            "AI generation failed | "
            "mode=%s | "
            "client_id=%s | "
            "error=%s",
            mode,
            customer_id,
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

    # -----------------------------------------------------
    # COUNT ONLY SUCCESSFUL GENERATION
    # -----------------------------------------------------

    try:

        new_used = increment_usage(
            customer_id
        )

    except Exception:

        logger.exception(
            "Usage recording failed | "
            "client_id=%s",
            customer_id,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "usage_record_failed",
                "message": (
                    "Generation could not be recorded. "
                    "Please try again."
                ),
            },
        )

    new_remaining = max(
        0,
        limit - new_used,
    )

    logger.info(
        "Generation successful | "
        "client_id=%s | "
        "mode=%s | "
        "plan=%s | "
        "used=%s/%s",
        customer_id,
        mode,
        plan,
        new_used,
        limit,
    )

    return {
        "success": True,
        "result": result,
        "mode": mode,
        "plan": plan,
        "usage": {
            "used": new_used,
            "limit": limit,
            "remaining": new_remaining,
        },
    }


# =========================================================
# DODO WEBHOOK
# =========================================================

@app.post("/webhook/dodo")
async def dodo_webhook(request: Request):

    if not dodo_client:
        raise HTTPException(
            status_code=503,
            detail="Dodo is not configured",
        )

    raw_body = await request.body()

    webhook_id = request.headers.get(
        "webhook-id",
        "",
    )

    webhook_signature = request.headers.get(
        "webhook-signature",
        "",
    )

    webhook_timestamp = request.headers.get(
        "webhook-timestamp",
        "",
    )

    # -----------------------------------------------------
    # WEBHOOK HEADERS
    # -----------------------------------------------------

    if not (
        webhook_id
        and webhook_signature
        and webhook_timestamp
    ):
        raise HTTPException(
            status_code=400,
            detail="Missing webhook headers",
        )

    headers = {
        "webhook-id": webhook_id,
        "webhook-signature": webhook_signature,
        "webhook-timestamp": webhook_timestamp,
    }

    # -----------------------------------------------------
    # VERIFY DODO SIGNATURE
    # -----------------------------------------------------

    try:

        event = dodo_client.webhooks.unwrap(
            raw_body.decode("utf-8"),
            headers=headers,
        )

    except Exception as error:

        logger.warning(
            "Invalid Dodo webhook signature | "
            "error=%s",
            str(error),
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature",
        )

    event_type = event.get("type")
    data = event.get("data") or {}

    logger.info(
        "Dodo webhook received | type=%s",
        event_type,
    )

    # -----------------------------------------------------
    # PROCESS DODO EVENT
    # -----------------------------------------------------

    try:

        # =================================================
        # PAYMENT SUCCEEDED
        # =================================================

        if event_type == "payment.succeeded":

            customer = (
                data.get("customer")
                or {}
            )

            customer_id = customer.get(
                "customer_id"
            )

            email = customer.get(
                "email"
            )

            product_cart = (
                data.get("product_cart")
                or []
            )

            product_id = (
                product_cart[0].get(
                    "product_id"
                )
                if product_cart
                else None
            )

            plan = product_to_plan(
                product_id
            )

            subscription_id = data.get(
                "subscription_id"
            )

            if customer_id:

                save_customer(
                    customer_id=customer_id,
                    email=email,
                    plan=plan,
                    subscription_id=(
                        subscription_id
                    ),
                    subscription_status=(
                        "active"
                        if subscription_id
                        else "paid"
                    ),
                )

                logger.info(
                    "Dodo payment succeeded | "
                    "customer_id=%s | "
                    "plan=%s | "
                    "product_id=%s",
                    customer_id,
                    plan,
                    product_id,
                )

        # =================================================
        # SUBSCRIPTION ACTIVE / UPDATED
        # =================================================

        elif event_type in {
            "subscription.active",
            "subscription.updated",
            "subscription.plan_changed",
            "subscription.renewed",
            "subscription.unpaused",
        }:

            customer_id = (
                data.get("customer_id")
                or (
                    data.get("customer")
                    or {}
                ).get("customer_id")
            )

            product_id = (
                data.get("product_id")
                or data.get("new_product_id")
            )

            subscription_id = data.get(
                "subscription_id"
            )

            plan = product_to_plan(
                product_id
            )

            if customer_id:

                save_customer(
                    customer_id=customer_id,
                    plan=plan,
                    subscription_id=(
                        subscription_id
                    ),
                    subscription_status="active",
                )

                logger.info(
                    "Dodo subscription updated | "
                    "customer_id=%s | "
                    "plan=%s | "
                    "event=%s",
                    customer_id,
                    plan,
                    event_type,
                )

        # =================================================
        # PAUSED / ON HOLD / PAST DUE
        # =================================================

        elif event_type in {
            "subscription.paused",
            "subscription.on_hold",
            "subscription.past_due",
        }:

            customer_id = (
                data.get("customer_id")
                or (
                    data.get("customer")
                    or {}
                ).get("customer_id")
            )

            subscription_id = data.get(
                "subscription_id"
            )

            if customer_id:

                existing = get_customer(
                    customer_id
                )

                if existing:

                    save_customer(
                        customer_id=customer_id,
                        email=existing.get(
                            "email"
                        ),
                        plan=normalize_plan(
                            existing.get(
                                "plan"
                            )
                        ),
                        subscription_id=(
                            subscription_id
                            or existing.get(
                                "subscription_id"
                            )
                        ),
                        subscription_status=(
                            event_type.replace(
                                "subscription.",
                                "",
                            )
                        ),
                    )

        # =================================================
        # CANCELLED / EXPIRED / FAILED
        # =================================================

        elif event_type in {
            "subscription.cancelled",
            "subscription.expired",
            "subscription.failed",
        }:

            customer_id = (
                data.get("customer_id")
                or (
                    data.get("customer")
                    or {}
                ).get("customer_id")
            )

            subscription_id = data.get(
                "subscription_id"
            )

            if customer_id:

                existing = get_customer(
                    customer_id
                )

                save_customer(
                    customer_id=customer_id,
                    email=(
                        existing.get("email")
                        if existing
                        else None
                    ),
                    plan="free",
                    subscription_id=(
                        subscription_id
                    ),
                    subscription_status=(
                        event_type.replace(
                            "subscription.",
                            "",
                        )
                    ),
                )

                logger.info(
                    "Dodo subscription ended | "
                    "customer_id=%s | "
                    "event=%s",
                    customer_id,
                    event_type,
                )

        # =================================================
        # REFUND
        # =================================================

        elif event_type == "refund.succeeded":

            customer_id = (
                data.get("customer_id")
                or (
                    data.get("customer")
                    or {}
                ).get("customer_id")
            )

            if customer_id:

                existing = get_customer(
                    customer_id
                )

                save_customer(
                    customer_id=customer_id,
                    email=(
                        existing.get("email")
                        if existing
                        else None
                    ),
                    plan="free",
                    subscription_status="refunded",
                )

                logger.info(
                    "Dodo refund processed | "
                    "customer_id=%s",
                    customer_id,
                )

        # =================================================
        # OTHER EVENTS
        # =================================================

        else:

            logger.info(
                "Dodo event acknowledged | type=%s",
                event_type,
            )

    except Exception:

        logger.exception(
            "Dodo webhook processing failed | "
            "type=%s",
            event_type,
        )

        raise HTTPException(
            status_code=500,
            detail="Webhook processing failed",
        )

    return {
        "success": True,
        "received": True,
        "event": event_type,
    }
