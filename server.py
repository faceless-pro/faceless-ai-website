import os
import time
import random

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai


# ============================================================
# ENVIRONMENT
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
    raise RuntimeError(
        "GEMINI_API_KEY is missing"
    )


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


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
        "https://faceless-ai-website.vercel.app"
    ],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):

    mode: str = Field(
        ...,
        min_length=1,
        max_length=50
    )

    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000
    )


# ============================================================
# MODE CONFIGURATION
# ============================================================

MODE_INSTRUCTIONS = {

    "offer": {
        "name": "Offer Builder",

        "instruction": """
Create a strong, clear and commercially useful offer.

Return:

1. Offer name
2. Target customer
3. Core problem
4. Main promise
5. Key benefits
6. What is included
7. Suggested CTA
8. One short positioning statement

Make the offer specific and easy to understand.
Avoid unrealistic guarantees.
"""
    },


    "landing": {
        "name": "Landing Page Copy",

        "instruction": """
Create high-quality landing page copy.

Return:

1. Hero headline
2. Subheadline
3. Problem section
4. Solution section
5. Key benefits
6. Features
7. Social-proof placeholder section
8. FAQ ideas
9. Final CTA

Make the copy clear, persuasive and suitable for a modern SaaS or service business.
Do not invent testimonials, customer numbers or fake results.
"""
    },


    "email": {
        "name": "Cold Email",

        "instruction": """
Create a concise B2B cold email.

Return:

Subject:
Email:

Then provide:
- A short follow-up email
- A second follow-up
- A simple CTA

Keep the emails natural and personalized.
Do not use fake claims.
Avoid spammy language.
"""
    },


    "dm": {
        "name": "Sales DM",

        "instruction": """
Create a natural sales DM sequence.

Return:

1. First message
2. Follow-up message
3. Value message
4. Soft CTA

Keep it conversational and non-pushy.
The goal is to start a real conversation rather than immediately forcing a sale.
"""
    },


    "ads": {
        "name": "Ad Campaign",

        "instruction": """
Create an advertising concept for the product or service.

Return:

1. Campaign angle
2. Main hook
3. Three alternative hooks
4. Primary ad copy
5. Short ad copy
6. CTA
7. Three creative concepts
8. Three audience angles

Make the ideas practical for social media advertising.
Do not claim guaranteed results.
"""
    },


    "content": {
        "name": "Content Pack",

        "instruction": """
Create a content pack around the user's topic.

Return:

1. Main content angle
2. Five strong hooks
3. Five short-form post ideas
4. Three educational posts
5. Three opinion/insight posts
6. Three CTA ideas
7. One seven-day content outline

Make the content useful, specific and easy to publish.
"""
    }

}


# ============================================================
# VALIDATE REQUEST
# ============================================================

def validate_request(
    request: GenerateRequest
):

    mode = request.mode.strip().lower()
    prompt = request.prompt.strip()

    if mode not in MODE_INSTRUCTIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid generation mode. "
                "Please select a supported workflow."
            )
        )

    if not prompt:

        raise HTTPException(
            status_code=400,
            detail="Please describe your product, service or idea."
        )

    if len(prompt) < 10:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide a little more information "
                "so the AI can create a useful result."
            )
        )

    if len(prompt) > 2000:

        raise HTTPException(
            status_code=400,
            detail="Your input must be under 2000 characters."
        )

    return mode, prompt


# ============================================================
# TEMPORARY GEMINI ERROR CHECK
# ============================================================

def is_temporary_gemini_error(
    error: Exception
) -> bool:

    text = str(error).upper()

    temporary_codes = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "TIMEOUT",
        "DEADLINE",
        "INTERNAL"
    ]

    return any(
        code in text
        for code in temporary_codes
    )


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_growth_asset(
    mode: str,
    user_prompt: str
) -> str:

    mode_config = MODE_INSTRUCTIONS[mode]

    system_instruction = f"""
You are FacelessAI Growth Copilot.

You help creators, freelancers, founders,
agencies and small businesses create useful
marketing assets.

Selected workflow:
{mode_config["name"]}

Important rules:

- Write in clear professional English.
- Be specific rather than generic.
- Give practical output that can actually be used.
- Do not invent testimonials.
- Do not invent customer numbers.
- Do not invent revenue figures.
- Do not promise guaranteed results.
- Do not make unsupported factual claims.
- Avoid excessive emojis.
- Avoid unnecessary filler.
- Use clean headings and formatting.
- Focus on the user's actual business.
- If information is missing, make a reasonable
  generic assumption and clearly keep it editable.

Workflow requirements:

{mode_config["instruction"]}

User's business / idea:

{user_prompt}

Now create the final marketing asset.
Return ONLY the useful finished output.
"""


    models = [
        PRIMARY_MODEL,
        FALLBACK_MODEL
    ]

    errors = []


    for model in models:

        for attempt in range(3):

            try:

                print(
                    f"Gemini: {model} | "
                    f"attempt {attempt + 1}/3",
                    flush=True
                )


                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=model,
                        contents=system_instruction
                    )
                )


                result = (
                    response.text or ""
                ).strip()


                if not result:

                    raise RuntimeError(
                        f"{model} returned an empty response."
                    )


                print(
                    "Gemini generation completed. "
                    f"Output length: {len(result)}",
                    flush=True
                )


                return result


            except Exception as exc:

                print(
                    "Gemini error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )


                errors.append(
                    f"{model}: {exc}"
                )


                if not is_temporary_gemini_error(
                    exc
                ):
                    break


                if attempt < 2:

                    delay = (
                        (2 ** attempt)
                        +
                        random.uniform(
                            0.5,
                            1.5
                        )
                    )


                    print(
                        f"Retrying in "
                        f"{delay:.1f}s...",
                        flush=True
                    )


                    time.sleep(delay)


    raise RuntimeError(
        "Gemini could not generate the requested "
        "marketing asset. "
        +
        " | ".join(
            errors[-4:]
        )
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
        "version": "5.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "service": "FacelessAI Growth Copilot API",
        "version": "5.0.0"
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
async def generate(
    request: GenerateRequest
):

    print(
        "REQUEST: /generate received.",
        flush=True
    )


    try:

        mode, prompt = validate_request(
            request
        )


        print(
            f"Mode: {mode}",
            flush=True
        )


        print(
            f"Prompt length: {len(prompt)}",
            flush=True
        )


        result = generate_growth_asset(
            mode,
            prompt
        )


        print(
            "SUCCESS: Sending AI result.",
            flush=True
        )


        return {
            "success": True,
            "mode": mode,
            "result": result
        }


    except HTTPException:

        raise


    except Exception as exc:

        print(
            "GENERATION ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "The AI could not generate your result "
                "right now. Please try again."
            )
        )


# ============================================================
# OPTIONAL: OLD VIDEO ENDPOINT MESSAGE
# ============================================================

@app.post("/generate-video")
async def old_video_endpoint():

    raise HTTPException(
        status_code=410,
        detail=(
            "Video generation has been replaced by "
            "FacelessAI Growth Copilot."
        )
    )
