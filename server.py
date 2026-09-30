import os
import uuid
import shutil
import time
import random
from pathlib import Path

import requests
import edge_tts

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# You can change this from Render Environment Variables.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not PEXELS_API_KEY:
    raise RuntimeError("PEXELS_API_KEY is missing")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FacelessAI API",
    version="2.0.0"
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
# STATIC VIDEO FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="static"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class VideoRequest(BaseModel):
    topic: str


# ============================================================
# TOPIC VALIDATION
# ============================================================

def validate_topic(topic: str) -> str:

    if topic is None:
        raise HTTPException(
            status_code=400,
            detail="Topic is required."
        )

    topic = topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Please enter a video topic."
        )

    if len(topic) > 500:
        raise HTTPException(
            status_code=400,
            detail="Topic must be under 500 characters."
        )

    return topic


# ============================================================
# GEMINI SCRIPT GENERATION
# ============================================================

def generate_script(topic: str) -> str:

    prompt = f"""
Create a high-retention short-form video narration.

Topic:
{topic}

Requirements:

- 80 to 120 words.
- Strong hook in the first sentence.
- Fast-paced and engaging.
- Simple spoken English.
- Useful and factual.
- Suitable for YouTube Shorts, TikTok and Instagram Reels.
- No emojis.
- No markdown.
- No headings.
- No bullet points.
- Return only the narration.
"""

    last_error = None

    for attempt in range(4):

        try:

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            script = (
                response.text or ""
            ).strip()

            if not script:

                raise RuntimeError(
                    "Gemini returned an empty script."
                )

            return script

        except Exception as exc:

            last_error = exc

            error_text = str(exc).upper()

            temporary_error = any(
                code in error_text
                for code in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                    "TIMEOUT"
                ]
            )

            if (
                not temporary_error
                or attempt == 3
            ):

                raise RuntimeError(
                    f"Gemini API error: {exc}"
                ) from exc

            delay = (
                (2 ** attempt)
                + random.uniform(0.5, 1.5)
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Gemini API error: {last_error}"
    )


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def generate_voice(
    text: str,
    output_path: Path
) -> None:

    communicator = edge_tts.Communicate(
        text=text,
        voice="en-US-AndrewNeural",
        rate="+5%",
        pitch="+0Hz"
    )

    await communicator.save(
        str(output_path)
    )


# ============================================================
# PEXELS SEARCH
# ============================================================

def search_pexels_video(
    query: str
) -> str:

    response = requests.get(
        "https://api.pexels.com/videos/search",
        headers={
            "Authorization": PEXELS_API_KEY
        },
        params={
            "query": query,
            "per_page": 15
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    videos = data.get(
        "videos",
        []
    )

    if not videos:

        raise RuntimeError(
            f"No Pexels videos found for: {query}"
        )


    # --------------------------------------------------------
    # Prefer portrait videos
    # --------------------------------------------------------

    portrait_files = []

    for video in videos:

        for file in video.get(
            "video_files",
            []
        ):

            width = file.get("width")
            height = file.get("height")
            link = file.get("link")

            if (
                width
                and height
                and link
                and height >= width
            ):

                portrait_files.append(file)


    if portrait_files:

        portrait_files.sort(
            key=lambda item: abs(
                (
                    (item.get("height") or 0)
                    /
                    max(
                        item.get("width") or 1,
                        1
                    )
                )
                -
                (16 / 9)
            )
        )

        return portrait_files[0]["link"]


    # --------------------------------------------------------
    # Fallback to any video
    # --------------------------------------------------------

    usable_files = []

    for video in videos:

        for file in video.get(
            "video_files",
            []
        ):

            if file.get("link"):

                usable_files.append(file)


    if usable_files:

        usable_files.sort(
            key=lambda item:
            (
                (item.get("width") or 0)
                *
                (item.get("height") or 0)
            ),
            reverse=True
        )

        return usable_files[0]["link"]


    raise RuntimeError(
        "Pexels returned no downloadable video."
    )


# ============================================================
# DOWNLOAD BACKGROUND VIDEO
# ============================================================

def download_video(
    url: str,
    output_path: Path
) -> None:

    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={
            "User-Agent": "FacelessAI/2.0"
        }
    ) as response:

        response.raise_for_status()

        content_type = (
            response.headers
            .get("content-type", "")
            .lower()
        )

        if "text/html" in content_type:

            raise RuntimeError(
                "Pexels returned HTML instead of a video."
            )

        with output_path.open(
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(chunk)


    if (
        not output_path.exists()
        or output_path.stat().st_size < 10000
    ):

        raise RuntimeError(
            "Downloaded video is empty or invalid."
        )


# ============================================================
# FORMAT VERTICAL CLIP
# ============================================================

def format_vertical_clip(
    clip
):

    scale = max(
        1080 / clip.w,
        1920 / clip.h
    )

    clip = clip.resized(
        scale
    )

    clip = clip.cropped(
        width=1080,
        height=1920,
        x_center=clip.w / 2,
        y_center=clip.h / 2
    )

    return clip


# ============================================================
# PREPARE VIDEO
# ============================================================

def prepare_clip(
    input_path: Path,
    target_duration: float
):

    source = VideoFileClip(
        str(input_path)
    )

    if (
        not source.duration
        or source.duration <= 0
    ):

        source.close()

        raise RuntimeError(
            "Downloaded video has no valid duration."
        )


    # --------------------------------------------------------
    # Video is long enough
    # --------------------------------------------------------

    if source.duration >= target_duration:

        clip = format_vertical_clip(
            source
        )

        if clip.duration > target_duration:

            final_clip = clip.subclipped(
                0,
                target_duration
            )

            clip.close()

            return final_clip

        return clip


    # --------------------------------------------------------
    # Video is shorter: loop it
    # --------------------------------------------------------

    source.close()

    clips = []
    elapsed = 0.0

    while elapsed < target_duration:

        clip = VideoFileClip(
            str(input_path)
        )

        clip = format_vertical_clip(
            clip
        )

        remaining = (
            target_duration
            -
            elapsed
        )

        if clip.duration > remaining:

            shortened = clip.subclipped(
                0,
                remaining
            )

            clip.close()

            clip = shortened

        clips.append(
            clip
        )

        elapsed += clip.duration


    return concatenate_videoclips(
        clips,
        method="compose"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "FacelessAI API",
        "status": "running",
        "version": "2.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "version": "2.0.0"
    }


# ============================================================
# GENERATE VIDEO
# ============================================================

@app.post("/generate-video")
async def generate_video(
    request: VideoRequest
):

    topic = validate_topic(
        request.topic
    )


    # --------------------------------------------------------
    # Job paths
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex

    job_dir = (
        TEMP_DIR
        /
        f"job_{job_id}"
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    audio_path = (
        job_dir
        /
        "voice.mp3"
    )

    background_path = (
        job_dir
        /
        "background.mp4"
    )

    final_path = (
        OUTPUT_DIR
        /
        f"{job_id}.mp4"
    )


    audio_clip = None
    video_clip = None


    try:

        # ====================================================
        # STEP 1 — GEMINI
        # ====================================================

        script = generate_script(
            topic
        )


        # ====================================================
        # STEP 2 — VOICE
        # ====================================================

        await generate_voice(
            script,
            audio_path
        )


        if (
            not audio_path.exists()
            or audio_path.stat().st_size == 0
        ):

            raise RuntimeError(
                "Voice generation failed."
            )


        # ====================================================
        # STEP 3 — AUDIO DURATION
        # ====================================================

        audio_clip = AudioFileClip(
            str(audio_path)
        )

        duration = audio_clip.duration

        if (
            not duration
            or duration <= 0
        ):

            raise RuntimeError(
                "Invalid audio duration."
            )


        # ====================================================
        # STEP 4 — PEXELS
        # ====================================================

        queries = []

        if len(topic) <= 80:

            queries.append(
                topic
            )

        queries.extend([
            "technology",
            "futuristic technology",
            "abstract technology",
            "business technology"
        ])


        video_url = None
        pexels_errors = []


        for query in queries:

            try:

                video_url = search_pexels_video(
                    query
                )

                if video_url:
                    break

            except Exception as exc:

                pexels_errors.append(
                    f"{query}: {exc}"
                )


        if not video_url:

            raise RuntimeError(
                "Could not find a usable Pexels video. "
                +
                " | ".join(
                    pexels_errors[-2:]
                )
            )


        # ====================================================
        # STEP 5 — DOWNLOAD
        # ====================================================

        download_video(
            video_url,
            background_path
        )


        # ====================================================
        # STEP 6 — 9:16
        # ====================================================

        video_clip = prepare_clip(
            background_path,
            duration
        )


        # ====================================================
        # STEP 7 — AUDIO
        # ====================================================

        video_clip = video_clip.with_audio(
            audio_clip
        )


        # ====================================================
        # STEP 8 — EXPORT
        # ====================================================

        video_clip.write_videofile(
            str(final_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            threads=2,
            logger=None
        )


        # ====================================================
        # VERIFY
        # ====================================================

        if (
            not final_path.exists()
            or final_path.stat().st_size == 0
        ):

            raise RuntimeError(
                "Final video file was not created."
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        return {
            "success": True,
            "message": "Video generated successfully.",
            "video_url": (
                f"/static/{final_path.name}"
            ),
            "script": script
        }


    except HTTPException:

        raise


    except Exception as exc:

        # IMPORTANT:
        # Return a normal JSON 500 response so the frontend
        # can show the actual backend error.
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc


    finally:

        # ----------------------------------------------------
        # Close MoviePy resources
        # ----------------------------------------------------

        if video_clip is not None:

            try:
                video_clip.close()
            except Exception:
                pass


        if audio_clip is not None:

            try:
                audio_clip.close()
            except Exception:
                pass


        # ----------------------------------------------------
        # Remove temporary files
        # ----------------------------------------------------

        try:

            shutil.rmtree(
                job_dir,
                ignore_errors=True
            )

        except Exception:

            pass
