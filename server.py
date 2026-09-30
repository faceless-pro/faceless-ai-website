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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FacelessAI API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# CLEAN TOPIC
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
# GEMINI SCRIPT
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

    # Retry temporary Gemini errors
    for attempt in range(4):

        try:

            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
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

            error_text = str(exc)

            # Temporary errors that are safe to retry
            transient_error = any(
                code in error_text
                for code in [
                    "503",
                    "429",
                    "500",
                    "502",
                    "504",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED"
                ]
            )

            # Do not retry permanent errors
            if (
                not transient_error
                or attempt == 3
            ):

                raise RuntimeError(
                    f"Gemini API error: {exc}"
                ) from exc

            # Exponential backoff + small random delay
            delay = (
                (2 ** attempt)
                + random.uniform(0, 1)
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
# PEXELS VIDEO SEARCH
# ============================================================

def search_pexels_video(
    query: str
) -> str:

    url = (
        "https://api.pexels.com/videos/search"
    )

    headers = {
        "Authorization": PEXELS_API_KEY
    }

    params = {
        "query": query,
        "per_page": 10
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
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

    for video in videos:

        files = video.get(
            "video_files",
            []
        )

        portrait_files = []

        for file in files:

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
                key=lambda item:
                abs(
                    (item.get("height") or 0)
                    - 1920
                )
            )

            return portrait_files[0]["link"]


    # --------------------------------------------------------
    # Fallback to any usable video
    # --------------------------------------------------------

    for video in videos:

        files = video.get(
            "video_files",
            []
        )

        usable_files = [
            file
            for file in files
            if file.get("link")
        ]

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
# DOWNLOAD VIDEO
# ============================================================

def download_video(
    url: str,
    output_path: Path
) -> None:

    response = requests.get(
        url,
        stream=True,
        timeout=90
    )

    response.raise_for_status()

    with output_path.open(
        "wb"
    ) as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)


# ============================================================
# PREPARE VERTICAL VIDEO
# ============================================================

def prepare_clip(
    input_path: Path,
    target_duration: float
):

    base_clip = VideoFileClip(
        str(input_path)
    )

    if not base_clip.duration:

        base_clip.close()

        raise RuntimeError(
            "Downloaded video has no duration."
        )


    # --------------------------------------------------------
    # Resize so video completely covers 1080x1920
    # --------------------------------------------------------

    scale = max(
        1080 / base_clip.w,
        1920 / base_clip.h
    )

    base_clip = base_clip.resized(
        scale
    )


    # --------------------------------------------------------
    # Center crop to 9:16
    # --------------------------------------------------------

    base_clip = base_clip.cropped(
        width=1080,
        height=1920,
        x_center=base_clip.w / 2,
        y_center=base_clip.h / 2
    )


    # --------------------------------------------------------
    # Loop if video is shorter than narration
    # --------------------------------------------------------

    if base_clip.duration < target_duration:

        original_duration = base_clip.duration

        repeat_count = (
            int(
                target_duration
                /
                original_duration
            )
            + 1
        )

        clips = []

        for _ in range(repeat_count):

            loop_clip = VideoFileClip(
                str(input_path)
            )

            loop_clip = loop_clip.resized(
                scale
            )

            loop_clip = loop_clip.cropped(
                width=1080,
                height=1920,
                x_center=loop_clip.w / 2,
                y_center=loop_clip.h / 2
            )

            clips.append(
                loop_clip
            )

        base_clip.close()

        combined = concatenate_videoclips(
            clips,
            method="compose"
        )

        base_clip = combined


    # --------------------------------------------------------
    # Cut exact duration
    # --------------------------------------------------------

    if base_clip.duration > target_duration:

        final_clip = base_clip.subclipped(
            0,
            target_duration
        )

        base_clip.close()

        return final_clip


    return base_clip


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "FacelessAI API",
        "status": "running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy"
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


    job_id = uuid.uuid4().hex

    job_dir = (
        OUTPUT_DIR
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
        # STEP 1 — GEMINI SCRIPT
        # ====================================================

        script = generate_script(
            topic
        )


        # ====================================================
        # STEP 2 — AI VOICE
        # ====================================================

        await generate_voice(
            script,
            audio_path
        )


        if not audio_path.exists():

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

        if not duration or duration <= 0:

            raise RuntimeError(
                "Invalid audio duration."
            )


        # ====================================================
        # STEP 4 — PEXELS SEARCH
        # ====================================================

        queries = [
            topic,
            "technology",
            "futuristic technology",
            "abstract technology"
        ]

        video_url = None
        last_error = None


        for query in queries:

            try:

                video_url = search_pexels_video(
                    query
                )

                if video_url:
                    break

            except Exception as exc:

                last_error = exc


        if not video_url:

            raise RuntimeError(
                "Could not find a Pexels video. "
                f"{last_error}"
            )


        # ====================================================
        # STEP 5 — DOWNLOAD
        # ====================================================

        download_video(
            video_url,
            background_path
        )


        if not background_path.exists():

            raise RuntimeError(
                "Background video download failed."
            )


        # ====================================================
        # STEP 6 — MAKE 9:16
        # ====================================================

        video_clip = prepare_clip(
            background_path,
            duration
        )


        # ====================================================
        # STEP 7 — ADD AUDIO
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
            preset="medium",
            threads=2,
            logger=None
        )


        # ====================================================
        # VERIFY OUTPUT
        # ====================================================

        if (
            not final_path.exists()
            or final_path.stat().st_size == 0
        ):

            raise RuntimeError(
                "Final video file was not created."
            )


        # ====================================================
        # RESPONSE
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

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


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
        # Delete temporary job folder
        # ----------------------------------------------------

        try:

            if job_dir.exists():

                shutil.rmtree(
                    job_dir,
                    ignore_errors=True
                )

        except Exception:

            pass
