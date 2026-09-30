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
from moviepy import VideoFileClip, AudioFileClip


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash"
)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not PEXELS_API_KEY:
    raise RuntimeError("PEXELS_API_KEY is missing")

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
# FASTAPI
# ============================================================

app = FastAPI(
    title="FacelessAI API",
    version="4.2.1"
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
# STATIC FILES
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
# GEMINI TEMPORARY ERROR CHECK
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
# GEMINI SCRIPT GENERATION
# ============================================================

def generate_script(topic: str) -> str:

    print(
        "STEP 1: Starting Gemini script generation...",
        flush=True
    )

    prompt = f"""
Create a high-retention short-form video narration.

Topic:
{topic}

Requirements:

- 80 to 120 words.
- Strong hook in the first sentence.
- Fast-paced and engaging.
- Natural spoken English.
- Easy for a global audience to understand.
- Useful and factual.
- Suitable for YouTube Shorts, TikTok and Instagram Reels.
- Keep curiosity throughout the narration.
- End with a memorable takeaway.
- No emojis.
- No markdown.
- No headings.
- No bullet points.
- Return ONLY the narration.
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
                    f"Gemini: {model} | attempt "
                    f"{attempt + 1}/3",
                    flush=True
                )

                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=model,
                        contents=prompt
                    )
                )

                script = (
                    response.text or ""
                ).strip()

                if not script:
                    raise RuntimeError(
                        f"{model} returned an empty script."
                    )

                print(
                    "STEP 2: Gemini completed. "
                    f"Script length: {len(script)}",
                    flush=True
                )

                return script

            except Exception as exc:

                print(
                    "Gemini error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )

                errors.append(
                    f"{model}: {exc}"
                )

                if not is_temporary_gemini_error(exc):
                    break

                if attempt < 2:

                    delay = (
                        (2 ** attempt)
                        + random.uniform(
                            0.5,
                            1.5
                        )
                    )

                    print(
                        f"Retrying Gemini in "
                        f"{delay:.1f}s...",
                        flush=True
                    )

                    time.sleep(delay)

    raise RuntimeError(
        "Gemini could not generate the script. "
        + " | ".join(errors[-4:])
    )


# ============================================================
# EDGE TTS
# ============================================================

async def generate_voice(
    text: str,
    output_path: Path
) -> None:

    print(
        "STEP 3: Starting Edge TTS...",
        flush=True
    )

    communicator = edge_tts.Communicate(
        text=text,
        voice="en-US-AndrewNeural",
        rate="+5%",
        pitch="+0Hz"
    )

    await communicator.save(
        str(output_path)
    )

    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Edge TTS did not create an audio file."
        )

    print(
        "STEP 4: Edge TTS completed.",
        flush=True
    )


# ============================================================
# PEXELS SEARCH
# ============================================================

def search_pexels_video(
    query: str
) -> str:

    print(
        f"Pexels: searching '{query}'...",
        flush=True
    )

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
            key=lambda item: (
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

    print(
        "STEP 6: Starting background video download...",
        flush=True
    )

    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={
            "User-Agent": "FacelessAI/4.2"
        }
    ) as response:

        response.raise_for_status()

        content_type = (
            response.headers
            .get(
                "content-type",
                ""
            )
            .lower()
        )

        if "text/html" in content_type:

            raise RuntimeError(
                "Pexels returned HTML instead of a video."
            )

        with output_path.open("wb") as file:

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

    print(
        "STEP 7: Background video downloaded.",
        flush=True
    )


# ============================================================
# FORMAT VIDEO TO 9:16
# ============================================================

def format_vertical_clip(clip):

    scale = max(
        720 / clip.w,
        1280 / clip.h
    )

    clip = clip.resized(scale)

    clip = clip.cropped(
        width=720,
        height=1280,
        x_center=clip.w / 2,
        y_center=clip.h / 2
    )

    return clip


# ============================================================
# PREPARE VIDEO CLIP
# ============================================================

def prepare_clip(
    input_path: Path,
    target_duration: float
):

    print(
        "STEP 8: Starting MoviePy processing...",
        flush=True
    )

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
            - elapsed
        )

        if clip.duration > remaining:

            shortened = clip.subclipped(
                0,
                remaining
            )

            clip.close()

            clip = shortened

        clips.append(clip)

        elapsed += clip.duration

    from moviepy import concatenate_videoclips

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
        "version": "4.2.1",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "version": "4.2.1",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL
    }


# ============================================================
# GENERATE VIDEO
# ============================================================

@app.post("/generate-video")
async def generate_video(
    request: VideoRequest
):

    print(
        "REQUEST: /generate-video received.",
        flush=True
    )

    topic = validate_topic(
        request.topic
    )

    print(
        f"REQUEST: Topic received "
        f"({len(topic)} characters).",
        flush=True
    )

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
        # 1. GENERATE SCRIPT
        # ====================================================

        script = generate_script(
            topic
        )

        # ====================================================
        # 2. GENERATE VOICE
        # ====================================================

        await generate_voice(
            script,
            audio_path
        )

        # ====================================================
        # 3. LOAD AUDIO
        # ====================================================

        print(
            "STEP 5: Loading audio...",
            flush=True
        )

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

        print(
            f"STEP 5: Audio loaded. "
            f"Duration: {duration:.2f}s",
            flush=True
        )

        # ====================================================
        # 4. SEARCH PEXELS
        # ====================================================

        print(
            "STEP 6: Starting Pexels search...",
            flush=True
        )

        queries = []

        if len(topic) <= 80:
            queries.append(topic)

        queries.extend([
            "technology",
            "artificial intelligence",
            "futuristic technology",
            "business technology",
            "digital technology"
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

                print(
                    "Pexels error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )

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

        print(
            "Pexels search completed successfully.",
            flush=True
        )

        # ====================================================
        # 5. DOWNLOAD BACKGROUND
        # ====================================================

        download_video(
            video_url,
            background_path
        )

        # ====================================================
        # 6. PREPARE VIDEO
        # ====================================================

        video_clip = prepare_clip(
            background_path,
            duration
        )

        print(
            "STEP 9: MoviePy processing completed.",
            flush=True
        )

        # ====================================================
        # 7. ATTACH AUDIO
        # ====================================================

        print(
            "STEP 10: Attaching audio...",
            flush=True
        )

        video_clip = video_clip.with_audio(
            audio_clip
        )

        # ====================================================
        # 8. EXPORT MP4
        # ====================================================

        print(
            "STEP 11: Starting MP4 export...",
            flush=True
        )

        video_clip.write_videofile(
    str(final_path),
    fps=20,
    codec="libx264",
    audio_codec="aac",
    bitrate="1000k",
    audio_bitrate="96k",
    preset="ultrafast",
    threads=1,
    logger=None
  )
        print(
            "STEP 12: MP4 export completed.",
            flush=True
        )

        # ====================================================
        # 9. VERIFY FILE
        # ====================================================

        if (
            not final_path.exists()
            or final_path.stat().st_size < 10000
        ):

            raise RuntimeError(
                "Final video file was not created "
                "or is invalid."
            )

        print(
            "STEP 13: Final video verified.",
            flush=True
        )

        # ====================================================
        # 10. PRINT ACTUAL VIDEO URL
        # ====================================================

        print(
            "VIDEO URL:",
            f"https://faceless-ai-website.onrender.com/static/{final_path.name}",
            flush=True
        )

        # ====================================================
        # 11. SUCCESS RESPONSE
        # ====================================================

        print(
            "SUCCESS: Sending response to frontend.",
            flush=True
        )

        return {
            "success": True,
            "message": "Video generated successfully.",
            "video_url": (
                f"/static/{final_path.name}"
            ),
            "script": script
        }

    # ========================================================
    # HTTP ERRORS
    # ========================================================

    except HTTPException:
        raise

    # ========================================================
    # GENERATION ERRORS
    # ========================================================

    except Exception as exc:

        print(
            "GENERATION ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if video_clip is not None:

            try:

                video_clip.close()

            except Exception as exc:

                print(
                    "Video cleanup warning: "
                    f"{exc}",
                    flush=True
                )

        if audio_clip is not None:

            try:

                audio_clip.close()

            except Exception as exc:

                print(
                    "Audio cleanup warning: "
                    f"{exc}",
                    flush=True
                )

        try:

            shutil.rmtree(
                job_dir,
                ignore_errors=True
            )

        except Exception as exc:

            print(
                "Cleanup warning: "
                f"{exc}",
                flush=True
            )
