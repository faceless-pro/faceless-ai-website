
 import os
import uuid
from pathlib import Path

import requests
import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")

if not PEXELS_API_KEY:
    raise RuntimeError("PEXELS_API_KEY is missing.")


# --------------------------------------------------
# GEMINI
# --------------------------------------------------

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="FacelessAI Video Generator",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="static"
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class VideoRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=500
    )

    voice: str = "en-US-ChristopherNeural"


# --------------------------------------------------
# GENERATE SCRIPT
# --------------------------------------------------

def generate_script(topic: str) -> str:

    prompt = f"""
Write a compelling 30-second short-form video script.

Topic:
{topic}

Requirements:
- Strong hook in the first sentence
- Simple spoken English
- Fast-paced and engaging
- Around 70 to 90 words
- Clear ending
- Output ONLY the spoken script
- No title
- No scene directions
- No timestamps
- No markdown
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    script = (response.text or "").strip()

    if not script:
        raise RuntimeError(
            "Gemini returned an empty script."
        )

    return script


# --------------------------------------------------
# GENERATE AI VOICE
# --------------------------------------------------

async def generate_voice(
    script: str,
    voice: str,
    audio_path: Path
):

    communicate = edge_tts.Communicate(
        script,
        voice
    )

    await communicate.save(
        str(audio_path)
    )

    if not audio_path.exists():
        raise RuntimeError(
            "Voice file was not created."
        )

    if audio_path.stat().st_size == 0:
        raise RuntimeError(
            "Generated voice file is empty."
        )


# --------------------------------------------------
# DOWNLOAD PEXELS VIDEO
# --------------------------------------------------

def download_background_video(
    topic: str,
    video_path: Path
):

    search_url = (
        "https://api.pexels.com/videos/search"
    )

    headers = {
        "Authorization": PEXELS_API_KEY
    }

    params = {
        "query": topic,
        "per_page": 10,
        "orientation": "portrait"
    }

    response = requests.get(
        search_url,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    videos = data.get("videos", [])

    if not videos:
        raise RuntimeError(
            "No suitable Pexels video found."
        )

    selected_file = None

    # Find a portrait MP4
    for video in videos:

        files = video.get(
            "video_files",
            []
        )

        mp4_files = [
            file
            for file in files
            if file.get("file_type") == "video/mp4"
            and file.get("link")
        ]

        if not mp4_files:
            continue

        portrait_files = [
            file
            for file in mp4_files
            if (
                (file.get("width") or 0)
                < (file.get("height") or 0)
            )
        ]

        candidates = (
            portrait_files
            if portrait_files
            else mp4_files
        )

        selected_file = max(
            candidates,
            key=lambda file: (
                (file.get("width") or 0)
                * (file.get("height") or 0)
            )
        )

        break

    if selected_file is None:
        raise RuntimeError(
            "Pexels returned no usable MP4 file."
        )

    video_url = selected_file["link"]

    with requests.get(
        video_url,
        stream=True,
        timeout=60
    ) as video_response:

        video_response.raise_for_status()

        with open(
            video_path,
            "wb"
        ) as output_file:

            for chunk in video_response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    output_file.write(chunk)

    if not video_path.exists():
        raise RuntimeError(
            "Video download failed."
        )

    if video_path.stat().st_size == 0:
        raise RuntimeError(
            "Downloaded video is empty."
        )


# --------------------------------------------------
# COMBINE VIDEO + AUDIO
# --------------------------------------------------

def create_final_video(
    video_path: Path,
    audio_path: Path,
    final_path: Path
):

    video_clip = None
    audio_clip = None
    combined_clip = None

    try:

        audio_clip = AudioFileClip(
            str(audio_path)
        )

        video_clip = VideoFileClip(
            str(video_path)
        )

        if not audio_clip.duration:
            raise RuntimeError(
                "Audio duration is invalid."
            )

        if not video_clip.duration:
            raise RuntimeError(
                "Video duration is invalid."
            )

        target_duration = audio_clip.duration

        # Video is longer than narration
        if video_clip.duration >= target_duration:

            final_video_clip = video_clip.subclipped(
                0,
                target_duration
            )

        # Video is shorter than narration
        else:

            repeat_count = int(
                target_duration / video_clip.duration
            ) + 1

            repeated_clips = [
                video_clip
                for _ in range(repeat_count)
            ]

            combined_clip = concatenate_videoclips(
                repeated_clips,
                method="chain"
            )

            final_video_clip = combined_clip.subclipped(
                0,
                target_duration
            )

        final_video_clip = final_video_clip.with_audio(
            audio_clip
        )

        final_video_clip.write_videofile(
            str(final_path),
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="medium",
            threads=2,
            logger=None
        )

        final_video_clip.close()

    finally:

        if combined_clip is not None:
            combined_clip.close()

        if video_clip is not None:
            video_clip.close()

        if audio_clip is not None:
            audio_clip.close()


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "FacelessAI Video Generator Active"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# --------------------------------------------------
# GENERATE VIDEO API
# --------------------------------------------------

@app.post("/generate-video")
async def generate_video(
    request: VideoRequest
):

    job_id = uuid.uuid4().hex

    audio_path = (
        OUTPUT_DIR
        / f"{job_id}_voice.mp3"
    )

    background_path = (
        OUTPUT_DIR
        / f"{job_id}_background.mp4"
    )

    final_path = (
        OUTPUT_DIR
        / f"{job_id}_final.mp4"
    )

    try:

        # 1. Generate script
        script = generate_script(
            request.topic
        )

        # 2. Generate voice
        await generate_voice(
            script=script,
            voice=request.voice,
            audio_path=audio_path
        )

        # 3. Download background
        download_background_video(
            topic=request.topic,
            video_path=background_path
        )

        # 4. Combine everything
        create_final_video(
            video_path=background_path,
            audio_path=audio_path,
            final_path=final_path
        )

        if not final_path.exists():
            raise RuntimeError(
                "Final video was not created."
            )

        if final_path.stat().st_size == 0:
            raise RuntimeError(
                "Final video is empty."
            )

        return {
            "status": "success",
            "script": script,
            "video_url": (
                f"/static/{final_path.name}"
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        # Delete temporary files
        try:
            audio_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        try:
            background_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass
