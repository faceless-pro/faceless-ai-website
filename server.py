import os
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import edge_tts

from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips


# =============================
# CONFIG
# =============================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not PEXELS_API_KEY:
    raise RuntimeError("PEXELS_API_KEY is missing")


gemini_client = genai.Client(api_key=GEMINI_API_KEY)


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# FASTAPI
# =============================

app = FastAPI(title="FacelessAI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="static",
)


# =============================
# REQUEST MODEL
# =============================

class VideoRequest(BaseModel):
    topic: str


# =============================
# HELPERS
# =============================

def clean_topic(topic: str) -> str:
    topic = topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic is required.",
        )

    if len(topic) > 500:
        raise HTTPException(
            status_code=400,
            detail="Topic is too long. Keep it under 500 characters.",
        )

    return topic


# =============================
# GEMINI SCRIPT GENERATION
# =============================

def generate_script(topic: str) -> str:

    prompt = f"""
Write a short, engaging script for a vertical social-media video.

Topic: {topic}

Rules:
- 80 to 120 words.
- Start with a strong hook.
- Use simple spoken English.
- Keep the pacing fast.
- Give useful information.
- No emojis.
- No headings.
- No bullet points.
- Return ONLY the narration script.
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Gemini API error: {exc}"
        ) from exc


    text = (response.text or "").strip()

    if not text:
        raise RuntimeError(
            "Gemini returned an empty script."
        )

    return text


# =============================
# EDGE TTS
# =============================

async def create_voice(
    text: str,
    output_path: Path,
) -> None:

    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-AndrewNeural",
        rate="+8%",
        pitch="+0Hz",
    )

    await communicate.save(str(output_path))


# =============================
# PEXELS SEARCH
# =============================

def search_pexels_video(query: str) -> str:

    url = "https://api.pexels.com/videos/search"

    headers = {
        "Authorization": PEXELS_API_KEY,
    }

    params = {
        "query": query,
        "per_page": 10,
        "orientation": "portrait",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    videos = data.get("videos", [])

    if not videos:
        raise RuntimeError(
            "No suitable Pexels videos were found."
        )


    # Prefer portrait videos
    for video in videos:

        files = video.get(
            "video_files",
            [],
        )

        portrait_files = [
            f
            for f in files
            if f.get("width")
            and f.get("height")
            and f["height"] >= f["width"]
        ]

        if portrait_files:

            portrait_files.sort(
                key=lambda f: abs(
                    (f.get("width") or 0) - 1080
                )
            )

            return portrait_files[0]["link"]


    # Fallback
    files = videos[0].get(
        "video_files",
        [],
    )

    if not files:
        raise RuntimeError(
            "Pexels returned a video without a downloadable file."
        )


    files.sort(
        key=lambda f:
        (f.get("width") or 0)
        *
        (f.get("height") or 0),
        reverse=True,
    )

    return files[0]["link"]


# =============================
# DOWNLOAD FILE
# =============================

def download_file(
    url: str,
    output_path: Path,
) -> None:

    response = requests.get(
        url,
        stream=True,
        timeout=60,
    )

    response.raise_for_status()


    with output_path.open("wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)


# =============================
# MAKE 9:16 VIDEO
# =============================

def make_vertical_clip(
    video_paths: list[Path],
    duration: float,
):

    clips = []

    try:

        for path in video_paths:

            clip = VideoFileClip(
                str(path)
            )

            if not clip.duration:
                clip.close()
                continue


            # Resize to cover 1080x1920
            scale = max(
                1080 / clip.w,
                1920 / clip.h,
            )

            clip = clip.resized(scale)


            # Center crop
            clip = clip.cropped(
                width=1080,
                height=1920,
                x_center=clip.w / 2,
                y_center=clip.h / 2,
            )


            clips.append(clip)


        if not clips:

            raise RuntimeError(
                "Could not open downloaded background videos."
            )


        combined = concatenate_videoclips(
            clips,
            method="compose",
        )


        # Repeat videos if needed
        if combined.duration < duration:

            repeats = (
                int(duration / combined.duration)
                + 1
            )

            repeated = concatenate_videoclips(
                [combined] * repeats,
                method="compose",
            )

            combined = repeated


        final_clip = combined.subclipped(
            0,
            duration,
        )

        return final_clip


    except Exception:

        for clip in clips:

            try:
                clip.close()
            except Exception:
                pass

        raise


# =============================
# ROOT
# =============================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "FacelessAI API",
    }


# =============================
# HEALTH
# =============================

@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


# =============================
# GENERATE VIDEO
# =============================

@app.post("/generate-video")
async def generate_video(
    request: VideoRequest,
):

    topic = clean_topic(
        request.topic
    )


    job_id = uuid.uuid4().hex


    work_dir = (
        OUTPUT_DIR
        /
        f"job_{job_id}"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    audio_path = (
        work_dir
        /
        "voice.mp3"
    )


    final_path = (
        OUTPUT_DIR
        /
        f"{job_id}.mp4"
    )


    background_paths: list[Path] = []

    audio_clip = None
    final_clip = None


    try:

        # -------------------------
        # 1. Generate script
        # -------------------------

        script = generate_script(
            topic
        )


        # -------------------------
        # 2. Generate voice
        # -------------------------

        await create_voice(
            script,
            audio_path,
        )


        # -------------------------
        # 3. Read audio duration
        # -------------------------

        audio_clip = AudioFileClip(
            str(audio_path)
        )


        if (
            not audio_clip.duration
            or audio_clip.duration <= 0
        ):

            raise RuntimeError(
                "Generated voice has no valid duration."
            )


        duration = audio_clip.duration


        # -------------------------
        # 4. Download backgrounds
        # -------------------------

        search_queries = [
            topic,
            "technology",
            "abstract background",
        ]


        for query in search_queries:

            try:

                video_url = (
                    search_pexels_video(
                        query
                    )
                )


                video_path = (
                    work_dir
                    /
                    f"background_{len(background_paths)}.mp4"
                )


                download_file(
                    video_url,
                    video_path,
                )


                background_paths.append(
                    video_path
                )


                if len(background_paths) >= 3:
                    break


            except Exception:

                continue


        if not background_paths:

            raise RuntimeError(
                "Could not download a background video from Pexels."
            )


        # -------------------------
        # 5. Create vertical video
        # -------------------------

        final_clip = make_vertical_clip(
            background_paths,
            duration,
        )


        # -------------------------
        # 6. Add voice
        # -------------------------

        final_clip = final_clip.with_audio(
            audio_clip
        )


        # -------------------------
        # 7. Export MP4
        # -------------------------

        final_clip.write_videofile(
            str(final_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=2,
            logger=None,
        )


        if (
            not final_path.exists()
            or final_path.stat().st_size == 0
        ):

            raise RuntimeError(
                "Video export failed."
            )


        # -------------------------
        # SUCCESS
        # -------------------------

        return {
            "success": True,
            "video_url": (
                f"/static/{final_path.name}"
            ),
            "script": script,
        }


    except HTTPException:

        raise


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


    finally:

        # Close audio
        if audio_clip is not None:

            try:
                audio_clip.close()
            except Exception:
                pass


        # Close final video
        if final_clip is not None:

            try:
                final_clip.close()
            except Exception:
                pass


        # Delete temporary files
        try:

            for file in work_dir.iterdir():

                if file.is_file():
                    file.unlink()


            work_dir.rmdir()

        except Exception:

            pass
