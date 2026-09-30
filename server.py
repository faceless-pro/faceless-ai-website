import os
import requests
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import edge_tts
from google import genai
from moviepy.editor import VideoFileClip, AudioFileClip

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")

client = genai.Client(api_key=GEMINI_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("output", exist_ok=True)
app.mount("/static", StaticFiles(directory="output"), name="static")

class VideoRequest(BaseModel):
    topic: str
    voice: str = "en-US-ChristopherNeural"

@app.get("/")
def index():
    return {"status": "Faceless AI Video Generator Active 🚀"}

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    try:
        topic_text = req.topic
        prompt = "Write a compelling 30-second video script about " + topic_text + ". Output ONLY plain text script without formatting or scene instructions."
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        script_text = response.text.strip()

        audio_path = "output/voice.mp3"
        communicate = edge_tts.Communicate(script_text, req.voice)
        await communicate.save(audio_path)

        headers = {"Authorization": PEXELS_KEY}
        pexels_url = "https://api.pexels.com/videos/search?query=" + topic_text + "&per_page=1&orientation=portrait"

        pexels_res = requests.get(pexels_url, headers=headers, timeout=15).json()

        if not pexels_res.get("videos") or len(pexels_res["videos"]) == 0:
            raise HTTPException(status_code=404, detail="No video found")

        video_download_url = pexels_res["videos"][0]["video_files"][0]["link"]
        video_path = "output/background.mp4"

        video_bytes = requests.get(video_download_url, timeout=30).content
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        final_path = "output/final_video.mp4"
        audio_clip = AudioFileClip(audio_path)
        video_clip = VideoFileClip(video_path)

        if video_clip.duration > audio_clip.duration:
            video_clip = video_clip.subclip(0, audio_clip.duration)

        if hasattr(video_clip, "with_audio"):
            final_clip = video_clip.with_audio(audio_clip)
        else:
            final_clip = video_clip.set_audio(audio_clip)

        final_clip.write_videofile(
            final_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            logger=None
        )

        audio_clip.close()
        video_clip.close()

        return {
            "status": "success",
            "script": script_text,
            "video_url": "/static/final_video.mp4"
        }

    except Exception as e:
        raise HTTPException(status_co
                            de=500, detail=str(e))
        
