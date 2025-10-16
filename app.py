from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import os
import shutil

app = FastAPI()

# Create upload folder if not exists
os.makedirs("uploads", exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key="Your_OpenAi_API_Key")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1️⃣ Transcribe audio using Whisper (new API)
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    text = transcript.text

    # 2️⃣ Summarize using GPT
    prompt = f"""
    Summarize the following meeting transcript into:
    1. Key Summary (2-3 sentences)
    2. Key Decisions (bullet points)
    3. Action Items (bullet points)

    Transcript:
    {text}
    """

    summary = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )

    summary_text = summary.choices[0].message.content

    return {
        "transcript": text,
        "summary": summary_text
    }
