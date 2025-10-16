from fastapi import FastAPI, UploadFile, File
from openai import OpenAI, OpenAIError  # <- import OpenAIError here
import os
import shutil

app = FastAPI()

os.makedirs("uploads", exist_ok=True)

client = OpenAI(api_key=os.getenv("YOUR_OPEN_AI_API_KEY"))

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Transcribe audio
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        text = transcript.text

        # Summarize transcript
        prompt = f"""
        Summarize the following meeting transcript into:
        1. Key Summary (2-3 sentences)
        2. Key Decisions (bullet points)
        3. Action Items (bullet points)

        Transcript:
        {text}
        """

        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        summary_text = summary.choices[0].message.content

        return {"transcript": text, "summary": summary_text}

    except OpenAIError as e:
        # Catch all OpenAI API errors, including rate limits
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
