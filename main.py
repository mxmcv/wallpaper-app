import os
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from cards import get_card_url

app = FastAPI(title="Magic Card Reveal API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are a card identifier. The user is naming a playing card.\n"
    "Your job is to convert their speech into a standardized filename format: '{rank}_{suit}'.\n"
    "Rules:\n"
    "1. Ranks: ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, jack, queen, king.\n"
    "2. Suits: clubs, diamonds, hearts, spades.\n"
    "3. Output: Return ONLY the formatted string (e.g., '7_diamonds', 'queen_hearts', 'ace_spades').\n"
    "4. If the user says 'Joker', return 'joker'.\n"
    "5. If no card is mentioned, return 'error'.\n"
    "6. Do not output markdown or explanation."
)

# --- HELPER: Lazy Load Client ---
def get_groq_client():
    """Safely initializes the Groq client only when needed."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: GROQ_API_KEY is missing from environment.")
        raise HTTPException(status_code=500, detail="Server configuration error: API Key missing.")
    return Groq(api_key=api_key)


# --- HELPER: Transcribe Audio ---
def _transcribe(audio_path: str) -> str:
    """Send an audio file to Groq Whisper and return the transcript."""
    client = get_groq_client()
    
    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        return str(transcription).strip()
    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# --- HELPER: Extract Card Code ---
def _extract_card_code(transcript: str) -> str:
    """Ask Llama 3 to normalize a transcript into a card code."""
    client = get_groq_client()

    try:
        chat = client.chat.completions.create(
            # --- FIX: UPDATED MODEL ID HERE ---
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
            max_tokens=30,
        )
        return (chat.choices[0].message.content or "error").strip().lower()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "error"


# --- ENDPOINT: Reveal Card ---
@app.post("/card-reveal")
async def card_reveal(file: UploadFile = File(...)):
    """Accept a multipart audio file and return the matching card image URL."""
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Upload must be an audio file.")

    suffix = os.path.splitext(file.filename or "audio.m4a")[1] or ".m4a"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        transcript = _transcribe(tmp_path)
        card_code = _extract_card_code(transcript)
        
        url, matched = get_card_url(card_code)

        return {
            "url": url,
            "card": card_code if matched else "back",
            "transcript": transcript,
        }

    except Exception as e:
        print(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --- ENDPOINT: Health Check ---
@app.get("/")
async def health():
    return {"status": "ok", "service": "magic-card-reveal"}