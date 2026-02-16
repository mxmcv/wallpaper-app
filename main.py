import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# Ensure this module exists in your project structure
from cards import get_card_url

app = FastAPI(title="Magic Card Reveal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FIX: MOVED CLIENT INITIALIZATION INSIDE A FUNCTION ---
def get_groq_client():
    """Safely get the Groq client only when needed."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # This logs the error to Vercel logs without crashing the whole app on startup
        print("CRITICAL: GROQ_API_KEY is missing.")
        raise HTTPException(status_code=500, detail="Server misconfiguration: API Key missing.")
    return Groq(api_key=api_key)

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


def _transcribe(audio_path: str) -> str:
    """Send an audio file to Groq Whisper and return the transcript."""
    client = get_groq_client()  # Initialize here
    
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return str(transcription).strip()


def _extract_card_code(transcript: str) -> str:
    """Ask Llama 3 to normalise a transcript into a card code."""
    client = get_groq_client() # Initialize here

    chat = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0,
        max_tokens=30,
    )
    return (chat.choices[0].message.content or "error").strip().lower()


@app.post("/card-reveal")
async def card_reveal(file: UploadFile = File(...)):
    """Accept a multipart audio file and return the matching card image URL."""
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Upload must be an audio file.")

    # Generate a unique temp file
    suffix = os.path.splitext(file.filename or "audio.m4a")[1] or ".m4a"
    
    # Use delete=False to ensure file persists for the duration of the request
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcript = _transcribe(tmp_path)
        card_code = _extract_card_code(transcript)
        # Ensure 'get_card_url' handles 'error' or unknown codes gracefully
        url, matched = get_card_url(card_code)

        return {
            "url": url,
            "card": card_code if matched else "back",
            "transcript": transcript,
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/")
async def health():
    return {"status": "ok", "service": "magic-card-reveal"}