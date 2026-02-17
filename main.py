# main.py
from __future__ import annotations

import os
import re
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

from cards import (
    extract_card_from_transcript,
    get_card_url,
    normalize_card_code,
)

app = FastAPI(title="Magic Card Reveal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CARD_BACK_URL = "https://raw.githubusercontent.com/mxmcv/wallpaper-app/core/images/cards/back.jpg"

# --- System prompt tuned for "messy convo" + strict output ---
SYSTEM_PROMPT = (
    "You are a card-extraction engine for a magic trick.\n"
    "Input will be a messy conversation transcript (banter, filler words, etc.) that includes a named playing card.\n"
    "Your job is to output the final chosen/settled playing card.\n\n"
    "OUTPUT FORMAT (STRICT): rank_suit\n"
    "Valid ranks: ace,2,3,4,5,6,7,8,9,10,jack,queen,king\n"
    "Valid suits: clubs,diamonds,hearts,spades\n"
    "Examples:\n"
    "Input: 'maybe the three... wait no the two of hearts' -> 2_hearts\n"
    "Input: 'bro it's obviously the two of hearts' -> 2_hearts\n"
    "Input: 'final answer: ace of spades' -> ace_spades\n\n"
    "Rules:\n"
    "- Output ONLY the code. No extra text.\n"
    "- If no card can be confidently identified, output: error\n"
)

# Bias Whisper toward hearing card vocabulary
WHISPER_BIAS_VOCAB = (
    "ace, two, three, four, five, six, seven, eight, nine, ten, "
    "jack, queen, king, clubs, diamonds, hearts, spades, "
    "2,3,4,5,6,7,8,9,10"
)


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")
    return Groq(api_key=api_key)


def _transcribe(audio_path: str) -> str:
    client = get_groq_client()
    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3",
                prompt=WHISPER_BIAS_VOCAB,
                response_format="text",
                language="en",
            )
        transcript = str(transcription).strip()
        print(f"[whisper] transcript: {transcript}")
        return transcript
    except Exception as e:
        print(f"[whisper] error: {e}")
        return ""


def _llm_extract_card_code(transcript: str) -> str:
    """
    LLM extraction is a *fallback*. We still validate and normalize after.
    """
    if not transcript:
        return "error"

    client = get_groq_client()
    try:
        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
        )

        raw = (chat.choices[0].message.content or "").strip().lower()
        print(f"[llm] raw: {raw}")

        # Hard-trim to first token line, just in case
        raw = raw.splitlines()[0].strip()

        # Must be either 'error' or a plausible rank_suit pattern
        if raw == "error":
            return "error"

        # very strict regex to avoid extra words
        if not re.fullmatch(r"(ace|[2-9]|10|jack|queen|king)_(clubs|diamonds|hearts|spades)", raw):
            return "error"

        return raw
    except Exception as e:
        print(f"[llm] error: {e}")
        return "error"


def _best_card_code(transcript: str) -> str:
    """
    1) Deterministic transcript scan (best when Whisper is okay)
    2) LLM extraction fallback
    3) Final normalization/validation (never reveal wrong card)
    """
    # Pass 1: deterministic scan from transcript
    code1 = extract_card_from_transcript(transcript)
    if code1:
        return code1

    # Pass 2: LLM fallback
    code2 = _llm_extract_card_code(transcript)
    if code2 != "error":
        # normalize_card_code also rejects invalid values
        normalized = normalize_card_code(code2)
        if normalized:
            return normalized

    # Pass 3: last-ditch deterministic normalize on full transcript
    code3 = normalize_card_code(transcript)
    if code3:
        return code3

    return "error"


@app.post("/card-reveal")
async def card_reveal(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.m4a")[1] or ".m4a"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        transcript = _transcribe(tmp_path)

        card_code = _best_card_code(transcript)
        url, matched = get_card_url(card_code)

        if not matched or card_code == "error":
            print("[result] no confident card found -> showing back")
            return {"url": CARD_BACK_URL, "card": "back", "transcript": transcript}

        print(f"[result] matched: {card_code}")
        return {"url": url, "card": card_code, "transcript": transcript}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/")
async def health():
    return {"status": "ok"}
