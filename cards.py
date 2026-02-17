# cards.py
from __future__ import annotations

import re
from typing import Optional, Tuple

GITHUB_BASE_URL = (
    "https://raw.githubusercontent.com/mxmcv/wallpaper-app/core/images/cards/"
)

RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]

VALID_CARDS: set[str] = {f"{rank}_{suit}" for rank in RANKS for suit in SUITS}

DEFAULT_IMAGE = f"{GITHUB_BASE_URL}back.jpg"

# --- Aliases / Whisper tolerance ---
RANK_ALIASES = {
    "a": "ace",
    "ace": "ace",
    "2": "2",
    "two": "2",
    "to": "2",     # whisper typo
    "too": "2",    # whisper typo
    "3": "3",
    "three": "3",
    "4": "4",
    "four": "4",
    "5": "5",
    "five": "5",
    "6": "6",
    "six": "6",
    "7": "7",
    "seven": "7",
    "8": "8",
    "eight": "8",
    "9": "9",
    "nine": "9",
    "10": "10",
    "ten": "10",
    "jack": "jack",
    "queen": "queen",
    "king": "king",
}

SUIT_ALIASES = {
    "club": "clubs",
    "clubs": "clubs",
    "diamond": "diamonds",
    "diamonds": "diamonds",
    "heart": "hearts",
    "hearts": "hearts",
    "spade": "spades",
    "spades": "spades",
}

FILLER_WORDS = {
    "of", "the", "a", "an", "my", "your", "was", "like"
}

# phrase weights to prefer "final answer / settle" type mentions
CONFIRMATION_PHRASES = [
    "it's actually",
    "its actually",
    "i swear",
    "actually is",
    "prove",
    "proof"
]


def _clean_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_card_code(raw: str) -> Optional[str]:
    """
    Convert messy input into canonical 'rank_suit' (e.g. '2_hearts').
    Returns None if it can't confidently parse into a valid 52-card entry.
    """
    if not raw:
        return None

    s = _clean_text(raw)
    if not s:
        return None

    tokens = [t for t in s.split(" ") if t and t not in FILLER_WORDS]
    if not tokens:
        return None

    rank = None
    suit = None

    for t in tokens:
        if rank is None and t in RANK_ALIASES:
            rank = RANK_ALIASES[t]
        if suit is None and t in SUIT_ALIASES:
            suit = SUIT_ALIASES[t]
        if rank and suit:
            break

    if not (rank and suit):
        return None

    code = f"{rank}_{suit}"
    return code if code in VALID_CARDS else None


def extract_card_from_transcript(transcript: str) -> Optional[str]:
    """
    Deterministically scan transcript for card mentions and pick the best candidate:
    - Prefer mentions near confirmation phrases (final answer / obviously / etc.)
    - Otherwise use the last valid mention
    Returns canonical 'rank_suit' or None.
    """
    if not transcript:
        return None

    text = _clean_text(transcript)
    if not text:
        return None

    # Split into rough "segments" for weighting (sentences-ish)
    segments = re.split(r"(?:\.\s+|\?\s+|\!\s+|\s{2,})", text)
    candidates: list[tuple[int, int, str]] = []  # (score, idx, code)

    for idx, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue

        # quick filter: if no suit word appears, skip (saves time)
        if not any(s in seg for s in SUIT_ALIASES.keys()):
            continue

        code = normalize_card_code(seg)
        if not code:
            # sometimes rank/suit are spread: try scanning tokens windows
            tokens = seg.split(" ")
            for i in range(len(tokens)):
                window = " ".join(tokens[i : i + 6])  # small rolling window
                code2 = normalize_card_code(window)
                if code2:
                    code = code2
                    break

        if not code:
            continue

        score = 0
        for phrase in CONFIRMATION_PHRASES:
            if phrase in seg:
                score += 5

        # slight preference for later segments (often the “settled” choice)
        score += idx

        candidates.append((score, idx, code))

    if not candidates:
        return None

    # Choose highest score; if tie, choose latest
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def get_card_url(card_code: str) -> Tuple[str, bool]:
    """Return (image_url, matched). Accepts canonical or messy card strings."""
    normalized = normalize_card_code(card_code)
    if normalized:
        return f"{GITHUB_BASE_URL}{normalized}.jpg", True
    return DEFAULT_IMAGE, False
