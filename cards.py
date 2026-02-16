GITHUB_BASE_URL = (
    "https://raw.githubusercontent.com/YOUR_USERNAME/magic-app/main/images/cards/"
)

RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]

VALID_CARDS: set[str] = {f"{rank}_{suit}" for rank in RANKS for suit in SUITS}

DEFAULT_IMAGE = f"{GITHUB_BASE_URL}back.jpg"


def get_card_url(card_code: str) -> tuple[str, bool]:
    """Return (image_url, matched).

    If *card_code* matches a valid 52-card deck entry the URL points to that
    card image; otherwise the default "back of card" image is returned.
    """
    code = card_code.strip().lower()
    if code in VALID_CARDS:
        return f"{GITHUB_BASE_URL}{code}.jpg", True
    return DEFAULT_IMAGE, False
