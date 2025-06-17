import json
import os
import re

PRONUNCIATIONS_DICT_PATH = os.getenv("PRONUNCIATION_DICT_PATH", "/app/api/pronunciations.json")

_pronunciations: dict[str, str] = {}


def load_pronunciations() -> None:
    """Load pronunciation dictionary from disk if available."""
    global _pronunciations
    if os.path.exists(PRONUNCIATIONS_DICT_PATH):
        try:
            with open(PRONUNCIATIONS_DICT_PATH, "r", encoding="utf-8") as f:
                _pronunciations = json.load(f)
        except Exception:
            _pronunciations = {}
    else:
        _pronunciations = {}


def save_pronunciations() -> None:
    """Persist pronunciation dictionary to disk."""
    with open(PRONUNCIATIONS_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(_pronunciations, f, ensure_ascii=False, indent=2)


def get_pronunciations() -> dict[str, str]:
    """Get current pronunciation dictionary."""
    return _pronunciations


def update_pronunciation(word: str, phonemes: str) -> None:
    """Add or update a word pronunciation and save."""
    if not word or not phonemes:
        raise ValueError("Word and phonemes must be provided")
    _pronunciations[word.lower()] = phonemes
    save_pronunciations()


def apply_pronunciations(text: str) -> str:
    """Apply dictionary pronunciations using custom phoneme syntax."""
    if not _pronunciations:
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        phon = _pronunciations.get(word.lower())
        if phon:
            return f"[{word}](/" + phon + "/)"
        return word

    return re.sub(r"\b\w+\b", repl, text)


# Load dictionary at import
load_pronunciations()
