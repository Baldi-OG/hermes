from spellchecker import SpellChecker
import re
from langchain_core.tools import tool

spell = SpellChecker(language='en')


@tool
def typical_error_patterns(text: str) -> dict:
    """Look for spelling and punctuation mistakes."""

    # punctuation heuristic
    errors = {
        "multiple_spaces": len(re.findall(r' {2,}', text)),
        "missing_space_after_comma": len(re.findall(r',[^ \n]', text)),
    }

    # spelling analysis
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()

    if not words:
        return {"total_typos": 0, "typos_per_100_words": 0.0, "formatting_errors": errors}

    misspelled = spell.unknown(words)

    return {
        "total_typos": len(misspelled),
        "typos_per_100_words": round((len(misspelled) / len(words)) * 100, 2),
        "formatting_errors": errors,
        "example_typos": list(misspelled)[:5]
    }
