import spacy
from langchain_core.tools import tool
from collections import Counter

nlp = spacy.load("en_core_web_sm")


@tool
def punctuation_frequency(text: str) -> dict:
    """Calculate relative frequency for punctuations."""
    doc = nlp(text)
    puncts = [token.text for token in doc if token.is_punct]
    if not puncts:
        return {"info": "no punctuation found."}

    total = len(puncts)
    counts = Counter(puncts)
    return {char: round(count / total, 4) for char, count in counts.items()}
