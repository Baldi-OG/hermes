import spacy
from langchain_core.tools import tool

nlp = spacy.load("en_core_web_sm")


@tool
def calculate_lexical_diversity(text: str) -> float:
    """Calculate lexical diversity"""
    # only count real words, no punctuation
    words = [token.text.lower() for token in nlp(text) if token.is_alpha]
    if not words:
        return 0.0
    return len(set(words)) / len(words)
