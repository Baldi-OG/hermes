import spacy
from langchain_core.tools import tool

nlp = spacy.load("en_core_web_sm")


@tool
def frequency_of_function_words(text: str) -> float:
    """Calculate frequency of function words in comparison to all words."""
    doc = nlp(text)
    words = [token for token in doc if token.is_alpha]
    if not words:
        return 0.0

    stop_words = [token for token in words if token.is_stop]
    return len(stop_words) / len(words)
