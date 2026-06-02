import spacy
from langchain_core.tools import tool

nlp = spacy.load("en_core_web_sm")


@tool
def average_sentence_length(text: str) -> float:
    """Calculates average sentence length."""
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    total_words = sum(len([t for t in sent if t.is_alpha])
                      for sent in sentences)
    return total_words / len(sentences)
