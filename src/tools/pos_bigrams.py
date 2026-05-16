import spacy
from langchain_core.tools import tool
from collections import Counter

nlp = spacy.load("en_core_web_sm")


@tool
def calculate_pos_bi_grams(text: str) -> dict:
    """Calculates top-5 POS Bi-grams."""
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]

    if len(pos_tags) < 2:
        return {}

    bigrams = zip(pos_tags, pos_tags[1:])
    ngram_counts = Counter(bigrams)

    top_5 = ngram_counts.most_common(5)
    return {"_".join(k): v for k, v in top_5}
