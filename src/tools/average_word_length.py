import spacy
from langchain_core.tools import tool
import numpy as np

nlp = spacy.load("en_core_web_sm")


@tool
def word_length_statistics(text: str) -> dict:
    """Calculates average word length."""
    doc = nlp(text)
    words = [token for token in doc if token.is_alpha]
    if not words:
        return {"info": "no words found."}

    length_array = np.array([len(token.text) for token in words])
    total_words = length_array.shape[0]
    mean = np.mean(length_array)
    median = np.median(length_array)
    std = np.std(length_array)
    percentiles = np.percentile(length_array, [range(0, 101, 10)]).squeeze()
    return {
        "total_words": total_words,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "percentiles": {f"{i*10}th": np.round(percentiles[i], 2).item() for i in range(11)},
    }
