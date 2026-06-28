import spacy
from langchain_core.tools import tool
import numpy as np

nlp = spacy.load("en_core_web_sm")


@tool
def sentence_length_statistics(text: str) -> dict:
    """Calculates average sentence length."""
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return {"info": "no sentences found."}
    length_array = np.array([len([t for t in sent if t.is_alpha]) for sent in sentences])

    total_words = length_array.shape[0]
    mean = np.mean(length_array).item()
    median = np.median(length_array).item()
    std = np.std(length_array).item()
    percentiles = np.percentile(length_array, [range(0, 101, 10)]).squeeze()
    return {
        "total_sentences": total_words,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "percentiles": {f"{i*10}th": np.round(percentiles[i], 2).item() for i in range(11)},
    }
