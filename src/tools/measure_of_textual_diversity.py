import spacy
from langchain_core.tools import tool
import numpy as np

nlp = spacy.load("en_core_web_sm")


@tool
def measure_of_textual_diversity(text: str, min_ttr=0.72) -> dict:
    """Calculates the mltd of the text, this measures how long it takes for the TTR(lexical diversity) to drop below a certain threshold as more words are added."""
    doc = nlp(text)
    words = [token for token in doc if token.is_alpha]
    if len(words) == 0:
        return {"info": "no words found."}

    unique_words = set()
    last_cutoff_index = 0
    previous_lengths = []
    for i in range(1, len(words) + 1):
        next_word = words[i - 1]
        unique_words.update(next_word.text.lower())
        ttr = len(unique_words) / (i - last_cutoff_index)
        if ttr < min_ttr:
            previous_lengths.append(i - last_cutoff_index)
            last_cutoff_index = i
            unique_words.clear()
    length_array = np.array(previous_lengths)
    mean = np.mean(length_array)
    median = np.median(length_array)
    std = np.std(length_array)
    percentiles = np.percentile(length_array, [range(0, 101, 10)]).squeeze()
    return {
        "total_words": len(words),
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "percentiles": {f"{i*10}th": np.round(percentiles[i], 2).item() for i in range(11)},
    }
