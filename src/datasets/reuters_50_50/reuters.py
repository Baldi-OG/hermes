import json
import pandas as pd
import logging

from tqdm import tqdm
from pathlib import Path
from tools import authorship_tools

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVE_PATH = BASE_DIR / "data" / "reuters_processed.parquet"

logger = logging.getLogger(__name__)


def _read_raw_files() -> pd.DataFrame:
    """Read all txt files and return structed data frame."""
    records = []

    for split in ["C50train", "C50test"]:
        split_path = DATA_DIR / split
        if not split_path.exists():
            raise FileNotFoundError(f"Directory not found: {split_path}")

        split_label = "train" if "train" in split else "test"

        for author_dir in sorted(split_path.iterdir()):
            if author_dir.is_dir():
                author_name = author_dir.name

                for txt_file in sorted(author_dir.glob("*.txt")):
                    try:
                        content = txt_file.read_text(encoding="utf-8", errors="replace")
                        records.append(
                            {
                                "text": content,
                                "author": author_name,
                                "split": split_label,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error while reading {txt_file}: {e}")

    return pd.DataFrame(records)


def process_and_save_dataset():
    """
    Process all document with all tools.
    """
    logger.info("NOTE: This procedure will only run once.")
    logger.info("Reading raw data...")
    df = _read_raw_files()
    total_docs = len(df)
    logger.info(f"{total_docs} documents read.")

    logger.info("\nProcess text with tools...")

    lex_diversities = []
    avg_sent_lengths = []
    punct_freqs = []
    func_word_freqs = []
    pos_ngrams = []
    dep_depths = []
    error_patterns = []
    word_length_statistics = []
    mltds = []

    for idx, text in tqdm(enumerate(df["text"])):
        avg_sent_lengths.append(authorship_tools[0].invoke(text))
        func_word_freqs.append(authorship_tools[1].invoke(text))
        lex_diversities.append(authorship_tools[2].invoke(text))
        pos_ngrams.append(json.dumps(authorship_tools[3].invoke(text)))
        punct_freqs.append(json.dumps(authorship_tools[4].invoke(text)))
        dep_depths.append(authorship_tools[5].invoke(text))
        error_patterns.append(json.dumps(authorship_tools[6].invoke(text)))
        word_length_statistics.append(json.dumps(authorship_tools[7].invoke(text)))
        mltds.append(json.dumps(authorship_tools[8].invoke(text)))

    df["lexical_diversity"] = lex_diversities
    df["average_sentence_length"] = avg_sent_lengths
    df["punctuation_frequency_json"] = punct_freqs
    df["function_words_frequency"] = func_word_freqs
    df["pos_ngrams_json"] = pos_ngrams
    df["sentence_dependency_depth"] = dep_depths
    df["typical_error_patterns_json"] = error_patterns
    df["word_length_statistics_json"] = word_length_statistics
    df["measure_of_textual_diversity_json"] = mltds

    logger.info(f"\nSave all processed files {SAVE_PATH}...")
    df.to_parquet(SAVE_PATH, index=False)
    logger.info("Done. Dataset is processed and compressed.")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns train and test split with all metrics. If these were not calculated before this system will begin to process them.

    Returns:
        tuple: (train_df, test_df)
    """
    if not SAVE_PATH.exists():
        process_and_save_dataset()

    df = pd.read_parquet(SAVE_PATH)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, test_df
