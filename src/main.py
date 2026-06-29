import sys
import os
import random
import re
import json
import logging
import hydra
import spacy
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import mlflow
import mlflow.langchain
import datasets.reuters_50_50.reuters as reuters_dataset

from config import CONFIG
from tools import authorship_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from visualization.confusion import plot_and_log_confusion_matrix

sys.path.append(os.path.abspath(".."))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", force=True
)
logging.getLogger("httpx").setLevel(logging.WARNING)

JSON_PREDICTION_KEYS = ("author", "prediction", "predicted_author")


class AuthorshipExperiment:
    def __init__(self, cfg: DictConfig, train_df, test_df):
        self.cfg = cfg
        self.train_df = train_df
        self.test_df = test_df

        # load hyperparams
        self.num_authors = cfg.experiment.num_authors_to_compare
        self.docs_per_author = cfg.experiment.docs_per_author
        self.max_retries = cfg.experiment.max_retries

        self.selected_authors = []
        self.author_profiles_json = "{}"
        self.agent_executor = None
        self.setup_mlflow()

    def setup_mlflow(self):
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        mlflow.langchain.autolog()

    def train_profiles(self):
        all_authors = self.train_df["author"].unique()
        self.selected_authors = random.sample(list(all_authors), k=self.num_authors)

        logging.info(f"Selected authors: {', '.join(self.selected_authors)}")

        filtered_train = self.train_df[
            self.train_df["author"].isin(self.selected_authors)
        ]
        # TODO: pretty sure some of our tool features are not used at all
        # convert these into proper dicts for aggregation
        filtered_train["word_length_statistics_json"] = (
            filtered_train["word_length_statistics_json"]
            .apply(json.loads)
        )
        filtered_train["measure_of_textual_diversity_json"] = (
            filtered_train["measure_of_textual_diversity_json"]
            .apply(json.loads)
        )
        # these features all contain multiple sub-features, so we need to normalize them into separate columns for aggregation
        nested_features = [
            "average_sentence_length",
            "word_length_statistics_json",
            "measure_of_textual_diversity_json",
        ]
        nested_dfs = [pd.json_normalize(filtered_train[feature]) for feature in nested_features]
        aggregated_nested_dfs = []
        for nested_df in nested_dfs:
            nested_df = nested_df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, setting errors to NaN
            tmp = (
                pd.concat(
                    [filtered_train["author"].reset_index(drop=True)
                    , nested_df.reset_index(drop=True)],
                    axis=1
                )
            )

            grouped = tmp.groupby("author").mean()

            aggregated_nested_dfs.append(grouped)
        profile_stats = (
            filtered_train.groupby("author")
            .agg(
                {
                    "lexical_diversity": "mean",
                    "sentence_dependency_depth": "mean",
                    "function_words_frequency": "mean",
                }
            )
            .round(3)
            .to_dict(orient="index")
        )
        for feature_name, aggregated_df in zip(nested_features, aggregated_nested_dfs):
            for author, row in aggregated_df.iterrows():
                profile_stats[author][feature_name] = row.to_dict()

        self.author_profiles_json = json.dumps(profile_stats, indent=2)
        self.author_names = list(profile_stats.keys())
        return profile_stats

    def init_agent(self):
        system_prompt = f"""

        You are an expert forensic linguist. Your task is to identify the author of the 'UNKNOWN TEXT' using your linguistic tools.
        Ignore the content of the text entirely and simply analyze the writing style, structure, and other This is NOT an interactive conversation.
        The input is a complete task.
        Never ask follow-up questions.
        Never ask what the user wants.
        Always produce a final prediction using the linguistic features, the tools at your disposal, and the author profiles provided below.
        
        Step 1: Use your tools to analyze the UNKNOWN TEXT.
        Step 2: Compare your findings with the following reference profiles of known authors:
        {self.author_profiles_json}
        
        Step 3: Make a decision. Explain your reasoning briefly, mentioning all metrics from your tools that influenced your decision.
        Step 4: You MUST end your response with the exact json format specified below.
        
        ---
        Respond with your explanation first.

        Then always end it with outputing ONLY the following JSON object:

        {'{'}
            "author": "<author>",
            "confidence": 0.0,
            "reasoning": "<brief explanation>"
        {'}'}
        ---
        """
        llm = ChatOpenAI(
            model=self.cfg.llm.model_name,
            temperature=self.cfg.llm.temperature,
            base_url=CONFIG.WEBIS_URL_WEBUI,
            api_key=CONFIG.WEBIS_KEY_WEBUI,
            request_timeout=self.cfg.llm.request_timeout,
            seed=self.cfg.experiment.random_seed,
        )
        self.agent_executor = create_agent(model=llm, tools=authorship_tools, system_prompt=system_prompt)

    def evaluate(self):
        logging.info(
            f"Start Evaluating ({self.num_authors} authors x {self.docs_per_author} docs)"
        )
        logging.info("-" * 50)

        correct_predictions = 0
        total_tests = 0
        y_true = []
        y_pred = []

        for author in self.selected_authors:
            author_test_docs = self.test_df[self.test_df["author"] == author].sample(
                n=self.docs_per_author
            )

            for _, row in author_test_docs.iterrows():
                total_tests += 1
                predicted_author = self._predict_single_document(
                    row["text"], row["author"], total_tests
                )

                safe_prediction = (
                    predicted_author
                    if predicted_author is not None
                    else "Failed/Unknown"
                )
                y_true.append(row["author"])
                y_pred.append(safe_prediction)

                if predicted_author == row["author"]:
                    logging.info(f"\t✅ Correct: {predicted_author}")
                    correct_predictions += 1
                elif predicted_author is not None:
                    logging.info(
                        f"\t❌ Incorrect: {predicted_author} (Expected: {row['author']})"
                    )
                else:
                    logging.error(
                        f"\t❌ Parsing failed completely after {self.max_retries} attempts."
                    )

        accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
        return accuracy, correct_predictions, total_tests, y_true, y_pred

    def _predict_single_document(
        self, unkown_text: str, true_author: str, test_id: int
    ):
        user_prompt = f"""
        Determine the author of this text.
        BEGIN UNKNOWN TEXT:

        {unkown_text}

        END UNKNOWN TEXT
        """

        logging.info(f"Test {test_id}: Analyze text by {true_author}...")
        predicted_author = None

        for attempt in range(self.max_retries):
            try:
                response = self.agent_executor.invoke({"messages": [("user", user_prompt)]})
                llm_output = response["messages"][-1].content
                print(llm_output)

                found_prediction = False

                # because LLMs can be unpredictable, they sometimes use different key names that we still want to parse
                for prediction_key in JSON_PREDICTION_KEYS:
                    match = re.search(rf'\{"{"}.*?"{prediction_key}":.*?"confidence":.*?\{"}"}', llm_output, re.DOTALL)
                    if match:
                        prediction = json.loads(match.group(0))

                        predicted_author = prediction[prediction_key]
                        if predicted_author not in self.selected_authors:
                            logging.warning(
                                f"\tAttempt {attempt+1} produced an author not in the selected authors: {predicted_author}. Output: {llm_output[-50:]}"
                            )
                            user_prompt += '\n\nERROR: The predicted author is not in the list of selected authors. Please provide only the json now.'
                            continue
                        confidence = prediction["confidence"]
                        print("Predicted Author:", predicted_author)
                        print("Confidence:", confidence)
                        found_prediction = True
                        break
                    else:
                        logging.warning(
                            f"\tAttempt {attempt+1} failed to parse. Output: {llm_output[-50:]}"
                        )
                        user_prompt += '\n\nERROR: You forgot to include {"author": "<author>", "confidence":0.0, "reasoning": "<brief explanation"} or formated it incorrectly. Please provide only the json now.'
                # if these all fail, try if the last line contains a markdown response with the author in **author** format
                if not found_prediction:
                    match = re.search(r'\*\*(.*?)\*\*', llm_output)
                    if match:
                        predicted_author = match.group(1)
                        if predicted_author not in self.selected_authors:
                            logging.warning(
                                f"\tAttempt {attempt+1} produced an author not in the selected authors: {predicted_author}. Output: {llm_output[-50:]}"
                            )
                            user_prompt += '\n\nERROR: The predicted author is not in the list of selected authors. Please provide only the json now.'
                            continue
                        confidence = None
                        print("Predicted Author (markdown):", predicted_author)
                        found_prediction = True
                if found_prediction:
                    break

            except Exception as e:
                logging.error(f"\t Error while calling agent: {e}")

        return predicted_author

    def run(self):
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "num_authors": self.num_authors,
                    "docs_per_author": self.docs_per_author,
                    "max_retries": self.max_retries,
                    "llm_model": self.cfg.llm.model_name,
                    "llm_temperature": self.cfg.llm.temperature,
                }
            )

            logging.info("Building author profiles...")
            profile_stats = self.train_profiles()
            mlflow.log_dict(profile_stats, "author_profiles.json")

            logging.info("Initializing Agent...")
            self.init_agent()

            logging.info("Running Evaluation...")
            accuracy, correct, total, y_true, y_pred = self.evaluate()
            plot_and_log_confusion_matrix(y_true, y_pred)
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "correct_predictions": correct,
                    "total_tests": total,
                }
            )

            logging.info("\n" + "=" * 50)
            logging.info("EVALUATION RESULTS")
            logging.info("=" * 50)
            logging.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
            logging.info("=" * 50)


@hydra.main(version_base=None, config_path="../conf/", config_name="experiment.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.experiment.random_seed)
    spacy.util.fix_random_seed(cfg.experiment.random_seed)
    np.random.seed(cfg.experiment.random_seed)      # set numpy seed aswell even though we don't use it yet, just to be safe if it is later included

    logging.info("Loading data...")
    train_df, test_df = reuters_dataset.load_data()

    experiment = AuthorshipExperiment(cfg, train_df, test_df)
    experiment.run()


if __name__ == "__main__":
    main()
