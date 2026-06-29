import sys
import os
import random
import re
import json
import logging
import hydra
import spacy
import time
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

MODEL_NAMES_FILTERED = models_list = [
    "gemma3-4b",
    "qwen3-0.6b-cpu",
    "tinyllama-1.1b-cpu",
    "smollm2-1.7b-cpu",
    "vicuna-7b",
    "gpt-oss-20b",
    "mistral-7b",
    "phi3-3.8b",
    "gemma3-270m-cpu",
    "qwen3-1.7b-cpu",
    "deepseek-r1-14b",
    "llama3.1-8b",
    "zephyr-7b",
    "gemma2-2b",
    "llama3-8b",
    "llama3.2-3b",
    "qwen3-1.7b",
    "qwen3-8b",
    "deepseek-r1-1.5b-cpu",
    "deepseek-r1-8b",
    "gemma3-1b-cpu"
]

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

        self.model_stats_dict = {}  # Dictionary to hold stats for each model

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
        total_retries = 0
        response_times = []

        for author in self.selected_authors:
            author_test_docs = self.test_df[self.test_df["author"] == author].sample(
                n=self.docs_per_author
            )

            for _, row in author_test_docs.iterrows():
                total_tests += 1
                start_time = time.time()
                predicted_author, retries_used = self._predict_single_document(
                    row["text"], row["author"], total_tests
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                total_retries += retries_used

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
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        avg_retries = total_retries / total_tests if total_tests > 0 else 0

        return accuracy, correct_predictions, total_tests, y_true, y_pred, avg_response_time, avg_retries

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
        retries_used = 0

        for attempt in range(self.max_retries):
            retries_used = attempt + 1
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

        return predicted_author, retries_used

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
            accuracy, correct, total, y_true, y_pred, avg_response_time, avg_retries = self.evaluate()
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

    def run_compare_models(self):
        # First, try to load any previously saved model stats so we can resume
        results_file = os.path.join(os.getcwd(), "outputs", "model_stats.json")
        # ensure output dir exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as fh:
                    self.model_stats_dict = json.load(fh)
                logging.info(f"Loaded existing model stats from {results_file}")
            except Exception as e:
                logging.warning(f"Failed to load existing model stats file {results_file}: {e}. Starting fresh.")
                self.model_stats_dict = {}

        # First, train profiles (only once, as it's the same for all models)
        logging.info("Building author profiles...")
        profile_stats = self.train_profiles()

        for model_name in MODEL_NAMES_FILTERED:
            logging.info(f"\n\nRunning experiment with model: {model_name}")
            logging.info("=" * 60)
            self.cfg.llm.model_name = model_name

            # If this model has already been evaluated (from disk or current run), skip it
            if model_name in self.model_stats_dict:
                logging.info(f"Model {model_name} already evaluated. Skipping.")
                # still write the file to ensure on-disk file is up-to-date
                try:
                    tmpfile = results_file + ".tmp"
                    with open(tmpfile, "w", encoding="utf-8") as fh:
                        json.dump(self.model_stats_dict, fh, indent=2)
                    os.replace(tmpfile, results_file)
                except Exception as e:
                    logging.warning(f"Failed to write model stats to disk: {e}")
                continue

            model_accuracies = []
            model_response_times = []
            model_retries = []

            # Run 5 requests for each model
            for run_num in range(5):
                logging.info(f"Run {run_num + 1}/5 for {model_name}")
                self.init_agent()
                accuracy, correct, total, y_true, y_pred, avg_response_time, avg_retries = self.evaluate()

                model_accuracies.append(accuracy)
                model_response_times.append(avg_response_time)
                model_retries.append(avg_retries)

                logging.info(f"Run {run_num + 1} - Accuracy: {accuracy:.2f}%, Avg Response Time: {avg_response_time:.2f}s, Avg Retries: {avg_retries:.2f}")

            # Calculate averages for this model
            avg_accuracy = sum(model_accuracies) / len(model_accuracies)
            avg_response_time_model = sum(model_response_times) / len(model_response_times)
            avg_retries_model = sum(model_retries) / len(model_retries)

            # Store in model_stats_dict
            self.model_stats_dict[model_name] = {
                "avg_accuracy": round(avg_accuracy, 2),
                "avg_response_time": round(avg_response_time_model, 2),
                "avg_retries": round(avg_retries_model, 2)
            }

            logging.info(f"Model {model_name} Averages:")
            logging.info(f"  - Accuracy: {avg_accuracy:.2f}%")
            logging.info(f"  - Response Time: {avg_response_time_model:.2f}s")
            logging.info(f"  - Retries: {avg_retries_model:.2f}")

            # after processing each model, serialize the full dict to disk atomically
            try:
                tmpfile = results_file + ".tmp"
                with open(tmpfile, "w", encoding="utf-8") as fh:
                    json.dump(self.model_stats_dict, fh, indent=2)
                os.replace(tmpfile, results_file)
            except Exception as e:
                logging.warning(f"Failed to write model stats to disk after evaluating {model_name}: {e}")

        # Print ordered results at the end
        logging.info("\n" + "=" * 60)
        logging.info("FINAL RESULTS - ALL MODELS")
        logging.info("=" * 60)

        # Sort by accuracy (descending)
        sorted_models = sorted(self.model_stats_dict.items(), key=lambda x: x[1]["avg_accuracy"], reverse=True)

        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            logging.info(f"{rank}. {model_name}")
            logging.info(f"   Accuracy: {stats['avg_accuracy']:.2f}%")
            logging.info(f"   Response Time: {stats['avg_response_time']:.2f}s")
            logging.info(f"   Retries: {stats['avg_retries']:.2f}")

        # Print as JSON
        logging.info("\nModel Stats Dictionary (JSON):")
        print(json.dumps(self.model_stats_dict, indent=2))


@hydra.main(version_base=None, config_path="../conf/", config_name="experiment.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.experiment.random_seed)
    spacy.util.fix_random_seed(cfg.experiment.random_seed)
    np.random.seed(cfg.experiment.random_seed)      # set numpy seed aswell even though we don't use it yet, just to be safe if it is later included

    logging.info("Loading data...")
    train_df, test_df = reuters_dataset.load_data()

    experiment = AuthorshipExperiment(cfg, train_df, test_df)
    experiment.run_compare_models()


if __name__ == "__main__":
    main()
