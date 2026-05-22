import sys
import os
import random
import re
import json
import logging
import hydra
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    force=True
)
logging.getLogger("httpx").setLevel(logging.WARNING)


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

    def setup_mlflow(self):
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        mlflow.langchain.autolog()

    def train_profiles(self):
        all_authors = self.train_df["author"].unique()
        self.selected_authors = random.sample(
            list(all_authors), k=self.num_authors)

        logging.info(f"Selected authors: {', '.join(self.selected_authors)}")

        filtered_train = self.train_df[self.train_df["author"].isin(
            self.selected_authors)]

        profile_stats = filtered_train.groupby("author").agg({
            "lexical_diversity": "mean",
            "average_sentence_length": "mean",
            "sentence_dependency_depth": "mean",
            "function_words_frequency": "mean"
        }).round(3).to_dict(orient="index")

        self.author_profiles_json = json.dumps(profile_stats, indent=2)
        return profile_stats

    def init_agent(self):
        llm = ChatOpenAI(
            model=self.cfg.llm.model_name,
            temperature=self.cfg.llm.temperature,
            base_url=CONFIG.WEBIS_URL_WEBUI,
            api_key=CONFIG.WEBIS_KEY_WEBUI,
            request_timeout=self.cfg.llm.request_timeout
        )
        self.agent_executor = create_agent(llm, authorship_tools)

    def evaluate(self):
        logging.info(
            f"Start Evaluating ({self.num_authors} authors x {self.docs_per_author} docs)")
        logging.info("-" * 50)

        correct_predictions = 0
        total_tests = 0
        y_true = []
        y_pred = []

        for author in self.selected_authors:
            author_test_docs = self.test_df[self.test_df["author"] == author].sample(
                n=self.docs_per_author)

            for _, row in author_test_docs.iterrows():
                total_tests += 1
                predicted_author = self._predict_single_document(
                    row["text"], row["author"], total_tests)

                safe_prediction = predicted_author if predicted_author is not None else "Failed/Unknown"
                y_true.append(row["author"])
                y_pred.append(safe_prediction)

                if predicted_author == row["author"]:
                    logging.info(f"\t✅ Correct: {predicted_author}")
                    correct_predictions += 1
                elif predicted_author is not None:
                    logging.info(
                        f"\t❌ Incorrect: {predicted_author} (Expected: {row['author']})")
                else:
                    logging.error(
                        f"\t❌ Parsing failed completely after {self.max_retries} attempts.")

        accuracy = (correct_predictions / total_tests) * \
            100 if total_tests > 0 else 0
        return accuracy, correct_predictions, total_tests, y_true, y_pred

    def _predict_single_document(self, unbekannter_text: str, echter_autor: str, test_id: int):
        prompt = f"""
        You are an expert forensic linguist. Your task is to identify the author of the 'UNKNOWN TEXT' using your linguistic tools.
        
        Step 1: Use your tools to analyze the UNKNOWN TEXT.
        Step 2: Compare your findings with the following reference profiles of known authors:
        {self.author_profiles_json}
        
        Step 3: Make a decision. Explain your reasoning briefly.
        Step 4: You MUST end your response with the exact name of the chosen author enclosed in XML tags.
        
        ---
        Example Format:
        [Your reasoning here...]
        Therefore, based on the high lexical diversity and sentence depth, the text was written by John Doe.
        <prediction>John Doe</prediction>
        ---
        
        UNKNOWN TEXT:
        "{unbekannter_text}"
        """

        logging.info(f"Test {test_id}: Analyze text by {echter_autor}...")
        predicted_author = None

        for attempt in range(self.max_retries):
            try:
                response = self.agent_executor.invoke(
                    {"messages": [("user", prompt)]})
                llm_output = response["messages"][-1].content

                match = re.search(
                    r"<prediction>\s*(.*?)\s*</prediction>", llm_output, re.IGNORECASE | re.DOTALL)

                if match:
                    predicted_author = match.group(1).strip()
                    break
                else:
                    logging.warning(
                        f"\tAttempt {attempt+1} failed to parse. Output: {llm_output[-50:]}")
                    prompt += "\n\nERROR: You forgot to include the <prediction>AuthorName</prediction> tag. Please provide only the prediction tag now."

            except Exception as e:
                logging.error(f"\t Error while calling agent: {e}")

        return predicted_author

    def run(self):
        self.setup_mlflow()

        with mlflow.start_run():
            mlflow.log_params({
                "num_authors": self.num_authors,
                "docs_per_author": self.docs_per_author,
                "max_retries": self.max_retries,
                "llm_model": self.cfg.llm.model_name,
                "llm_temperature": self.cfg.llm.temperature
            })

            logging.info("Building author profiles...")
            profile_stats = self.train_profiles()
            mlflow.log_dict(profile_stats, "author_profiles.json")

            logging.info("Initializing Agent...")
            self.init_agent()

            logging.info("Running Evaluation...")
            accuracy, correct, total, y_true, y_pred = self.evaluate()
            plot_and_log_confusion_matrix(y_true, y_pred)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "correct_predictions": correct,
                "total_tests": total
            })

            logging.info("\n" + "=" * 50)
            logging.info("EVALUATION RESULTS")
            logging.info("=" * 50)
            logging.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
            logging.info("=" * 50)


@hydra.main(version_base=None, config_path="../conf/", config_name="experiment.yaml")
def main(cfg: DictConfig):
    logging.info("Loading data...")
    train_df, test_df = reuters_dataset.load_data()

    experiment = AuthorshipExperiment(cfg, train_df, test_df)
    experiment.run()


if __name__ == "__main__":
    main()
