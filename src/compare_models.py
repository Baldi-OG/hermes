import logging
import time

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import CONFIG
from tools import authorship_tools
import datasets.reuters_50_50.reuters as reuters_dataset

models_list = [
    "gemma3-4b",
    "qwen3-0.6b-cpu",
    "tinyllama-1.1b-cpu",
    "smollm2-1.7b-cpu",
    "vicuna-7b",
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

JSON_PREDICTION_KEYS = ("author", "prediction", "predicted_author")

class ModelEvalExperiment:

    def __init__(self, model_name, eval_df):
        self.model_name = model_name
        self.eval_df = eval_df

        self.max_retries = 2

        self.llm = None
        self.agent_executor = None

        self.init_agent()

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

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.1,
            base_url=CONFIG.WEBIS_URL_WEBUI,
            api_key=CONFIG.WEBIS_KEY_WEBUI,
            request_timeout=200,
            model_kwargs={"seed": 161}
        )

        self.agent_executor = create_agent(self.llm, authorship_tools)


def run(self):
    """
    Evaluate the model on authorship attribution task.
    Metrics: average response time, average retries, and accuracy.
    """
    response_times = []
    retries_list = []
    correct_count = 0
    total_count = len(self.eval_df)

    logging.info(f"Starting evaluation on {total_count} samples...")

    for idx, row in self.eval_df.iterrows():
        text = row.get('text', row.get('content', ''))
        true_author = row.get('author', row.get('label', None))

        if not text or not true_author:
            logging.warning(f"Skipping row {idx}: missing text or author")
            continue

        # Track response time and retries
        start_time = time.time()
        retries = 0
        correct_format = False
        predicted_author = None

        while retries <= self.max_retries and not correct_format:
            try:
                response = self.agent_executor.invoke({
                    "input": f"Identify the author of the following text:\n{text[:500]}"
                })

                result_text = response.get('output', '')

                # Try to extract author name from response
                if isinstance(result_text, dict):
                    predicted_author = result_text.get('author')
                else:
                    # Simple parsing - look for author in response
                    predicted_author = result_text.strip()

                if predicted_author:
                    correct_format = True

            except Exception as e:
                logging.warning(f"Error in retry {retries} for row {idx}: {e}")
                retries += 1
                continue

            retries += 1

        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        retries_list.append(retries)

        # Check accuracy
        if correct_format and predicted_author and predicted_author.lower() == true_author.lower():
            correct_count += 1

        if (idx + 1) % max(1, total_count // 5) == 0:
            logging.info(f"Progress: {idx + 1}/{total_count} samples processed")

    # Calculate and log metrics
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_retries = sum(retries_list) / len(retries_list) if retries_list else 0
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Model: {self.model_name}")
    logging.info(f"Average Response Time: {avg_response_time:.2f} seconds")
    logging.info(f"Average Retries Till Correct Format: {avg_retries:.2f}")
    logging.info(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    logging.info(f"{'=' * 60}\n")

    return {
        "model": self.model_name,
        "avg_response_time": avg_response_time,
        "avg_retries": avg_retries,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count
    }


if __name__ == '__main__':

    for model in models_list:
        logging.info(f"Evaluating model: {model}")

        logging.info("Loading data...")
        train_df, test_df = reuters_dataset.load_data()

        experiment = ModelEvalExperiment(model, test_df[0:10])  # Evaluate on a subset of the test data for speed
        experiment.run()
