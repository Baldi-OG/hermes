import sys
import os
import random
import re
import json
import logging

from config import CONFIG
from tools import authorship_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import datasets.reuters_50_50.reuters as reuters_dataset

sys.path.append(os.path.abspath(".."))
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True
)

# Evaluation Config TODO: use hydra for managing this
NUM_AUTHORS_TO_COMPARE = 3
DOCS_PER_AUTHOR = 2

# load data and select authors
logging.info("Loading data...")
train_df, test_df = reuters_dataset.load_data()

all_authors = train_df["author"].unique()
selected_authors = random.sample(list(all_authors), k=NUM_AUTHORS_TO_COMPARE)

logging.info(f"Selected authors in benchmark: {', '.join(selected_authors)}")

# build author profiles
filtered_train = train_df[train_df["author"].isin(selected_authors)]

profile_stats = filtered_train.groupby("author").agg({
    "lexical_diversity": "mean",
    "average_sentence_length": "mean",
    "sentence_dependency_depth": "mean",
    "function_words_frequency": "mean"
}).round(3).to_dict(orient="index")
author_profiles_json = json.dumps(profile_stats, indent=2)

# setup llm and agent
llm = ChatOpenAI(
    model="qwen3-30b-a3b",
    temperature=0.1,
    base_url=CONFIG.WEBIS_URL_WEBUI,
    api_key=CONFIG.WEBIS_KEY_WEBUI,
    request_timeout=200
)
agent_executor = create_agent(llm, authorship_tools)

# Start Evaluation
logging.info(f"Start Evaluating ({NUM_AUTHORS_TO_COMPARE} authors x {DOCS_PER_AUTHOR} docs = {NUM_AUTHORS_TO_COMPARE * DOCS_PER_AUTHOR} tests)")
logging.info("-" * 50)

correct_predictions = 0
total_tests = 0

for author in selected_authors:
    author_test_docs = test_df[test_df["author"] == author].sample(n=DOCS_PER_AUTHOR)
    
    for idx, row in author_test_docs.iterrows():
        total_tests += 1
        unbekannter_text = row["text"]
        echter_autor = row["author"]
        
        prompt = f"""
        You are an expert forensic linguist. Your task is to identify the author of the 'UNKNOWN TEXT' using your linguistic tools.
        
        Step 1: Use your tools to analyze the UNKNOWN TEXT.
        Step 2: Compare your findings with the following reference profiles of known authors:
        {author_profiles_json}
        
        Step 3: Make a decision. Explain your reasoning briefly.
        Step 4: You MUST end your response with the exact name of the chosen author enclosed in tags like this:
        [PREDICTION: AuthorName]
        
        UNKNOWN TEXT:
        "{unbekannter_text}"
        """
        
        logging.info(f"Test {total_tests}: Analyse text by {echter_autor}...")
        
        try:
            response = agent_executor.invoke({"messages": [("user", prompt)]})
            llm_output = response["messages"][-1].content
            
            match = re.search(r"\[PREDICTION:\s*(.*?)\]", llm_output, re.IGNORECASE)
            
            if match:
                predicted_author = match.group(1).strip()
                if predicted_author == echter_autor:
                    logging.info(f"\t✅ Correct LLM prediction: {predicted_author}")
                    correct_predictions += 1
                else:
                    logging.info(f"\t❌ Incorrect LLM prediction: {predicted_author} (correct: {echter_autor})")
            else:
                logging.info(f"\tParsing Error.")
                logging.info(f"\tLLM Output: {llm_output}")
                
        except Exception as e:
            logging.info(f"\t⚠️ Error while calling: {e}")

# final report
accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0

logging.info("\n" + "=" * 50)
logging.info("EVALUATION RESULTS")
logging.info("=" * 50)
logging.info(f"Number of compared authors : {NUM_AUTHORS_TO_COMPARE}")
logging.info(f"Testsper auhor             : {DOCS_PER_AUTHOR}")
logging.info(f"Sum of tests               : {total_tests}")
logging.info(f"Number of correct responses: {correct_predictions}")
logging.info(f"Accuracy                   : {accuracy:.2f}%")
logging.info("=" * 50)