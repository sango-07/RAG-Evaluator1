import os
import ragas
import pandas as pd
from datasets import Dataset, load_dataset
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics.critique import harmfulness
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
    context_relevancy)
from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._answer_similarity import answer_similarity


# Ragas Evaluation
def ragas_eval(metrics, openai_api_key, df):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    df.rename(columns={"context": "contexts", "ground_truths": "ground_truth"}, inplace=True)
    df["contexts"] = df["contexts"].apply(lambda x: [x])
    eval_data = Dataset.from_pandas(df)

    metric_mappings = {
        "answer_correctness": answer_correctness,
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "context_relevancy": context_relevancy,
        "answer_similarity": answer_similarity,
    }



    for metric in metrics:
        if metric in metric_mappings:
            result = evaluate(eval_data, metrics=[metric_mappings[metric]], llm=llm, embeddings=embeddings, raise_exceptions=False)
            df2 = result.to_pandas()

            new_columns = [col for col in df2.columns if col not in df.columns]

            # Append only the new columns to final_df
            for col in new_columns:
                 df[col] = df2[col]

    return df