# Phoenix Evaluation
import os
from getpass import getpass
import nest_asyncio
nest_asyncio.apply()

import matplotlib.pyplot as plt
import openai
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.metrics import classification_report

from phoenix.evals import (
    HALLUCINATION_PROMPT_RAILS_MAP,
    HALLUCINATION_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)
import phoenix.evals.default_templates as templates
from phoenix.evals import (
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)

from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)
from phoenix.evals import (
    CODE_READABILITY_PROMPT_RAILS_MAP,
    CODE_READABILITY_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)


from phoenix.evals import (
    TOXICITY_PROMPT_RAILS_MAP,
    TOXICITY_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)

from phoenix.evals import (
    QA_PROMPT_RAILS_MAP,
    QA_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)

from phoenix.evals.default_templates import (
    REFERENCE_LINK_CORRECTNESS_PROMPT_RAILS_MAP,
    REFERENCE_LINK_CORRECTNESS_PROMPT_TEMPLATE
)
from phoenix.evals import (
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
    llm_generate,
    USER_FRUSTRATION_PROMPT_RAILS_MAP,
    USER_FRUSTRATION_PROMPT_TEMPLATE,
)
from phoenix.evals import (
    SQL_GEN_EVAL_PROMPT_TEMPLATE,
    SQL_GEN_EVAL_PROMPT_RAILS_MAP
)


def phoenix_eval(metrics, openai_api_key, df):
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = OpenAIModel(model="gpt-3.5-turbo", temperature=0.25)

    # Rename columns to match expected input names for evaluation
    df.rename(columns={"question": "input", "answer": "output", "cleaned_context": "reference"}, inplace=True)

    # Define a dictionary of metric configurations
    metric_mappings = {
        "hallucination": (HALLUCINATION_PROMPT_TEMPLATE, HALLUCINATION_PROMPT_RAILS_MAP, "Hallucination"),
        "toxicity": (TOXICITY_PROMPT_TEMPLATE, TOXICITY_PROMPT_RAILS_MAP, "Toxicity"),
        "relevance": (RAG_RELEVANCY_PROMPT_TEMPLATE, RAG_RELEVANCY_PROMPT_RAILS_MAP, "Relevancy"),
        "Q&A": (QA_PROMPT_TEMPLATE, QA_PROMPT_RAILS_MAP, "Q&A_eval"),
    }

    # Loop over each metric in the provided metrics list
    for metric in metrics:
        if metric in metric_mappings:
            template, rails_map, column_name = metric_mappings[metric]
            rails = list(rails_map.values())
            
            # Perform classification and add results to a new column for the current metric
            classifications = llm_classify(dataframe=df, template=template, model=model, rails=rails, concurrency=20)["label"].tolist()
            df[column_name] = classifications
        else:
            print(f"Warning: Metric '{metric}' is not supported.")

    # Rename columns back to their original names
    df.rename(columns={"input": "question", "output": "answer", "reference": "context"}, inplace=True)

    return df
