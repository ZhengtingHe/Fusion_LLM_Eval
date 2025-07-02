import requests
import json
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import DisplayConfig, AsyncConfig
from tqdm import tqdm
import requests
import re
from typing import List, Dict, Tuple, Any
import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from custom_metrics import answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric

from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import pydantic_core
import asyncio


display_config = DisplayConfig(
    show_indicator=True,
    print_results=False,
)
async_config = AsyncConfig(
    max_concurrent=20
)
RAG_test_dataset = EvaluationDataset()
RAG_test_dataset.add_test_cases_from_json_file(
    file_path='./RAG-test-dataset/20250701_123749.json',
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    retrieval_context_key_name="retrieval_context",
    context_key_name="context",
)

evaluation_output = evaluate(
    RAG_test_dataset.test_cases,
    metrics=[
        answer_relevancy_metric, 
        faithfulness_metric, 
        contextual_relevancy_metric
        ],
    display_config=display_config,
    async_config=async_config,
    )
evaluation_output_df = pd.DataFrame(
    [
        {
            "input": evaluation_output.test_results[i].input,
            "actual_output": evaluation_output.test_results[i].actual_output,
            "expected_output": evaluation_output.test_results[i].expected_output,
            "retrieval_context": evaluation_output.test_results[i].retrieval_context,
            "answer_relevancy_score": evaluation_output.test_results[i].metrics_data[0].score,
            "faithfulness_score": evaluation_output.test_results[i].metrics_data[1].score,
            "contextual_relevancy_score": evaluation_output.test_results[i].metrics_data[2].score,
        } 
    for i in range(len(evaluation_output.test_results))
    ])
evaluation_output.to_parquet('RAG-results/evaluation_output_of_RAG.parquet')
# answer_relevancy_scores = np.array([evaluation_output.test_results[k].metrics_data[0].score for k in range(len(evaluation_output.test_results))])
# faithfulness_scores = np.array([evaluation_output.test_results[k].metrics_data[1].score for k in range(len(evaluation_output.test_results))])
# contextual_relevancy_scores = np.array([evaluation_output.test_results[k].metrics_data[2].score for k in range(len(evaluation_output.test_results))])