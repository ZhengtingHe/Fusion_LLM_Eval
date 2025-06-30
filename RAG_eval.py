import requests
import json
from deepeval.dataset import EvaluationDataset
from tqdm import tqdm
import requests
import re
from typing import List, Dict, Tuple, Any

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from custom_metrics import answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric

from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

RAG_test_dataset = EvaluationDataset()
RAG_test_dataset.add_test_cases_from_json_file(
    file_path='./RAG-test-dataset/20250627_141131.json',
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
        ]
    )
# answer_relevancy_scores = np.array([evaluation_output.test_results[k].metrics_data[0].score for k in range(len(evaluation_output.test_results))])
# faithfulness_scores = np.array([evaluation_output.test_results[k].metrics_data[1].score for k in range(len(evaluation_output.test_results))])
# contextual_relevancy_scores = np.array([evaluation_output.test_results[k].metrics_data[2].score for k in range(len(evaluation_output.test_results))])