import requests
import json
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import DisplayConfig, AsyncConfig, evaluate
from deepeval.test_case import LLMTestCase
# 明确导入原始的度量类
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualRecallMetric
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import pydantic_core
import asyncio

# --- 开始修改: 定义具有重试逻辑的子类 ---

# 定义一个通用的重试逻辑函数，避免代码重复
async def measure_with_retry(metric_instance, test_case: LLMTestCase, max_retries=3, delay=2, **kwargs):
    for attempt in range(max_retries):
        try:
            # 调用父类的 a_measure 方法
            return await super(metric_instance.__class__, metric_instance).a_measure(test_case, **kwargs)
        except pydantic_core.ValidationError as e:
            print(f"Validation Error on attempt {attempt + 1}/{max_retries} for metric '{metric_instance.__class__.__name__}'. Input: '{test_case.input[:50]}...'. Retrying...")
            if attempt + 1 >= max_retries:
                print(f"Max retries reached. Failing test case for metric '{metric_instance.__class__.__name__}'.")
                metric_instance.score = 0.0
                metric_instance.reason = f"Failed after {max_retries} retries due to: {e}"
                if hasattr(metric_instance, 'verdicts'):
                    metric_instance.verdicts = []
                return 0.0
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"An unexpected error occurred during metric measurement: {e}")
            metric_instance.score = 0.0
            metric_instance.reason = f"An unexpected error occurred: {e}"
            return 0.0
    return 0.0

# 1. 为 AnswerRelevancyMetric 创建一个带重试功能的子类
class RetryingAnswerRelevancyMetric(AnswerRelevancyMetric):
    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:
        return await measure_with_retry(self, test_case, **kwargs)

# 2. 为 FaithfulnessMetric 创建一个带重试功能的子类
class RetryingFaithfulnessMetric(FaithfulnessMetric):
    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:
        return await measure_with_retry(self, test_case, **kwargs)

# 3. 为 ContextualRelevancyMetric 创建一个带重试功能的子类
class RetryingContextualRelevancyMetric(ContextualRelevancyMetric):
    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:
        return await measure_with_retry(self, test_case, **kwargs)

# 4. 为 ContextualRecallMetric 创建一个带重试功能的子类
class RetryingContextualRecallMetric(ContextualRecallMetric):
    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:
        return await measure_with_retry(self, test_case, **kwargs)
# --- 修改结束 ---


# --- 接下来，使用这些新的子类来实例化你的度量标准 ---
# 假设你的 custom_metrics 是这样定义的，我们现在直接在这里创建实例
# ⭐ 重要: 请确保这里的参数 (threshold, model等) 和你原始 custom_metrics.py 文件中的一致
from custom_metrics import deepseek_model  # 确保你有正确的模型实例
answer_relevancy_metric = RetryingAnswerRelevancyMetric(
    threshold=0.7,
    model=deepseek_model,
    include_reason=False,
)
faithfulness_metric = RetryingFaithfulnessMetric(
    threshold=0.7,
    model=deepseek_model,
    include_reason=False,
)
contextual_relevancy_metric = RetryingContextualRelevancyMetric(
    threshold=0.7,
    model=deepseek_model,
    include_reason=False,
)
contextual_recall_metric = RetryingContextualRecallMetric(
    threshold=0.7,
    model=deepseek_model,
    include_reason=False,
)


# --- 您的原始代码（基本保持不变） ---

display_config = DisplayConfig(
    show_indicator=True,
    print_results=False,
)
async_config = AsyncConfig(
    max_concurrent=20
)
RAG_test_dataset = EvaluationDataset()

try:
    RAG_test_dataset.add_test_cases_from_json_file(
        file_path='./RAG-test-dataset/20250701_123749.json',
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        retrieval_context_key_name="retrieval_context",
        context_key_name="context",
    )
except FileNotFoundError:
    print("错误：找不到指定的数据集JSON文件。")
    exit()
import json
with open('goldens/QA_goldens.jsonl', 'r', encoding="utf-8") as f:
    goldens_QA = [json.loads(line) for line in f]
for i, case in enumerate(RAG_test_dataset.test_cases):
    assert case.input == goldens_QA[i]['问题']
    case.expected_output = goldens_QA[i]['参考答案']
print(f"开始评测 {len(RAG_test_dataset.test_cases)} 个测试用例...")

evaluation_output = evaluate(
    RAG_test_dataset.test_cases,
    metrics=[
        answer_relevancy_metric, 
        faithfulness_metric, 
        contextual_relevancy_metric,
        contextual_recall_metric
    ],
    display_config=display_config,
    async_config=async_config,
)

# 创建 DataFrame 的逻辑是正确的，无需修改
evaluation_output_df = pd.DataFrame(
    [
        {
            "input": result.input,
            "actual_output": result.actual_output,
            "expected_output": result.expected_output,
            "retrieval_context": result.retrieval_context,
            "answer_relevancy_score": result.metrics_data[0].score,
            "faithfulness_score": result.metrics_data[1].score,
            "contextual_relevancy_score": result.metrics_data[2].score,
            "contextual_recall_score": result.metrics_data[3].score,
        } 
        for result in evaluation_output.test_results
    ]
)

# 确保输出目录存在
output_dir = Path('RAG-results')
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'evaluation_output_of_RAG.parquet'

evaluation_output_df.to_parquet(output_file)

print(f"\n评测完成。结果已保存至 {output_file}")
print("DataFrame 表头:")
print(evaluation_output_df.head())