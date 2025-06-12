import numpy as np
import pandas as pd
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import DisplayConfig, AsyncConfig
display_config = DisplayConfig(
    show_indicator=True,
    print_results=False,
    verbose_mode=False,
)
async_config = AsyncConfig(
    max_concurrent=20
)
import toml
with open('config.toml', 'r', encoding='utf-8') as toml_file:
    config = toml.load(toml_file)

model_names = config['model_names']
alter_names = model_names.copy() # Create a copy to modify

for i, name in enumerate(alter_names):
    if name in config['alternative_names']: 
        alter_names[i] = config['alternative_names'][name]

for name in alter_names:
    print(name)
INPUT_EXCEL_FILE = "goldens" / Path(config['QA_file_name'])
question_dfs = pd.read_excel(INPUT_EXCEL_FILE, sheet_name=None, index_col=0)
DOMAIN = list(question_dfs.keys())
num_questions_per_domain = question_dfs[DOMAIN[0]].shape[0]
print(f"共有{len(DOMAIN)}个领域，每个领域有{num_questions_per_domain}个问题")

QA_df = {}
for i, model in enumerate(model_names):
    QA_FILE = "QA" / Path(f"{model}_answers.xlsx")
    QA_df[model_names[i]] = pd.read_excel(QA_FILE, sheet_name=None, index_col=0)
from custom_metrics import get_dataset, correctness_metric, relevance_metric
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
inference_model = model_names[1]
reference_answer = model_names[5]
case_dataset = get_dataset(
        infer_model=inference_model,
        ref_model=reference_answer,
        question_dataframe=question_dfs,
        QA_dataframe=QA_df,
        domains=DOMAIN
    )
evaluation_output = evaluate(case_dataset, 
                             [correctness_metric], 
                             display_config=display_config,
                             async_config=async_config,
                             )
evaluation_output_df = pd.DataFrame(
    [
        {
            "input": evaluation_output.test_results[i].input,
            "actual_output": evaluation_output.test_results[i].input,
            "expected_output": evaluation_output.test_results[i].expected_output,
            "score": evaluation_output.test_results[i].metrics_data[0].score,
            "reason": evaluation_output.test_results[i].metrics_data[0].reason,
        } 
    for i in range(len(evaluation_output.test_results))
    ])
evaluation_output_df.to_excel(f"evaluation_output_of_{inference_model}_by_{reference_answer}.xlsx", index=False)