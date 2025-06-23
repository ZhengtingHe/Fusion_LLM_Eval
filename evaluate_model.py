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

# for name in alter_names:
#     print(name)
INPUT_EXCEL_FILE = "goldens" / Path(config['QA_file_name'])
question_dfs = pd.read_excel(INPUT_EXCEL_FILE, sheet_name=None, index_col=0)
DOMAIN = list(question_dfs.keys())
num_questions_per_domain = question_dfs[DOMAIN[0]].shape[0]
print(f"共有{len(DOMAIN)}个领域，每个领域有{num_questions_per_domain}个问题")

from util import get_QA_from_different_models
QA_df = get_QA_from_different_models()

from custom_metrics import get_dataset, correctness_metric, relevance_metric
from deepeval import evaluate

inference_model = model_names[1]
reference_answer = model_names[6]
case_dataset = get_dataset(
        infer_model=inference_model,
        ref_model=reference_answer,
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
from util import sanitize_model_name
# Ensure the model names are sanitized for file naming
sanitized_inference_model = sanitize_model_name(inference_model)
sanitized_reference_answer = sanitize_model_name(reference_answer)
evaluation_output_file_name = Path(f"evaluation_output_of_{sanitized_inference_model}_by_{sanitized_reference_answer}.parquet")
project_dir = Path(__file__).parent
output_path = project_dir / "eval_output"
evaluation_output_df.to_parquet(output_path / evaluation_output_file_name)

trained_inference_model = model_names[2]
reference_answer = model_names[6]
case_dataset = get_dataset(
        infer_model=trained_inference_model,
        ref_model=reference_answer,
        QA_dataframe=QA_df,
        domains=DOMAIN
    )
trained_evaluation_output = evaluate(case_dataset, 
                             [correctness_metric], 
                             display_config=display_config,
                            #  hyperparameters={"Temperature": 0.1, "Max Tokens": 50000, "System Prompt": "You MUST NOT add any extra commentary outside the JSON"}
                             )
trained_evaluation_output_df = pd.DataFrame(
    [
        {
            "input": trained_evaluation_output.test_results[i].input,
            "actual_output": trained_evaluation_output.test_results[i].input,
            "expected_output": trained_evaluation_output.test_results[i].expected_output,
            "score": trained_evaluation_output.test_results[i].metrics_data[0].score,
            "reason": trained_evaluation_output.test_results[i].metrics_data[0].reason,
        } 
    for i in range(len(trained_evaluation_output.test_results))
    ])
sanitized_trained_inference_model = sanitize_model_name(trained_inference_model)
trained_evaluation_output_file_name = Path(f"trained_evaluation_output_of_{sanitized_trained_inference_model}_by_{sanitized_reference_answer}.parquet")
trained_evaluation_output_df.to_parquet(output_path / trained_evaluation_output_file_name)
