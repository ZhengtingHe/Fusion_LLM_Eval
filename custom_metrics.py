import toml
with open('config.toml', 'r', encoding='utf-8') as toml_file:
    config = toml.load(toml_file)
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
import json, logging

# Silence the urllib3 logger by setting its level to WARNING
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logger to the lowest level you want to capture

# --- Handler for WARNINGS ---
# This remains mostly the same from your original code.
warn_handler = logging.FileHandler('invalid_json.log', encoding='utf-8')
warn_handler.setLevel(logging.WARNING)
# warn_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
warn_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
warn_handler.setFormatter(warn_formatter)

# --- Handler for INFO ---
# We'll create a new handler for the info log.
info_handler = logging.FileHandler('info.log', encoding='utf-8')
info_handler.setLevel(logging.INFO)
# info_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
info_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
info_handler.setFormatter(info_formatter)

# This filter will ensure that only INFO messages are handled by the info_handler
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

info_handler.addFilter(InfoFilter())

# Add both handlers to the logger
logger.addHandler(warn_handler)
logger.addHandler(info_handler)
# The following line would now be ignored by the logging system
# because we set the "urllib3" logger's level to WARNING.
logging.getLogger("urllib3").info('This is a test from urllib3 and it will not be logged.')
# correctness_metric = GEval(
#     name="Correctness",
#     evaluation_steps=[
#         "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
#         "You should also heavily penalize omission of detail",
#         "Vague language, or contradicting OPINIONS, are OK",
#     ],
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
# )

# from langchain_deepseek import ChatDeepSeek
# deepseek_json_chat = ChatDeepSeek(
#     # model="DeepSeek-R1-0528-AWQ",
#     # base_url='http://180.213.184.177:30084/v1',
#     # api_key='Empty',

#     model="deepseek-reasoner",
#     base_url="https://api.deepseek.com",
#     api_key="sk-47c6b7385f4d4e47af1969f6c99f2d4d",
#     temperature=0.1,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# ).with_structured_output(method="json_mode")

from deepeval.models.base_model import DeepEvalBaseLLM
# class CustomOpenAI(DeepEvalBaseLLM):
#     def __init__(
#         self,
#         model
#     ):
#         self.model = model

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         chat_model = self.load_model()
#         return chat_model.invoke(prompt).content

#     async def a_generate(self, prompt: str) -> str:
#         chat_model = self.load_model()
#         res = await chat_model.ainvoke(prompt)
#         return res.content

#     def get_model_name(self):
#         return "Custom Model with OpenAI SDK"

# deepseek_model = CustomOpenAI(deepseek_json_chat)
from langchain_openai import ChatOpenAI     # pip install langchain-openai >=0.1.0
from pydantic import BaseModel
from openai import OpenAI
class score(BaseModel):
    score: int
    reason: str
json_schema = score.model_json_schema()
deepseek_json_chat = ChatOpenAI(
    model=config['model'],  
    base_url=config['base_url'],  
    api_key=config['api_key'],                 
    temperature=0,
    # model_kwargs={
    #     # "response_format": {"type": "json_object"},   # ★ 启用 JSON-Mode
    #     "extra_body": {"guided_json": json_schema}, # Using directly a JSON Schema
    # },
)

from util import repair_json_string

class CustomOpenAI(DeepEvalBaseLLM):
    
    def __init__(self, model, debug=False):
        self._model = model
        self._debug = debug
        self._model_name = config['model']

    def load_model(self):
        return self._model          # ChatOpenAI 实例

    def generate(self, prompt: str) -> str:
        content = self._model.invoke(prompt).content
        repaired_content = repair_json_string(content)
        if self.debug:
            # logging.info(f"Prompt send to LLM:{prompt}")

            try:
                json.loads(repaired_content)  # 尝试解析 JSON
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON caused by prompt: {prompt}")
                logging.warning(f"Invalid JSON response: {repaired_content}")
                logging.warning(f"Response before reparing: {content}")
                raise e
            return repaired_content
        else:
            return repaired_content

    async def a_generate(self, prompt: str) -> str:
        res = await self._model.ainvoke(prompt)
        content = res.content
        repaired_content = repair_json_string(content)
        if self._debug:
            # logging.info(f"Prompt send to LLM:{prompt}")
            # logging.info(f"Response from LLM: {content}")
            logging.info(f"Usage metadata: {res.usage_metadata}")
            try:
                json.loads(repaired_content)
                logging.info(f"Valid JSON: {content}")  
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON caused by prompt: {prompt}")
                logging.warning(f"Invalid JSON response: {repaired_content}")
                logging.warning(f"Response before reparing: {content}")
                raise e
            return repaired_content
        else:
            return repaired_content

    def get_model_name(self):
        return f"{self._model_name} (JSON)"

deepseek_model = CustomOpenAI(deepseek_json_chat, debug=True)  # 设置 debug=True 以启用 JSON 验证和日志记录

correctness_metric = GEval(
    name              = "正确率",
    evaluation_steps  = [
        "用简体中文陈述给分原因",
        "检查最终结论是否与预期中的结论一致",
        "推导思路相近但结论不同的情况，认为是错误的",
        "如果出现公式和物理量，需要检查代表内容是否一致",
        # "如果实际输出中包含预期输出中没有的内容，只要不影响结论并且与问题有关，认为是正确的",
    ],
    evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,LLMTestCaseParams.EXPECTED_OUTPUT],
    model=deepseek_model
)

derivation_metric = GEval(
    name="过程分",
    evaluation_steps=[
        "用简体中文陈述给分原因",
        "检查推导过程是否正确",
        "如果推导过程不完整，或者有错误的推导步骤，认为是错误的",
        "推导中使用了不同符号或公式，但代表的物理量一致，认为是正确的",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_model
)

relevance_metric = GEval(
    name="相关性",
    evaluation_steps=[
        "用简体中文陈述给分原因",
        "确保实际输出回答了输入的问题",
        "实际输出不应该包含与输入领域无关的信息",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    model=deepseek_model
)

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
def get_dataset(infer_model, ref_model, question_dataframe, QA_dataframe, domains):
    questions = []
    actual_output = []
    expected_output = []

    # 检查 domains 是否为列表，如果不是，则转换为列表
    if not isinstance(domains, list):
        domains = [domains]

    for domain in domains:
        questions.extend(question_dataframe[domain]['问题'].tolist())
        actual_output.extend(QA_dataframe[infer_model][domain][infer_model + '_answer'].tolist())
        expected_output.extend(QA_dataframe[ref_model][domain][ref_model + '_answer'].tolist())
    test_cases = []
    for i, question in enumerate(questions):
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output[i],
            expected_output=expected_output[i]
        )
        test_cases.append(test_case)   
    dataset = EvaluationDataset(test_cases=test_cases)
    return dataset

if __name__ == "__main__":
    # 测试 correctness_metric
    test_case1 = LLMTestCase(
        input="水能喝吗", 
        actual_output="冰镇可乐更好喝",
        expected_output="水能喝",
        )
    test_case2 = LLMTestCase(
        input='氢元素的同位素有哪些',
        actual_output='氢元素的同位素有氕、氘和氚',
        expected_output='氢元素一共有7 种已知的同位素，其中有3种是天然存在的，分别是氕、氘和氚。',
        )
    test_case3 = LLMTestCase(
        input='中国建国是哪一天',
        actual_output='中国建国是1949年10月1日',
        expected_output='中国（中华人民共和国）于1949年10月1日建立',
        )

    evaluate(test_cases=[test_case1, test_case2, test_case3], metrics=[correctness_metric, derivation_metric, relevance_metric])