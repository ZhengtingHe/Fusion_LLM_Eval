from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
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

deepseek_json_chat = ChatOpenAI(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    api_key="sk-47c6b7385f4d4e47af1969f6c99f2d4d",                   # 本地端点无需鉴权可留空
    temperature=0,
    model_kwargs={
        "response_format": {"type": "json_object"}   # ★ 启用 JSON-Mode
    },
)

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self._model = model

    def load_model(self):
        return self._model          # ChatOpenAI 实例

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self._model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "DeepSeek Chat (JSON)"

deepseek_model = CustomOpenAI(deepseek_json_chat)

correctness_metric = GEval(
    name              = "正确率",
    evaluation_steps  = [
        "检查最终结论是否与预期中的结论一致",
        "推导思路相近但结论不同的情况，认为是错误的",
        "如果出现公式和物理量，需要检查代表内容是否一致",
    ],
    evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT,LLMTestCaseParams.EXPECTED_OUTPUT],
    model=deepseek_model
)

dereivation_metric = GEval(
    name="过程分",
    evaluation_steps=[
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

    evaluate(test_cases=[test_case1, test_case2, test_case3], metrics=[correctness_metric, dereivation_metric, relevance_metric])