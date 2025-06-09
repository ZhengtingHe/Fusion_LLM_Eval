import toml
with open('config.toml', 'r', encoding='utf-8') as toml_file:
    config = toml.load(toml_file)
from langchain_openai import ChatOpenAI     # pip install langchain-openai >=0.1.0
from pydantic import BaseModel
from openai import OpenAI
class score(BaseModel):
    score: int
    reason: str
json_schema = score.model_json_schema()
deepseek_json_chat = ChatOpenAI(
    model=config['model'],  # DeepSeek-R1-0528-AWQ
    base_url=config['base_url'],  # http://
    api_key=config['api_key'],                   # 本地端点无需鉴权可留空
    temperature=0,
    model_kwargs={
        # "response_format": {"type": "json_object"},   # ★ 启用 JSON-Mode
        # "extra_body": {"guided_json": json_schema}, # Using directly a JSON Schema
    },
)
res = deepseek_json_chat.invoke(
    "请给出一个包含score和reason的JSON对象，score为整数，reason为字符串。"
)
print(res.content)  # 输出模型返回的JSON字符串