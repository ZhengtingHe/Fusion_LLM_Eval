from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# 你的 vLLM 服务器 IP 和端口
VLLM_API_BASE = "http://180.213.184.182:30083/v1" # 注意这里是 /v1，不是 /v1/chat/completions

llm = ChatOpenAI(
    model="XiaomiMiMo/MiMo-VL-7B-RL",  # 这里的 model 名称应与你 vLLM 启动时 `served-model-name` 或默认加载的 model name 一致
    openai_api_key="sk-no-key-required", # vLLM 通常不需要 API Key
    openai_api_base=VLLM_API_BASE,
    temperature=0.7,
    max_tokens=256,
    streaming=True # 开启流式输出，如果 vLLM 支持
)

# 3. 构建多模态消息（重要！）


# 定义图片 URL
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

# 构建 HumanMessage，其中包含文本和图片内容
messages = [
    HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
]

# 4. 发送请求并处理响应

try:
    # 对于流式输出，需要迭代
    print("Streaming response:")
    for chunk in llm.stream(messages):
        print(chunk.content or "", end="", flush=True)
    print("\n--- End of streaming response ---")

    # 如果不需要流式，可以直接调用invoke
    # response = llm.invoke(messages)
    # print("\nFull response:")
    # print(response.content)

except Exception as e:
    print(f"An error occurred: {e}")
    # 打印更多错误信息，尤其是来自服务器的原始错误
    # 如果是 HTTP 错误，langchain_openai 可能会封装它