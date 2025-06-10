import pandas as pd
import json
import asyncio
from pathlib import Path
import toml
from tqdm import tqdm
from collections import defaultdict

# LangChain specific imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- ⚙️ 1. CONFIGURATION (无变化) ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
with open(project_root / 'config.toml', 'r', encoding='utf-8') as toml_file:
    config = toml.load(toml_file)
goldens_path = project_root / 'goldens'
input_jsonl_path = goldens_path / config['goldens_question_jsonl_file']
print(f"Input JSONL file: {input_jsonl_path}")
output_excel_path = goldens_path / config['goldens_QA_excel_file']

# LLM API Configuration
BASE_URL = config['base_url']
API_KEY = config['api_key']
MODEL_NAME = config['model']
SYSTEM_PROMPT: str = Path(project_root / "system_prompt.md").read_text(encoding="utf-8")

# --- ⛓️ 2. LANGCHAIN SETUP (无变化) ---
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{question}")
])
model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.5,
)
output_parser = StrOutputParser()
chain = prompt | model | output_parser


# --- 🚀 3. MAIN ASYNC EXECUTION LOGIC (全新修改) ---
async def main():
    """
    Main async function using asyncio.gather for robust parallel processing.
    """
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f]
        print(f"✅ Successfully loaded {len(questions)} questions from '{input_jsonl_path}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found. Please run the data generation script first.")
        return

    inputs = [{"question": q.get("问题", "")} for q in questions]

    # --- 全新的带进度条的并发逻辑 ---

    # 1. 初始化tqdm进度条
    pbar = tqdm(total=len(inputs), desc="获取回答中")

    # 2. 定义一个简单的回调函数，用于在每个任务完成时更新进度条
    def update_progress(future):
        pbar.update(1)

    # 3. 创建所有任务，并为每个任务附加完成回调
    tasks = []
    for inp in inputs:
        # 创建任务
        task = asyncio.create_task(chain.ainvoke(inp))
        # 附加回调
        task.add_done_callback(update_progress)
        tasks.append(task)

    print(f"🚀 Sending {len(inputs)} requests to the LLM in parallel...")

    # 4. 使用 asyncio.gather 并发运行所有任务
    #    return_exceptions=True 是一个关键参数，它能确保即使部分任务失败，
    #    gather也不会立即中断，而是将异常作为结果返回。
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 5. 所有任务完成后，关闭进度条
    pbar.close()

    # --- 后续逻辑不变，但增加了对异常的处理 ---

    print("\n✅ All questions have been processed.")

    # 将结果合并回原始数据
    for i, question_data in enumerate(questions):
        result = answers[i]
        # 检查返回的结果是否是一个异常对象
        if isinstance(result, Exception):
            question_data["回答"] = f"ERROR: {result}"
        else:
            question_data["回答"] = result

    # --- 💾 4. SAVING TO EXCEL (全新修改) ---

    # 4a. 按“领域”对结果进行分组
    grouped_by_domain = defaultdict(list)
    for qa_pair in questions:
        domain = qa_pair.get("领域", "未分类") # 如果没有领域，则归为“未分类”
        grouped_by_domain[domain].append(qa_pair)

    # 4b. 使用ExcelWriter将每个组写入一个单独的sheet
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            print(f"✍️ Saving results to '{output_excel_path}' with multiple sheets...")
            
            # 遍历每个领域及其对应的数据
            for domain, data_list in grouped_by_domain.items():
                # 为当前领域创建一个DataFrame
                df_domain = pd.DataFrame(data_list)
                
                # 定义并筛选列顺序（不再需要“领域”列，因为它已经是sheet名）
                cols = ["问题", "回答", "客观分类", "知识点", "主观难易度", "应用场景"]
                df_domain = df_domain[[c for c in cols if c in df_domain.columns]]
                
                # Excel的sheet名有字符限制（如长度不能超过31），这里做一个简单的截断
                sheet_name = domain[:31]
                
                # 将DataFrame写入指定名称的sheet
                df_domain.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print(f"✅ Successfully saved results. Each domain is now a separate sheet in the Excel file.")
    except Exception as e:
        print(f"❌ ERROR: Failed to save results to Excel. {e}")


if __name__ == "__main__":
    asyncio.run(main())