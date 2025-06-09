import pandas as pd
import json
from pathlib import Path # 导入Path类
import toml
with open('config.toml', 'r', encoding='utf-8') as toml_file:
    config = toml.load(toml_file)
output_jsonl_file = Path(config['ouput_question_jsonl_file'])
assert output_jsonl_file.suffix == '.jsonl'
# --- 使用 pathlib 动态计算路径 ---

# Path(__file__) 创建一个指向当前脚本的路径对象
# .resolve() 获取绝对路径
# .parent 获取该路径的父目录
# script_dir -> Path object for /path/to/your/project/Fusion_QA_Gen/goldens
script_dir = Path(__file__).resolve().parent

# .parent 可以链式调用，轻松地向上导航到项目根目录
# project_root -> Path object for /path/to/your/project/Fusion_QA_Gen
project_root = script_dir.parent

# 使用更直观的 / 操作符来拼接路径
# excel_file_path -> Path object for .../Fusion_QA_Gen/source/QA_tables.xlsx
excel_file_path = project_root / 'goldens' / config['input_question_excel_file']

# 构建输出文件的路径
# output_filename -> Path object for .../Fusion_QA_Gen/goldens/output.jsonl
output_filename = script_dir / output_jsonl_file

# --- 核心逻辑不变 ---

try:
    # pandas可以直接接受Path对象作为参数
    xls = pd.read_excel(excel_file_path, sheet_name=None)

    # open函数也可以直接使用Path对象
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # 遍历读取到的所有工作表
        for sheet_name, df in xls.items():
            domain = sheet_name
            df = df.fillna("")

            # 遍历当前工作表中的每一行
            for index, row in df.iterrows():
                record = {
                    "问题": row.get("问题"),
                    "客观分类": row.get("客观分类"),
                    "知识点": row.get("知识点"),
                    "主观难易度": row.get("主观难易度"),
                    "应用场景": row.get("应用场景"),
                    "领域": domain
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ 数据处理完成！")
    # Path对象在打印时会自动显示为操作系统的标准路径格式
    print(f"   - 输入文件: '{excel_file_path}'")
    print(f"   - 输出文件: '{output_filename}'")

except FileNotFoundError:
    print(f"❌ 错误: 文件 '{excel_file_path}' 未找到。请再次检查您的项目结构和文件名是否正确。")
except Exception as e:
    print(f"处理文件时发生了一个错误: {e}")