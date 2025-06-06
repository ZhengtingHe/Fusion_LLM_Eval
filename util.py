# import re
# from typing import Any, Dict, Union

# # 预编译正则表达式以提高效率
# _RE_MARKDOWN_CODE_BLOCK = re.compile(r'^(\s*```\w*\s*\n?)|(\s*```\s*$)', re.MULTILINE)
# _RE_REASON_MISSING_QUOTES = re.compile(
#     r'("reason"\s*:\s*)([^"\n][^\n]*)(\n?\s*[,\]}])', 
#     flags=re.S
# )
# _RE_DUEL_QUOTES = re.compile(
#     r'("reason"\s*:\s*)""(.*?)""(?=\s*[,\n}])',
#     flags=re.S
# )
# _RE_LOSE_BACKSLASH = re.compile(
#     r'(?<!\\)\\(?![\\/"bfnrtu])'
# )

# def repair_json_string(content: str) -> str:
#     """
#     修复JSON字符串，包含三层处理：
#     1. 移除Markdown代码块标记
#     2. 修复缺失引号的reason字段
#     3. 修复多余双引号和反斜杠问题
    
#     参数:
#         content (str): 要修复的JSON字符串
        
#     返回:
#         str: 修复后的JSON字符串
#     """
#     # 步骤1: 移除Markdown代码块标记
#     cleaned = _RE_MARKDOWN_CODE_BLOCK.sub('', content).strip()
    
#     # 步骤2: 修复缺失引号的reason字段
#     fixed = _RE_REASON_MISSING_QUOTES.sub(
#         lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
#         cleaned
#     )
    
#     # 步骤3: 修复多余双引号和反斜杠问题
#     fixed = _RE_DUEL_QUOTES.sub(lambda m: f'{m.group(1)}"{m.group(2)}"', fixed)
#     fixed = _RE_LOSE_BACKSLASH.sub(r'\\\\', fixed)
    
#     return fixed

import re

# 0) 预编译的原有正则
# 如果你看不懂这些正则，请丢给AI帮你解释
_RE_MARKDOWN_CODE_BLOCK = re.compile(r'^(\s*```\w*\s*\n?)|(\s*```\s*$)', re.MULTILINE)
_RE_REASON_MISSING_QUOTES = re.compile(r'("reason"\s*:\s*)([^"\n][^\n]*)(\n?\s*[,\]}])', re.S)
_RE_DUEL_QUOTES = re.compile(r'("reason"\s*:\s*)""(.*?)""(?=\s*[,\n}])', re.S)
_RE_LOSE_BACKSLASH = re.compile(r'(?<!\\)\\(?![\\/"bfnrtu])')

# 1) 新增：匹配 reason 整个字符串，方便单独处理内容
_RE_REASON_VALUE = re.compile(r'("reason"\s*:\s*")((?:\\.|[^\\])*)(")', re.S)
#                  └────1────┘└─────────2────────┘└3┘

def _escape_inner_quotes(match: re.Match) -> str:
    prefix, value, suffix = match.groups()
    # 只动未转义的 "      (?<!\\)"
    value = re.sub(r'(?<!\\)"', r'\"', value)
    return f'{prefix}{value}{suffix}'

def repair_json_string(content: str) -> str:
    """按照 4 步修复潜在的伪 JSON 字符串"""
    # ① 去掉 ```code block``` 标记
    cleaned = _RE_MARKDOWN_CODE_BLOCK.sub('', content).strip()

    # ② 如果 reason 后面本来没引号，加上引号
    fixed = _RE_REASON_MISSING_QUOTES.sub(
        lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
        cleaned,
    )

    # ③ 把 ""value"" → "value"
    fixed = _RE_DUEL_QUOTES.sub(lambda m: f'{m.group(1)}"{m.group(2)}"', fixed)

    # ④ 处理 value 里遗漏的反斜杠 / 引号
    fixed = _RE_REASON_VALUE.sub(_escape_inner_quotes, fixed)
    fixed = _RE_LOSE_BACKSLASH.sub(r'\\\\', fixed)

    return fixed


if __name__ == "__main__":
    test_cases = [
        '```json\n{"info": {"name": "刘五"}}\n```',
        '```\n{"info": {"name": "刘五"}}\n```',
        '{"info": {"name": "刘五"}}',
        '   ```json\n{"info": {"name": "刘五"}}\n```   ',
        '```json{"info": {"name": "刘五"}}```',
        'Some text ```json\n{"info": {"name": "刘五"}}\n``` more text'
        ]

    for i, test_str in enumerate(test_cases):
        cleaned = repair_json_string(test_str)
        print(f"测试用例 {i+1}:")
        print(f"原始: {repr(test_str)}")
        print(f"清理后: {repr(cleaned)}")
        print("-" * 50)
    good = '''
    {
    "score": 0,
    "reason": "The text does not follow the evaluation steps provided."
    }
    '''
    bad = '''
    {
        "score": 6,
        "reason": The Actual Output correctly identifies the importance ...
    }
    '''
    print(repair_json_string(good))
    print(repair_json_string(bad))

    bad = '''
    {
    "score": 6,
    "reason": "缺少 $P_\mathrm{aux}$ 等参数"
    }
    '''

    print(repair_json_string(bad))
    problematic_json = '''
    {
        "score": 6,
        "reason": "The Actual Output discusses p-¹¹B's advantages in fuel availability, reaction products, neutron production, material activation, and energy conversion, but it does not explicitly state the core conclusion that p-¹¹B's low neutron production is the primary reason for its "clean" label, as emphasized in the Expected Output. Additionally, the Actual Output contains some inaccuracies in reaction products and lacks the quantitative analysis and specific reaction equations provided in the Expected Output."
    }
    '''
    print(repair_json_string(problematic_json))