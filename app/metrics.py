from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower()
    replacements = {
        "二零二六": "2026",
        "十秒": "10秒",
        "臺": "台",
        "測": "测",
        "試": "试",
        "語": "语",
        "識": "识",
        "別": "别",
        "錄": "录",
        "線": "线",
        "後": "后",
        "續": "续",
        "應": "应",
        "體": "体",
        "數": "数",
        "據": "据",
        "寫": "写",
        "檔": "档",
        "務": "务",
        "話": "话",
        "頻": "频",
        "請": "请",
        "記": "记",
        "準": "准",
        "確": "确",
        "時": "时",
        "會": "会",
        "議": "议",
        "聲": "声",
        "關": "关",
        "錯": "错",
        "萬": "万",
        "發": "发",
        "處": "处",
        "條": "条",
        "內": "内",
        "與": "与",
        "轉": "转",
        "離": "离",
        "經": "经",
        "這": "这",
        "個": "个",
        "樣": "样",
        "於": "于",
        "評": "评",
        "動": "动",
        "備": "备",
        "傳": "传",
        "組": "组",
        "穩": "稳",
        "復": "复",
        "現": "现",
        "結": "结",
        "當": "当",
        "優": "优",
        "證": "证",
        "礎": "础",
        "課": "课",
        "筆": "笔",
        "輯": "辑",
        "驗": "验",
        "頓": "顿",
        "網": "网",
        "絡": "络",
        "賴": "赖",
        "匯": "汇",
        "總": "总",
        "載": "载",
        "較": "较",
        "長": "长",
        "實": "实",
        "鍊": "链",
        "鏈": "链",
        "煉": "链",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[\s,，。.!！？?、:：;；\"“”'‘’（）()《》<>【】\[\]-]", "", text)


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(expected: str, actual: str) -> tuple[float, int, int]:
    normalized_expected = normalize_text(expected)
    normalized_actual = normalize_text(actual)
    distance = edit_distance(normalized_expected, normalized_actual)
    expected_chars = max(1, len(normalized_expected))
    return distance / expected_chars, distance, expected_chars

