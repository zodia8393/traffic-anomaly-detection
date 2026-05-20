"""AI Hub #71566 차종 카테고리 매핑."""

AIHUB_TO_13TYPE: dict[str, str] = {
    "승용차": "승용",
    "SUV": "승용",
    "트럭 1t": "화물(소)",
    "트럭 5t": "화물(대)",
    "덤프트럭": "화물(대)",
    "탱크로리": "특수",
    "소형버스": "승합(소)",
    "대형버스": "승합(대)",
    "특수차량": "특수",
}

AIHUB_TO_5CLASS: dict[str, str] = {
    "승용차": "승용",
    "SUV": "승용",
    "트럭 1t": "화물",
    "트럭 5t": "화물",
    "덤프트럭": "화물",
    "탱크로리": "특수",
    "소형버스": "버스",
    "대형버스": "버스",
    "특수차량": "특수",
}


def map_category(aihub_cat: str, target: str = "13type") -> str:
    if target == "13type":
        table = AIHUB_TO_13TYPE
    elif target == "5class":
        table = AIHUB_TO_5CLASS
    else:
        raise ValueError(f"Unknown target: {target!r}. Use '13type' or '5class'.")
    mapped = table.get(aihub_cat)
    if mapped is None:
        raise KeyError(f"Unknown AI Hub category: {aihub_cat!r}")
    return mapped
