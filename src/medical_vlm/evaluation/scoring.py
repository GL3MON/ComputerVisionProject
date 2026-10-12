"""Original PMC-VQA prompt and scoring rules, kept stable for comparison."""

import re


def format_question(sample):
    options = "\n".join(sample["options"])

    return f"""{sample["question"]}

{options}

Select the correct answer.

Analyze the medical image carefully and determine the correct answer and put your final answer inside <answer></answer>."""


def extract_answer(response):
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return response.strip()


def normalize_answer(answer):
    answer = str(answer).strip()

    match = re.match(
        r"^([A-E])(?:\s*[:.)-]|\s+|$)",
        answer,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return answer.lower()
