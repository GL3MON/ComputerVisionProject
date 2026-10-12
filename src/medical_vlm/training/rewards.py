"""Strict checklist verifier independent of the trainer."""

import json
import re
import requests

VERIFIER_TEMPLATE = """You are an extremely strict answer-completeness verifier.

Your task is to determine whether the candidate FINAL ANSWER contains EVERY
required component of the GROUND-TRUTH ANSWER. You need to verrify that the candidate's answer is complete and does not omit any required information.


Rules:

1. Missing even ONE required component means incorrect.
2. If the ground truth contains multiple required items, check EVERY item
   separately.
3. Do not give partial credit.
4. Do not infer a missing answer from the candidate's reasoning.
5. Synonyms are allowed only when they clearly express the same concept.
6. Extra explanation is allowed.
7. The candidate must actually answer the question.
8. Do not mark an item present merely because it is implied.
9. The "evidence" field must be a SHORT VERBATIM QUOTE from the candidate
   final answer showing that required item.
10. Keep evidence under 15 words.
11. If an item is absent, set present=false and evidence="".
12. If the ground truth is a short single answer, that exact answer or a
    clearly equivalent expression must be present in the candidate.
13. Do not reward a candidate just because it discusses related information.
14. Do not reward a partially correct list.
15. ALL required items must be present for the answer to be correct.

GROUND-TRUTH ANSWER:
{ground_truth}

CANDIDATE FINAL ANSWER:
{completion}

Return ONLY a JSON object.
Do NOT use Markdown fences.
Do NOT output anything before or after the JSON.

Use exactly this structure:

{{
  "correct": true,
  "items": [
    {{
      "required": "required component",
      "present": true,
      "evidence": "short quote from candidate"
    }}
    ... all necessary components
  ],
  "reason": "brief explanation"
}}

Set "correct" to true ONLY when EVERY item has "present": true.
"""


def completion_to_text(completion):
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        pieces = []

        for item in completion:
            if not isinstance(item, dict):
                continue

            content = item.get("content", "")

            if isinstance(content, str):
                pieces.append(content)

        return "".join(pieces)

    return str(completion)


def prompt_to_text(prompt):
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        pieces = []

        for message in prompt:
            if not isinstance(message, dict):
                continue

            content = message.get("content", "")

            if isinstance(content, str):
                pieces.append(content)

            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        pieces.append(item.get("text", ""))

        return "\n".join(pieces)

    return str(prompt)


def split_reasoning_and_answer(completion):
    text = completion.strip()

    if "</think>" in text:
        reasoning, answer = text.split(
            "</think>",
            1,
        )

        reasoning = reasoning.strip()

        if "<think>" in reasoning:
            reasoning = reasoning.split(
                "<think>",
                1,
            )[1].strip()

        return reasoning, answer.strip()

    if "<think>" in text:
        reasoning = text.split(
            "<think>",
            1,
        )[1].strip()

        return reasoning, ""

    return "", text


def has_reasoning(completion):
    reasoning, _ = split_reasoning_and_answer(completion)

    return bool(reasoning.strip())


def parse_verifier_output(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if not match:
            return (
                False,
                [],
                "Invalid JSON",
            )

        try:
            result = json.loads(match.group(0))

        except json.JSONDecodeError:
            return (
                False,
                [],
                "Invalid JSON",
            )

    if not isinstance(result, dict):
        return (
            False,
            [],
            "Verifier output is not a JSON object",
        )

    items = result.get("items")

    if not isinstance(items, list) or not items:
        return (
            False,
            [],
            "No checklist returned",
        )

    for item in items:
        if not isinstance(item, dict):
            return (
                False,
                items,
                "Invalid checklist item",
            )

        required = str(item.get("required", "")).strip()

        present = item.get(
            "present",
            False,
        )

        if not required:
            return (
                False,
                items,
                "Checklist item has no required field",
            )

        if present is not True:
            return (
                False,
                items,
                f"Missing required component: {required}",
            )

    # IMPORTANT:
    # The checklist is authoritative.
    # We intentionally DO NOT trust result["correct"].
    return (
        True,
        items,
        str(
            result.get(
                "reason",
                "",
            )
        ),
    )


class VerifierReward:
    def __init__(self, url, model, max_tokens=512):
        self.url, self.model, self.max_tokens = url, model, max_tokens

    def verifier_reward(
        self,
        prompts,
        completions,
        ground_truth,
        log_extra=None,
        log_metric=None,
        **kwargs,
    ):
        completion_texts = [completion_to_text(x) for x in completions]

        rewards = [-1.0 for _ in completion_texts]

        reasoning_texts = []
        final_answer_texts = []
        reasoning_flags = []

        for completion in completion_texts:
            reasoning, answer = split_reasoning_and_answer(completion)

            reasoning_texts.append(reasoning)
            final_answer_texts.append(answer)

            reasoning_flags.append(bool(reasoning.strip()))

        verifier_correct = [None for _ in completion_texts]

        verifier_reason = [None for _ in completion_texts]

        verifier_items = [None for _ in completion_texts]

        verify_indices = [i for i, has_think in enumerate(reasoning_flags) if has_think]

        # No reasoning -> -1 automatically.
        # These examples are NOT sent to the verifier.

        if verify_indices:
            verifier_prompts = [
                VERIFIER_TEMPLATE.format(
                    ground_truth=ground_truth[i],
                    completion=final_answer_texts[i],
                )
                for i in verify_indices
            ]

            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": verifier_prompts,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                },
                timeout=600,
            )

            response.raise_for_status()

            data = response.json()

            choices = sorted(
                data["choices"],
                key=lambda x: x["index"],
            )

            if len(choices) != len(verify_indices):
                raise RuntimeError(
                    f"Verifier returned {len(choices)} results for {len(verify_indices)} requests"
                )

            for i, choice in zip(
                verify_indices,
                choices,
            ):
                verifier_output = choice["text"].strip()

                correct, items, reason = parse_verifier_output(verifier_output)

                verifier_correct[i] = correct
                verifier_reason[i] = reason
                verifier_items[i] = items

                rewards[i] = 1.0 if correct else 0.1

        if log_metric is not None:
            n = len(rewards)

            if n > 0:
                log_metric(
                    "verifier_mean_reward",
                    sum(rewards) / n,
                )

                log_metric(
                    "verifier_correct_rate",
                    sum(r == 1.0 for r in rewards) / n,
                )

                log_metric(
                    "verifier_incorrect_rate",
                    sum(r == 0.1 for r in rewards) / n,
                )

                log_metric(
                    "no_reasoning_rate",
                    sum(r == -1.0 for r in rewards) / n,
                )

        if log_extra is not None:
            log_extra(
                "ground_truth",
                [str(x) for x in ground_truth],
            )

            log_extra(
                "reasoning",
                reasoning_texts,
            )

            log_extra(
                "final_answer",
                final_answer_texts,
            )

            log_extra(
                "has_reasoning",
                reasoning_flags,
            )

            log_extra(
                "verifier_correct",
                ["NA" if x is None else x for x in verifier_correct],
            )

            log_extra(
                "verifier_reason",
                ["NA" if x is None else x for x in verifier_reason],
            )

            log_extra(
                "verifier_items",
                verifier_items,
            )

            log_extra(
                "reward",
                rewards,
            )

        return rewards
