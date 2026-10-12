"""Dependency-injected OpenAI-compatible multimodal inference."""

from dataclasses import dataclass
import time
from medical_vlm.data.media import image_to_base64
from medical_vlm.evaluation.scoring import format_question, extract_answer, normalize_answer


@dataclass(frozen=True)
class InferenceSettings:
    model: str
    max_tokens: int = 4096


def create_client(base_url, api_key="EMPTY"):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=300.0, max_retries=2)


class ChatInference:
    def __init__(self, client, settings):
        self.client = client
        self.settings = settings

    def generate(self, sample, temperature=0.7):

        image_b64 = image_to_base64(sample["image"])
        prompt = format_question(sample)

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=self.settings.max_tokens,
            temperature=temperature,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                },
            },
        )

        latency = time.perf_counter() - start

        message = response.choices[0].message

        reasoning = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )

        final_answer = message.content or ""

        if not final_answer and reasoning:
            final_answer = reasoning

        return {
            "reasoning": reasoning,
            "finish_reason": response.choices[0].finish_reason,
            "response": final_answer,
            "latency": latency,
            "completion_tokens": getattr(
                response.usage,
                "completion_tokens",
                None,
            ),
            "prompt_tokens": getattr(
                response.usage,
                "prompt_tokens",
                None,
            ),
        }

    def generate_n(self, sample, k):
        attempts = []
        for attempt in range(k):
            temp = 0.6 if attempt == 0 else min(0.6 + attempt * 0.15, 1.0)
            result = self.generate(sample, temperature=temp)
            predicted = extract_answer(result["response"])
            normalized = normalize_answer(predicted)
            result["predicted"] = predicted
            result["normalized"] = normalized
            attempts.append(result)
        return attempts
