import base64
import io
import random

from datasets import load_dataset
from openai import OpenAI
from PIL import Image

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL,
)

models = client.models.list()
MODEL_NAME = models.data[0].id

print("Using model:", MODEL_NAME)

dataset = load_dataset("OctoMed/PMC-VQA")
test_dataset = dataset["test"]

sample = random.choice(test_dataset)

def image_to_base64(image):
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

options = "\n".join(sample["options"])
prompt = f"""{sample["question"]}

{options}

Select the correct answer.

Analyze the medical image carefully and determine the correct answer and put your final answer inside <answer></answer>."""

image_b64 = image_to_base64(sample["image"])

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }
    ],
    max_completion_tokens=4096,
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)

message = response.choices[0].message

print("=" * 60)
print("QUESTION")
print("=" * 60)
print(sample["question"])
print("\nOPTIONS:")
for opt in sample["options"]:
    print(f"  {opt}")
print(f"\nGROUND TRUTH: {sample['answer']}")

print("\n" + "=" * 60)
print("REASONING")
print("=" * 60)
print(getattr(message, "reasoning", None))

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(message.content)

print("\n" + "=" * 60)
print("RAW MESSAGE")
print("=" * 60)
print(message)

print("\n" + "=" * 60)
print("USAGE")
print("=" * 60)
print(response.usage)