from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

models = client.models.list()
MODEL_NAME = models.data[0].id

print("Model:", MODEL_NAME)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "What is 17 × 24? Think carefully and give the final answer.",
        }
    ],
    max_completion_tokens=512,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
    },
)

message = response.choices[0].message

print("\n========== REASONING ==========")
print(getattr(message, "reasoning", None))

print("\n========== FINAL ANSWER ==========")
print(message.content)

print("\n========== RAW MESSAGE ==========")
print(message)