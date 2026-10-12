"""Qualitative generation callback used by SFT training."""

import torch
import wandb
from transformers import TrainerCallback


class SampleGenerationCallback(TrainerCallback):
    """Generates 1 sample completion on a fixed held-out question every eval,
    and logs it to wandb (and prints it) so you can qualitatively read
    outputs alongside the eval loss curve.
    """

    def __init__(self, processor, tokenizer, sample_question, max_new_tokens=256):
        self.processor = processor
        self.tokenizer = tokenizer
        self.sample_question = sample_question
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return

        was_training = model.training
        model.eval()

        # generation needs kv-cache; training runs with use_cache=False
        prev_use_cache = getattr(model.config, "use_cache", None)
        model.config.use_cache = True

        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.sample_question}],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        print(
            f"\n[sample generation @ step {state.global_step}]\n"
            f"Q: {self.sample_question}\nA: {generated_text}\n"
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "sample_generation": wandb.Table(
                        columns=["step", "question", "generation"],
                        data=[[state.global_step, self.sample_question, generated_text]],
                    )
                },
                step=state.global_step,
            )

        model.config.use_cache = prev_use_cache
        if was_training:
            model.train()
