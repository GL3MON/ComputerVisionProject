"""GRPO orchestration with explicit configuration and a reusable verifier."""

import argparse
import os
from medical_vlm.core.config import load_config, filter_supported_kwargs
from medical_vlm.core.paths import configure_wandb, project_root, resolve_path
from medical_vlm.data.verifiable import format_example
from medical_vlm.training.rewards import VerifierReward

DEFAULTS = {
    "base_model": "Qwen/Qwen3-VL-2B-Thinking",
    "sft_checkpoint": "",
    "dataset_name": "FreedomIntelligence/medical-o1-verifiable-problem",
    "output_dir": "outputs/qwen3-vl-2b-medical-grpo",
    "num_generations": 4,
    "max_completion_length": 2048,
    "verifier_url": "http://127.0.0.1:8000/v1/completions",
    "verifier_model": "Qwen/Qwen3-VL-2B-Instruct",
    "verifier_max_tokens": 512,
    "wandb_project": "medical-o1-grpo",
    "wandb_run_name": "qwen3-vl-2b-medical-grpo",
    "training_args": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-05,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "beta": 0.0,
        "use_vllm": False,
        "report_to": "wandb",
        "log_completions": True,
        "num_completions_to_print": 4,
        "log_unique_prompts": True,
        "log_multimodal": True,
        "remove_unused_columns": False,
        "seed": 42,
    },
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
}


def build_policy(config):
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import Qwen3VLForConditionalGeneration

    if config["sft_checkpoint"]:
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            config["base_model"],
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        policy = PeftModel.from_pretrained(
            base, resolve_path(config["sft_checkpoint"]), is_trainable=True
        )
        return policy, None
    return config["base_model"], LoraConfig(task_type="CAUSAL_LM", **config["lora"])


def main():
    parser = argparse.ArgumentParser(description="Train GRPO with strict answer verification")
    parser.add_argument("--config", default=str(project_root() / "configs/training/grpo.yaml"))
    parser.add_argument("--resume", help="Resume GRPO separately from SFT initialization")
    args = parser.parse_args()
    config = load_config(args.config, DEFAULTS)
    for key in [
        "base_model",
        "sft_checkpoint",
        "output_dir",
        "num_generations",
        "max_completion_length",
        "verifier_url",
        "verifier_model",
        "verifier_max_tokens",
        "wandb_project",
        "wandb_run_name",
    ]:
        if key.upper() in os.environ:
            value = os.environ[key.upper()]
            config[key] = int(value) if isinstance(DEFAULTS[key], int) else value
    from datasets import load_dataset
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    configure_wandb()
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    dataset = load_dataset(config["dataset_name"], split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    processor = AutoProcessor.from_pretrained(config["base_model"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model, peft_config = build_policy(config)
    kwargs = dict(config["training_args"])
    kwargs.update(
        output_dir=str(resolve_path(config["output_dir"])),
        run_name=config["wandb_run_name"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
    )
    reward = VerifierReward(
        config["verifier_url"], config["verifier_model"], config["verifier_max_tokens"]
    )
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**filter_supported_kwargs(GRPOConfig, kwargs)),
        train_dataset=dataset,
        reward_funcs=reward.verifier_reward,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=str(resolve_path(args.resume)) if args.resume else None)
    target = resolve_path(config["output_dir"]) / "final"
    trainer.save_model(str(target))
    processor.save_pretrained(target)


if __name__ == "__main__":
    main()
