"""SFT orchestration with separately testable loading and preprocessing."""

import argparse
import os
from medical_vlm.core.config import load_config, filter_supported_kwargs
from medical_vlm.core.paths import configure_wandb, project_root, resolve_path
from medical_vlm.data.reasoning import prepare_datasets

DEFAULTS = {
    "model_name": "Qwen/Qwen3-VL-2B-Thinking",
    "dataset_name": "FreedomIntelligence/medical-o1-reasoning-SFT",
    "output_dir": "./outputs/qwen3-vl-2b-medical-lora",
    "max_length": 4096,
    "wandb_project": "qwen3-vl-medical-lora",
    "wandb_run_name": "qwen3-vl-2b-medical-lora-r16",
    "eval_split_ratio": 0.02,
    "generation": {
        "max_new_tokens": 256,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
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
    "training_args": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        # fix: added eval so eval loss + sample generations get logged.
        # both eval_strategy (newer transformers) and evaluation_strategy
        # (older transformers) are listed; filter_supported_kwargs() will
        # keep whichever one your installed version actually accepts.
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "per_device_eval_batch_size": 1,
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "max_grad_norm": 1.0,
        "ddp_find_unused_parameters": False,
        "save_safetensors": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL LoRA adapter, config optionally supplied via YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root() / "configs/training/sft.yaml"),
        help="Path to a YAML file overriding any of the default training args, "
        "LoRA config, model/dataset names, output dir, or wandb settings.",
    )
    return parser.parse_args()


def load_processor(model_name):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def build_model(config, tokenizer):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration

    try:
        import flash_attn  # noqa: F401

        attention = "flash_attention_2"
    except ImportError:
        attention = "sdpa"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_name"], dtype=torch.bfloat16, attn_implementation=attention
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def train(config):
    import wandb
    from accelerate import PartialState
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from medical_vlm.training.callbacks import SampleGenerationCallback

    configure_wandb()
    output = resolve_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    is_main = PartialState().is_main_process
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    if is_main:
        wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
    try:
        processor = load_processor(config["model_name"])
        model = build_model(config, processor.tokenizer)
        train_dataset, eval_dataset, question = prepare_datasets(config, processor)
        arguments = dict(config["training_args"])
        arguments.update(
            output_dir=str(output),
            run_name=config["wandb_run_name"],
            report_to="wandb",
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                **filter_supported_kwargs(TrainingArguments, arguments, is_main)
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=processor.tokenizer, model=model, padding=True, label_pad_token_id=-100
            ),
            callbacks=[
                SampleGenerationCallback(
                    processor, processor.tokenizer, question, config["generation"]["max_new_tokens"]
                )
            ],
        )
        trainer.train()
        trainer.save_model(str(output))
        processor.save_pretrained(output)
        return output
    finally:
        if is_main:
            wandb.finish()


def main():
    args = parse_args()
    config = load_config(args.config, DEFAULTS)
    print(f"Saved adapter: {train(config)}")


if __name__ == "__main__":
    main()
