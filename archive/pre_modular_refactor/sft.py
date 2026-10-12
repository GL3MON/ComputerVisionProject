import argparse
import inspect
import os

# Keep local W&B run files in the shared output tree.
_WANDB_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "logs")
os.makedirs(_WANDB_LOG_DIR, exist_ok=True)
os.environ.setdefault("WANDB_DIR", _WANDB_LOG_DIR)

import torch
import wandb
import yaml
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)


# --- defaults, overridable via --config yaml file ---
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


def deep_merge(base, override):
    """Recursively merge override dict into base dict, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def filter_supported_kwargs(cls, kwargs, warn=True):
    """Drop kwargs that `cls.__init__` doesn't accept, warning about each one.

    Guards against transformers-version drift, e.g. `save_safetensors` not
    existing on some TrainingArguments versions.
    """
    accepted = set(inspect.signature(cls.__init__).parameters)
    supported = {}
    dropped = []
    for key, value in kwargs.items():
        if key in accepted:
            supported[key] = value
        else:
            dropped.append(key)
    if dropped and warn:
        print(
            f"[config] Dropping unsupported {cls.__name__} argument(s) for "
            f"your installed transformers version: {dropped}"
        )
    return supported


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
            output_ids[0][inputs["input_ids"].shape[1]:],
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


def load_config(config_path):
    cfg = DEFAULTS
    if config_path:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_merge(DEFAULTS, user_cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL LoRA adapter, config optionally supplied via YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML file overriding any of the default training args, "
        "LoRA config, model/dataset names, output dir, or wandb settings.",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    cfg = load_config(args.config)

    MODEL_NAME = cfg["model_name"]
    DATASET_NAME = cfg["dataset_name"]
    OUTPUT_DIR = cfg["output_dir"]
    MAX_LENGTH = cfg["max_length"]
    WANDB_PROJECT = cfg["wandb_project"]
    WANDB_RUN_NAME = cfg["wandb_run_name"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # fix: under `accelerate launch` this script runs once per process/GPU.
    # Only the main process should own the wandb run (and be noisy with
    # prints) — otherwise you get N duplicate runs/logs for N processes.
    is_main_process = PartialState().is_main_process

    # wandb needs the project name set before Trainer initializes it via report_to
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    if is_main_process:
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=cfg)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    tokenizer = processor.tokenizer

    tokenizer.padding_side = "right"

    # fix: Qwen tokenizers don't always ship with a pad token set,
    # which breaks DataCollatorForSeq2Seq padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fix: flash_attention_2 raises ImportError if flash-attn isn't installed
    # in the environment; fall back to sdpa instead of hard-failing
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        if is_main_process:
            print("flash-attn not found, falling back to sdpa attention")
        attn_impl = "sdpa"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    model.config.use_cache = False

    # fix: pad_token_id wasn't propagated to the model config, which can
    # cause generation/attention mask mismatches
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = load_dataset(
        DATASET_NAME, 'en'
    )

    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        full_dataset = dataset[list(dataset.keys())[0]]

    if is_main_process:
        print("Dataset size:", len(full_dataset))
        print("Columns:", full_dataset.column_names)

    # split into train / eval before tokenizing, so eval never leaks into
    # training and the raw text is still available for the sample-generation
    # callback below
    split = full_dataset.train_test_split(
        test_size=cfg["eval_split_ratio"],
        seed=42,
    )
    train_raw = split["train"]
    eval_raw = split["test"]

    if is_main_process:
        print(
            f"Train / eval split: {len(train_raw)} / {len(eval_raw)} "
            f"({cfg['eval_split_ratio']:.0%} eval)"
        )

    # held out for the qualitative sample-generation callback, taken before
    # any columns get removed by .map() below
    sample_question = str(eval_raw[0]["Question"])

    def preprocess_example(example):

        question = str(example["Question"])
        reasoning = str(example["Complex_CoT"])
        response = str(example["Response"])

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    }
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": response,
            },
        ]

        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    }
                ],
            },
        ]

        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )



        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        prompt_length = len(prompt_tokens["input_ids"])

        labels = input_ids.copy()

        for i in range(
            min(prompt_length, len(labels))
        ):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = train_raw.map(
        preprocess_example,
        remove_columns=train_raw.column_names,
        desc="Formatting train dataset",
    )

    train_dataset = train_dataset.filter(
        lambda x: any(
            label != -100
            for label in x["labels"]
        )
    )

    eval_dataset = eval_raw.map(
        preprocess_example,
        remove_columns=eval_raw.column_names,
        desc="Formatting eval dataset",
    )

    eval_dataset = eval_dataset.filter(
        lambda x: any(
            label != -100
            for label in x["labels"]
        )
    )

    if is_main_process:
        print("Final training examples:", len(train_dataset))
        print("Final eval examples:", len(eval_dataset))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # fix: enable wandb logging instead of disabling all reporting
        report_to="wandb",
        run_name=WANDB_RUN_NAME,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        # everything else (epochs, batch size, lr, scheduler, optim,
        # logging/save cadence, etc.) comes from the training_args section
        # of the config, so it can be overridden via --config yaml.
        #
        # fix: different transformers versions accept different
        # TrainingArguments fields (e.g. `save_safetensors` doesn't exist
        # on some versions), so silently drop any config keys the
        # installed version doesn't recognize instead of hard-crashing.
        **filter_supported_kwargs(TrainingArguments, cfg["training_args"], is_main_process),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            SampleGenerationCallback(
                processor=processor,
                tokenizer=tokenizer,
                sample_question=sample_question,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
            )
        ],
    )

    trainer.train()

    trainer.save_model(
        OUTPUT_DIR
    )

    processor.save_pretrained(
        OUTPUT_DIR
    )

    if is_main_process:
        wandb.finish()

        print(
            f"LoRA adapter saved to: {OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()