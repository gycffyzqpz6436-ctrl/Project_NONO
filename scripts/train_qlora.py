from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from nono_lora.config import load_yaml, section
from nono_lora.training import (
    to_prompt_completion,
    training_data_files,
    validate_precision_settings,
)
from scripts.validate_jsonl import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NONO QLoRA adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    return parser.parse_args()


def torch_dtype(name: str):
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"unsupported compute dtype: {name}")
    return values[name]


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    model_cfg = section(config, "model")
    quant_cfg = section(config, "quantization")
    lora_cfg = section(config, "lora")
    train_cfg = section(config, "training")
    data_cfg = section(config, "data")
    data_paths = [Path(str(data_cfg["train_file"]))]
    if data_cfg.get("validation_file"):
        data_paths.append(Path(str(data_cfg["validation_file"])))
    data_errors = validate(data_paths, None, None)
    if data_errors:
        raise ValueError(
            "Training data validation failed:\n"
            + "\n".join(f"- {error}" for error in data_errors)
        )

    model_name = str(model_cfg["name"])
    compute_dtype_name = str(quant_cfg.get("compute_dtype", "bfloat16"))
    compute_dtype = torch_dtype(compute_dtype_name)
    bf16 = bool(train_cfg.get("bf16", False))
    fp16 = bool(train_cfg.get("fp16", False))
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
    validate_precision_settings(
        bf16=bf16,
        fp16=fp16,
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
    )
    if bf16 and compute_dtype != torch.bfloat16:
        raise ValueError(
            "training.bf16=true requires quantization.compute_dtype=bfloat16"
        )
    if fp16 and compute_dtype != torch.float16:
        raise ValueError(
            "training.fp16=true requires quantization.compute_dtype=float16"
        )
    use_4bit = bool(quant_cfg.get("load_in_4bit", True))
    if use_4bit and not cuda_available:
        raise RuntimeError(
            "QLoRA training requires a supported accelerator. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
        )
    max_steps = int(train_cfg.get("max_steps", -1))
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if cuda_available
        else "CUDA unavailable"
    )
    print("Training startup configuration:")
    print(f"  GPU name: {gpu_name}")
    print(f"  model dtype: {compute_dtype}")
    print(f"  bitsandbytes compute dtype: {compute_dtype}")
    print(f"  bf16: {bf16}")
    print(f"  fp16: {fp16}")
    print(f"  max_steps: {max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(model_cfg.get("trust_remote_code", False))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise RuntimeError(
            f"{model_name} has no chat template; choose an instruct/chat base model."
        )

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_cfg.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=compute_dtype,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    model.config.use_cache = False

    files = training_data_files(data_cfg)
    dataset = load_dataset("json", data_files=files)
    original_columns = dataset["train"].column_names
    dataset = dataset.map(
        lambda example: to_prompt_completion(example["messages"], tokenizer),
        remove_columns=original_columns,
        desc="Applying chat template and isolating assistant completions",
    )

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    has_eval = "validation" in dataset
    output_dir = str(train_cfg.get("output_dir", "outputs/nono-qlora"))
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("epochs", 3)),
        max_steps=max_steps,
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 2.0e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_strategy=str(train_cfg.get("save_strategy", "epoch")),
        eval_strategy=str(
            train_cfg.get("eval_strategy", "epoch" if has_eval else "no")
        ),
        max_length=int(train_cfg.get("max_length", 1024)),
        completion_only_loss=True,
        packing=bool(train_cfg.get("packing", False)),
        gradient_checkpointing=bool(
            train_cfg.get("gradient_checkpointing", True)
        ),
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        fp16=fp16,
        bf16=bf16,
        seed=int(train_cfg.get("seed", 42)),
        report_to=str(train_cfg.get("report_to", "none")),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
