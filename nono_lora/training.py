from __future__ import annotations

from typing import Any


def validate_precision_settings(
    *,
    bf16: bool,
    fp16: bool,
    cuda_available: bool,
    bf16_supported: bool,
) -> None:
    """Reject mixed-precision combinations that would select the wrong scaler."""
    if bf16 and fp16:
        raise ValueError("training.bf16 and training.fp16 cannot both be true")
    if bf16 and (not cuda_available or not bf16_supported):
        raise RuntimeError(
            "training.bf16=true, but the active CUDA device does not support BF16"
        )


def training_data_files(data_config: dict[str, Any]) -> dict[str, str]:
    """Build the DatasetDict inputs without ever including the held-out test set."""
    files = {"train": str(data_config["train_file"])}
    if data_config.get("validation_file"):
        files["validation"] = str(data_config["validation_file"])
    return files


def to_prompt_completion(
    messages: list[dict[str, str]], tokenizer: Any
) -> dict[str, str]:
    """Apply the model chat template and isolate the final assistant response."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("messages must end with an assistant response")
    prompt_messages = messages[:-1]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_conversation = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_conversation.startswith(prompt):
        raise ValueError(
            "chat template output for the full conversation does not start "
            "with the generation prompt"
        )
    completion = full_conversation[len(prompt) :]
    if not completion:
        raise ValueError("chat template produced an empty assistant completion")
    return {"prompt": prompt, "completion": completion}
