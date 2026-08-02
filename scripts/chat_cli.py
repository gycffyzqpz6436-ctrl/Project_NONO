from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from nono_lora.config import load_yaml, section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a trained NONO adapter.")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/nono-qlora"))
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument("--system-prompt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    inference = section(config, "inference")
    quant = section(config, "quantization")
    adapter_config = PeftConfig.from_pretrained(args.adapter)
    base_model = adapter_config.base_model_name_or_path
    dtype = torch.float16
    if str(quant.get("compute_dtype", "float16")) == "bfloat16":
        dtype = torch.bfloat16
    use_4bit = bool(quant.get("load_in_4bit", True)) and torch.cuda.is_available()
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
        if use_4bit
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    system_prompt = None
    if args.system_prompt:
        system_prompt = args.system_prompt.read_text(encoding="utf-8-sig").strip()
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    print("NONO CLI: /reset で履歴消去、/exit で終了")
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text == "/exit":
            break
        if user_text == "/reset":
            history = history[:1] if system_prompt else []
            print("履歴を消去しました。")
            continue
        if not user_text:
            continue
        history.append({"role": "user", "content": user_text})
        inputs = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(inference.get("max_new_tokens", 256)),
                do_sample=True,
                temperature=float(inference.get("temperature", 0.8)),
                top_p=float(inference.get("top_p", 0.9)),
                repetition_penalty=float(
                    inference.get("repetition_penalty", 1.05)
                ),
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"NONO> {response}")
        history.append({"role": "assistant", "content": response})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
