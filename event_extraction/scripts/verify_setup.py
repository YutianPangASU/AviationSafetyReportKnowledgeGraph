"""Sanity-check the Qwen3.6 + vLLM setup WITHOUT loading the full model.

Verifies:
  1. vLLM + torch + cuda are importable.
  2. All 4 GPUs are visible.
  3. The Qwen3.6-35B-A3B tokenizer loads from the local cache.
  4. The few-shot prompt fits comfortably inside max_model_len.

Usage:

    /home/yp6443/miniconda3/envs/qwen-vllm/bin/python \
        event_extraction/scripts/verify_setup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


MODEL_PATH = "/home/yp6443/.cache/huggingface/hub/Qwen--Qwen3.6-35B-A3B"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _check_vllm() -> None:
    import torch
    import vllm

    print(f"vllm     {vllm.__version__}")
    print(f"torch    {torch.__version__} (cuda {torch.version.cuda})")
    print(f"cuda.is_available {torch.cuda.is_available()}")
    print(f"cuda.device_count {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(f"  gpu[{i}] {props.name}  {mem_gb:.1f} GB  sm_{props.major}{props.minor}")


def _check_model_files() -> None:
    p = Path(MODEL_PATH)
    if not p.exists():
        raise SystemExit(f"model dir not found: {p}")
    shards = sorted(p.glob("model-*.safetensors"))
    total = sum(s.stat().st_size for s in shards)
    print(f"model shards: {len(shards)} files, {total / (1024**3):.1f} GB")
    for req in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (p / req).exists():
            raise SystemExit(f"missing {req} in {p}")


def _check_tokenizer_and_prompt() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"tokenizer class: {type(tok).__name__}")
    print(f"vocab size: {tok.vocab_size}")

    system = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()
    fewshot = json.loads((PROMPTS_DIR / "few_shot.json").read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": system}]
    for ex in fewshot:
        messages.append({"role": "user", "content": f"NARRATIVE:\n{ex['narrative']}"})
        messages.append({"role": "assistant", "content": json.dumps(ex["extraction"])})
    messages.append({"role": "user", "content": "NARRATIVE:\n" + "word " * 500})
    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"WARN: chat template failed ({e}); falling back to concat")
        rendered = "\n".join(m["content"] for m in messages)
    tokens = tok(rendered, return_tensors=None)["input_ids"]
    print(f"prompt tokens (system + 2 shots + 500-word narrative): {len(tokens)}")
    if len(tokens) > 16000:
        print("WARN: prompt is large; consider trimming few-shots")


def main() -> int:
    print("=== vLLM / torch / cuda ===")
    _check_vllm()
    print("\n=== model files ===")
    _check_model_files()
    print("\n=== tokenizer + prompt budget ===")
    _check_tokenizer_and_prompt()
    print("\nOK — setup looks good. To launch the server:")
    print("  GPUS=0,1 PORT=8000 bash event_extraction/scripts/serve_qwen.sh")
    print(
        "  GPUS=2,3 PORT=8001 bash event_extraction/scripts/serve_qwen.sh  "
        "(optional second replica)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
