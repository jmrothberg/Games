#!/usr/bin/env python3
"""
TRT Chat (Terminal) - Plain stdin/stdout LLM chat for DGX Spark.
Same model loading logic as trt_chat.py but no TUI — just print().
Easy to copy/paste errors and debug.
"""

import os
import sys
import gc
import json
import time
from pathlib import Path

# DGX Spark unified memory: CPU+GPU share 128 GB physical DRAM.
# These must be set before any model loading.
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

MODELS_DIR = Path("/home/jonathan/Models_Transformer")

# ── Backend imports ──────────────────────────────────────────────────────────
TRTLLM_AVAILABLE = False
try:
    import tensorrt_llm
    from tensorrt_llm import LLM as TRT_LLM, SamplingParams as TRT_SamplingParams
    TRTLLM_AVAILABLE = True
except ImportError:
    pass

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from threading import Thread
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_model_config(model_path: Path) -> dict:
    """Read config.json and return quant/dtype metadata."""
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return {"quantization_config": None, "quant_method": None,
                "torch_dtype": None, "architectures": [], "disk_gb": 0.0}
    cfg = json.load(open(cfg_path))
    qcfg = cfg.get("quantization_config")
    quant_method = None
    if qcfg:
        quant_method = qcfg.get("quant_method", "").lower() or None

    # torch_dtype may be at top level or nested under text_config
    torch_dtype = cfg.get("torch_dtype")
    if not torch_dtype:
        for sub in ("text_config", "language_config", "llm_config"):
            if sub in cfg and isinstance(cfg[sub], dict):
                torch_dtype = cfg[sub].get("dtype") or cfg[sub].get("torch_dtype")
                if torch_dtype:
                    break

    try:
        disk_gb = sum(
            f.stat().st_size for f in model_path.iterdir()
            if f.suffix in (".safetensors", ".bin")
        ) / (1024**3)
    except Exception:
        disk_gb = 0.0

    return {
        "quantization_config": qcfg,
        "quant_method": quant_method,
        "torch_dtype": torch_dtype,
        "architectures": cfg.get("architectures", []),
        "disk_gb": disk_gb,
    }


def quant_label(model_path: Path) -> str:
    """Short label: FP8, NVFP4, GPTQ, AWQ, BF16, etc."""
    info = read_model_config(model_path)
    qm = info["quant_method"]
    if qm:
        _LABEL_MAP = {"compressed-tensors": "FP8", "modelopt": "NVFP4"}
        return _LABEL_MAP.get(qm, qm.upper())
    dt = info["torch_dtype"]
    if dt == "bfloat16":
        return "BF16"
    if dt == "float16":
        return "FP16"
    if dt == "float32":
        return "FP32"
    return "BF16"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return "No GPU"
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    name = torch.cuda.get_device_name(0)
    return f"{name} | {alloc:.1f}/{total:.1f} GB"


def scan_models():
    models = []
    if not MODELS_DIR.exists():
        return models
    for item in sorted(MODELS_DIR.iterdir()):
        if item.is_dir() and (item / "config.json").exists():
            models.append(item)
    return models


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    """Load model, return (trt_llm, model, tokenizer, backend_name)."""
    name = model_path.name
    mcfg = read_model_config(model_path)
    qm = mcfg["quant_method"]
    arch = mcfg["architectures"]
    dtype_str = mcfg["torch_dtype"]
    label = quant_label(model_path)

    print(f"  Config: arch={arch}, quant={qm or 'none'}, dtype={dtype_str}, type={label}")

    clear_gpu()
    start = time.time()

    # ── Try TRT-LLM first ────────────────────────────────────────────────
    if TRTLLM_AVAILABLE:
        try:
            trt_kwargs = {"model": str(model_path)}
            if qm:
                trt_kwargs["dtype"] = "auto"
            else:
                trt_kwargs["dtype"] = "bfloat16"
            print(f"  TRT-LLM: dtype={trt_kwargs['dtype']}")
            trt_llm = TRT_LLM(**trt_kwargs)
            elapsed = time.time() - start
            print(f"  Loaded {name} ({label}) via TensorRT-LLM in {elapsed:.1f}s")
            return trt_llm, None, None, "TensorRT-LLM"
        except Exception as e:
            clear_gpu()
            err_str = str(e)
            if "executor worker" in err_str.lower():
                print(f"  TRT-LLM executor worker error — arch={arch}, quant={qm or 'none'}")
                print(f"  Model may be too large or unsupported architecture for TRT-LLM.")
            else:
                print(f"  TRT-LLM failed: {e}")
            print(f"  Falling back to Transformers...")

    # ── Transformers fallback ────────────────────────────────────────────
    if TRANSFORMERS_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if qm:
            load_dtype = "auto"
        else:
            load_dtype = torch.bfloat16

        disk_gb = mcfg["disk_gb"]

        print(f"  Transformers: torch_dtype={load_dtype}, quant_method={qm or 'none'}, "
              f"disk={disk_gb:.1f} GB")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map={"": "cuda:0"},
            torch_dtype=load_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        elapsed = time.time() - start
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  Loaded {name} ({label}) via Transformers in {elapsed:.1f}s ({vram:.1f} GB VRAM)")
        else:
            print(f"  Loaded {name} ({label}) via Transformers in {elapsed:.1f}s")
        return None, model, tokenizer, "Transformers"

    print("  ERROR: No backend available.")
    return None, None, None, "None"


# ── Generation ───────────────────────────────────────────────────────────────

def generate_trtllm(trt_llm, messages):
    prompt = build_prompt(messages)
    params = TRT_SamplingParams(max_tokens=12048, temperature=0.7, top_p=0.9)
    start = time.time()
    outputs = trt_llm.generate([prompt], sampling_params=params)
    text = outputs[0].outputs[0].text
    elapsed = time.time() - start
    tokens = len(text.split())
    return text, f"({tokens} tokens, {elapsed:.1f}s, ~{tokens/elapsed:.0f} tok/s)"


def generate_transformers(model, tokenizer, messages):
    chat = list(messages)
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = build_prompt(messages)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": 12048,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    print("\nAssistant: ", end="", flush=True)

    start = time.time()
    thread.start()
    full_response = ""
    for chunk in streamer:
        if chunk:
            print(chunk, end="", flush=True)
            full_response += chunk
    thread.join()
    elapsed = time.time() - start

    output_tokens = len(tokenizer.encode(full_response))
    tps = output_tokens / elapsed if elapsed > 0 else 0
    stats = f"({output_tokens} tokens, {elapsed:.1f}s, ~{tps:.0f} tok/s)"
    print(f"\n  {stats}")
    return full_response, stats


def build_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"
    prompt += "Assistant: "
    return prompt


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    backend = "TensorRT-LLM" if TRTLLM_AVAILABLE else ("Transformers" if TRANSFORMERS_AVAILABLE else "None")
    print(f"═══ TRT Chat (Terminal) ═══")
    print(f"Backend: {backend} | GPU: {gpu_status()}")
    print()

    if not TRTLLM_AVAILABLE and not TRANSFORMERS_AVAILABLE:
        print("ERROR: Neither tensorrt-llm nor transformers is installed.")
        sys.exit(1)

    models = scan_models()
    if not models:
        print(f"No models found in {MODELS_DIR}")
        sys.exit(1)

    # ── Model selection ──────────────────────────────────────────────────
    trt_llm = None
    hf_model = None
    tokenizer = None
    messages = []

    def show_models():
        print(f"\n{'#':>3}  {'Type':<8}  {'Size':>8}  Model")
        print(f"{'─'*3}  {'─'*8}  {'─'*8}  {'─'*40}")
        for i, m in enumerate(models, 1):
            try:
                size = sum(f.stat().st_size for f in m.iterdir() if f.is_file()) / (1024**3)
                size_str = f"{size:.1f} GB"
            except Exception:
                size_str = "?"
            print(f"{i:>3}  {quant_label(m):<8}  {size_str:>8}  {m.name}")
        print()

    show_models()

    while True:
        try:
            line = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if line.lower() in ("model", "models"):
            # Unload current
            trt_llm = None
            hf_model = None
            tokenizer = None
            messages.clear()
            clear_gpu()
            show_models()
            continue

        if line.lower() == "clear":
            messages.clear()
            print("Chat history cleared.")
            continue

        if line.lower() == "gpu":
            print(f"GPU: {gpu_status()}")
            continue

        # ── Model selection by number ────────────────────────────────────
        if trt_llm is None and hf_model is None:
            try:
                idx = int(line) - 1
                if 0 <= idx < len(models):
                    print(f"\nLoading {models[idx].name}...")
                    try:
                        trt_llm, hf_model, tokenizer, bk = load_model(models[idx])
                        if trt_llm is None and hf_model is None:
                            print("Failed to load model.")
                        else:
                            print(f"Ready ({bk}). Type your message.\n")
                    except Exception as e:
                        import traceback
                        print(f"ERROR loading model: {e}")
                        traceback.print_exc()
                        trt_llm = None
                        hf_model = None
                        tokenizer = None
                        clear_gpu()
                else:
                    print(f"Enter 1-{len(models)}.")
            except ValueError:
                print(f"No model loaded. Enter a number (1-{len(models)}).")
            continue

        # ── Chat ─────────────────────────────────────────────────────────
        messages.append({"role": "user", "content": line})
        try:
            if trt_llm is not None:
                text, stats = generate_trtllm(trt_llm, messages)
                print(f"\nAssistant: {text}")
                print(f"  {stats}")
                messages.append({"role": "assistant", "content": text})
            elif hf_model is not None:
                text, stats = generate_transformers(hf_model, tokenizer, messages)
                messages.append({"role": "assistant", "content": text})
        except Exception as e:
            import traceback
            print(f"\nERROR during generation: {e}")
            traceback.print_exc()
            # Remove the failed user message
            messages.pop()


if __name__ == "__main__":
    main()
