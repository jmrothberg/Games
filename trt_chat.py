#!/usr/bin/env python3
"""
TRT Chat - GPU-accelerated LLM chat for DGX Spark (Blackwell GB10)
Uses TensorRT-LLM when available, falls back to Transformers.
Textual TUI with rich rendering.
"""

import os
import sys
import gc
import time
from pathlib import Path

from textual.app import App
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual.binding import Binding
from textual import work
from textual.worker import Worker, get_current_worker
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

import torch

MODELS_DIR = Path("/home/jonathan/Models_Transformer")

# Try TensorRT-LLM first, fall back to Transformers
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


def get_backend_name():
    if TRTLLM_AVAILABLE:
        return "TensorRT-LLM"
    elif TRANSFORMERS_AVAILABLE:
        return "Transformers"
    return "None"


def scan_models():
    """Scan Models_Transformer directory for valid HuggingFace models."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for item in sorted(MODELS_DIR.iterdir()):
        if item.is_dir() and (item / "config.json").exists():
            models.append(item)
    return models


def clear_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class ChatApp(App):
    """Main TUI chat application."""

    CSS = """
    #gpu-bar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    #chat-log {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    #input-box {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.models = scan_models()
        self.model = None
        self.tokenizer = None
        self.trt_llm = None
        self.loaded_model_name = ""
        self.messages = []
        self.generating = False

    def compose(self):
        yield Header(show_clock=True)
        yield Static(id="gpu-bar")
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        yield Static(id="status-bar")
        yield Input(placeholder="Type a message (or 'model' to switch, 'quit' to exit)...", id="input-box")
        yield Footer()

    def on_mount(self):
        self.update_gpu_bar()
        self.set_interval(5.0, self.update_gpu_bar)
        log = self.query_one("#chat-log", RichLog)

        backend = get_backend_name()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        log.write(Panel(
            f"[bold cyan]TRT Chat[/] — [green]{backend}[/] backend on [yellow]{gpu_name}[/]\n"
            f"Type [bold]model[/] to switch models, [bold]quit[/] to exit, [bold]clear[/] to reset chat.",
            title="Welcome", border_style="cyan"
        ))
        self.show_model_list()

    def update_gpu_bar(self):
        bar = self.query_one("#gpu-bar", Static)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pct = (alloc / total) * 100 if total > 0 else 0
            gpu_name = torch.cuda.get_device_name(0)
            bar.update(f" GPU: {gpu_name} | VRAM: {alloc:.1f}/{total:.1f} GB ({pct:.0f}%) | Backend: {get_backend_name()}")
        else:
            bar.update(" GPU: Not available | Backend: CPU")

    def show_model_list(self):
        log = self.query_one("#chat-log", RichLog)
        if not self.models:
            log.write("[red]No models found in {MODELS_DIR}[/]")
            return
        table = Table(title="Models", expand=True, border_style="cyan", padding=(0, 1))
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("Model", style="bold white")
        table.add_column("Size", style="dim", width=10, justify="right")
        for i, m in enumerate(self.models, 1):
            try:
                size = sum(f.stat().st_size for f in m.iterdir() if f.is_file()) / (1024**3)
                size_str = f"{size:.1f} GB"
            except Exception:
                size_str = "?"
            table.add_row(str(i), m.name, size_str)
        log.write(table)
        log.write("[dim]Enter a number to load a model.[/]")

    def update_status(self, msg: str):
        self.query_one("#status-bar", Static).update(f" {msg}")

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        event.input.clear()
        if not text:
            return

        log = self.query_one("#chat-log", RichLog)

        if text.lower() == "quit":
            self.unload_model()
            self.exit()
            return

        if text.lower() == "model":
            self.unload_model()
            self.show_model_list()
            return

        if text.lower() == "clear":
            self.messages.clear()
            log.clear()
            log.write("[dim]Chat history cleared.[/]")
            if self.loaded_model_name:
                log.write(f"[dim]Model: {self.loaded_model_name}[/]")
            return

        # If no model loaded, treat input as model selection number
        if self.model is None and self.trt_llm is None:
            try:
                idx = int(text) - 1
                if 0 <= idx < len(self.models):
                    self.load_model_async(self.models[idx])
                else:
                    log.write(f"[red]Invalid selection. Enter 1-{len(self.models)}.[/]")
            except ValueError:
                log.write(f"[red]No model loaded. Enter a number (1-{len(self.models)}) to load one.[/]")
            return

        if self.generating:
            log.write("[yellow]Still generating... please wait.[/]")
            return

        # Chat message
        log.write(Text.assemble(("\nYou: ", "bold green"), (text, "white")))
        self.messages.append({"role": "user", "content": text})
        self.generate_response()

    @work(thread=True, exclusive=True, group="load")
    def load_model_async(self, model_path: Path):
        """Load a model in a background thread."""
        worker = get_current_worker()
        log = self.query_one("#chat-log", RichLog)
        name = model_path.name

        self.app.call_from_thread(self.update_status, f"Loading {name}...")
        self.app.call_from_thread(log.write, f"\n[yellow]Loading {name}...[/]")

        clear_gpu()
        start = time.time()

        try:
            if TRTLLM_AVAILABLE:
                self.trt_llm = TRT_LLM(model=str(model_path))
                self.loaded_model_name = name
                elapsed = time.time() - start
                self.app.call_from_thread(log.write,
                    f"[green]Loaded {name} via TensorRT-LLM in {elapsed:.1f}s[/]")
            elif TRANSFORMERS_AVAILABLE:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                # Let transformers auto-detect dtype/quantization from config.
                # Only force bfloat16 for models with no dtype AND no quantization set.
                import json as _json
                _cfg = _json.load(open(model_path / "config.json"))
                _has_dtype_or_quant = _cfg.get("torch_dtype") is not None or _cfg.get("quantization_config") is not None
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    device_map={"": "cuda:0"},
                    torch_dtype="auto" if _has_dtype_or_quant else torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.tokenizer = tokenizer
                self.model = model
                self.loaded_model_name = name
                elapsed = time.time() - start
                if torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated(0) / 1024**3
                    self.app.call_from_thread(log.write,
                        f"[green]Loaded {name} via Transformers in {elapsed:.1f}s ({vram:.1f} GB VRAM)[/]")
                else:
                    self.app.call_from_thread(log.write,
                        f"[green]Loaded {name} via Transformers in {elapsed:.1f}s[/]")
            else:
                self.app.call_from_thread(log.write,
                    "[red]No backend available. Install transformers or tensorrt-llm.[/]")
                return

            self.messages.clear()
            self.app.call_from_thread(log.write, "[dim]Ready. Type your message.[/]")
            self.app.call_from_thread(self.update_status, f"Model: {name}")
            self.app.call_from_thread(self.update_gpu_bar)

        except Exception as e:
            self.app.call_from_thread(log.write, f"[red]Failed to load {name}: {e}[/]")
            self.app.call_from_thread(self.update_status, "No model loaded")
            self.model = None
            self.tokenizer = None
            self.trt_llm = None
            clear_gpu()

    def unload_model(self):
        log = self.query_one("#chat-log", RichLog)
        if self.loaded_model_name:
            log.write(f"[dim]Unloading {self.loaded_model_name}...[/]")
        self.model = None
        self.tokenizer = None
        self.trt_llm = None
        self.loaded_model_name = ""
        self.messages.clear()
        clear_gpu()
        self.update_status("No model loaded")
        self.update_gpu_bar()

    @work(thread=True, exclusive=True, group="generate")
    def generate_response(self):
        """Generate LLM response in a background thread with streaming."""
        worker = get_current_worker()
        self.generating = True
        log = self.query_one("#chat-log", RichLog)

        self.app.call_from_thread(self.update_status, "Generating...")

        try:
            if self.trt_llm is not None:
                self._generate_trtllm(log)
            elif self.model is not None:
                self._generate_transformers(log)
        except Exception as e:
            self.app.call_from_thread(log.write, f"\n[red]Error: {e}[/]")
        finally:
            self.generating = False
            self.app.call_from_thread(self.update_status, f"Model: {self.loaded_model_name}")
            self.app.call_from_thread(self.update_gpu_bar)

    def _generate_trtllm(self, log):
        """Generate using TensorRT-LLM."""
        prompt = self._build_prompt()
        params = TRT_SamplingParams(max_tokens=2048, temperature=0.7, top_p=0.9)
        start = time.time()
        outputs = self.trt_llm.generate([prompt], sampling_params=params)
        text = outputs[0].outputs[0].text
        elapsed = time.time() - start
        tokens = len(text.split())

        self.messages.append({"role": "assistant", "content": text})
        self.app.call_from_thread(log.write,
            Text.assemble(("\nAssistant: ", "bold cyan"), (text, "white")))
        self.app.call_from_thread(log.write,
            f"[dim]({tokens} tokens, {elapsed:.1f}s, ~{tokens/elapsed:.0f} tok/s)[/]")

    def _generate_transformers(self, log):
        """Generate using Transformers with streaming."""
        chat = list(self.messages)
        if self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = self._build_prompt()

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)

        # Print the "Assistant: " label once before streaming starts
        self.app.call_from_thread(log.write,
            Text("Assistant:", style="bold cyan"), scroll_end=True)

        start = time.time()
        thread.start()
        full_response = ""
        # Buffer chunks and only call log.write() on complete lines (each write adds a newline)
        line_buf = ""
        for chunk in streamer:
            if chunk:
                full_response += chunk
                elapsed_so_far = time.time() - start
                tps = len(full_response.split()) / elapsed_so_far if elapsed_so_far > 0 else 0
                self.app.call_from_thread(self.update_status,
                    f"Generating... {len(full_response.split())} words, {tps:.0f} w/s")
                line_buf += chunk
                if '\n' in line_buf:
                    parts = line_buf.split('\n')
                    for part in parts[:-1]:
                        self.app.call_from_thread(log.write, part, scroll_end=True)
                    line_buf = parts[-1]
        if line_buf:  # flush remaining partial line
            self.app.call_from_thread(log.write, line_buf, scroll_end=True)
        thread.join()
        elapsed = time.time() - start

        output_tokens = len(self.tokenizer.encode(full_response))
        tps = output_tokens / elapsed if elapsed > 0 else 0

        self.messages.append({"role": "assistant", "content": full_response})
        self.app.call_from_thread(log.write,
            f"[dim]({output_tokens} tokens, {elapsed:.1f}s, ~{tps:.0f} tok/s)[/]")

    def _build_prompt(self):
        """Fallback prompt builder if no chat template."""
        prompt = ""
        for msg in self.messages:
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


def main():
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Models will run on CPU (very slow).")
    if not TRTLLM_AVAILABLE and not TRANSFORMERS_AVAILABLE:
        print("ERROR: Neither tensorrt-llm nor transformers is installed.")
        print("  pip install transformers  OR  pip install tensorrt-llm")
        sys.exit(1)
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
