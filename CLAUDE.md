# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CodeRunner IDE is a single-file (~12K lines) Tkinter desktop application that connects to local LLMs and cloud APIs to generate, run, and iteratively fix code. The entire application lives in `CodeRunner_IDE_clean.py`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python CodeRunner_IDE_clean.py

# API keys (optional, for cloud backends only)
cp .env.example .env   # then edit .env
```

There are no tests, linter configs, or build steps. The project is a single Python script with optional dependencies.

## Architecture

### Single-File Monolith

Everything is in `CodeRunner_IDE_clean.py`. The two main classes are:

- **`OllamaGUI`** — The main application window (300+ methods). Manages chat, LLM backends, message routing, threading, and the three-panel layout (Chat | IDE | System/Debug Console).
- **`IDEWindow`** — The code editor. Handles syntax highlighting, inline diff view, Run/Run&Fix execution, Accept/Reject workflow, and SEARCH/REPLACE block parsing.
- **`BrowserErrorServer`** — A local HTTP server (port 8765) that captures JavaScript errors from generated HTML games running in the browser.

### Backend System

Eight LLM backends, each implemented as methods within `OllamaGUI`: Ollama, GGUF (llama-cpp-python), MLX (Apple Silicon), vLLM, Transformers, Blackwell, Claude API, OpenAI API. Platform-aware auto-detection: MLX on macOS, Transformers on Linux, Claude as fallback. Backends are lazily loaded.

### Core Workflow: The Fix Loop

1. **Generate** — User asks LLM to write code → full program returned → "Move to IDE" button transfers it to the editor
2. **Run & Fix (F6)** — Executes code → captures errors → sends errors to LLM → shows color-coded diff → user Accept (`Ctrl+Enter`) / Reject (`Escape`)
3. **LLM Fix** — User describes what's wrong → sends only a fix-specific system prompt + full code (NOT full chat history) → LLM returns only changed functions → merged back via `SequenceMatcher` opcodes

This two-phase approach (full program for generation, changed-functions-only for fixes) is central to the design. Fix mode deliberately strips chat history to prevent weak local LLMs from regenerating the entire program.

### Diff & Merge Strategy

- Uses `difflib.SequenceMatcher` opcodes for intelligent merging of partial LLM responses
- Supports SEARCH/REPLACE blocks (`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE`) with fuzzy fallback
- Falls back to function-name matching when merging partial returns

### Message Routing (three distinct channels)

- `display_chat_system_message()` → Chat panel (user-facing conversation)
- `display_status_message()` → System console (status info, code analysis, non-critical)
- `display_system_message()` → Debug console (runtime errors, stderr/stdout)

### Threading Model

Main Tkinter GUI thread + background threads for LLM calls. Thread-safe message queues for UI updates.

### Key Directories

- `Generated_games/` — Output directory for LLM-generated HTML games (some served via GitHub Pages)
- `temp_html/` — Temporary HTML files during execution (gitignored)

### Configuration

- `.env` / `anthropic_key.txt` / `openai_key.txt` — API keys (never committed)
- `config.json` — Model configuration (currently Qwen3 VL MOE)
- Default sampling: Temperature 0.1, Top-p 0.5, Top-k 40, Max tokens 16,000

## Key Patterns to Know

- **All optional dependencies are try/except imported** — the app gracefully degrades if packages like `jedi`, `pygments`, `chromadb`, etc. are missing
- **System prompts differ by mode** — "Python Programmer" and "HTML Programmer" for generation; a separate "Fix Mode" system prompt replaces the generation one during fixes
- **Chat is editable** — Users can cut/paste in the chat display to modify history; `Send` rebuilds the message list from visible text
- **15 built-in game presets** — Dropdown prompts for one-shot game generation (Space Invaders, Asteroids, etc.)
