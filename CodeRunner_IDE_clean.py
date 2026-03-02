# =============================================================================
# CodeRunner IDE (Clean) - Code-focused AI IDE
# =============================================================================
# Based on CodeRunner IDE v2.8.26
# Stripped down to code-only features (no chat mode, no RAG/vector search)
#
# WORKFLOW:
#   1. Ask LLM to write code in the chat box
#   2. Click "Move to IDE" to load code into the editor (first time)
#   3. Press F5 to Run, or F6 for Run & Fix
#   4. For manual fixes: type what's wrong in chat, click "LLM Fix"
#   5. Review diff → Accept (Ctrl+Enter) or Reject (Escape)
#   6. Repeat!
#
# KEY POINT: After the first "Move to IDE", edits always show as a DIFF.
#            You never lose your existing code.
#
# FIX MODE: When "LLM Fix" is clicked, only the system message + fix prompt
#           are sent to the LLM (not the full conversation history). This
#           keeps weak local LLMs focused on the fix instead of regenerating.
#           Full history is restored after the response completes.
#
# Features:
# - 8 LLM backends: Ollama, Transformers, Claude, OpenAI, GGUF, vLLM, MLX, Blackwell
# - Platform-aware defaults: MLX on macOS, Transformers on Linux
# - Run (F5) + Run & Fix (F6) + LLM Fix button on chat and IDE
# - Inline diff view with Accept (Ctrl+Enter) / Reject (Escape)
# - Partial merge: LLM returns only changed functions, merged into full code
# - Chat edit sync: cut/paste in chat affects what LLM sees on Send
# - Token counters, speed metrics, and response timer
# - Smart debugging: clickable error lines, browser error capture
# - Debug console: runtime errors only; status info goes to system messages
# - 15 built-in game presets for one-shot generation
#
# Installation: pip install -r requirements.txt
# =============================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer

import os

# Load environment variables from .env file (for API keys - keeps secrets out of code)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not installed, will use env vars directly

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_DISABLE_CACHE"] = "1" 

# Try to import ollama (optional dependency)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not installed - Ollama backend will not be available")

import re
import tkinter as tk
from tkinter import scrolledtext, Menu, BooleanVar, Frame, Button, Label, filedialog, messagebox, StringVar, ttk, DoubleVar, simpledialog, Checkbutton, Toplevel, Text, Scrollbar, Entry, IntVar
import threading
import json
import os
import base64
from datetime import datetime
import subprocess
import tempfile
import signal
import http.server
import socketserver
import urllib.parse
from PIL import Image, ImageTk, ImageGrab
import io
import uuid
import glob
import time
import difflib

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("GGUF not available. Install with: pip install llama-cpp-python")

# Try to import MLX for Apple Silicon (macOS only)
import platform
if platform.system() == "Darwin":  # macOS only
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        MLX_AVAILABLE = True
    except ImportError:
        MLX_AVAILABLE = False
        print("MLX not available. Install with: pip install mlx mlx-lm")
else:
    MLX_AVAILABLE = False
    print("MLX skipped (Apple Silicon only)")

# Try to import mlx-vlm for vision-language models (macOS only)
if platform.system() == "Darwin":
    try:
        from mlx_vlm import load as vlm_load, stream_generate as vlm_stream_generate
        from mlx_vlm import apply_chat_template as vlm_apply_chat_template
        MLX_VLM_AVAILABLE = True
        print("mlx-vlm available for vision-language models")
    except ImportError:
        MLX_VLM_AVAILABLE = False
        print("mlx-vlm not available. Install with: pip install mlx-vlm")
else:
    MLX_VLM_AVAILABLE = False

# Try to import vLLM for GPU acceleration
# NOTE: vLLM works on x86 and ARM (e.g. DGX Spark Blackwell) with proper PyTorch/CUDA setup
try:
    import vllm
    from vllm import LLM, SamplingParams
    # vLLM only makes sense on GPU - disable if no CUDA available
    import torch as _torch_check
    if not _torch_check.cuda.is_available():
        VLLM_AVAILABLE = False
        print(f"vLLM installed (v{vllm.__version__}) but disabled - no CUDA GPU (torch {_torch_check.__version__} is cpu-only)")
        print("  Install CUDA-enabled PyTorch to use vLLM")
    else:
        VLLM_AVAILABLE = True
        print(f"vLLM available (v{vllm.__version__}) with CUDA support")
    del _torch_check
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Install with: pip install vllm")
except Exception as e:
    VLLM_AVAILABLE = False
    print(f"vLLM available but has compatibility issues: {str(e)}")
    print("vLLM may require specific PyTorch versions that conflict with CUDA support")

# Import enhanced IDE features
try:
    import jedi
    JEDI_AVAILABLE = True
except ImportError:
    JEDI_AVAILABLE = False
    print("Jedi not available. Install with: pip install jedi for code intelligence")

try:
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.token import Token
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("Pygments not available. Install with: pip install pygments for better syntax highlighting")

try:
    import flake8.api.legacy as flake8_api
    FLAKE8_AVAILABLE = True
except ImportError:
    FLAKE8_AVAILABLE = False
    print("Flake8 not available. Install with: pip install flake8 for real-time linting")

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False
    print("Black not available. Install with: pip install black for code formatting")

# Import additional power features for LLM code testing
try:
    import ast
    import astor
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False
    print("AST analysis not available. Install with: pip install astor for code analysis")

try:
    import mypy.api
    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False
    print("MyPy not available. Install with: pip install mypy for type checking")

try:
    from bandit.core import manager
    from bandit.core import config
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False
    print("Bandit not available. Install with: pip install bandit for security scanning")

try:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    print("Radon not available. Install with: pip install radon for complexity analysis")

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False
    print("Coverage not available. Install with: pip install coverage for test coverage")

try:
    from icecream import ic
    ic.configureOutput(prefix='🔍 DEBUG | ')
    ICECREAM_AVAILABLE = True
except ImportError:
    ICECREAM_AVAILABLE = False
    print("IceCream not available. Install with: pip install icecream for better debugging")

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.table import Table
    RICH_AVAILABLE = True
    rich_console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available. Install with: pip install rich for better output formatting")

# Transformers already imported at top of file for proper CUDA initialization (matches gptoss.py)
# Import additional transformers classes needed for this app
from transformers import AutoConfig, BitsAndBytesConfig, MistralForCausalLM
import torch

# Try to import MistralCommonBackend (may not be available in newer transformers versions)
try:
    from transformers import MistralCommonBackend
    MISTRAL_COMMON_AVAILABLE = True
except ImportError:
    MISTRAL_COMMON_AVAILABLE = False
    print("MistralCommonBackend not available in this transformers version - Devstral models will use standard tokenization")

TRANSFORMERS_AVAILABLE = True

# RAG imports removed (code-only version)
CHROMADB_AVAILABLE = False
LANGCHAIN_SPLITTERS_AVAILABLE = False

import glob
import requests  # For Claude API calls and web search
import queue  # For MCP client response queues

# LangChain imports removed (code-only version)
LANGCHAIN_COMMUNITY_AVAILABLE = False
from pathlib import Path
import time
import sys

# =============================================================================
# MCP (Model Context Protocol) Client — embedded from obedient_beast/mcp_client.py
# Manages MCP server subprocesses for tool calling (Brave Search, fetch, etc.)
# =============================================================================
try:
    from dataclasses import dataclass, field
    from typing import Optional as _Optional

    @dataclass
    class MCPTool:
        """Represents a tool from an MCP server."""
        server: str
        name: str
        description: str
        input_schema: dict

    @dataclass
    class MCPServer:
        """Represents a running MCP server subprocess."""
        name: str
        command: list
        process: _Optional[subprocess.Popen] = None
        tools: list = field(default_factory=list)
        request_id: int = 0
        response_queue: queue.Queue = field(default_factory=queue.Queue)
        reader_thread: _Optional[threading.Thread] = None

    class MCPClient:
        """Client for managing multiple MCP servers via JSON-RPC 2.0 over stdio."""

        def __init__(self, config_file):
            self.config_file = Path(config_file)
            self.servers: dict = {}

        def load_config(self) -> dict:
            if not self.config_file.exists():
                return {"servers": {}}
            return json.loads(self.config_file.read_text())

        def start_servers(self):
            config = self.load_config()
            for name, server_config in config.get("servers", {}).items():
                if not server_config.get("enabled", True):
                    continue
                try:
                    self._start_server(name, server_config)
                    print(f"[MCP] Started server: {name}")
                except Exception as e:
                    print(f"[MCP] Failed to start {name}: {e}")

        def _start_server(self, name: str, config: dict):
            command = config["command"]
            if isinstance(command, str):
                command = command.split()
            env = os.environ.copy()
            for key, value in config.get("env", {}).items():
                env[key] = value
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1
            )
            server = MCPServer(name=name, command=command, process=process)

            def read_responses():
                while True:
                    try:
                        line = process.stdout.readline()
                        if not line:
                            break
                        response = json.loads(line)
                        server.response_queue.put(response)
                    except Exception as e:
                        if process.poll() is not None:
                            break
                        print(f"[MCP] {name} read error: {e}")

            server.reader_thread = threading.Thread(target=read_responses, daemon=True)
            server.reader_thread.start()
            self.servers[name] = server
            self._initialize(name)
            self._discover_tools(name)

        def _send_request(self, server_name: str, method: str, params: dict = None) -> dict:
            server = self.servers.get(server_name)
            if not server or not server.process:
                raise RuntimeError(f"Server {server_name} not running")
            server.request_id += 1
            request = {"jsonrpc": "2.0", "id": server.request_id, "method": method}
            if params:
                request["params"] = params
            server.process.stdin.write(json.dumps(request) + "\n")
            server.process.stdin.flush()
            timeout = 60  # Allow extra time for first-run npm downloads
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"No response from {server_name} after {timeout}s")
                try:
                    response = server.response_queue.get(timeout=remaining)
                    if "id" not in response:
                        continue
                    return response
                except queue.Empty:
                    raise TimeoutError(f"No response from {server_name} after {timeout}s")

        def _initialize(self, server_name: str):
            self._send_request(server_name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "coderunner-ide", "version": "1.0.0"}
            })
            server = self.servers[server_name]
            notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            server.process.stdin.write(json.dumps(notification) + "\n")
            server.process.stdin.flush()

        def _discover_tools(self, server_name: str):
            response = self._send_request(server_name, "tools/list")
            server = self.servers[server_name]
            server.tools = []
            for tool_data in response.get("result", {}).get("tools", []):
                tool = MCPTool(
                    server=server_name,
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {})
                )
                server.tools.append(tool)

        def list_tools(self):
            all_tools = []
            for server in self.servers.values():
                all_tools.extend(server.tools)
            return all_tools

        def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
            response = self._send_request(server_name, "tools/call", {
                "name": tool_name, "arguments": arguments
            })
            result = response.get("result", {})
            content = result.get("content", [])
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        text_parts.append(f"[Image: {item.get('mimeType', 'image')}]")
                return "\n".join(text_parts) or str(result)
            return str(result)

        def stop_servers(self):
            for name, server in self.servers.items():
                if server.process:
                    try:
                        server.process.terminate()
                        server.process.wait(timeout=5)
                    except Exception as e:
                        print(f"[MCP] Error stopping {name}: {e}")
                        try:
                            server.process.kill()
                        except Exception:
                            pass
            self.servers = {}

    MCP_AVAILABLE = True
except Exception as _mcp_err:
    MCP_AVAILABLE = False
    print(f"MCP client not available: {_mcp_err}")

MAX_AGENT_TURNS = 5  # Maximum tool-call agent loop iterations per user message

# =============================================================================
# Helper: fetch an image URL and save to Generated_games/assets/
# =============================================================================
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Generated_games", "assets")

def fetch_image_as_file(url, max_bytes=5*1024*1024):
    """Download an image from *url*, save to Generated_games/assets/, return the relative path.

    The returned path is relative to Generated_games/ (e.g. "assets/sprite_abc123.png")
    so HTML files in Generated_games/ can reference it directly as a src attribute.
    """
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.content
        if len(data) > max_bytes:
            return f"Error: image too large ({len(data)} bytes, max {max_bytes})"

        # Detect extension from Content-Type or URL
        mime = resp.headers.get("Content-Type", "").split(";")[0].strip()
        ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
                   "image/webp": ".webp", "image/svg+xml": ".svg"}
        ext = ext_map.get(mime, "")
        if not ext:
            url_ext = url.rsplit(".", 1)[-1].lower().split("?")[0]
            ext = f".{url_ext}" if url_ext in ("png", "jpg", "jpeg", "gif", "webp", "svg") else ".png"

        # Ensure assets directory exists
        os.makedirs(_ASSETS_DIR, exist_ok=True)

        # Generate a short unique filename
        short_id = uuid.uuid4().hex[:8]
        # Try to derive a readable name from the URL
        url_name = url.rsplit("/", 1)[-1].split("?")[0].split("#")[0]
        url_name = re.sub(r'[^a-zA-Z0-9_-]', '_', url_name)[:40]
        filename = f"{url_name}_{short_id}{ext}"
        filepath = os.path.join(_ASSETS_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(data)

        # Return path relative to Generated_games/ so HTML can use it directly
        return f"assets/{filename}"
    except Exception as e:
        return f"Error fetching image: {e}"

# =============================================================================
# MCP-aware tool definitions for LLMs
# =============================================================================
def get_mcp_tool_definitions(api_type="openai", mcp_client=None):
    """Build tool definitions from MCP servers + local helpers.

    Returns a list formatted for either OpenAI (Ollama/llama-cpp) or Claude API.
    Falls back to the legacy web_search tool when MCP is unavailable.
    """
    tools = []

    if mcp_client:
        for mcp_tool in mcp_client.list_tools():
            prefixed_name = f"mcp_{mcp_tool.server}_{mcp_tool.name}"
            if api_type == "claude":
                tools.append({
                    "name": prefixed_name,
                    "description": f"[MCP:{mcp_tool.server}] {mcp_tool.description}",
                    "input_schema": mcp_tool.input_schema
                })
            else:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": prefixed_name,
                        "description": f"[MCP:{mcp_tool.server}] {mcp_tool.description}",
                        "parameters": mcp_tool.input_schema
                    }
                })

    # Add local fetch_image tool
    if api_type == "claude":
        tools.append({
            "name": "fetch_image",
            "description": "Download an image from a URL and save it to the assets/ folder. Returns a relative file path (e.g. 'assets/sprite_abc123.png') that can be used directly in HTML src attributes. The HTML file and assets folder live in the same directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the image to download"}
                },
                "required": ["url"]
            }
        })
    else:
        tools.append({
            "type": "function",
            "function": {
                "name": "fetch_image",
                "description": "Download an image from a URL and save it to the assets/ folder. Returns a relative file path (e.g. 'assets/sprite_abc123.png') that can be used directly in HTML src attributes. The HTML file and assets folder live in the same directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL of the image to download"}
                    },
                    "required": ["url"]
                }
            }
        })

    # Fallback: if no MCP tools, add legacy web_search
    if not mcp_client or not mcp_client.servers:
        if api_type == "claude":
            tools.append({
                "name": "web_search",
                "description": "Search the web for current information about any topic.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query."}},
                    "required": ["query"]
                }
            })
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information about any topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "The search query."}},
                        "required": ["query"]
                    }
                }
            })

    return tools

# Default model to use
model = 'qwen3:235b'
model = 'qwen3:235b-a22b-q8_0'
model = 'llama4:17b-maverick-128e-instruct-q8_0'

# =============================================================================
# SYSTEM PROMPTS — one per radio-button mode in the UI
# Python/HTML prompts include SEARCH/REPLACE instruction for the fix workflow.
# =============================================================================

# Removed: therapist_system_message (code-only version)
# Removed: helpful_system_message (code-only version)
# Kept for backward-compat references that may still exist:
helpful_system_message = None
therapist_system_message = None

# Python programmer system message - not encoded for easy editing
python_system_message = """You are an expert Python programmer. You excel at:
1. Implementing game applications using PyGame.
2. Structuring code for readability and maintainability.
3. Send key information to the console that can be used to debug the code.
4. When writing a NEW program, include all features — do not simplify.
"""

# HTML programmer system message
html_system_message = """You are an expert HTML/JavaScript programmer. You excel at:
1. Implementing game applications using HTML5 Canvas and JavaScript.
2. Structuring code for readability and maintainability.
3. Send key information to the console that can be used to debug the code.
4. When writing a NEW program, include all features — do not simplify.
5. DO NOT USE EXTERNAL SOUND FILES UNLESS specified by the user JUST comment in code where they would go, but make sure works without.
6. DO NOT USE EXTERNAL IMAGE FILES create YOUR OWN images using Canvas drawing.
"""

# Claude API configuration - reads from environment variable or file
def get_claude_api_key():
    """Get Claude API key from environment variable or file (similar to OpenAI handling)"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback to file if env var not set
        key_path = os.path.join(os.path.dirname(__file__), "anthropic_key.txt")
        if os.path.exists(key_path):
            with open(key_path) as f:
                api_key = f.read().strip()
    return api_key

CLAUDE_API_KEY = get_claude_api_key()

def get_web_search_tool_definition(api_type="openai"):
    """Get the tool definition for web search that LLMs can call
    
    Args:
        api_type: "openai" for Ollama/llama-cpp or "claude" for Claude API
    """
    if api_type == "claude":
        # Claude expects a different tool format
        return {
            "name": "web_search",
            "description": "Search the web for current information about any topic. Use this when you need up-to-date information not in your training data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and include relevant keywords."
                    }
                },
                "required": ["query"]
            }
        }
    else:
        # OpenAI-style format for Ollama and llama-cpp
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information about any topic. Use this when you need up-to-date information not in your training data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific and include relevant keywords."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

def safe_web_search(query, max_results=5):
    """Safely search the web using multiple strategies"""
    try:
        # First try DuckDuckGo's instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract useful information
        results = []
        
        # Get instant answer if available
        if data.get('Answer'):
            results.append(f"Answer: {data['Answer']}")
        
        # Get abstract with source
        if data.get('Abstract'):
            abstract = data['Abstract']
            source = data.get('AbstractSource', 'Unknown')
            results.append(f"Summary ({source}): {abstract}")
        
        # Get definition if available
        if data.get('Definition'):
            results.append(f"Definition: {data['Definition']}")
        
        # Get infobox data if available
        if data.get('Infobox'):
            infobox = data['Infobox']
            if infobox.get('content'):
                for item in infobox['content'][:3]:
                    if item.get('label') and item.get('value'):
                        results.append(f"{item['label']}: {item['value']}")
        
        # Get related topics with more detail
        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:max_results]:
                if isinstance(topic, dict):
                    if topic.get('Text'):
                        text = topic['Text']
                        # Clean up the text
                        if '...' in text:
                            text = text.split('...')[0].strip()
                        if text and len(text) > 20:  # Only include meaningful results
                            results.append(f"• {text}")
                    # Check for subtopics
                    elif topic.get('Topics'):
                        for subtopic in topic['Topics'][:2]:
                            if subtopic.get('Text'):
                                text = subtopic['Text']
                                if '...' in text:
                                    text = text.split('...')[0].strip()
                                if text and len(text) > 20:
                                    results.append(f"• {text}")
        
        # If we still have no results, provide a fallback search suggestion
        if not results:
            # Try Wikipedia search as fallback
            wiki_url = f"https://en.wikipedia.org/w/api.php"
            wiki_params = {
                'action': 'opensearch',
                'search': query,
                'limit': 3,
                'format': 'json'
            }
            try:
                wiki_response = requests.get(wiki_url, params=wiki_params, timeout=5)
                wiki_data = wiki_response.json()
                if len(wiki_data) > 3 and wiki_data[2]:  # Has descriptions
                    for i, desc in enumerate(wiki_data[2][:3]):
                        if desc:
                            title = wiki_data[1][i] if i < len(wiki_data[1]) else ""
                            results.append(f"Wikipedia - {title}: {desc}")
            except:
                pass  # Fallback failed, continue
        
        if not results:
            return f"No search results found for '{query}'. The query may be too specific or the search APIs are limited. Try rephrasing or breaking down the query."
        
        return "\n".join(results)
        
    except requests.RequestException as e:
        return f"Web search failed: Connection error ({str(e)})"
    except Exception as e:
        return f"Web search failed: {str(e)}"

# RAG functions for ChromaDB integration
def create_chroma_collection(collection_name, persist_dir):
    """Create or get a ChromaDB collection"""
    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB not available. Install with: pip install chromadb")

    client = chromadb.PersistentClient(path=persist_dir)
    # Use OpenAI compatible embedding function by default
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    # Try to get existing collection or create new one
    try:
        # First try to get the collection
        try:
            collection = client.get_collection(name=collection_name, embedding_function=ef)
            return client, collection
        except ValueError as e:
            # Collection doesn't exist yet, create it
            collection = client.create_collection(name=collection_name, embedding_function=ef)
            return client, collection
        except chromadb.errors.NotFoundError:
            # This is a specific error when collection doesn't exist
            collection = client.create_collection(name=collection_name, embedding_function=ef)
            return client, collection
    except Exception as e:
        print(f"Error creating/getting collection: {str(e)}")
        # Try a simpler collection name if it failed
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name)
        try:
            collection = client.create_collection(name=safe_name, embedding_function=ef)
            print(f"Created collection with sanitized name: {safe_name}")
            return client, collection
        except Exception as e2:
            print(f"Failed even with sanitized name: {str(e2)}")
            # Last resort - use a generic name with timestamp
            fallback_name = f"collection_{int(time.time())}"
            collection = client.create_collection(name=fallback_name, embedding_function=ef)
            print(f"Created fallback collection: {fallback_name}")
            return client, collection

def process_image_ocr(image_path, status_callback=None):
    """
    Extract text from images using OCR via pytesseract
    macOS users need to:
    1. Install Tesseract OCR: `brew install tesseract`
    2. Install pytesseract: `pip install pytesseract`
    """
    try:
        from PIL import Image
        import pytesseract
        
        # Special handling for macOS typical Tesseract installation paths
        if sys.platform == 'darwin':
            # Common macOS tesseract locations
            tesseract_paths = [
                '/usr/local/bin/tesseract',
                '/opt/homebrew/bin/tesseract',
                '/usr/bin/tesseract'
            ]
            
            # Try each path
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        if status_callback:
            status_callback(f"Processing image with OCR: {os.path.basename(image_path)}")
        
        # Open the image
        image = Image.open(image_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        
        if text.strip():
            return text.strip()
        else:
            if status_callback:
                status_callback(f"No text found in image: {os.path.basename(image_path)}")
            return None
    except Exception as e:
        if status_callback:
            status_callback(f"OCR error: {str(e)}")
        return None

def load_documents_from_folder(folder_path, status_callback=None):
    """Load documents from a folder with various document loaders based on file type"""
    documents = []
    paths = []
    
    if status_callback:
        status_callback(f"Scanning folder: {folder_path}")
    
    # Track indexing statistics
    stats = {
        "attempted": 0,
        "loaded": 0,
        "failed": 0,
        "directories_processed": set(),
        "skipped_extensions": set(),
        "skipped_complex_files": 0,
        "limited_directories": 0,        # Count directories where we limited files
        "skipped_files_in_large_dirs": 0, # Count files skipped due to directory size limit
        "loaded_by_type": {}
    }
    
    # Maximum files to process per directory (when a directory has more than this number)
    MAX_FILES_PER_DIRECTORY = 100
    
    # Easy formats - these can be processed reliably
    easy_formats = {
        # Simple text formats
        '.txt': 'text',
        '.md': 'text',
        '.csv': 'text',
        '.json': 'text',
        '.xml': 'text',
        '.html': 'text',
        '.htm': 'text',
        '.yaml': 'text',
        '.yml': 'text',
        '.ini': 'text',
        '.conf': 'text',
        '.log': 'text',
        
        # Code formats
        '.py': 'code',
        '.js': 'code',
        '.java': 'code',
        '.c': 'code',
        '.cpp': 'code',
        '.h': 'code',
        '.cs': 'code',
        '.go': 'code',
        '.rb': 'code',
        '.php': 'code',
        '.ts': 'code',
        '.jsx': 'code',
        '.tsx': 'code',
        '.css': 'code',
        '.sh': 'code',
        '.bash': 'code',
        '.sql': 'code',
        
        # Document formats (need additional libraries)
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.rtf': 'text'
    }
    
    # Formats to skip entirely (complex formats that cause issues)
    complex_formats = [
        # Office spreadsheets/presentations (need specialized handlers)
        '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods',
        # Temporary files often created by Office apps
        '~$', '.tmp',
        # Images 
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp', '.heic', '.raw', '.svg',
        # Archives
        '.zip', '.rar', '.tar', '.gz', '.7z',
        # Binary formats
        '.exe', '.bin', '.dat', '.dll', '.so'
    ]
    
    if status_callback:
        status_callback("Starting file scan with no directory limits - limiting to 100 files per large directory...")
    
    try:
        # Walk the entire directory structure without limits
        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories and problematic system folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['.Trash', '.tmp', '.cache', 'node_modules']]
            
            # Add directory to processed list
            stats["directories_processed"].add(root)
            
            # Update progress periodically
            if stats["directories_processed"] and len(stats["directories_processed"]) % 20 == 0:
                if status_callback:
                    status_callback(f"Processed {len(stats['directories_processed'])} directories, found {stats['loaded']} usable documents...")
            
            # Check if directory has too many files and report it
            if len(files) > MAX_FILES_PER_DIRECTORY:
                stats["limited_directories"] += 1
                if status_callback:
                    status_callback(f"Directory {root} has {len(files)} files - limiting to {MAX_FILES_PER_DIRECTORY}")
                # Calculate how many files we're skipping
                stats["skipped_files_in_large_dirs"] += (len(files) - MAX_FILES_PER_DIRECTORY)
                # Limit files to process
                files = files[:MAX_FILES_PER_DIRECTORY]
            
            # Process each file (with directory size limit applied)
            for file in files:
                stats["attempted"] += 1
                file_path = os.path.join(root, file)
                
                # Skip hidden and system files
                if file.startswith('.') or file in ['.DS_Store', 'Thumbs.db', 'desktop.ini']:
                    continue
                
                # Skip temporary Office files
                if '~$' in file:
                    stats["skipped_complex_files"] += 1
                    continue
                
                # Extract extension and lowercase it
                ext = os.path.splitext(file)[1].lower()
                
                # Check if any complex format is in the file name
                is_complex = False
                for complex_ext in complex_formats:
                    if complex_ext in file.lower():
                        is_complex = True
                        break
                
                if is_complex:
                    stats["skipped_complex_files"] += 1
                    continue
                
                # Process based on file type
                if ext in easy_formats:
                    file_type = easy_formats[ext]
                    
                    # Process text-based files
                    if file_type == 'text' or file_type == 'code':
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                                if content.strip():  # Skip empty files
                                    documents.append(content)
                                    paths.append(file_path)
                                    stats["loaded"] += 1
                                    
                                    # Track file type stats
                                    stats["loaded_by_type"][file_type] = stats["loaded_by_type"].get(file_type, 0) + 1
                        except Exception as e:
                            if status_callback and len(documents) < 5:  # Only show errors for first few files
                                status_callback(f"Skipping text file {file_path}: {str(e)}")
                            stats["failed"] += 1
                    
                    # Process PDF files
                    elif file_type == 'pdf':
                        try:
                            # Try to use PyPDFLoader from langchain
                            pdf_text = ""
                            try:
                                from langchain_community.document_loaders import PyPDFLoader
                                loader = PyPDFLoader(file_path)
                                pages = loader.load()
                                for page in pages:
                                    pdf_text += page.page_content + "\n\n"
                            except ImportError:
                                # Fallback to PyPDF2 if available
                                try:
                                    import PyPDF2
                                    with open(file_path, 'rb') as f:
                                        reader = PyPDF2.PdfReader(f)
                                        for page in reader.pages:
                                            pdf_text += page.extract_text() + "\n\n"
                                except ImportError:
                                    # Fallback to pdfminer if available
                                    try:
                                        from pdfminer.high_level import extract_text  # pyright: ignore[reportMissingImports]
                                        pdf_text = extract_text(file_path)
                                    except ImportError:
                                        if status_callback:
                                            status_callback("PDF processing libraries not found. Install PyPDF2 or pdfminer.six to process PDFs.")
                                        continue
                            
                            if pdf_text.strip():
                                documents.append(pdf_text)
                                paths.append(file_path)
                                stats["loaded"] += 1
                                
                                # Track file type stats
                                stats["loaded_by_type"]["pdf"] = stats["loaded_by_type"].get("pdf", 0) + 1
                        except Exception as e:
                            if status_callback and len(documents) < 5:
                                status_callback(f"Skipping PDF {file_path}: {str(e)}")
                            stats["failed"] += 1
                    
                    # Process Word documents
                    elif file_type == 'docx':
                        try:
                            # Try docx2txt first
                            docx_text = ""
                            try:
                                import docx2txt
                                docx_text = docx2txt.process(file_path)
                            except ImportError:
                                # Fallback to python-docx if available
                                try:
                                    import docx  # pyright: ignore[reportMissingImports]
                                    doc = docx.Document(file_path)
                                    docx_text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
                                except ImportError:
                                    if status_callback:
                                        status_callback("Word document processing libraries not found. Install docx2txt or python-docx to process .docx files.")
                                    continue
                            
                            if docx_text.strip():
                                documents.append(docx_text)
                                paths.append(file_path)
                                stats["loaded"] += 1
                                
                                # Track file type stats
                                stats["loaded_by_type"]["docx"] = stats["loaded_by_type"].get("docx", 0) + 1
                        except Exception as e:
                            if status_callback and len(documents) < 5:
                                status_callback(f"Skipping Word doc {file_path}: {str(e)}")
                            stats["failed"] += 1
                else:
                    stats["skipped_extensions"].add(ext)
        
        if status_callback:
            status_callback(f"Document loading complete! Found {stats['loaded']} documents.")
            
            # Report on directory size limiting
            if stats["limited_directories"] > 0:
                status_callback(f"Limited {stats['limited_directories']} large directories to {MAX_FILES_PER_DIRECTORY} files each.")
                status_callback(f"Skipped approximately {stats['skipped_files_in_large_dirs']} files in large directories.")
            
            # If no PDF/Word docs were loaded but those extensions were requested
            if "pdf" not in stats["loaded_by_type"] and ".pdf" in easy_formats:
                status_callback("Note: No PDF files were successfully loaded. You may need to install PyPDF2 or pdfminer.six.")
            
            if "docx" not in stats["loaded_by_type"] and (".docx" in easy_formats or ".doc" in easy_formats):
                status_callback("Note: No Word documents were successfully loaded. You may need to install docx2txt or python-docx.")
    
    except Exception as e:
        if status_callback:
            status_callback(f"Error during file processing: {str(e)}")
    
    # Display stats summary
    if status_callback:
        status_callback(f"Document loading summary:")
        status_callback(f"- Total documents loaded: {len(documents)}")
        
        # Show breakdown by file type
        if stats["loaded_by_type"]:
            type_summary = ", ".join([f"{count} {doc_type}" for doc_type, count in stats["loaded_by_type"].items()])
            status_callback(f"- Document types: {type_summary}")
        
        if stats["limited_directories"] > 0:
            status_callback(f"- Large directories limited: {stats['limited_directories']} (max {MAX_FILES_PER_DIRECTORY} files per directory)")
            status_callback(f"- Files skipped in large dirs: {stats['skipped_files_in_large_dirs']}")
            
        status_callback(f"- Documents attempted: {stats['attempted']}")
        status_callback(f"- Loading failures: {stats['failed']}")
        status_callback(f"- Complex files skipped: {stats['skipped_complex_files']}")
        status_callback(f"- Directories processed: {len(stats['directories_processed'])}")
        
        if stats["skipped_extensions"]:
            status_callback(f"- Skipped file extensions: {', '.join(stats['skipped_extensions'])}")
    
    return documents, paths

def split_documents(documents, paths, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for indexing"""
    try:
        # With 96GB RAM, we can handle much larger document collections
        # Set very high limits only for extreme cases to prevent system crashes
        MAX_DOCUMENT_SIZE = 10000000  # 10MB per document - only limit truly massive files
        MAX_CHUNK_COUNT = 1000000  # Up to 1 million chunks - ChromaDB should handle this with 96GB RAM
        
        # Dynamic chunk sizing based on document types rather than arbitrary limits
        # This is for optimal retrieval quality, not memory constraints
        
        # Try to use LangChain's text splitter if available
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        all_metadatas = []
        total_chars = sum(len(doc) for doc in documents)
        
        # Process each document based on file extension
        for i, doc in enumerate(documents):
            # Only truncate extreme documents (10MB+)
            if len(doc) > MAX_DOCUMENT_SIZE:
                print(f"Truncating extremely large document: {paths[i]} ({len(doc)/1000000:.2f}MB)")
                truncated_doc = doc[:MAX_DOCUMENT_SIZE] + "\n\n[CONTENT TRUNCATED - DOCUMENT TOO LARGE]"
                doc = truncated_doc
            
            file_path = paths[i]
            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            # Adjust chunk size based on file type for optimal retrieval
            adjusted_chunk_size = chunk_size
            adjusted_overlap = chunk_overlap
            
            # Use smaller chunks for code files to preserve function/method context
            if file_ext in ['.py', '.js', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php']:
                adjusted_chunk_size = min(chunk_size, 500)  # Smaller chunks for code
                adjusted_overlap = min(chunk_overlap, 100)
                
                # For code files, try to split more intelligently
                chunks = split_code_document(doc, adjusted_chunk_size, adjusted_overlap)
            # Use larger chunks for structured data files
            elif file_ext in ['.json', '.xml', '.csv']:
                adjusted_chunk_size = min(chunk_size, 800)  # Structured data needs context
                chunks = text_splitter.split_text(doc)
            # Standard text documents
            else:
                chunks = text_splitter.split_text(doc)
            
            # Add metadata to each chunk
            for j, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": file_path,
                    "chunk": j,
                    "filename": file_name,
                    "filetype": file_ext[1:] if file_ext.startswith('.') else file_ext,
                    "total_chunks": len(chunks)
                })
                
                # Only enforce chunk limit for extreme cases to prevent system crashes
                # With 96GB this should rarely be hit
                if len(all_chunks) >= MAX_CHUNK_COUNT:
                    print(f"WARNING: Reached extreme chunk count ({MAX_CHUNK_COUNT}). This may impact system stability.")
                    break
            
            # Break main loop if we hit the max chunk count
            if len(all_chunks) >= MAX_CHUNK_COUNT:
                break
        
        return all_chunks, all_metadatas
    
    except Exception as e:
        print(f"Error using LangChain text splitter: {e}. Falling back to simple chunking.")
        return simple_split_documents(documents, paths, chunk_size, chunk_overlap)

def split_code_document(doc, chunk_size=500, chunk_overlap=100):
    """Split code documents more intelligently to preserve function/class boundaries"""
    # First try to split by class/function definitions
    chunks = []
    lines = doc.split('\n')
    
    # Look for common code block patterns
    block_patterns = [
        r'^\s*(def|class|function|public|private|protected|async|static|void|int|function\*|const|let|var)\s+\w+',
        r'^\s*(#|//)\s*[-=]{3,}',  # Comment section headers with dashes/equals
        r'^\s*"""',                # Python docstring
        r'^\s*\/\*',               # C-style comment block
        r'^\s*\*\/'                # End of C-style comment block
    ]
    
    # Compile patterns
    block_patterns = [re.compile(pattern) for pattern in block_patterns]
    
    # Find potential block boundaries
    block_boundaries = []
    for i, line in enumerate(lines):
        for pattern in block_patterns:
            if pattern.match(line):
                block_boundaries.append(i)
                break
    
    # If we found good boundaries, use them
    if len(block_boundaries) > 2:  # Need at least a few boundaries to be useful
        # Add start and end
        if 0 not in block_boundaries:
            block_boundaries.insert(0, 0)
        if len(lines) - 1 not in block_boundaries:
            block_boundaries.append(len(lines))
            
        # Sort boundaries
        block_boundaries = sorted(list(set(block_boundaries)))
        
        # Create chunks based on boundaries, merging very small blocks
        current_chunk = []
        current_size = 0
        
        for i in range(len(block_boundaries) - 1):
            start = block_boundaries[i]
            end = block_boundaries[i + 1]
            
            block_lines = lines[start:end]
            block_text = '\n'.join(block_lines)
            block_size = len(block_text)
            
            # If adding this block exceeds chunk size and we already have content,
            # store current chunk and start a new one
            if current_size + block_size > chunk_size and current_size > 0:
                chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = len('\n'.join(current_chunk))
            
            # Add block to current chunk
            current_chunk.extend(block_lines)
            current_size += block_size
        
        # Add the final chunk if there's anything left
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
    
    # If the block-based approach didn't work well, fall back to simple line-based chunking
    if not chunks or max(len(chunk) for chunk in chunks) > chunk_size * 2:
        # Reset and use simpler approach
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed chunk size, store chunk and start new one
            if current_size + line_size > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with overlap - try to keep logical lines together
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(line) + 1 for line in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add the final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
    
    return chunks

def simple_split_documents(documents, paths, chunk_size=1000, chunk_overlap=200):
    """Simple alternative to split documents without LangChain dependency"""
    all_chunks = []
    all_metadatas = []
    
    for i, doc in enumerate(documents):
        file_path = paths[i]
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # First try splitting by paragraphs
        paragraphs = re.split(r'\n\s*\n', doc)
        
        # If document has very few paragraphs, try line-based splitting
        if len(paragraphs) < 3:
            paragraphs = doc.split('\n')
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk size, store current chunk and start new one
            if current_length + len(para) > chunk_size and current_length > 0:
                # Store the current chunk
                chunk_text = '\n\n'.join(current_chunk)
                all_chunks.append(chunk_text)
                all_metadatas.append({
                    "source": file_path,
                    "chunk": len(all_chunks) - 1,
                    "filename": file_name,
                    "filetype": file_ext[1:] if file_ext.startswith('.') else file_ext
                })
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - chunk_overlap // len(current_chunk[0]) if current_chunk else 0)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(p) for p in current_chunk)
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_length += len(para)
            
        # Add the final chunk if there's anything left
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            all_chunks.append(chunk_text)
            all_metadatas.append({
                "source": file_path,
                "chunk": len(all_chunks) - 1,
                "filename": file_name,
                "filetype": file_ext[1:] if file_ext.startswith('.') else file_ext
            })
    
    # If we still have no chunks or very large documents, use character-based chunking as last resort
    if not all_chunks or max(len(chunk) for chunk in all_chunks) > chunk_size * 2:
        # Fall back to simple character-based chunking
        all_chunks = []
        all_metadatas = []
        
        for i, doc in enumerate(documents):
            file_path = paths[i]
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Simple character-based chunking
            for j in range(0, len(doc), chunk_size - chunk_overlap):
                chunk = doc[j:j + chunk_size]
                if chunk.strip():  # Skip empty chunks
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "source": file_path,
                        "chunk": j // (chunk_size - chunk_overlap),
                        "filename": file_name,
                        "filetype": file_ext[1:] if file_ext.startswith('.') else file_ext
                    })
    
    return all_chunks, all_metadatas

def index_folder_to_chromadb(folder_path, collection_name, persist_dir, status_callback=None):
    """Index a folder of documents to ChromaDB"""
    if not CHROMADB_AVAILABLE:
        if status_callback:
            status_callback("ChromaDB not available. Install with: pip install chromadb")
        return False
    if status_callback:
        status_callback("Loading documents...")
    
    start_time = time.time()
    
    # Load documents
    documents, paths = load_documents_from_folder(folder_path, status_callback=status_callback)
    
    if not documents:
        if status_callback:
            status_callback("No documents found in folder.")
        return False
    
    doc_count = len(documents)
    if status_callback:
        status_callback(f"Found {doc_count} documents. Splitting into chunks...")
        
        # Display memory usage info if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            total_memory = psutil.virtual_memory().total
            status_callback(f"Current memory usage: {memory_info.rss / (1024 * 1024):.1f}MB, System total: {total_memory / (1024 * 1024 * 1024):.1f}GB")
        except:
            pass  # Skip if psutil not available
    
    # Split documents with progress reporting
    chunks = []
    metadatas = []
    
    # Process in batches to avoid memory issues even with 96GB RAM
    # Using reasonable batch sizes for progress reporting, not due to memory limitations
    batch_size = 100  # Larger batch size for high-RAM systems
    total_batches = (doc_count + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, doc_count)
        
        if status_callback:
            status_callback(f"Processing batch {batch_num+1}/{total_batches} (documents {start_idx+1}-{end_idx})...")
        
        # Get current batch
        batch_docs = documents[start_idx:end_idx]
        batch_paths = paths[start_idx:end_idx]
        
        # Split this batch
        try:
            batch_chunks, batch_metadatas = split_documents(batch_docs, batch_paths)
            chunks.extend(batch_chunks)
            metadatas.extend(batch_metadatas)
            
            if status_callback:
                status_callback(f"Batch {batch_num+1} created {len(batch_chunks)} chunks")
                
                # Show memory usage after each batch
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    status_callback(f"Current memory usage: {memory_info.rss / (1024 * 1024):.1f}MB")
                except:
                    pass
        except Exception as e:
            if status_callback:
                status_callback(f"Error splitting batch {batch_num+1}: {str(e)}")
    
    chunk_count = len(chunks)
    if status_callback:
        status_callback(f"Created {chunk_count} chunks total. Adding to ChromaDB...")
    
    if chunk_count == 0:
        if status_callback:
            status_callback("No chunks were created. Check document formats and content.")
        return False
    
    # Create unique IDs
    ids = [f"doc_{i}_{uuid.uuid4().hex[:8]}" for i in range(chunk_count)]
    
    # Create or get collection
    try:
        client, collection = create_chroma_collection(collection_name, persist_dir)
    except Exception as e:
        if status_callback:
            status_callback(f"Error creating ChromaDB collection: {str(e)}")
        return False
    
    # Add documents in batches - using sensible batch sizes for progress reporting
    # For 96GB RAM, we can use larger batches
    embedding_batch_size = 500  # Increased for high-RAM systems (was 100)
    batches_to_process = (chunk_count + embedding_batch_size - 1) // embedding_batch_size
    
    success_count = 0
    error_count = 0
    
    for i in range(0, chunk_count, embedding_batch_size):
        end = min(i + embedding_batch_size, chunk_count)
        batch_num = i // embedding_batch_size + 1
        
        if status_callback:
            status_callback(f"Adding embedding batch {batch_num}/{batches_to_process} (chunks {i+1}-{end})...")
        
        try:
            collection.add(
                documents=chunks[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            success_count += (end - i)
            
            # Show memory usage after each batch
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                status_callback(f"Memory after batch {batch_num}: {memory_info.rss / (1024 * 1024):.1f}MB")
            except:
                pass
                
        except Exception as e:
            if status_callback:
                status_callback(f"Error adding batch {batch_num}: {str(e)}")
            
            # Try adding one by one if batch fails
            if status_callback:
                status_callback("Attempting to add documents individually...")
            
            for j in range(i, end):
                try:
                    collection.add(
                        documents=[chunks[j]],
                        metadatas=[metadatas[j]],
                        ids=[ids[j]]
                    )
                    success_count += 1
                except Exception as e2:
                    error_count += 1
                    # Only log every 10 errors to avoid flooding
                    if error_count <= 10 or error_count % 10 == 0:
                        if status_callback:
                            status_callback(f"Error adding document {j}: {str(e2)}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if status_callback:
        status_callback(f"Indexing complete in {elapsed:.1f} seconds.")
        status_callback(f"Added {success_count} chunks to ChromaDB collection '{collection_name}'.")
        if error_count > 0:
            status_callback(f"Failed to add {error_count} chunks due to errors.")
            
        # Final memory stats
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            status_callback(f"Final memory usage: {memory_info.rss / (1024 * 1024):.1f}MB")
        except:
            pass
    
    return success_count > 0

def query_chromadb(query, collection_name, persist_dir, n_results=5):
    """Query the ChromaDB collection for relevant documents"""
    if not CHROMADB_AVAILABLE:
        return None, "ChromaDB not available. Install with: pip install chromadb"

    # Get collection
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    try:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
    except ValueError:
        return None, "Collection not found. Please index documents first."
    
    # Query collection
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results, None

def get_default_system_message():
    """Return the default system message (Python programmer in code-only version)"""
    return python_system_message

def find_gguf_models(base_path=None):
    """Find all GGUF models, showing only the first file of multi-part series"""

    if base_path is None:
        # Auto-detect based on platform
        import platform
        if platform.system() == "Linux":
            base_path = "/data/GGUF_Models"
        else:  # macOS and others
            base_path = "/Users/jonathanrothberg/GGUF_Models"
    
    if not os.path.exists(base_path):
        print(f"GGUF directory {base_path} does not exist!")
        return []
    
    # Find all .gguf files recursively (case insensitive)
    gguf_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.gguf'):
                full_path = os.path.join(root, file)
                gguf_files.append(full_path)
    
    # Filter to show only first files of series (e.g., only -00001- files)
    filtered_models = {}
    
    for file_path in gguf_files:
        filename = os.path.basename(file_path)
        
        # Check if this is a multi-part file (contains -00001- or _00001- patterns)
        if (('-00' in filename and '-of-' in filename) or ('_00' in filename and '-of-' in filename)):
            # Extract the base name (everything before -00001- or _00001-)
            if '-00' in filename:
                parts = filename.split('-00')
                separator = '-00'
            else:
                parts = filename.split('_00')
                separator = '_00'
                
            if len(parts) >= 2:
                base_name = parts[0]
                # Only keep if this is the first file (-00001- or _00001-)
                if '-00001-' in filename or '_00001-' in filename:
                    filtered_models[base_name] = file_path
        else:
            # Single file model
            base_name = filename.replace('.gguf', '').replace('.GGUF', '')
            filtered_models[base_name] = file_path
    
    return list(filtered_models.values())

def detect_macos_and_find_mlx_models():
    """
    Detect if running on macOS and find MLX models in the MLX_Models folder.
    Returns the best available model path or None if not on macOS or no models found.
    """
    # Check if running on macOS
    import platform
    if platform.system() != "Darwin":
        return None

    # Check multiple possible MLX_Models directory locations
    possible_dirs = [
        Path("/Users/jonathanrothberg/MLX_Models"),  # Original path
        Path.home() / "MLX_Models",  # User's home directory
    ]

    mlx_models_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            mlx_models_dir = dir_path
            break

    # Check if any MLX_Models directory exists
    if mlx_models_dir is None:
        return None

    # Find all model directories that actually contain safetensors files
    model_dirs = []
    for item in mlx_models_dir.iterdir():
        if item.is_dir():
            # Check if directory contains actual safetensors files (not just index)
            safetensors_files = list(item.glob("*.safetensors"))
            if safetensors_files:  # Only include if it has actual model files
                model_dirs.append(item)

    if not model_dirs:
        return None

    # Sort by preference: prioritize 6bit model, then thinking models, then others
    def model_priority(model_path):
        name = str(model_path).lower()
        if "6bit" in name:
            return 0  # Highest priority - 6bit preferred
        elif "thinking" in name:
            return 1  # Second priority - thinking models
        elif "qwen" in name:
            return 2
        elif "llama" in name:
            return 3
        else:
            return 4

    model_dirs.sort(key=model_priority)

    # Return the best candidate (first one after sorting)
    best_model = model_dirs[0]
    return str(best_model)

def get_models_transformer_path():
    """Get the appropriate Models_Transformer path based on platform and environment."""
    import platform
    from pathlib import Path

    # Check if we're running in the CodeRunner container (models mounted at /app/models)
    if Path("/app/models").exists() and Path("/app/models").is_dir():
        return "/app/models"

    # Use /data/Models_Transformer only on AMD Ubuntu (x86_64), keep /home/jonathan/Models_Transformer for ARM Ubuntu and others
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        return "/data/Models_Transformer"
    else:  # ARM Ubuntu, macOS and others
        return "/home/jonathan/Models_Transformer"

def get_available_vllm_models():
    """
    Get list of available vLLM models in Models_Transformer directories.
    Checks both standard path and media path for models.
    Returns list of model paths or empty list if not available.
    """
    try:
        # Check if vLLM is available
        if not VLLM_AVAILABLE:
            return []

        # Define paths to check for models
        model_paths = [
            get_models_transformer_path(),  # Standard path
            "/media/jonathan/data/Models_Transformer"  # Media path
        ]

        # Get all model directories from all paths
        model_dirs = []
        for models_path in model_paths:
            vllm_models_dir = Path(models_path)

            # Check if directory exists
            if not vllm_models_dir.exists() or not vllm_models_dir.is_dir():
                print(f"Models_Transformer directory not found: {vllm_models_dir}")
                continue

            # Get all model directories (folders that contain model files)
            for item in vllm_models_dir.iterdir():
                if item.is_dir():
                    # Check for common model file extensions
                    has_model_files = (
                        list(item.glob("*.safetensors")) or  # Safetensors format
                        list(item.glob("*.bin")) or         # Binary format
                        list(item.glob("*.pt")) or          # PyTorch format
                        list(item.glob("pytorch_model.bin")) or  # HuggingFace format
                        list(item.glob("model-*.safetensors"))   # Sharded safetensors
                    )
                    if has_model_files:
                        model_dirs.append(str(item))

        return model_dirs

    except Exception as e:
        print(f"Error detecting vLLM models: {str(e)}")
        return []

def get_available_transformers_models():
    """
    Get list of available Transformers models in Models_Transformer directories.
    Checks both standard path and media path for models.
    Returns list of model paths or empty list if not available.
    """
    try:
        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            return []

        # Define paths to check for models
        model_paths = [
            get_models_transformer_path(),  # Standard path
            "/media/jonathan/data/Models_Transformer"  # Media path
        ]

        # Get all model directories from all paths
        model_dirs = []
        for models_path in model_paths:
            transformers_models_dir = Path(models_path)

            # Check if directory exists
            if not transformers_models_dir.exists() or not transformers_models_dir.is_dir():
                print(f"Models_Transformer directory not found: {transformers_models_dir}")
                continue

            # Get all model directories (folders that contain model files)
            for item in transformers_models_dir.iterdir():
                if item.is_dir():
                    # Check for common model file extensions
                    has_model_files = (
                        list(item.glob("*.safetensors")) or  # Safetensors format
                        list(item.glob("*.bin")) or         # Binary format
                        list(item.glob("*.pt")) or          # PyTorch format
                        list(item.glob("pytorch_model.bin")) or  # HuggingFace format
                        list(item.glob("model-*.safetensors"))   # Sharded safetensors
                    )
                    if has_model_files:
                        model_dirs.append(str(item))

        return model_dirs

    except Exception as e:
        print(f"Error detecting Transformers models: {str(e)}")
        return []

# ============================================================================
# MINI MAX MODEL SUPPORT
# ============================================================================
#
# MiniMax models (mlx-community/MiniMax-M2-mlx-8bit-gs32) are now supported
# in this application. The following changes were made to enable MiniMax support:
#
# 1. Added minimax.py model implementation to mlx_lm.models directory
#    - Downloaded from: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py
#    - Supports MiniMax's unique architecture with sigmoid scoring and QK norms
#
# 2. Updated MODEL_REMAPPING in mlx_lm/utils.py to include:
#    - "minimax": "minimax" mapping
#
# 3. MiniMax models can now be loaded using the standard MLX load() function
#    - Works with all existing MLX functions in this application
#    - Supports both local model paths and HuggingFace identifiers
#
# Example usage:
#   model, tokenizer = load("mlx-community/MiniMax-M2-mlx-8bit-gs32")
#   model, tokenizer = load("/Users/jonathanrothberg/MLX_Models/MiniMax-M2-8bit")
#
# ============================================================================

def is_mlx_vlm_model(model_path):
    """Check if an MLX model is a vision-language model by inspecting its config.json.

    Returns True if the model has vision capabilities, False otherwise.
    Never raises — returns False on any error.
    """
    try:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return False

        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check for vision_config key (most VLMs like Qwen2-VL, LLaVA)
        if "vision_config" in config:
            return True

        # Check for image_token_id (some VLMs)
        if "image_token_id" in config:
            return True

        # Check model_type for VL/vision indicators
        model_type = config.get("model_type", "").lower()
        if any(tag in model_type for tag in ["_vl", "_vision", "llava", "pixtral"]):
            return True

        # Check architectures for VL/Vision/Llava
        architectures = config.get("architectures", [])
        for arch in architectures:
            arch_lower = arch.lower()
            if any(tag in arch_lower for tag in ["vl", "vision", "llava", "pixtral"]):
                return True

        return False
    except Exception:
        return False


def get_available_mlx_models():
    """
    Get list of available MLX models on macOS.
    Returns list of (model_path, is_vlm) tuples or empty list if not available.

    NOTE: Now includes support for MiniMax models (mlx-community/MiniMax-M2-mlx-8bit-gs32)
    """
    try:
        # Check if MLX is available
        if not MLX_AVAILABLE:
            return []

        # Detect macOS and find models
        mlx_model = detect_macos_and_find_mlx_models()
        if not mlx_model:
            return []

        # Check multiple possible MLX_Models directory locations
        possible_dirs = [
            Path("/Users/jonathanrothberg/MLX_Models"),  # Original path
            Path.home() / "MLX_Models",  # User's home directory
        ]

        model_dirs = []
        vlm_flags = {}  # {path: bool} — True if VLM
        for mlx_models_dir in possible_dirs:
            if mlx_models_dir.exists() and mlx_models_dir.is_dir():
                for item in mlx_models_dir.iterdir():
                    if item.is_dir():
                        safetensors_files = list(item.glob("*.safetensors"))
                        if safetensors_files:
                            path_str = str(item)
                            model_dirs.append(path_str)
                            vlm_flags[path_str] = is_mlx_vlm_model(path_str)

        return model_dirs, vlm_flags

    except Exception as e:
        print(f"Error detecting MLX models: {str(e)}")
        return [], {}

def get_available_models():
    """Get list of available Ollama models"""
    try:
        # Use direct API call to get all models
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            # Ensure we have at least the default model
            if not models or len(models) == 0:
                models = [model]
            return models
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return [model]  # Return default model as fallback

def get_claude_models():
    """Get list of available Claude models"""
    try:
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data:
                claude_models = [model["id"] for model in models_data["data"]]
                if claude_models:
                    return sorted(claude_models)
            return []
        else:
            print(f"Error fetching Claude models: Status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Claude models: {e}")
        return []


def get_openai_models():
    """Get list of available OpenAI chat models (filtered to chat-capable only)
    Minimal helper mirroring Claude model fetch. Uses OPENAI_API_KEY from env, returns sorted IDs.
    """
    try:
        import os
        import requests
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # If no env var, try to read a local key file (surgical, optional)
            key_path = os.path.join(os.path.dirname(__file__), "openai_key.txt")
            if os.path.exists(key_path):
                with open(key_path, "r") as f:
                    api_key = f.read().strip()
        if not api_key:
            return []

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            all_models = [m.get("id", "") for m in models_data.get("data", [])]
            include_prefixes = ["gpt-5", "gpt-4", "gpt-3.5", "gpt-4o"]
            exclude_substrings = ["-search", "-transcribe", "-realtime", "-tts", "embedding", "-embed", "-audio"]
            filtered = [
                m for m in all_models
                if any(m.startswith(p) for p in include_prefixes)
                and not any(ex in m for ex in exclude_substrings)
            ]
            return sorted(filtered)
        else:
            return []
    except Exception:
        return []


# =============================================================================
# BROWSER ERROR SERVER — captures JS errors from HTML games via HTTP POST
# =============================================================================

class BrowserErrorServer:
    """Simple HTTP server to capture browser console errors and send them to debug console"""

    def __init__(self, ide_instance, port=8765):
        self.ide_instance = ide_instance
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start the error server in a background thread"""
        if self.thread and self.thread.is_alive():
            return  # Already running

        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        # print(f"🔧 Error capture server started on port {self.port}")  # Commented out to reduce noise

    def stop(self):
        """Stop the error server"""
        if self.server:
            self.server.shutdown()
            self.server = None

    def _run_server(self):
        """Run the HTTP server"""
        class ErrorHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, ide_instance=None, **kwargs):
                self.ide_instance = ide_instance
                super().__init__(*args, **kwargs)

            def do_POST(self):
                """Handle POST requests with error data"""
                # print(f"📨 Received POST request to: {self.path}")  # Commented out to reduce noise
                if self.path == '/report_error':
                    # print("✅ Processing error report")  # Commented out to reduce noise
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = urllib.parse.parse_qs(post_data.decode('utf-8'))
                    # print(f"📝 Error data: {data}")  # Commented out to reduce noise

                    # Extract error information
                    error_type = data.get('type', ['unknown'])[0]
                    error_message = data.get('message', ['No message'])[0]
                    error_source = data.get('source', ['unknown'])[0]
                    error_line = data.get('line', ['unknown'])[0]
                    error_column = data.get('column', ['unknown'])[0]
                    error_stack = data.get('stack', [''])[0]

                    # Send to debug console (safe)
                    target = getattr(self.ide_instance, 'add_to_debug_console', None)
                    if callable(target):
                        self.ide_instance.add_to_debug_console("="*60)
                        self.ide_instance.add_to_debug_console("BROWSER ERROR CAPTURED")
                        self.ide_instance.add_to_debug_console("="*60)
                        self.ide_instance.add_to_debug_console(f"Type: {error_type}")
                        self.ide_instance.add_to_debug_console(f"Message: {error_message}")
                    if error_source != 'unknown':
                        self.ide_instance.add_to_debug_console(f"Source: {error_source}")
                    if error_line != 'unknown':
                        self.ide_instance.add_to_debug_console(f"Line: {error_line}")
                    if error_column != 'unknown':
                        self.ide_instance.add_to_debug_console(f"Column: {error_column}")
                    if error_stack:
                        self.ide_instance.add_to_debug_console(f"Stack: {error_stack}")

                    # Send status message to system console
                    if hasattr(self.ide_instance, 'display_status_message'):
                        self.ide_instance.display_status_message(f"🚨 Browser error captured - see debug console")

                    # Send success response (with CORS for file:// origin)
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    self.end_headers()
                    self.wfile.write(b'Error logged')
                else:
                    self.send_response(404)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

            def do_GET(self):
                """Serve a tiny JS file for error capture when requested at /capture.js"""
                if self.path == '/capture.js':
                    payload = (
                        "window.addEventListener('error',function(e){try{var x=new XMLHttpRequest();x.open('POST','http://localhost:8765/report_error',true);x.setRequestHeader('Content-Type','application/x-www-form-urlencoded');var d='type='+encodeURIComponent('JavaScript Error')+'&message='+encodeURIComponent(e.message)+'&source='+encodeURIComponent(e.filename||'')+'&line='+encodeURIComponent(e.lineno||'')+'&column='+encodeURIComponent(e.colno||'')+'&stack='+encodeURIComponent(e.error&&e.error.stack||'');x.send(d);}catch(_){}});" 
                        + "window.addEventListener('unhandledrejection',function(e){try{var x=new XMLHttpRequest();x.open('POST','http://localhost:8765/report_error',true);x.setRequestHeader('Content-Type','application/x-www-form-urlencoded');var d='type='+encodeURIComponent('Unhandled Promise Rejection')+'&message='+encodeURIComponent((e.reason&&e.reason.toString())||'')+'&stack='+encodeURIComponent((e.reason&&e.reason.stack)||'');x.send(d);}catch(_){}});" 
                        + "(function(){var oe=console.error;console.error=function(){try{var m=Array.prototype.map.call(arguments,function(a){return (typeof a==='object')?JSON.stringify(a):String(a)}).join(' ');var x=new XMLHttpRequest();x.open('POST','http://localhost:8765/report_error',true);x.setRequestHeader('Content-Type','application/x-www-form-urlencoded');x.send('type='+encodeURIComponent('Console Error')+'&message='+encodeURIComponent(m));}catch(_){ } oe&&oe.apply(console,arguments);};})();"
                    ).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/javascript')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

            def do_OPTIONS(self):
                """Handle CORS preflight if a browser sends it"""
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def log_message(self, format, *args):
                # Suppress server logs
                pass

        try:
            # Allow quick reuse if port was recently used
            class ReusableTCPServer(socketserver.TCPServer):
                allow_reuse_address = True

            with ReusableTCPServer(("", self.port), lambda *args, **kwargs: ErrorHandler(*args, ide_instance=self.ide_instance, **kwargs)) as httpd:
                self.server = httpd
                httpd.serve_forever()
        except Exception as e:
            print(f"Error server failed: {e}")


# =============================================================================
# IDE WINDOW — code editor with toolbar, diff view, Run & Fix, Accept/Reject
# =============================================================================

class IDEWindow:
    """Simple IDE window for code editing - streamlined interface
    
    KEY FEATURES:
    1. Code Editing: Full-featured text editor with syntax highlighting and line numbers
    2. File Operations: New, Save to File, Load File
    3. Navigation: Find text, Go to Line
    4. Essential Chat Integration:
       - "Run": Executes code and shows output in chat
       - "Send to Chat": Sends your current code to chat as a user message
       - "Ask LLM to Fix": Asks the LLM to fix/improve the current code
    
    SIMPLE WORKFLOW:
    1. Write/edit code in the IDE
    2. Click "Run" to test it
    3. Click "Send to Chat" to continue conversation with your code
    4. Click "Ask LLM to Fix" if you need help with errors
    
    This streamlined interface focuses on the most useful features:
    - Edit code with syntax highlighting
    - Run and test directly
    - Send your work back to chat easily
    - Get LLM help when needed
    """
    
    def __init__(self, parent_gui):
        self.parent = parent_gui
        self.root = None
        self.editor = None
        self.current_content = ""  # The accepted/current code
        self.proposed_content = ""  # LLM-proposed changes
        self.diff_tags = []  # Tags for highlighting diffs
        self.is_showing_diff = False  # Whether diff view is active
        self.diff_hunks = []  # Per-hunk state for accept/reject
        self.diff_opcodes = []
        self.diff_current_lines = []
        self.diff_proposed_lines = []
        self.current_hunk_idx = 0
        self._hunk_display_ranges = {}
        self.help_window = None  # Track help window

        # Lightweight browser error server (always running for reliability)
        # Route to main GUI so debug console methods are available
        self.error_server = BrowserErrorServer(self.parent)
        self.error_server.start()

        # Create IDE window
        self.create_ide_window()
        
    def create_ide_window(self):
        """Create the IDE window interface
        
        The IDE provides practical editing features:
        - Fix Selected: Fix errors with debug context
        - Add Feature: Add functionality to existing code
        - Accept/Reject: Review changes before applying
        
        This approach is more practical than complex FIM tokens
        or trying to parse mixed explanations and code.
        """
        self.root = Toplevel(self.parent.root)
        self.root.title("CodeRunner IDE — Editor")
        self.root.geometry("1450x700")
        self.root.withdraw()  # Hide initially
        
        # Override close button to hide instead of destroy
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        
        # Main frame
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions bar at the very top
        instructions_frame = Frame(main_frame, bg="lightgray", height=25)
        instructions_frame.pack(fill=tk.X, pady=(0, 5))
        instructions_frame.pack_propagate(False)
        
        # Brief keyboard shortcuts reference
        instructions_text = "F5=Run | Ask LLM to Fix (targeted) | check 'Return full code' for rewrites | Ctrl+Enter=Accept All | Esc=Reject All | Ctrl+Up/Down=Step hunks"
        Label(instructions_frame, text=instructions_text, bg="lightgray", fg="darkblue",
              font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=5, pady=2)
        
        # Help button on the right
        Button(instructions_frame, text="❓ Help", command=self.show_help_popup, 
               bg="lightyellow", font=("TkDefaultFont", 9)).pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Toolbar container (for two rows)
        toolbar_container = Frame(main_frame)
        toolbar_container.pack(fill=tk.X, pady=(0, 10))
        
        # FIRST ROW TOOLBAR
        toolbar = Frame(toolbar_container)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # File operations — compact layout
        bf = ("TkDefaultFont", 8)
        Button(toolbar, text="New", command=self.new_file, font=bf).pack(side=tk.LEFT, padx=1)
        self.save_button = Button(toolbar, text="Save", command=self.save_file, state=tk.DISABLED, font=bf)
        self.save_button.pack(side=tk.LEFT, padx=1)
        Button(toolbar, text="Save As", command=self.save_as_file, font=bf).pack(side=tk.LEFT, padx=1)
        Button(toolbar, text="Load", command=self.load_file, font=bf).pack(side=tk.LEFT, padx=1)
        Button(toolbar, text="Find", command=self.show_find_dialog, font=bf).pack(side=tk.LEFT, padx=1)
        Button(toolbar, text="GoTo", command=self.go_to_line, font=bf).pack(side=tk.LEFT, padx=1)

        Frame(toolbar, width=1, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=3)

        # Run
        run_frame = Frame(toolbar)
        run_frame.pack(side=tk.LEFT, padx=1)
        Button(run_frame, text="Run F5", command=self.run_current_code, bg="lightblue", font=bf).pack(side=tk.LEFT)
        self.timed_execution = BooleanVar(value=True)
        Checkbutton(run_frame, text="Timed", variable=self.timed_execution, font=bf).pack(side=tk.LEFT, padx=1)

        Frame(toolbar, width=1, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=3)

        # Fix
        Button(toolbar, text="LLM Fix", command=self.fix_current_code, bg="lightyellow",
               font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT, padx=1)
        self.return_full_code = BooleanVar(value=False)
        Checkbutton(toolbar, text="Full code", variable=self.return_full_code,
                    font=bf).pack(side=tk.LEFT, padx=1)

        Frame(toolbar, width=1, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=3)

        # Accept/Reject all (disabled until diff is shown)
        self.accept_btn = Button(toolbar, text="Accept", command=self.accept_changes,
                                 bg="lightgreen", fg="darkgreen", state=tk.DISABLED,
                                 font=("TkDefaultFont", 8, "bold"))
        self.accept_btn.pack(side=tk.LEFT, padx=1)
        self.reject_btn = Button(toolbar, text="Reject", command=self.reject_changes,
                                 bg="lightpink", fg="darkred", state=tk.DISABLED,
                                 font=("TkDefaultFont", 8, "bold"))
        self.reject_btn.pack(side=tk.LEFT, padx=1)

        # Per-hunk navigation (hidden until diff is shown)
        self.hunk_frame = Frame(toolbar)
        self.prev_hunk_btn = Button(self.hunk_frame, text="<", command=lambda: self.navigate_hunk(-1), font=bf)
        self.prev_hunk_btn.pack(side=tk.LEFT, padx=0)
        self.hunk_label = Label(self.hunk_frame, text="", font=("TkDefaultFont", 8, "bold"))
        self.hunk_label.pack(side=tk.LEFT, padx=1)
        self.next_hunk_btn = Button(self.hunk_frame, text=">", command=lambda: self.navigate_hunk(1), font=bf)
        self.next_hunk_btn.pack(side=tk.LEFT, padx=0)
        self.accept_hunk_btn = Button(self.hunk_frame, text="Yes", command=self.accept_current_hunk,
                                       bg="lightgreen", font=("TkDefaultFont", 8, "bold"))
        self.accept_hunk_btn.pack(side=tk.LEFT, padx=(2, 0))
        self.reject_hunk_btn = Button(self.hunk_frame, text="No", command=self.reject_current_hunk,
                                       bg="lightpink", font=("TkDefaultFont", 8, "bold"))
        self.reject_hunk_btn.pack(side=tk.LEFT, padx=1)
        self.apply_hunks_btn = Button(self.hunk_frame, text="Done", command=self.apply_hunk_decisions,
                                       bg="lightblue", font=("TkDefaultFont", 8, "bold"))
        self.apply_hunks_btn.pack(side=tk.LEFT, padx=1)

        # Status label
        self.status_label = Label(toolbar, text="Ready", fg="blue")
        self.status_label.pack(side=tk.RIGHT)
        
        # Code editor with syntax highlighting
        editor_frame = Frame(main_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers frame
        self.line_frame = Frame(editor_frame, width=50, bg="lightgray")
        self.line_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.line_frame.pack_propagate(False)
        
        self.line_text = Text(self.line_frame, width=4, bg="lightgray", fg="black", 
                              state=tk.DISABLED, wrap=tk.NONE, font=("Consolas", 10))
        self.line_text.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for editor and scrollbar
        editor_scroll_frame = Frame(editor_frame)
        editor_scroll_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Main editor without built-in scrollbar
        self.editor = Text(editor_scroll_frame, wrap=tk.NONE, font=("Consolas", 10), undo=True)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Shared scrollbar
        scrollbar = Scrollbar(editor_scroll_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Connect scrollbar to both text widgets with proper synchronization
        scrollbar.config(command=self.sync_yview)
        self.editor.config(yscrollcommand=self.on_editor_scroll)
        self.line_text.config(yscrollcommand=self.on_line_scroll)
        
        # Store scrollbar reference for callbacks
        self.scrollbar = scrollbar
        
        # Configure syntax highlighting tags
        self.setup_syntax_highlighting()
        
        # Bind events
        self.editor.bind('<KeyRelease>', self.on_text_change)
        self.editor.bind('<Key>', self.on_key_press)
        # Don't bind Button-1 to on_text_change to prevent scroll jumps
        
        # Setup context menu for copy/paste
        self.create_ide_context_menu()
        
        # Add keyboard shortcuts for common operations
        self.setup_keyboard_shortcuts()
        
        # Update line numbers initially
        self.update_line_numbers()
        
        # Initialize enhanced IDE features
        self.setup_enhanced_features()

        # Bind F1 to help
        self.editor.bind('<F1>', lambda e: self.show_help_popup())
        self.root.bind('<F1>', lambda e: self.show_help_popup())

        # Show welcome text in editor on first open
        welcome = """# Welcome to CodeRunner IDE!
#
# STEP 1 — Get your first code:
#   Type in Chat: "Write a Space Invaders game in Pygame"
#   Click "Move to IDE" to load the code here
#   Press F5 to Run
#
# STEP 2 — Fix or change your code:
#   Type what's wrong in Chat ("enemies don't move")
#   Click "Ask LLM to Fix" in the toolbar above
#   Review the diff -> Accept (Ctrl+Enter) or Reject (Esc)
#
# You NEVER lose your code. All changes shown as diffs.
#
# Press F1 or click Help for more info
"""
        self.editor.insert("1.0", welcome)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            # Create tooltip window
            tooltip = Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            # Add text
            label = Label(tooltip, text=text, background="lightyellow", 
                         relief="solid", borderwidth=1, font=("TkDefaultFont", 9))
            label.pack()
            
            # Store reference
            widget.tooltip = tooltip
            
        def on_leave(event):
            # Destroy tooltip if it exists
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for all IDE operations"""
        # File operations
        self.editor.bind('<Control-n>', lambda e: self.new_file())
        self.editor.bind('<Control-o>', lambda e: self.load_file())
        self.editor.bind('<Control-s>', lambda e: self.save_file())
        self.editor.bind('<Control-Shift-S>', lambda e: self.save_as_file())
        
        # Edit operations
        self.editor.bind('<Control-z>', lambda e: self.editor.edit_undo())
        self.editor.bind('<Control-y>', lambda e: self.editor.edit_redo())
        self.editor.bind('<Control-a>', lambda e: self.select_all_ide())
        
        # Search operations
        self.editor.bind('<Control-f>', lambda e: self.show_find_dialog())
        self.editor.bind('<Control-g>', lambda e: self.go_to_line())
        
        # Code operations
        self.editor.bind('<F5>', lambda e: self.run_current_code())
        self.editor.bind('<Control-Return>', lambda e: self.accept_changes())
        self.editor.bind('<Escape>', lambda e: self.reject_changes())
        # Hunk navigation: Up/Down to step through hunks, a=accept, r=reject
        self.editor.bind('<Control-Up>', lambda e: self.navigate_hunk(-1))
        self.editor.bind('<Control-Down>', lambda e: self.navigate_hunk(1))
        self.editor.bind('<Control-bracketleft>', lambda e: self.navigate_hunk(-1))
        self.editor.bind('<Control-bracketright>', lambda e: self.navigate_hunk(1))

        # Mac compatibility
        self.editor.bind('<Command-n>', lambda e: self.new_file())
        self.editor.bind('<Command-o>', lambda e: self.load_file())
        self.editor.bind('<Command-s>', lambda e: self.save_file())
        self.editor.bind('<Command-Shift-S>', lambda e: self.save_as_file())
        self.editor.bind('<Command-z>', lambda e: self.editor.edit_undo())
        self.editor.bind('<Command-y>', lambda e: self.editor.edit_redo())
        self.editor.bind('<Command-a>', lambda e: self.select_all_ide())
        self.editor.bind('<Command-f>', lambda e: self.show_find_dialog())
        self.editor.bind('<Command-g>', lambda e: self.go_to_line())
        self.editor.bind('<Command-Return>', lambda e: self.accept_changes())
        
    def create_ide_context_menu(self):
        """Create comprehensive right-click context menu for IDE editor"""
        self.ide_context_menu = Menu(self.root, tearoff=0)

        # === EDIT OPERATIONS ===
        edit_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="✂️ Edit", menu=edit_menu)
        edit_menu.add_command(label="↩️ Undo", command=lambda: self.editor.edit_undo(), accelerator="Ctrl+Z")
        edit_menu.add_command(label="↪️ Redo", command=lambda: self.editor.edit_redo(), accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="✂️ Cut", command=self.cut_ide_selection, accelerator="Ctrl+X")
        edit_menu.add_command(label="📋 Copy", command=self.copy_ide_selection, accelerator="Ctrl+C")
        edit_menu.add_command(label="📄 Paste", command=self.paste_ide_clipboard, accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="🔍 Select All", command=self.select_all_ide, accelerator="Ctrl+A")

        # === FILE OPERATIONS ===
        file_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="📁 File", menu=file_menu)
        file_menu.add_command(label="📄 New File", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="📂 Open File", command=self.load_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="💾 Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="💾 Save As...", command=self.save_as_file, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="📤 Move to Chat", command=self.move_program_to_chat)
        
        # === CODE OPERATIONS ===
        code_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="⚡ Code", menu=code_menu)
        code_menu.add_command(label="▶️ Run Code", command=self.run_current_code, accelerator="F5")
        code_menu.add_command(label="🔧 Ask LLM to Fix", command=self.fix_current_code)

        # === REVIEW ===
        review_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="✅ Review", menu=review_menu)
        review_menu.add_command(label="Accept Changes", command=self.accept_changes, accelerator="Ctrl+Enter")
        review_menu.add_command(label="Reject Changes", command=self.reject_changes, accelerator="Escape")

        # === SEARCH & NAVIGATION ===
        search_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="🔍 Search", menu=search_menu)
        search_menu.add_command(label="🔍 Find Text", command=self.show_find_dialog, accelerator="Ctrl+F")
        search_menu.add_command(label="📍 Go to Line", command=self.go_to_line, accelerator="Ctrl+G")

        # === VIEW OPTIONS ===
        view_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="👁️ View", menu=view_menu)
        if PYGMENTS_AVAILABLE:
            view_menu.add_command(label="🎨 Refresh Highlighting", command=self.refresh_highlighting)
        view_menu.add_command(label="📏 Update Line Numbers", command=self.update_line_numbers)

        # === HELP ===
        self.ide_context_menu.add_separator()
        self.ide_context_menu.add_command(label="❓ Help", command=self.show_help_popup)

        # === QUICK ACTIONS (always visible) ===
        self.ide_context_menu.add_separator()
        self.ide_context_menu.add_command(label="🔧 Ask LLM to Fix", command=self.fix_current_code)
        self.ide_context_menu.add_command(label="✅ Accept", command=self.accept_changes)
        self.ide_context_menu.add_command(label="❌ Reject", command=self.reject_changes)
        self.ide_context_menu.add_command(label="💾 Save", command=self.save_file)

        # Bind right-click and Control+click for Mac compatibility
        self.editor.bind("<Button-2>", lambda e: self.show_ide_context_menu(e))
        self.editor.bind("<Button-3>", lambda e: self.show_ide_context_menu(e))
        self.editor.bind("<Control-Button-1>", lambda e: self.show_ide_context_menu(e))
        
    def show_ide_context_menu(self, event):
        """Show the IDE context menu at cursor position"""
        try:
            # Update menu based on context (selection, etc.)
            self.update_context_menu()
            
            # Move cursor to click position
            self.editor.mark_set("insert", f"@{event.x},{event.y}")
            
            # Show the menu
            self.ide_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.ide_context_menu.grab_release()
            
    def update_context_menu(self):
        """Update context menu based on current selection and context"""
        # Check if text is selected
        try:
            selected_text = self.editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            has_selection = bool(selected_text.strip())
        except tk.TclError:
            has_selection = False
            
        # Clear and rebuild the menu with context-sensitive items
        self.ide_context_menu.delete(0, tk.END)
        
        # If text is selected, show selection-specific options first
        if has_selection:
            self.ide_context_menu.add_command(label="📋 Copy Selected", command=self.copy_ide_selection)
            self.ide_context_menu.add_command(label="✂️ Cut Selected", command=self.cut_ide_selection)
            if ICECREAM_AVAILABLE:
                self.ide_context_menu.add_command(label="🐛 Debug Selected Variable", 
                                                 command=self.add_debug_prints)
            self.ide_context_menu.add_separator()
        
        # Standard menu structure
        # === EDIT OPERATIONS ===
        edit_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="✂️ Edit", menu=edit_menu)
        edit_menu.add_command(label="↩️ Undo", command=lambda: self.editor.edit_undo(), accelerator="Ctrl+Z")
        edit_menu.add_command(label="↪️ Redo", command=lambda: self.editor.edit_redo(), accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="✂️ Cut", command=self.cut_ide_selection, accelerator="Ctrl+X")
        edit_menu.add_command(label="📋 Copy", command=self.copy_ide_selection, accelerator="Ctrl+C")
        edit_menu.add_command(label="📄 Paste", command=self.paste_ide_clipboard, accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="🔍 Select All", command=self.select_all_ide, accelerator="Ctrl+A")
        
        # === FILE OPERATIONS ===
        file_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="📁 File", menu=file_menu)
        file_menu.add_command(label="📄 New File", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="📂 Open File", command=self.load_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="💾 Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="💾 Save As...", command=self.save_as_file, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="📤 Move to Chat", command=self.move_program_to_chat)

        # === CODE OPERATIONS ===
        code_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="⚡ Code", menu=code_menu)
        code_menu.add_command(label="▶️ Run Code", command=self.run_current_code, accelerator="F5")
        code_menu.add_command(label="🔧 Ask LLM to Fix", command=self.fix_current_code)

        # === REVIEW ===
        review_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="✅ Review", menu=review_menu)
        review_menu.add_command(label="Accept Changes", command=self.accept_changes, accelerator="Ctrl+Enter")
        review_menu.add_command(label="Reject Changes", command=self.reject_changes, accelerator="Escape")

        # === SEARCH & NAVIGATION ===
        search_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="🔍 Search", menu=search_menu)
        search_menu.add_command(label="🔍 Find Text", command=self.show_find_dialog, accelerator="Ctrl+F")
        search_menu.add_command(label="📍 Go to Line", command=self.go_to_line, accelerator="Ctrl+G")

        # === VIEW OPTIONS ===
        view_menu = Menu(self.ide_context_menu, tearoff=0)
        self.ide_context_menu.add_cascade(label="👁️ View", menu=view_menu)
        if PYGMENTS_AVAILABLE:
            view_menu.add_command(label="🎨 Refresh Highlighting", command=self.refresh_highlighting)
        view_menu.add_command(label="📏 Update Line Numbers", command=self.update_line_numbers)

        # === HELP ===
        self.ide_context_menu.add_separator()
        self.ide_context_menu.add_command(label="❓ Help", command=self.show_help_popup)

        # === QUICK ACTIONS (always visible) ===
        self.ide_context_menu.add_separator()
        self.ide_context_menu.add_command(label="🔧 Ask LLM to Fix", command=self.fix_current_code)
        self.ide_context_menu.add_command(label="✅ Accept", command=self.accept_changes)
        self.ide_context_menu.add_command(label="❌ Reject", command=self.reject_changes)
        self.ide_context_menu.add_command(label="💾 Save", command=self.save_file)
            
    def copy_ide_selection(self):
        """Copy selected text from IDE editor to clipboard"""
        try:
            selected_text = self.editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.status_label.config(text="Text copied to clipboard", fg="blue")
        except tk.TclError:
            # No selection
            self.status_label.config(text="No text selected", fg="orange")
            
    def paste_ide_clipboard(self):
        """Paste clipboard text to IDE editor"""
        try:
            clipboard_text = self.root.clipboard_get()

            # Preserve original indentation when pasting (don't apply HTML formatting)
            self.editor.insert(tk.INSERT, clipboard_text)
            # Update line numbers after pasting - FIX for missing line numbers
            self.update_line_numbers()
            self.status_label.config(text="Text pasted from clipboard", fg="blue")
        except tk.TclError:
            # Empty clipboard or other error
            self.status_label.config(text="Nothing to paste", fg="orange")


    def cut_ide_selection(self):
        """Cut selected text from IDE editor"""
        try:
            selected_text = self.editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.editor.delete(tk.SEL_FIRST, tk.SEL_LAST)
            # Update line numbers after cutting - FIX for missing line numbers
            self.update_line_numbers()
            self.status_label.config(text="Text cut to clipboard", fg="blue")
        except tk.TclError:
            # No selection
            self.status_label.config(text="No text selected", fg="orange")
            
    def select_all_ide(self):
        """Select all text in IDE editor"""
        self.editor.tag_add(tk.SEL, "1.0", tk.END)
        self.editor.mark_set(tk.INSERT, "1.0")
        self.editor.see(tk.INSERT)
        self.status_label.config(text="All text selected", fg="blue")

    def setup_syntax_highlighting(self):
        """Setup basic Python syntax highlighting"""
        # Keywords
        self.editor.tag_configure("keyword", foreground="blue", font=("Consolas", 10, "bold"))
        # Strings
        self.editor.tag_configure("string", foreground="green")
        # Comments
        self.editor.tag_configure("comment", foreground="gray", font=("Consolas", 10, "italic"))
        # Functions
        self.editor.tag_configure("function", foreground="purple", font=("Consolas", 10, "bold"))
        # Classes
        self.editor.tag_configure("class", foreground="darkgreen", font=("Consolas", 10, "bold"))
        # Numbers
        self.editor.tag_configure("number", foreground="darkred")
        
        # Diff highlighting tags
        self.editor.tag_configure("diff_add", background="lightgreen", foreground="darkgreen")
        self.editor.tag_configure("diff_remove", background="lightcoral", foreground="darkred")
        self.editor.tag_configure("diff_change", background="lightyellow", foreground="darkorange")
        self.editor.tag_configure("diff_accepted", background="#90EE90", foreground="black")
        self.editor.tag_configure("diff_rejected", background="#D3D3D3", foreground="gray", overstrike=True)
        self.editor.tag_configure("hunk_highlight", borderwidth=2, relief="solid")
        
    def apply_syntax_highlighting(self):
        """Apply basic Python syntax highlighting"""
        content = self.editor.get("1.0", tk.END)
        
        # Clear existing tags
        for tag in ["keyword", "string", "comment", "function"]:
            self.editor.tag_remove(tag, "1.0", tk.END)
        
        # Python keywords
        keywords = ["def", "class", "import", "from", "if", "else", "elif", "for", "while", 
                   "try", "except", "finally", "with", "as", "return", "yield", "break", 
                   "continue", "pass", "and", "or", "not", "in", "is", "True", "False", "None"]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line_start = f"{i+1}.0"
            
            # Highlight keywords
            for keyword in keywords:
                start = 0
                while True:
                    pos = line.find(keyword, start)
                    if pos == -1:
                        break
                    # Check if it's a whole word
                    if (pos == 0 or not line[pos-1].isalnum()) and \
                       (pos + len(keyword) == len(line) or not line[pos + len(keyword)].isalnum()):
                        start_idx = f"{i+1}.{pos}"
                        end_idx = f"{i+1}.{pos + len(keyword)}"
                        self.editor.tag_add("keyword", start_idx, end_idx)
                    start = pos + 1
            
            # Highlight strings
            in_string = False
            string_char = None
            for j, char in enumerate(line):
                if char in ['"', "'"] and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                        string_start = j
                    elif char == string_char:
                        in_string = False
                        start_idx = f"{i+1}.{string_start}"
                        end_idx = f"{i+1}.{j+1}"
                        self.editor.tag_add("string", start_idx, end_idx)
            
            # Highlight comments
            comment_pos = line.find('#')
            if comment_pos != -1:
                start_idx = f"{i+1}.{comment_pos}"
                end_idx = f"{i+1}.{len(line)}"
                self.editor.tag_add("comment", start_idx, end_idx)
            
            # Highlight function definitions
            if 'def ' in line:
                def_pos = line.find('def ')
                if def_pos != -1:
                    paren_pos = line.find('(', def_pos)
                    if paren_pos != -1:
                        start_idx = f"{i+1}.{def_pos + 4}"
                        end_idx = f"{i+1}.{paren_pos}"
                        self.editor.tag_add("function", start_idx, end_idx)
    
    def on_text_change(self, event=None):
        """Handle text changes"""
        self.update_line_numbers()
        # Apply syntax highlighting with a small delay to avoid lag
        self.editor.after_idle(self.apply_syntax_highlighting)
        
    def on_key_press(self, event=None):
        """Handle key press events"""
        # Auto-indent on Enter
        if event.keysym == 'Return':
            current_line = self.editor.get("insert linestart", "insert")
            indent = len(current_line) - len(current_line.lstrip())
            if current_line.strip().endswith(':'):
                indent += 4
            self.editor.after_idle(lambda: self.editor.insert("insert", " " * indent))
        
        # Convert Tab to 4 spaces for proper Python indentation
        elif event.keysym == 'Tab':
            self.editor.insert("insert", "    ")  # Insert 4 spaces instead of tab
            return "break"  # Prevent default tab behavior
    
    def sync_yview(self, *args):
        """Synchronize the yview of both line numbers and editor"""
        # Apply the same scrolling to both text widgets
        self.editor.yview(*args)
        self.line_text.yview(*args)
        
    def on_editor_scroll(self, *args):
        """Handle editor scroll and sync with line numbers"""
        # Update scrollbar
        self.scrollbar.set(*args)
        # Sync line numbers to match editor position
        self.line_text.yview_moveto(args[0])
        
    def on_line_scroll(self, *args):
        """Handle line numbers scroll and sync with editor"""
        # Update scrollbar  
        self.scrollbar.set(*args)
        # Sync editor to match line numbers position
        self.editor.yview_moveto(args[0])
        
    def update_line_numbers(self):
        """Update line numbers display"""
        # Save current scroll position
        current_view = self.editor.yview()
        
        content = self.editor.get("1.0", tk.END)
        lines = content.count('\n')
        
        line_numbers = '\n'.join(str(i) for i in range(1, lines + 1))
        
        self.line_text.config(state=tk.NORMAL)
        self.line_text.delete("1.0", tk.END)
        self.line_text.insert("1.0", line_numbers)
        self.line_text.config(state=tk.DISABLED)
        
        # Restore scroll position
        self.editor.yview_moveto(current_view[0])
        self.line_text.yview_moveto(current_view[0])
        # ============ ENHANCED IDE FEATURES ============
    def setup_enhanced_features(self):
        """Initialize enhanced IDE features if libraries are available"""
        # Setup Jedi autocomplete
        if JEDI_AVAILABLE:
            self.setup_jedi_autocomplete()
            
        # Setup Pygments syntax highlighting  
        if PYGMENTS_AVAILABLE:
            self.setup_pygments_highlighting()
            
        # Setup real-time linting
        if FLAKE8_AVAILABLE:
            self.setup_flake8_linting()
            
    def setup_jedi_autocomplete(self):
        """Setup Jedi-powered autocomplete"""
        self.autocomplete_window = None
        self.jedi_script = None
        
        # Bind Ctrl+Space for autocomplete
        self.editor.bind('<Control-space>', self.show_autocomplete)
        # Also show autocomplete on dot
        # self.editor.bind('.', self.on_dot_typed)  # Commented out to prevent auto-dropdown on period
        
    def show_autocomplete(self, event=None):
        """Show autocomplete suggestions using Jedi"""
        if not JEDI_AVAILABLE:
            return
            
        # Get current code and cursor position
        code = self.editor.get("1.0", tk.END)
        cursor_pos = self.editor.index(tk.INSERT)
        line, col = map(int, cursor_pos.split('.'))
        
        try:
            # Create Jedi script
            script = jedi.Script(code, path='<ide>')
            completions = script.complete(line, col)
            
            if completions:
                self.display_autocomplete_popup(completions)
        except Exception as e:
            print(f"Autocomplete error: {e}")
            
        return "break"
        
    def on_dot_typed(self, event):
        """Trigger autocomplete when dot is typed"""
        self.editor.insert(tk.INSERT, ".")
        self.editor.after(100, self.show_autocomplete)
        return "break"
        
    def display_autocomplete_popup(self, completions):
        """Display autocomplete suggestions in a popup window"""
        # Close existing popup if any
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
            
        # Create popup window
        self.autocomplete_window = Toplevel(self.root)
        self.autocomplete_window.wm_overrideredirect(True)
        
        # Position popup at cursor
        x, y, cx, cy = self.editor.bbox(tk.INSERT)
        x += self.editor.winfo_rootx()
        y += self.editor.winfo_rooty() + 20
        self.autocomplete_window.geometry(f"+{x}+{y}")
        
        # Create listbox for suggestions
        listbox = tk.Listbox(self.autocomplete_window, height=min(len(completions), 10))
        listbox.pack()
        
        # Add completions to listbox
        for completion in completions[:20]:  # Limit to 20 suggestions
            display_text = f"{completion.name}"
            if completion.type:
                display_text += f" [{completion.type}]"
            listbox.insert(tk.END, display_text)
            
        # Bind selection
        def on_select(event=None):
            if listbox.curselection():
                idx = listbox.curselection()[0]
                selected = completions[idx]
                self.insert_completion(selected.name)
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
            
        listbox.bind('<Double-Button-1>', on_select)
        listbox.bind('<Return>', on_select)
        
        # Close on Escape
        self.autocomplete_window.bind('<Escape>', lambda e: self.autocomplete_window.destroy())
        
        # Focus on listbox
        listbox.focus_set()
        if listbox.size() > 0:
            listbox.selection_set(0)
            
    def insert_completion(self, completion_text):
        """Insert the selected completion"""
        # Get current word start
        current_pos = self.editor.index(tk.INSERT)
        line_start = f"{current_pos.split('.')[0]}.0"
        line_text = self.editor.get(line_start, current_pos)
        
        # Find word start
        word_start = len(line_text)
        for i in range(len(line_text) - 1, -1, -1):
            if not (line_text[i].isalnum() or line_text[i] == '_'):
                word_start = i + 1
                break
        else:
            word_start = 0
            
        # Delete partial word and insert completion
        word_start_pos = f"{current_pos.split('.')[0]}.{word_start}"
        self.editor.delete(word_start_pos, current_pos)
        self.editor.insert(word_start_pos, completion_text)
        
    def setup_pygments_highlighting(self):
        """Setup Pygments-based syntax highlighting"""
        # This will replace the basic regex highlighting
        self.editor.bind('<KeyRelease>', self.apply_pygments_highlighting)
        
    def apply_pygments_highlighting(self, event=None):
        """Apply Pygments syntax highlighting"""
        if not PYGMENTS_AVAILABLE:
            self.apply_syntax_highlighting()  # Fall back to basic highlighting
            return
            
        # Remove all existing tags
        for tag in self.editor.tag_names():
            if tag not in ('sel', 'insert'):
                self.editor.tag_remove(tag, "1.0", tk.END)
                
        # Get code
        code = self.editor.get("1.0", tk.END)
        
        # Tokenize with Pygments
        try:
            lexer = PythonLexer()
            tokens = list(lex(code, lexer))
            
            # Apply highlighting based on token types
            pos = "1.0"
            for token_type, value in tokens:
                # Calculate end position
                lines = value.count('\n')
                if lines:
                    end_pos = f"{int(pos.split('.')[0]) + lines}.{len(value.split('\n')[-1])}"
                else:
                    end_pos = f"{pos.split('.')[0]}.{int(pos.split('.')[1]) + len(value)}"
                    
                # Apply appropriate tag
                if token_type in Token.Keyword:
                    self.editor.tag_add("keyword", pos, end_pos)
                elif token_type in Token.String:
                    self.editor.tag_add("string", pos, end_pos)
                elif token_type in Token.Comment:
                    self.editor.tag_add("comment", pos, end_pos)
                elif token_type in Token.Name.Function:
                    self.editor.tag_add("function", pos, end_pos)
                elif token_type in Token.Name.Class:
                    self.editor.tag_add("class", pos, end_pos)
                elif token_type in Token.Number:
                    self.editor.tag_add("number", pos, end_pos)
                    
                # Move position
                pos = end_pos
                
        except Exception as e:
            print(f"Pygments highlighting error: {e}")
            self.apply_syntax_highlighting()  # Fall back to basic
            
    def setup_flake8_linting(self):
        """Setup real-time flake8 linting"""
        # Initialize linting_enabled if not already set
        if not hasattr(self, 'linting_enabled'):
            self.linting_enabled = BooleanVar(value=True)
            
        # Add underline tag for errors
        self.editor.tag_configure("lint_error", underline=True, underlinefg="red")
        self.editor.tag_configure("lint_warning", underline=True, underlinefg="orange")
        
        # Schedule linting
        self.editor.bind('<KeyRelease>', self.schedule_linting)
        self.lint_timer = None
        
    def schedule_linting(self, event=None):
        """Schedule linting after a delay to avoid constant checking"""
        # Only schedule if linting is enabled
        if not hasattr(self, 'linting_enabled') or not self.linting_enabled.get():
            return
            
        if self.lint_timer:
            self.root.after_cancel(self.lint_timer)
        self.lint_timer = self.root.after(1000, self.run_flake8_check)  # 1 second delay
        
    def run_flake8_check(self):
        """Run flake8 check on current code"""
        if not FLAKE8_AVAILABLE:
            return

        # ONLY run linting for Python mode, not HTML mode
        if hasattr(self, 'parent') and self.parent.system_mode.get() == "html_programmer":
            # Clear any existing lint tags when in HTML mode
            self.editor.tag_remove("lint_error", "1.0", tk.END)
            self.editor.tag_remove("lint_warning", "1.0", tk.END)
            return

        # Clear previous lint tags
        self.editor.tag_remove("lint_error", "1.0", tk.END)
        self.editor.tag_remove("lint_warning", "1.0", tk.END)

        # Get code
        code = self.editor.get("1.0", tk.END)

        # Save to temp file for flake8
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
            
        try:
            # Run flake8
            style_guide = flake8_api.get_style_guide()
            report = style_guide.check_files([temp_file])
            
            # Process results
            for error in report._application.file_checker_manager.results:
                line_num = error[0]
                col_num = error[1]
                error_code = error[2]
                
                # Add underline to problematic line
                start_pos = f"{line_num}.{col_num}"
                end_pos = f"{line_num}.end"
                
                if error_code.startswith('E'):
                    self.editor.tag_add("lint_error", start_pos, end_pos)
                else:
                    self.editor.tag_add("lint_warning", start_pos, end_pos)
                    
        except Exception as e:
            print(f"Linting error: {e}")
        finally:
            # Clean up temp file
            os.unlink(temp_file)
            
    def format_code_with_black(self):
        """Format code using Black formatter"""
        if not BLACK_AVAILABLE:
            self.status_label.config(text="Black not installed", fg="red")
            return
            
        try:
            # Get current code
            code = self.editor.get("1.0", tk.END)
            
            # Format with black
            formatted_code = black.format_str(code, mode=black.Mode())
            
            # Replace editor content
            self.editor.delete("1.0", tk.END)
            self.editor.insert("1.0", formatted_code)
            
            # Update display
            self.update_line_numbers()
            self.apply_pygments_highlighting() if PYGMENTS_AVAILABLE else self.apply_syntax_highlighting()
            
            self.status_label.config(text="Code formatted with Black", fg="green")
            
        except Exception as e:
            self.status_label.config(text=f"Format error: {str(e)[:50]}", fg="red")
            
    def toggle_linting(self):
        """Toggle real-time linting on/off"""
        if not FLAKE8_AVAILABLE:
            return
            
        self.linting_enabled.set(not self.linting_enabled.get())
        
        if self.linting_enabled.get():
            if hasattr(self, 'lint_button'):
                self.lint_button.config(text="Lint: ON", bg="lightcyan")
            self.status_label.config(text="Linting enabled", fg="green")
            # Run linting immediately
            self.run_flake8_check()
        else:
            if hasattr(self, 'lint_button'):
                self.lint_button.config(text="Lint: OFF", bg="lightgray")
            self.status_label.config(text="Linting disabled", fg="orange")
            # Clear all lint tags
            self.editor.tag_remove("lint_error", "1.0", tk.END)
            self.editor.tag_remove("lint_warning", "1.0", tk.END)
            
    def refresh_highlighting(self):
        """Manually refresh syntax highlighting"""
        if PYGMENTS_AVAILABLE:
            self.apply_pygments_highlighting()
            self.status_label.config(text="Syntax highlighting refreshed", fg="blue")
        else:
            self.apply_syntax_highlighting()
            self.status_label.config(text="Basic highlighting refreshed", fg="blue")
            
    def show_help_popup(self):
        """Show a help popup with all keyboard shortcuts and features"""
        # Check if help window already exists
        if hasattr(self, 'help_window') and self.help_window and self.help_window.winfo_exists():
            # Just bring existing window to front
            self.help_window.lift()
            self.help_window.focus_set()
            return
            
        self.help_window = Toplevel(self.root)
        self.help_window.title("CodeRunner Help")
        self.help_window.geometry("600x550")
        self.help_window.resizable(True, True)  # Allow resizing for convenience
        
        # Position the help window to the right of IDE window
        # Get IDE window position and size
        self.root.update_idletasks()  # Ensure geometry is updated
        ide_x = self.root.winfo_x()
        ide_y = self.root.winfo_y()
        ide_width = self.root.winfo_width()
        
        # Position help window to the right of IDE
        help_x = ide_x + ide_width + 20  # 20 pixels gap
        help_y = ide_y
        self.help_window.geometry(f"600x550+{help_x}+{help_y}")
        
        # DON'T use transient or grab_set so window stays independent
        # help_window.transient(self.root)  # REMOVED - allows independent window
        # help_window.grab_set()  # REMOVED - allows switching between windows
        
        # Create scrolled text widget
        help_frame = Frame(self.help_window, padx=10, pady=10)
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        help_text = scrolledtext.ScrolledText(help_frame, wrap=tk.WORD, width=70, height=30)
        help_text.pack(fill=tk.BOTH, expand=True)
        
        # Help content
        help_content = """
CODERUNNER IDE — QUICK START
==============================

STEP 1: GET YOUR FIRST CODE
-----------------------------
1. Type what you want in the Chat box:
   "Write a Space Invaders game in Python using Pygame"
2. Click "Move to IDE" — code loads into the editor
3. Press F5 to Run

STEP 2: FIX OR CHANGE YOUR CODE
---------------------------------
Your code is now in the IDE. Two ways to change it:

  A) Run first, then fix:
     1. Press F5 (Run) — see what happens
     2. Type what's wrong in the Chat box
        ("enemies don't move", "add a score counter")
     3. Click "Ask LLM to Fix" in the IDE toolbar
     4. LLM returns fixed code → diff appears → Accept/Reject

  B) Move to IDE again:
     If you ask for new code in chat and click "Move to IDE",
     it shows a DIFF (not replace). Accept to merge, Reject to keep.

KEY POINT: After the first load, you NEVER lose your code.
All changes are shown as diffs that you Accept or Reject.

KEYBOARD SHORTCUTS
------------------
* F5            : Run code
* Ctrl+Enter    : Accept changes
* Escape        : Reject changes
* Ctrl+S        : Save file
* Ctrl+N        : New file
* Ctrl+O        : Open file
* Ctrl+F        : Find text

THE TWO BUTTONS THAT MATTER
----------------------------
* Run (F5)          : Runs your code. Check debug console for errors.
* Ask LLM to Fix    : Type what's wrong in chat, then click this.
                      LLM gets your code + errors + your description.

"RETURN FULL CODE" CHECKBOX
----------------------------
* OFF (default)     : LLM explains the bug and shows just the fix.
                      Fast, uses fewer tokens. You apply the fix yourself.
* ON                : LLM returns the complete fixed program.
                      Shows as a diff — Accept or Reject. Use for big changes.

ACCEPT & REJECT  (when "Return full code" is ON)
-------------------------------------------------
When the LLM proposes changes:
  Green lines  = new code added
  Yellow lines = code changed
  Red lines    = code removed

* Accept (Ctrl+Enter) : Apply the changes
* Reject (Escape)     : Throw away the changes, keep your original

TIPS
----
* Select "Python / Pygame" or "HTML / JavaScript" code mode
* The debug console shows errors — useful context for the LLM
* Save your work with Ctrl+S before making big changes
"""
        
        # Insert help text
        help_text.insert("1.0", help_content)
        help_text.config(state=tk.DISABLED)
        
        # Create button frame for multiple buttons
        button_frame = Frame(help_frame)
        button_frame.pack(pady=10)
        
        # Close button
        def close_help():
            self.help_window.destroy()
            self.help_window = None  # Clear reference
            
        close_btn = Button(button_frame, text="Close Help", command=close_help)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        # Button to switch back to IDE
        def focus_ide():
            self.root.lift()
            self.root.focus_set()
            
        ide_btn = Button(button_frame, text="Back to IDE", command=focus_ide, bg="lightblue")
        ide_btn.pack(side=tk.LEFT, padx=5)
        
        # Keep on top checkbox
        self.help_on_top = BooleanVar(value=False)
        def toggle_on_top():
            self.help_window.attributes('-topmost', self.help_on_top.get())
            
        Checkbutton(button_frame, text="Keep on Top", variable=self.help_on_top, 
                   command=toggle_on_top).pack(side=tk.LEFT, padx=10)
        
        # Bind Escape to close
        self.help_window.bind('<Escape>', lambda e: close_help())
        
        # Handle window close button (X)
        self.help_window.protocol("WM_DELETE_WINDOW", close_help)
        
        # Add instruction at top of help window
        info_label = Label(help_frame, text="💡 This window stays open! Switch between Help and IDE as needed.", 
                          fg="blue", font=("TkDefaultFont", 10, "bold"))
        info_label.pack(before=help_text, pady=(0, 5))
        
    # ============ ADVANCED TESTING & ANALYSIS FEATURES ============
    def analyze_code_with_ast(self):
        """Analyze code using AST for safer inspection of LLM-generated code"""
        if not AST_AVAILABLE:
            msg = "AST/Astor not installed!\n\nInstall with:\npip install astor\n\nThen restart the IDE."
            messagebox.showwarning("AST Not Available", msg)
            return

        # Don't run AST analysis on HTML content
        if hasattr(self, 'parent') and self.parent.system_mode.get() == "html_programmer":
            messagebox.showinfo("AST Analysis", "AST analysis is for Python code only.\nSwitch to Python mode to analyze code structure.")
            return

        try:
            code = self.editor.get("1.0", tk.END).strip()
            
            # Check if editor is empty
            if not code or len(code) < 10:
                sample_code = '''import os
import sys

# Global variable
VERSION = "1.0.0"

def greet(name="World"):
    """Say hello to someone"""
    return f"Hello, {name}!"

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"I'm {self.name}, {self.age} years old"

# Potentially dangerous operations
def risky_function():
    user_input = input("Enter code: ")
    eval(user_input)  # Dangerous!
    exec("print('executed')")  # Also dangerous!

if __name__ == "__main__":
    p = Person("Alice", 30)
    print(p.introduce())'''
                
                response = messagebox.askyesno("No Code to Analyze",
                    "The editor is empty.\n\nWould you like to load sample code to see how AST analysis works?")
                
                if response:
                    self.editor.delete("1.0", tk.END)
                    self.editor.insert("1.0", sample_code)
                    self.apply_syntax_highlighting()
                    code = sample_code
                else:
                    return
                    
            tree = ast.parse(code)
            
            # Create analysis window
            analysis_window = Toplevel(self.root)
            analysis_window.title("AST Code Analysis")
            analysis_window.geometry("700x500")
            
            # Create scrolled text for results
            results_text = scrolledtext.ScrolledText(analysis_window, wrap=tk.WORD)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Analyze code structure
            analysis = "🔍 CODE STRUCTURE ANALYSIS\n" + "="*50 + "\n\n"
            
            # Find imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            if imports:
                analysis += "📦 IMPORTS:\n"
                for imp in imports:
                    analysis += f"  • {imp}\n"
                analysis += "\n"
            
            # Find functions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    functions.append(f"{node.name}({', '.join(args)})")
            
            if functions:
                analysis += "🔧 FUNCTIONS:\n"
                for func in functions:
                    analysis += f"  • {func}\n"
                analysis += "\n"
            
            # Find classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    classes.append((node.name, methods))
            
            if classes:
                analysis += "📐 CLASSES:\n"
                for cls_name, methods in classes:
                    analysis += f"  • {cls_name}\n"
                    for method in methods:
                        analysis += f"    - {method}()\n"
                analysis += "\n"
            
            # Find global variables
            globals_vars = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            globals_vars.append(target.id)
            
            if globals_vars:
                analysis += "📊 GLOBAL VARIABLES:\n"
                for var in set(globals_vars):  # Use set to remove duplicates
                    analysis += f"  • {var}\n"
                analysis += "\n"
            
            # Check for dangerous operations
            dangerous = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', '__import__', 'compile']:
                            dangerous.append(f"Line {node.lineno}: {node.func.id}()")
            
            if dangerous:
                analysis += "⚠️ POTENTIALLY DANGEROUS OPERATIONS:\n"
                for danger in dangerous:
                    analysis += f"  • {danger}\n"
                analysis += "\n"
            
            # Add helpful summary
            if not imports and not functions and not classes and not globals_vars:
                analysis += "ℹ️ This code appears to be simple statements without defined structures.\n"
                analysis += "Try adding functions, classes, or imports to see more analysis.\n\n"
            
            # Add AST tree representation (truncated for readability)
            analysis += "🌳 AST TREE PREVIEW:\n" + "="*50 + "\n"
            try:
                tree_dump = ast.dump(tree, indent=2)
                # Truncate if too long
                if len(tree_dump) > 2000:
                    tree_dump = tree_dump[:2000] + "\n... (truncated for readability)"
                analysis += tree_dump
            except:
                analysis += "AST tree visualization not available"
            
            results_text.insert("1.0", analysis)
            results_text.config(state=tk.DISABLED)
            
            # Close button
            Button(analysis_window, text="Close", command=analysis_window.destroy).pack(pady=5)
            
            self.status_label.config(text="AST analysis complete", fg="green")
            
        except SyntaxError as e:
            self.status_label.config(text=f"Syntax error: {e}", fg="red")
            messagebox.showerror("Syntax Error", f"Cannot analyze - syntax error:\n{e}\n\nFix the syntax error and try again.")
        except Exception as e:
            self.status_label.config(text=f"Analysis error: {str(e)[:50]}", fg="red")
            messagebox.showerror("AST Analysis Error", f"Error during analysis:\n{str(e)}")
            
    def run_security_scan(self):
        """Run Bandit security scan on code"""
        if not BANDIT_AVAILABLE:
            msg = "Bandit not installed!\n\nInstall with:\npip install bandit\n\nThen restart the IDE."
            messagebox.showwarning("Bandit Not Available", msg)
            return
            
        try:
            code = self.editor.get("1.0", tk.END).strip()
            
            # Check if editor is empty
            if not code or len(code) < 10:
                sample_code = '''import os
import pickle
import subprocess

# SECURITY ISSUES EXAMPLE CODE
# This code has intentional security issues for demonstration

def unsafe_password():
    """Hardcoded password - security issue!"""
    password = "admin123"  # B105: Hardcoded password
    return password

def unsafe_sql(user_input):
    """SQL injection vulnerability"""
    query = "SELECT * FROM users WHERE id = '%s'" % user_input  # B608: SQL injection
    return query

def unsafe_exec(code_string):
    """Using eval/exec is dangerous"""
    result = eval(code_string)  # B307: Use of eval
    exec("print('hello')")  # B102: Use of exec
    return result

def unsafe_pickle(data):
    """Unsafe deserialization"""
    return pickle.loads(data)  # B301: Pickle usage

def unsafe_command(filename):
    """Command injection vulnerability"""
    os.system("cat " + filename)  # B605: Shell injection
    subprocess.call("ls " + filename, shell=True)  # B602: Shell=True

def weak_random():
    """Weak random number generation"""
    import random
    return random.random()  # B311: Weak random for security

# Try/except without proper handling
try:
    risky_operation()
except:  # B001: Bare except
    pass'''
                
                response = messagebox.askyesno("No Code to Scan",
                    "The editor is empty.\n\nWould you like to load sample code with security issues to see how scanning works?")
                
                if response:
                    self.editor.delete("1.0", tk.END)
                    self.editor.insert("1.0", sample_code)
                    self.apply_syntax_highlighting()
                    code = sample_code
                else:
                    return
            
            # Save to temp file for Bandit
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run bandit
            from bandit.core import manager as bandit_manager
            from bandit.core import config as bandit_config
            b_mgr = bandit_manager.BanditManager(bandit_config.BanditConfig(), 'file')
            b_mgr.discover_files([temp_file])
            b_mgr.run_tests()
            
            # Get results
            issues = b_mgr.get_issue_list()
            
            # Display results in a window
            results_window = Toplevel(self.root)
            results_window.title("Security Scan Results")
            results_window.geometry("700x500")
            
            results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            if issues:
                result_text = "🔒 SECURITY SCAN RESULTS\n" + "="*50 + "\n\n"
                result_text += f"Found {len(issues)} potential security issues:\n\n"
                
                # Group by severity
                high_issues = [i for i in issues if i.severity == "HIGH"]
                medium_issues = [i for i in issues if i.severity == "MEDIUM"]
                low_issues = [i for i in issues if i.severity == "LOW"]
                
                if high_issues:
                    result_text += "🔴 HIGH SEVERITY:\n" + "-"*30 + "\n"
                    for issue in high_issues:
                        result_text += f"Line {issue.lineno}: {issue.test}\n"
                        result_text += f"  {issue.text}\n\n"
                
                if medium_issues:
                    result_text += "🟡 MEDIUM SEVERITY:\n" + "-"*30 + "\n"
                    for issue in medium_issues:
                        result_text += f"Line {issue.lineno}: {issue.test}\n"
                        result_text += f"  {issue.text}\n\n"
                
                if low_issues:
                    result_text += "🟢 LOW SEVERITY:\n" + "-"*30 + "\n"
                    for issue in low_issues:
                        result_text += f"Line {issue.lineno}: {issue.test}\n"
                        result_text += f"  {issue.text}\n\n"
                
                result_text += "="*50 + "\n"
                result_text += "💡 RECOMMENDATIONS:\n"
                result_text += "• Review and fix HIGH severity issues immediately\n"
                result_text += "• Consider fixing MEDIUM issues before production\n"
                result_text += "• LOW issues may be acceptable depending on context\n"
                
                results_text.insert("1.0", result_text)
                self.status_label.config(text=f"Security scan found {len(issues)} issues", fg="orange")
            else:
                result_text = "✅ SECURITY SCAN COMPLETE\n" + "="*50 + "\n\n"
                result_text += "No security issues found!\n\n"
                result_text += "Your code appears to be free of common security vulnerabilities.\n\n"
                result_text += "Note: This scan checks for common patterns but is not exhaustive.\n"
                result_text += "Always follow security best practices:\n"
                result_text += "• Never hardcode passwords or secrets\n"
                result_text += "• Validate and sanitize all user input\n"
                result_text += "• Use parameterized queries for databases\n"
                result_text += "• Avoid eval() and exec() with user input\n"
                result_text += "• Use secure random number generation for security\n"
                
                results_text.insert("1.0", result_text)
                self.status_label.config(text="No security issues found", fg="green")
            
            results_text.config(state=tk.DISABLED)
            Button(results_window, text="Close", command=results_window.destroy).pack(pady=5)
                
            # Clean up
            os.unlink(temp_file)
            
        except Exception as e:
            messagebox.showerror("Security Scan Error", f"Error during security scan:\n{str(e)}")
            self.status_label.config(text=f"Security scan error: {str(e)[:30]}", fg="red")
            
    def run_type_checking(self):
        """Run MyPy type checking on code"""
        if not MYPY_AVAILABLE:
            msg = "MyPy not installed!\n\nInstall with:\npip install mypy\n\nThen restart the IDE."
            messagebox.showwarning("MyPy Not Available", msg)
            return

        # Don't run type checking on HTML content
        if hasattr(self, 'parent') and self.parent.system_mode.get() == "html_programmer":
            messagebox.showinfo("Type Checking", "Type checking is for Python code only.\nSwitch to Python mode to check types.")
            return

        try:
            code = self.editor.get("1.0", tk.END).strip()
            
            # Check if editor is empty
            if not code or len(code) < 10:
                sample_code = '''from typing import List, Optional, Dict, Union

def add_numbers(a: int, b: int) -> int:
    """Correctly typed function"""
    return a + b

def bad_return_type(x: int) -> str:
    """Wrong return type - will be caught by mypy"""
    return x  # Error: returning int, expected str

def missing_return(x: int) -> int:
    """Missing return statement"""
    print(x)  # Error: missing return

def process_items(items: List[str]) -> Dict[str, int]:
    """Process a list of items"""
    result: Dict[str, int] = {}
    for item in items:
        result[item] = len(item)
    return result

class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
    
    def greet(self, other: 'Person') -> str:
        return f"Hello {other.name}, I'm {self.name}"

def optional_param(value: Optional[int] = None) -> int:
    if value is None:
        return 0
    return value * 2

# Type error examples
result1 = add_numbers("5", "10")  # Error: str instead of int
result2 = bad_return_type(42)  # Will show type mismatch
person = Person("Alice", "thirty")  # Error: str instead of int for age'''
                
                response = messagebox.askyesno("No Code to Type Check",
                    "The editor is empty.\n\nWould you like to load sample code with type hints to see how type checking works?")
                
                if response:
                    self.editor.delete("1.0", tk.END)
                    self.editor.insert("1.0", sample_code)
                    self.apply_syntax_highlighting()
                    code = sample_code
                else:
                    return
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run mypy
            stdout, stderr, exit_status = mypy.api.run([temp_file])
            
            # Display results in a window
            results_window = Toplevel(self.root)
            results_window.title("Type Checking Results")
            results_window.geometry("700x500")
            
            results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            result_text = "🔍 TYPE CHECKING RESULTS\n" + "="*50 + "\n\n"
            
            if stdout and stdout.strip():
                # Parse mypy output for better formatting
                lines = stdout.strip().split('\n')
                errors = []
                warnings = []
                notes = []
                
                for line in lines:
                    if 'error:' in line:
                        errors.append(line)
                    elif 'warning:' in line:
                        warnings.append(line)
                    elif 'note:' in line:
                        notes.append(line)
                    elif 'Success:' in line:
                        result_text += "✅ " + line + "\n"
                
                if errors:
                    result_text += "❌ ERRORS:\n" + "-"*30 + "\n"
                    for error in errors:
                        result_text += error + "\n"
                    result_text += "\n"
                
                if warnings:
                    result_text += "⚠️ WARNINGS:\n" + "-"*30 + "\n"
                    for warning in warnings:
                        result_text += warning + "\n"
                    result_text += "\n"
                
                if notes:
                    result_text += "📝 NOTES:\n" + "-"*30 + "\n"
                    for note in notes:
                        result_text += note + "\n"
                    result_text += "\n"
                
                if exit_status == 0:
                    result_text += "✅ Type checking passed!\n"
                    self.status_label.config(text="Type checking passed", fg="green")
                else:
                    result_text += f"Found {len(errors)} type errors\n"
                    self.status_label.config(text="Type checking found issues", fg="orange")
                    
                result_text += "\n" + "="*50 + "\n"
                result_text += "💡 TYPE CHECKING TIPS:\n"
                result_text += "• Add type hints to function parameters and returns\n"
                result_text += "• Use Optional[T] for nullable types\n"
                result_text += "• Import types from 'typing' module\n"
                result_text += "• Run regularly to catch type errors early\n"
            else:
                result_text += "✅ No type issues found!\n\n"
                result_text += "Your code passes all type checks.\n\n"
                result_text += "Consider adding more type hints to improve:\n"
                result_text += "• Code documentation\n"
                result_text += "• IDE autocomplete\n"
                result_text += "• Early error detection\n"
                self.status_label.config(text="No type issues found", fg="green")
            
            results_text.insert("1.0", result_text)
            results_text.config(state=tk.DISABLED)
            Button(results_window, text="Close", command=results_window.destroy).pack(pady=5)
                
            # Clean up
            os.unlink(temp_file)
            
        except Exception as e:
            messagebox.showerror("Type Checking Error", f"Error during type checking:\n{str(e)}")
            self.status_label.config(text=f"Type check error: {str(e)[:30]}", fg="red")
            
    def analyze_complexity(self):
        """Analyze code complexity using Radon"""
        if not RADON_AVAILABLE:
            msg = "Radon not installed!\n\nInstall with:\npip install radon\n\nThen restart the IDE."
            messagebox.showwarning("Radon Not Available", msg)
            return

        # Don't run complexity analysis on HTML content
        if hasattr(self, 'parent') and self.parent.system_mode.get() == "html_programmer":
            messagebox.showinfo("Complexity Analysis", "Complexity analysis is for Python code only.\nSwitch to Python mode to analyze code complexity.")
            return

        try:
            code = self.editor.get("1.0", tk.END).strip()
            
            # Check if editor is empty or has no meaningful code
            if not code or len(code) < 10:
                # Provide sample code
                sample_code = '''def calculate_grade(score):
    """Calculate letter grade from score"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def complex_function(data, flag1, flag2):
    """Example of more complex function"""
    result = []
    for item in data:
        if flag1:
            if flag2 and item > 10:
                for i in range(item):
                    if i % 2 == 0:
                        result.append(i * 2)
            else:
                result.append(item)
        else:
            if item < 5:
                result.append(item * -1)
    return result

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b != 0:
            return a / b
        return None'''
                
                # Ask user if they want to use sample code
                response = messagebox.askyesno("No Code to Analyze", 
                    "The editor is empty.\n\nWould you like to load sample code to see how complexity analysis works?")
                
                if response:
                    self.editor.delete("1.0", tk.END)
                    self.editor.insert("1.0", sample_code)
                    self.apply_syntax_highlighting()
                    code = sample_code
                else:
                    return
            
            # Analyze complexity
            blocks = radon_cc.cc_visit(code)
            
            # Create results window
            results_window = Toplevel(self.root)
            results_window.title("Code Complexity Analysis")
            results_window.geometry("700x500")
            
            results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            analysis = "📊 COMPLEXITY ANALYSIS\n" + "="*50 + "\n\n"
            
            if blocks:
                analysis += "📋 FUNCTIONS/METHODS FOUND:\n" + "-"*30 + "\n\n"
                
                # Complexity ratings
                for block in blocks:
                    if block.complexity <= 5:
                        rating = "✅ Simple"
                        advice = "Well structured, easy to understand"
                    elif block.complexity <= 10:
                        rating = "🟡 Moderate"
                        advice = "Consider simplifying if possible"
                    else:
                        rating = "🔴 Complex"
                        advice = "Should be refactored into smaller functions"
                        
                    analysis += f"📍 {block.name} (line {block.lineno}):\n"
                    analysis += f"  Complexity Score: {block.complexity} - {rating}\n"
                    analysis += f"  Type: {block.classname or 'Function'}\n"
                    analysis += f"  Advice: {advice}\n\n"
            else:
                analysis += "ℹ️ No functions or methods found in the code.\n"
                analysis += "Add some functions to see complexity analysis.\n\n"
            
            # Calculate maintainability index
            try:
                mi = radon_metrics.mi_visit(code, True)
                analysis += "\n📈 MAINTAINABILITY INDEX: {:.2f}\n".format(mi)
                analysis += "-"*30 + "\n"
                if mi >= 20:
                    analysis += "  ✅ Good maintainability (20+)\n"
                    analysis += "  Code is easy to maintain and modify\n"
                elif mi >= 10:
                    analysis += "  🟡 Moderate maintainability (10-20)\n"
                    analysis += "  Some refactoring could improve maintainability\n"
                else:
                    analysis += "  🔴 Poor maintainability (<10)\n"
                    analysis += "  Significant refactoring recommended\n"
            except Exception as e:
                analysis += "\n(Maintainability index not available for this code)\n"
            
            # Halstead metrics
            try:
                h = radon_metrics.h_visit(code)
                if h[0]:  # Check if metrics exist
                    h = h[0]  # Get first result
                    analysis += "\n📏 HALSTEAD METRICS:\n"
                    analysis += "-"*30 + "\n"
                    analysis += f"  Difficulty: {h.difficulty:.2f}\n"
                    analysis += f"  Effort: {h.effort:.2f}\n"
                    analysis += f"  Time to implement: {h.time:.2f} seconds\n"
                    analysis += f"  Bugs estimate: {h.bugs:.3f}\n"
            except:
                pass
            
            # Add explanation
            analysis += "\n" + "="*50 + "\n"
            analysis += "📚 UNDERSTANDING COMPLEXITY:\n"
            analysis += "-"*30 + "\n"
            analysis += "• Complexity 1-5: Simple, linear code\n"
            analysis += "• Complexity 6-10: Moderate, some branches\n"  
            analysis += "• Complexity 11+: Complex, many paths\n\n"
            analysis += "Lower complexity = easier to test and maintain!\n"
            
            results_text.insert("1.0", analysis)
            results_text.config(state=tk.DISABLED)
            
            Button(results_window, text="Close", command=results_window.destroy).pack(pady=5)
            
            self.status_label.config(text="Complexity analysis complete", fg="green")
            
        except Exception as e:
            messagebox.showerror("Complexity Analysis Error", f"Error analyzing code:\n{str(e)}")
            
    def show_debug_help(self):
        """Show help dialog for debug feature with example"""
        # Create help window
        help_win = Toplevel(self.root)
        help_win.title("How to Use Debug Feature")
        help_win.geometry("500x400")
        help_win.resizable(False, False)
        
        # Add instructions
        frame = Frame(help_win, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        Label(frame, text="🔍 DEBUG FEATURE GUIDE", font=("TkDefaultFont", 12, "bold")).pack(pady=(0, 10))
        
        # Instructions
        instructions = Text(frame, wrap=tk.WORD, height=20, width=60)
        instructions.pack(fill=tk.BOTH, expand=True)
        
        guide_text = """HOW TO USE THE DEBUG BUTTON:
=====================================

The Debug button adds smart print statements to see variable values.

STEP-BY-STEP:
------------
1. FIRST: Select the variable or expression you want to debug
   Example: Select just the text "my_variable" or "result" 
   
2. THEN: Click the Debug button

3. RESULT: Adds ic(your_variable) which prints:
   ic| your_variable: actual_value

EXAMPLE:
--------
Code before:
    x = 10
    y = x * 2
    result = x + y

To debug 'result':
1. Select the word "result" (just highlight it)
2. Click Debug button
3. Code becomes:

    x = 10
    y = x * 2
    result = x + y
    ic(result)

When run, it prints:
    🔍 DEBUG | result: 30

BENEFITS OVER print():
---------------------
• Shows variable NAME and VALUE
• Colorized output
• Works with any expression
• Automatic formatting

TIP: You can select expressions too!
Select "x + y" and Debug will add ic(x + y)
"""
        
        instructions.insert("1.0", guide_text)
        instructions.config(state=tk.DISABLED)
        
        # Buttons
        btn_frame = Frame(frame)
        btn_frame.pack(pady=10)
        
        Button(btn_frame, text="Got it!", command=help_win.destroy, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        Button(btn_frame, text="Try Debug Now", command=lambda: [help_win.destroy(), self.add_debug_prints()], 
               bg="lightblue").pack(side=tk.LEFT, padx=5)
        
        # Center the window
        help_win.transient(self.root)
        help_win.grab_set()
        help_win.focus_set()
        
    def add_debug_prints(self):
        """Add IceCream debug prints to selected code"""
        if not ICECREAM_AVAILABLE:
            msg = "IceCream not installed!\n\nInstall with:\npip install icecream\n\nThen restart the IDE."
            messagebox.showwarning("IceCream Not Available", msg)
            return
            
        try:
            # Get selected text
            try:
                selected = self.editor.get(tk.SEL_FIRST, tk.SEL_LAST)
                start_idx = self.editor.index(tk.SEL_FIRST)
                end_idx = self.editor.index(tk.SEL_LAST)
            except tk.TclError:
                # Show helpful message
                msg = ("No text selected!\n\n"
                      "HOW TO USE:\n"
                      "1. Select a variable name (e.g., 'x')\n"
                      "2. Click Debug button\n"
                      "3. This adds ic(x) to show its value\n\n"
                      "Example: Select 'result' then click Debug")
                messagebox.showinfo("How to Use Debug", msg)
                return
                
            # Check if something is selected
            if not selected.strip():
                messagebox.showinfo("Empty Selection", "Please select a variable or expression to debug")
                return
                
            # Add ic() debug statement
            debug_code = f"ic({selected})"
            
            # Get the line where selection ends
            end_line = int(end_idx.split('.')[0])
            
            # Insert on the next line with proper indentation
            # Get current line to match indentation
            line_start = f"{end_line}.0"
            line_end = f"{end_line}.end"
            current_line = self.editor.get(line_start, line_end)
            
            # Calculate indentation
            indent = len(current_line) - len(current_line.lstrip())
            indent_str = " " * indent
            
            # Insert debug statement on next line
            insert_pos = f"{end_line}.end"
            self.editor.insert(insert_pos, f"\n{indent_str}{debug_code}")
            
            # Also add import at top if not present
            content = self.editor.get("1.0", tk.END)
            if "from icecream import ic" not in content and "import icecream" not in content:
                # Add import at the beginning
                self.editor.insert("1.0", "from icecream import ic\n")
                self.status_label.config(text=f"Added: ic({selected}) and import", fg="green")
            else:
                self.status_label.config(text=f"Added: ic({selected})", fg="green")
            
            # Update syntax highlighting
            self.apply_syntax_highlighting()
            
        except Exception as e:
            messagebox.showerror("Debug Error", f"Error adding debug statement:\n{str(e)}")
    # ============ END ADVANCED TESTING & ANALYSIS FEATURES ============
    # ============ END ENHANCED IDE FEATURES ============
        
    def show_window(self):
        """Show the IDE window"""
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            # Window was destroyed, recreate it
            self.create_ide_window()
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        
    def hide_window(self):
        """Hide the IDE window"""
        self.root.withdraw()
        
    def set_content(self, content, filename=None):
        """Set the content of the editor

        This method is called when:
        - A file is loaded from disk
        - Code is moved from chat to IDE
        - New content is set programmatically

        It ensures the parent GUI knows about the loaded code
        so the LLM can be properly instructed to only generate
        targeted fixes/features, not regenerate the entire program.
        """
        self.current_content = content
        self.current_filename = filename


        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", content)
        self.update_line_numbers()
        self.apply_syntax_highlighting()
        self.status_label.config(text="Content loaded - Ready for targeted edits", fg="blue")
        # Enable Save button only if a filename exists (supports explicit Save semantics)
        if hasattr(self, 'save_button'):
            self.save_button.config(state=(tk.NORMAL if self.current_filename else tk.DISABLED))
        
        # Notify parent that code is loaded in IDE
        # This triggers code analysis and LLM instructions
        if hasattr(self.parent, 'notify_ide_content_loaded'):
            self.parent.notify_ide_content_loaded(content, filename)
        
    def get_content(self):
        """Get the current content of the editor"""
        return self.editor.get("1.0", tk.END + "-1c")
    
    # ---------- Diff view: shows color-coded proposed changes ----------

    def show_diff(self, proposed_content):
        """Show proposed changes with per-hunk Accept/Reject navigation.

        Hunks are contiguous groups of changes.  The user can step through
        them with Prev/Next buttons and Accept or Reject each independently.
        'Accept All' / 'Reject All' (the old buttons) still work for bulk.
        """
        self.proposed_content = proposed_content
        self.is_showing_diff = True

        current_lines = self.current_content.split('\n')
        proposed_lines = proposed_content.split('\n')

        # Notify parent
        if hasattr(self.parent, 'add_to_debug_console'):
            self.parent.add_to_debug_console("\n" + "="*60)
            self.parent.add_to_debug_console("PROPOSED CHANGES — use Prev/Next to review each hunk")
            self.parent.add_to_debug_console("="*60 + "\n")

        # --- Compute opcodes and group into hunks ---
        matcher = difflib.SequenceMatcher(None, current_lines, proposed_lines)
        opcodes = matcher.get_opcodes()

        # A hunk = contiguous non-equal opcodes.  We also store the
        # surrounding equal blocks so we can rebuild the file later.
        self.diff_hunks = []      # list of hunk dicts
        self.diff_opcodes = opcodes
        self.diff_current_lines = current_lines
        self.diff_proposed_lines = proposed_lines

        hunk_opcodes = []
        for op in opcodes:
            tag = op[0]
            if tag == 'equal':
                if hunk_opcodes:
                    self.diff_hunks.append({
                        'opcodes': list(hunk_opcodes),
                        'status': 'pending',  # 'pending', 'accepted', 'rejected'
                    })
                    hunk_opcodes = []
            else:
                hunk_opcodes.append(op)
        if hunk_opcodes:
            self.diff_hunks.append({
                'opcodes': list(hunk_opcodes),
                'status': 'pending',
            })

        self.current_hunk_idx = 0

        # --- Build display content ---
        self._rebuild_diff_display()

        # Enable accept/reject buttons
        if hasattr(self, 'accept_btn'):
            self.accept_btn.config(state=tk.NORMAL, text="Accept All", bg="#00cc00")
            self.root.after(2000, lambda: self.accept_btn.config(bg="lightgreen"))
        if hasattr(self, 'reject_btn'):
            self.reject_btn.config(state=tk.NORMAL, text="Reject All")

        # Show hunk navigation if there are hunks
        if self.diff_hunks:
            self.hunk_frame.pack(side=tk.LEFT, padx=(5, 0))
            self._update_hunk_label()

        self.status_label.config(
            text=f"{len(self.diff_hunks)} change(s) — step through with Prev/Next, Accept/Reject each",
            fg="orange", font=("TkDefaultFont", 10))

        # Make editor read-only during diff review
        self.editor.config(state=tk.DISABLED)

        # Bring IDE window to front
        self.root.lift()
        self.root.focus_set()

    def _rebuild_diff_display(self):
        """Rebuild the editor content from opcodes, colouring each hunk
        according to its status (pending / accepted / rejected)."""
        current_lines = self.diff_current_lines
        proposed_lines = self.diff_proposed_lines

        # Clear existing tags
        self.editor.config(state=tk.NORMAL)
        for tag in ["diff_add", "diff_remove", "diff_change",
                     "diff_accepted", "diff_rejected", "hunk_highlight"]:
            self.editor.tag_remove(tag, "1.0", tk.END)

        display_lines = []   # (text, tag_or_None, hunk_index_or_None)
        hunk_idx = 0

        for op in self.diff_opcodes:
            tag_name = op[0]
            i1, i2, j1, j2 = op[1], op[2], op[3], op[4]

            if tag_name == 'equal':
                for line in current_lines[i1:i2]:
                    display_lines.append((line, None, None))
            else:
                # Find which hunk this opcode belongs to
                cur_hunk = None
                for hi, h in enumerate(self.diff_hunks):
                    if op in h['opcodes']:
                        cur_hunk = h
                        hunk_idx = hi
                        break

                status = cur_hunk['status'] if cur_hunk else 'pending'

                if tag_name == 'replace':
                    if status == 'rejected':
                        # Show original lines as kept (greyed out strike-through)
                        for line in current_lines[i1:i2]:
                            display_lines.append((line, "diff_rejected", hunk_idx))
                    elif status == 'accepted':
                        for line in proposed_lines[j1:j2]:
                            display_lines.append((line, "diff_accepted", hunk_idx))
                    else:
                        for line in current_lines[i1:i2]:
                            display_lines.append(("- " + line, "diff_remove", hunk_idx))
                        for line in proposed_lines[j1:j2]:
                            display_lines.append((line, "diff_change", hunk_idx))
                elif tag_name == 'delete':
                    if status == 'rejected':
                        for line in current_lines[i1:i2]:
                            display_lines.append((line, "diff_rejected", hunk_idx))
                    elif status == 'accepted':
                        pass  # Lines deleted — don't show
                    else:
                        for line in current_lines[i1:i2]:
                            display_lines.append(("- " + line, "diff_remove", hunk_idx))
                elif tag_name == 'insert':
                    if status == 'rejected':
                        pass  # Rejected insert — don't show
                    elif status == 'accepted':
                        for line in proposed_lines[j1:j2]:
                            display_lines.append((line, "diff_accepted", hunk_idx))
                    else:
                        for line in proposed_lines[j1:j2]:
                            display_lines.append((line, "diff_add", hunk_idx))

        # Write to editor
        self.editor.delete("1.0", tk.END)

        # Track which display lines belong to each hunk (for highlight)
        self._hunk_display_ranges = {}  # hunk_idx -> (first_line, last_line)

        for idx, (text, tag, hi) in enumerate(display_lines):
            if idx > 0:
                self.editor.insert(tk.END, "\n")
            self.editor.insert(tk.END, text)
            line_num = idx + 1
            if tag:
                self.editor.tag_add(tag, f"{line_num}.0", f"{line_num}.end")
            if hi is not None:
                if hi not in self._hunk_display_ranges:
                    self._hunk_display_ranges[hi] = [line_num, line_num]
                else:
                    self._hunk_display_ranges[hi][1] = line_num

        self.update_line_numbers()

        # Highlight current hunk
        self._highlight_current_hunk()
        self.editor.config(state=tk.DISABLED)

    def _highlight_current_hunk(self):
        """Scroll to and highlight the current hunk."""
        self.editor.tag_remove("hunk_highlight", "1.0", tk.END)
        if not self.diff_hunks or self.current_hunk_idx >= len(self.diff_hunks):
            return
        rng = self._hunk_display_ranges.get(self.current_hunk_idx)
        if rng:
            first, last = rng
            self.editor.tag_add("hunk_highlight", f"{first}.0", f"{last}.end")
            self.editor.see(f"{first}.0")

    def _update_hunk_label(self):
        """Update the 'Hunk 2/5' counter label."""
        total = len(self.diff_hunks)
        if total == 0:
            self.hunk_label.config(text="No changes")
            return
        cur = self.current_hunk_idx + 1
        status = self.diff_hunks[self.current_hunk_idx]['status']
        status_str = {'pending': '', 'accepted': ' OK', 'rejected': ' X'}[status]
        self.hunk_label.config(text=f"Hunk {cur}/{total}{status_str}")

    def navigate_hunk(self, direction):
        """Move to previous (-1) or next (+1) hunk."""
        if not self.diff_hunks:
            return
        self.current_hunk_idx = max(0, min(len(self.diff_hunks) - 1,
                                            self.current_hunk_idx + direction))
        self._highlight_current_hunk()
        self._update_hunk_label()

    def accept_current_hunk(self):
        """Accept the current hunk and advance to the next pending one."""
        if not self.diff_hunks:
            return
        self.diff_hunks[self.current_hunk_idx]['status'] = 'accepted'
        self._rebuild_diff_display()
        self._advance_to_next_pending()
        self._update_hunk_label()

    def reject_current_hunk(self):
        """Reject the current hunk and advance to the next pending one."""
        if not self.diff_hunks:
            return
        self.diff_hunks[self.current_hunk_idx]['status'] = 'rejected'
        self._rebuild_diff_display()
        self._advance_to_next_pending()
        self._update_hunk_label()

    def _advance_to_next_pending(self):
        """Move to the next pending hunk, or stay if none left."""
        for i in range(len(self.diff_hunks)):
            idx = (self.current_hunk_idx + i) % len(self.diff_hunks)
            if self.diff_hunks[idx]['status'] == 'pending':
                self.current_hunk_idx = idx
                self._highlight_current_hunk()
                return
        # All hunks decided — prompt to apply
        self.status_label.config(text="All hunks reviewed — click 'Apply All' to finish", fg="green")
        self._highlight_current_hunk()

    def apply_hunk_decisions(self):
        """Build final content from per-hunk Accept/Reject decisions."""
        current_lines = self.diff_current_lines
        proposed_lines = self.diff_proposed_lines
        result_lines = []

        for op in self.diff_opcodes:
            tag = op[0]
            i1, i2, j1, j2 = op[1], op[2], op[3], op[4]

            if tag == 'equal':
                result_lines.extend(current_lines[i1:i2])
            else:
                # Find which hunk this opcode belongs to
                hunk = None
                for h in self.diff_hunks:
                    if op in h['opcodes']:
                        hunk = h
                        break

                accepted = hunk and hunk['status'] == 'accepted'

                if tag == 'replace':
                    if accepted:
                        result_lines.extend(proposed_lines[j1:j2])
                    else:
                        result_lines.extend(current_lines[i1:i2])
                elif tag == 'delete':
                    if not accepted:
                        result_lines.extend(current_lines[i1:i2])
                    # If accepted, lines are deleted (not included)
                elif tag == 'insert':
                    if accepted:
                        result_lines.extend(proposed_lines[j1:j2])
                    # If rejected, new lines are not included

        # Apply as new content
        final_content = '\n'.join(result_lines)
        self.current_content = final_content
        self.proposed_content = ""
        self.is_showing_diff = False
        self.diff_hunks = []

        self.editor.config(state=tk.NORMAL)
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", self.current_content)

        for tag in ["diff_add", "diff_remove", "diff_change",
                     "diff_accepted", "diff_rejected", "hunk_highlight"]:
            self.editor.tag_remove(tag, "1.0", tk.END)

        self.apply_syntax_highlighting()
        self.update_line_numbers()

        if hasattr(self, 'accept_btn'):
            self.accept_btn.config(state=tk.DISABLED, text="Accept Fix")
        if hasattr(self, 'reject_btn'):
            self.reject_btn.config(state=tk.DISABLED, text="Reject Fix")
        self.hunk_frame.pack_forget()

        self.status_label.config(text="Changes applied", fg="green")

        if hasattr(self.parent, 'notify_ide_content_loaded'):
            self.parent.notify_ide_content_loaded(self.current_content)

    def accept_changes(self):
        """Accept ALL hunks and apply (bulk accept)."""
        if not self.is_showing_diff:
            return
        # Mark all hunks as accepted
        for h in getattr(self, 'diff_hunks', []):
            h['status'] = 'accepted'
        self.apply_hunk_decisions()

    def reject_changes(self):
        """Reject ALL changes and restore original content."""
        if not self.is_showing_diff:
            return

        self.proposed_content = ""
        self.is_showing_diff = False
        self.diff_hunks = []

        self.editor.config(state=tk.NORMAL)
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", self.current_content)

        for tag in ["diff_add", "diff_remove", "diff_change",
                     "diff_accepted", "diff_rejected", "hunk_highlight"]:
            self.editor.tag_remove(tag, "1.0", tk.END)

        self.apply_syntax_highlighting()
        self.update_line_numbers()

        if hasattr(self, 'accept_btn'):
            self.accept_btn.config(state=tk.DISABLED, text="Accept Fix")
        if hasattr(self, 'reject_btn'):
            self.reject_btn.config(state=tk.DISABLED, text="Reject Fix")
        if hasattr(self, 'hunk_frame'):
            self.hunk_frame.pack_forget()

        self.status_label.config(text="Changes rejected - original code restored", fg="red")

    def new_file(self):
        """Create a new file"""
        if messagebox.askyesno("New File", "Clear current content?"):
            self.set_content("")
            self.status_label.config(text="New file created", fg="blue")
            self.parent.display_status_message("📝 New file created - content cleared")
            # New file has no name yet; keep Save disabled until Save As is used
            if hasattr(self, 'save_button'):
                self.save_button.config(state=tk.DISABLED)
    
    def _get_mode_default_ext_and_types(self):
        """Return defaultextension and filetypes based on current mode (HTML vs Python)."""
        try:
            mode = self.system_mode.get()
        except Exception:
            mode = "programmer"

        if mode == "html_programmer":
            return ".html", [("HTML files", "*.html"), ("All files", "*.*")]
        elif mode == "programmer":
            return ".py", [("Python files", "*.py"), ("All files", "*.*")]
        else:
            # Fallback: show both file types
            return ".py", [("Python files", "*.py"), ("HTML files", "*.html"), ("All files", "*.*")]

    def save_as_file(self):
        """Save As: prompt for filename and save. Also enable Save thereafter."""
        try:
            def_ext, file_types = self._get_mode_default_ext_and_types()
            file_path = filedialog.asksaveasfilename(
                defaultextension=def_ext,
                filetypes=file_types
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.get_content())
                self.current_filename = file_path  # Track name so Save works without prompting
                if hasattr(self, 'save_button'):
                    self.save_button.config(state=tk.NORMAL)
                self.status_label.config(text=f"Saved: {os.path.basename(file_path)}", fg="green")
                self.parent.display_status_message(f"💾 File saved as: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file: {str(e)}")

    def save_file(self):
        """Save: if we have a filename, overwrite; else fall back to Save As."""
        try:
            if getattr(self, 'current_filename', None):
                with open(self.current_filename, 'w', encoding='utf-8') as f:
                    f.write(self.get_content())
                self.status_label.config(text=f"Saved: {os.path.basename(self.current_filename)}", fg="green")
                self.parent.display_status_message(f"💾 File saved: {os.path.basename(self.current_filename)}")
            else:
                self.save_as_file()
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file: {str(e)}")
    
    def load_file(self):
        """Load content from file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Python files", "*.py"), ("HTML files", "*.html")]
            )
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.set_content(content, file_path)
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}", fg="green")
                self.parent.display_status_message(f"📂 File loaded: {os.path.basename(file_path)}")
                # Loaded file has a name; enable Save
                if hasattr(self, 'save_button'):
                    self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load file: {str(e)}")

    def move_program_to_chat(self):
        """Move the current IDE program content to the chat input field"""
        content = self.get_content()
        if content.strip():
            # Clear the current chat input and insert the program
            #self.parent.user_input.delete("1.0", tk.END)
            self.parent.user_input.insert("1.0", content)
            self.parent.user_input.focus_set()
            self.parent.display_status_message("📤 Program moved to chat input")
        else:
            self.parent.display_status_message("⚠️ No content to move to chat")

    def run_current_code(self):
        """Run the current code content using parent's execution system"""
        content = self.get_content()
        if content.strip():
            # Check current mode to determine execution type
            mode = self.parent.system_mode.get()
            code_type = "HTML" if mode == "html_programmer" else "Python"
            
            # Update status to show running
            self.status_label.config(text=f"Running {code_type} code...", fg="orange")
            self.root.update_idletasks()  # Force UI update
            
            # Use parent's code execution system with timeout preference
            use_timeout = self.timed_execution.get()
            self.parent.run_last_code_block(content, use_timeout=use_timeout)
            
            # Update status after execution
            self.status_label.config(text=f"Check DEBUG console for output", fg="blue")
        else:
            self.status_label.config(text="No code to run", fg="gray")
            

    def fix_current_code(self):
        """Send code to LLM for fixing. User describes the problem in chat,
        the LLM figures out what's broken and returns only the changes."""
        content = self.get_content()
        if content.strip():
            self.parent.fix_code_from_ide(content)
            self.status_label.config(text="Fix request sent to LLM", fg="blue")
        else:
            self.status_label.config(text="No code to fix", fg="gray")

    # ---------- Run & Fix (F6): the main workflow button ----------

    def run_and_auto_fix(self):
        """Run code, detect errors, and automatically send to LLM for fixing.
        Flow: execute -> check stderr/browser errors -> send to LLM -> show diff"""
        content = self.get_content()
        if not content.strip():
            self.status_label.config(text="No code to run", fg="gray")
            return

        # Determine mode
        mode = self.parent.system_mode.get()
        is_html = (mode == "html_programmer")
        code_type = "HTML" if is_html else "Python"

        self.status_label.config(text=f"Run & Fix: running {code_type}...", fg="orange")
        # Flash Run & Fix button orange while running
        if hasattr(self, 'run_fix_btn'):
            self.run_fix_btn.config(bg="orange")
            self.root.after(2000, lambda: self.run_fix_btn.config(bg="gold"))
        self.root.update_idletasks()

        # Run the code
        use_timeout = self.timed_execution.get()
        self.parent.run_last_code_block(content, use_timeout=use_timeout)

        if is_html:
            # HTML errors arrive async via BrowserErrorServer; poll for ~3 seconds
            self.root.after(3000, lambda: self._check_html_errors_and_fix(content))
        else:
            # Python execution is synchronous; check stderr after brief delay
            self.root.after(500, lambda: self._check_python_errors_and_fix(content))

    def _check_python_errors_and_fix(self, original_code):
        """Check for Python errors after execution and trigger auto-fix"""
        stderr = getattr(self.parent, 'last_run_stderr', None)
        if stderr and stderr.strip():
            self.status_label.config(text="Fixing...", fg="red", font=("TkDefaultFont", 10, "bold"))
            self.root.update_idletasks()

            # Enable auto-propose bypass so the fix flows through
            self.parent._auto_fix_in_progress = True

            # Store original code for diff
            self.parent.ide_original_code = original_code

            # Send fix request
            self.parent.fix_code_from_ide(original_code)
        else:
            self.status_label.config(text="No errors!", fg="green", font=("TkDefaultFont", 10, "bold"))
            # Reset status after 2 seconds
            self.root.after(2000, lambda: self.status_label.config(text="Ready", fg="blue", font=("TkDefaultFont", 10)))

    def _check_html_errors_and_fix(self, original_code):
        """Check for HTML/browser errors and trigger auto-fix"""
        # Read debug console for browser error markers
        self.parent.debug_console.config(state=tk.NORMAL)
        debug_content = self.parent.debug_console.get("1.0", tk.END).strip()
        self.parent.debug_console.config(state=tk.DISABLED)

        error_markers = [
            "BROWSER ERROR CAPTURED", "JavaScript Error", "Uncaught",
            "TypeError", "ReferenceError", "SyntaxError"
        ]
        has_errors = any(marker in debug_content for marker in error_markers)

        if has_errors:
            self.status_label.config(text="Fixing...", fg="red", font=("TkDefaultFont", 10, "bold"))
            self.root.update_idletasks()

            # Enable auto-propose bypass
            self.parent._auto_fix_in_progress = True

            # Store original code for diff
            self.parent.ide_original_code = original_code

            # Send fix request
            self.parent.fix_code_from_ide(original_code)
        else:
            self.status_label.config(text="No errors!", fg="green", font=("TkDefaultFont", 10, "bold"))
            # Reset status after 2 seconds
            self.root.after(2000, lambda: self.status_label.config(text="Ready", fg="blue", font=("TkDefaultFont", 10)))

    def clear_ide(self):
        """Clear the IDE content and send a simple system message"""
        # Clear the editor content directly without triggering notifications
        self.editor.delete('1.0', tk.END)
        
        # Clear stored references directly (set to None, not empty string)
        self.current_content = ""
        self.current_filename = None
        self.parent.ide_current_filename = None  
        self.parent.ide_current_content = None
        
        # Update line numbers
        self.update_line_numbers()
        
        # Don't send ANY messages anywhere - just the status label is enough
        
        # Update status
        self.status_label.config(text="IDE cleared", fg="green")
    
    def show_find_dialog(self):
        """Show find dialog for searching text in the editor"""
        # Create find dialog if it doesn't exist
        if not hasattr(self, 'find_dialog') or not self.find_dialog.winfo_exists():
            self.find_dialog = Toplevel(self.root)
            self.find_dialog.title("Find")
            self.find_dialog.geometry("300x80")
            self.find_dialog.resizable(False, False)
            
            # Find frame
            find_frame = Frame(self.find_dialog, padx=10, pady=10)
            find_frame.pack(fill=tk.BOTH, expand=True)
            
            Label(find_frame, text="Find:").grid(row=0, column=0, sticky=tk.W)
            self.find_entry = Entry(find_frame, width=25)
            self.find_entry.grid(row=0, column=1, padx=(5, 0))
            self.find_entry.bind('<Return>', lambda e: self.find_text())
            
            # Buttons frame
            buttons_frame = Frame(find_frame)
            buttons_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
            
            Button(buttons_frame, text="Find Next", command=self.find_text, width=10).pack(side=tk.LEFT, padx=(0, 5))
            Button(buttons_frame, text="Close", command=self.find_dialog.destroy, width=10).pack(side=tk.LEFT)
            
            # Initialize search position
            self.search_pos = "1.0"
        
        # Show dialog and focus on entry
        self.find_dialog.lift()
        self.find_entry.focus_set()
        self.find_entry.select_range(0, tk.END)
    
    def find_text(self):
        """Find and highlight text in the editor"""
        search_term = self.find_entry.get()
        if not search_term:
            return
        
        # Clear previous highlighting
        self.editor.tag_remove("search_highlight", "1.0", tk.END)
        
        # Configure search highlight tag
        self.editor.tag_configure("search_highlight", background="yellow", foreground="black")
        
        # Search for text starting from current position
        pos = self.editor.search(search_term, self.search_pos, tk.END)
        if pos:
            # Calculate end position
            end_pos = f"{pos}+{len(search_term)}c"
            
            # Highlight found text
            self.editor.tag_add("search_highlight", pos, end_pos)
            
            # Move cursor and scroll to found text
            self.editor.mark_set(tk.INSERT, pos)
            self.editor.see(pos)
            
            # Update search position for next search
            self.search_pos = end_pos
            
            self.status_label.config(text=f"Found: {search_term}", fg="blue")
        else:
            # If not found, start from beginning
            if self.search_pos != "1.0":
                self.search_pos = "1.0"
                self.find_text()  # Try again from beginning
            else:
                self.status_label.config(text=f"Not found: {search_term}", fg="red")


    
    def go_to_line(self):
        """Show dialog to jump to specific line number"""
        from tkinter import simpledialog
        line_num = simpledialog.askinteger("Go to Line", "Enter line number:", minvalue=1)
        if line_num:
            self.go_to_line_direct(line_num)
    
    def go_to_line_direct(self, line_num):
        """Jump directly to specific line number"""
        try:
            # Move cursor to the line
            self.editor.mark_set(tk.INSERT, f"{line_num}.0")
            self.editor.see(f"{line_num}.0")
            
            # Clear previous line highlighting
            self.editor.tag_remove("current_line", "1.0", tk.END)
            
            # Configure and add line highlighting
            self.editor.tag_configure("current_line", background="lightyellow")
            self.editor.tag_add("current_line", f"{line_num}.0", f"{line_num}.end")
            
            # Focus on editor
            self.editor.focus_set()
            
            self.status_label.config(text=f"Jumped to line {line_num}", fg="blue")
        except Exception as e:
            self.status_label.config(text=f"Invalid line number: {line_num}", fg="red")


def cleanup_handler(signum, frame):
    """Cleanup function called on Ctrl+C"""
    print("\n🧹 Cleaning up CodeRunner resources...")

    try:
        # Stop MCP servers if running
        if MCP_AVAILABLE:
            try:
                # Find the app instance and stop its MCP client
                import gc
                for obj in gc.get_referrers(OllamaGUI):
                    if isinstance(obj, OllamaGUI) and hasattr(obj, 'mcp_client') and obj.mcp_client:
                        obj.mcp_client.stop_servers()
                        print("✅ MCP servers stopped")
                        break
            except Exception:
                pass

        # Clear GPU memory first
        clear_gpu_memory()

        # Kill any leftover vLLM processes
        subprocess.run(["pkill", "-f", "vllm"],
                      capture_output=True, timeout=5)

        # Kill any CodeRunner Python processes
        subprocess.run(["pkill", "-f", "CodeRunner_IDE"],
                      capture_output=True, timeout=5)

        # Kill Ollama if running
        try:
            subprocess.run(["sudo", "systemctl", "stop", "ollama"],
                          capture_output=True, timeout=5)
        except:
            pass

        print("✅ Cleanup completed. GPU memory cleared. Exiting gracefully.")

    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

    # Force exit
    sys.exit(0)



def clear_gpu_memory():
    """Clear GPU memory by forcing garbage collection and emptying CUDA cache"""
    try:
        import torch
        import gc

        if torch.cuda.is_available():
            print("🧹 Clearing GPU memory...")

            # Force garbage collection first
            gc.collect()


            # Try to empty CUDA cache for each GPU
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(i)
                except Exception as gpu_error:
                    print(f"⚠️ GPU {i} cleanup warning: {gpu_error}")

            # Get memory stats after cleanup attempt
            try:
                memory_info = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    memory_info.append(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

                print("✅ GPU memory cleared")
                for info in memory_info:
                    print(f"   {info}")

            except Exception as stats_error:
                print(f"⚠️ Could not get memory stats: {stats_error}")
                print("✅ GPU memory clearing attempted (stats unavailable)")

        else:
            # Check if GPU exists but PyTorch CUDA support is missing (e.g. cpu-only torch on DGX Spark)
            import subprocess, shutil
            if shutil.which("nvidia-smi"):
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_name = result.stdout.strip()
                        print(f"ℹ️ GPU detected ({gpu_name}) but torch.cuda.is_available()=False")
                        print(f"   PyTorch version: {torch.__version__} - may need CUDA-enabled build")
                        print(f"   GC cleanup only (no CUDA cache to clear)")
                        gc.collect()
                    else:
                        print("ℹ️ No CUDA GPUs available - nothing to clear")
                except Exception:
                    print("ℹ️ No CUDA GPUs available - nothing to clear")
            else:
                print("ℹ️ No CUDA GPUs available - nothing to clear")

    except Exception as e:
        print(f"⚠️ GPU cleanup warning: {e}")


# =============================================================================
# MAIN APPLICATION — three-panel chat + IDE + debug console
# Handles all LLM backends, UI layout, message routing, and the fix workflow.
# =============================================================================

class OllamaGUI:
    """Main chat interface with three-window system and integrated IDE
    
    SYSTEM MESSAGE ROUTING GUIDE:
    
    Use the following methods for deterministic routing:
    
    1. display_chat_system_message(msg) - Messages that go to CHAT HISTORY:
       - Errors and exceptions
       - Code loading notifications  
       - Execution outputs
       - Important state changes (chat saved/loaded)
       - Anything the LLM needs for context
    
    2. display_status_message(msg) - Messages that go to SYSTEM CONSOLE:
       - Model/backend status updates
       - Available options lists
       - General informational messages
       - Progress indicators
       - Non-critical confirmations
    
    3. display_message("System", msg, to_chat=True/False) - Explicit routing
       - Use when you need fine control
       - to_chat=True forces to chat history
       - to_chat=False forces to system console
    
    MIGRATION: Replace all display_message("System", ...) calls with the
    appropriate specific method above for deterministic routing.
    
    THREE-WINDOW SYSTEM (NEW FEATURE):
    1. Main Chat Window: 
       - User messages and assistant responses
       - Critical system messages (errors, code loading, execution results)
       - Messages that provide context for the LLM
    
    2. System Console Window:
       - Non-critical system messages (status updates, model switches)
       - Timestamped for tracking events
       - Reduces clutter in main chat history
       - Can be cleared independently
    
    3. Debug Console Window:
       - Backend debugging information
       - Detailed logs and technical details
       - For troubleshooting issues
    
    IMPORTANT SYSTEM MESSAGES (stay in chat):
    - "📝 Code loaded" - LLM needs to know what code is loaded
    - "Running code block" - Execution notifications
    - "Error:" messages - Critical for debugging
    - "Chat loaded/saved" - Important state changes
    - Execution outputs and results
    
    This separation ensures the LLM sees important context while keeping
    the interface clean and organized.
    """
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("CodeRunner IDE")
        self.root.geometry("1600x1320")  # Increased width by 300px (1100 -> 1400); height unchanged at 1320.

        # Chat state variables
        self.model_var = StringVar(value=model)
        self.system_mode = StringVar(value="html_programmer")  # Default to HTML/JavaScript (code-only)
        self.system_message = {'role': 'system', 'content': html_system_message}
        self.messages = [self.system_message]  # Initialize chat history with system message
        self.hide_thinking = BooleanVar(value=False)  # Control whether to hide thinking from chat history
        self.do_thinking = BooleanVar(value=False)  # Control whether to request thinking from model
        self.previous_thinking_state = True  # Track previous state of thinking toggle (True = thinking shown/saved)
        self.first_message = True  # Flag for first message sent
        self.current_file = None  # Track the currently loaded/saved file
        self.generation_active = False  # Flag to track active generation
        self.stop_generation = False  # Flag to signal stopping generation
        self.ollama_available = True  # Flag to track if Ollama is available
        self.temperature = DoubleVar(value=0.1)  # Default temperature (low = deterministic, one-shot on local LLMs)
        self.top_p = DoubleVar(value=0.5)  # Default top_p for nucleus sampling
        self.top_k = IntVar(value=40)  # Default top_k for top-k sampling
        self.last_run_code = None
        self.last_run_stdout = None
        self.last_run_stderr = None

        # MCP (Model Context Protocol) client for tool-calling agent loop
        self.mcp_client = None
        self._mcp_initialized = False
        self._pending_tool_results = []  # Collected tool calls from streaming
        
        # Backend selection
        # Default backend: MLX on macOS, Transformers on Linux
        if platform.system() == "Darwin" and MLX_AVAILABLE:
            _default_backend = "mlx"
        elif platform.system() == "Linux" and TRANSFORMERS_AVAILABLE:
            _default_backend = "transformers"
        else:
            _default_backend = "claude"
        self.backend_var = StringVar(value=_default_backend)  # "ollama", "llama_cpp", "mlx", "vllm", "transformers", or "claude"
        self.llama_cpp_model = None  # Will hold loaded llama-cpp-python model
        self.available_gguf_models = []  # List of available GGUF models

        self.selected_gguf_path = StringVar()  # Currently selected GGUF model path
        self.max_tokens_var = DoubleVar(value=16000)  # Default max tokens

        # MLX backend variables
        self.mlx_model = None  # Will hold loaded MLX model
        self.mlx_tokenizer = None  # Will hold MLX tokenizer
        self.available_mlx_models = []  # List of available MLX models
        self.selected_mlx_path = StringVar()  # Currently selected MLX model path
        self.mlx_is_vlm = False  # Whether the loaded MLX model is a vision-language model
        self.mlx_vlm_model = None  # Will hold loaded mlx-vlm model
        self.mlx_vlm_processor = None  # Will hold mlx-vlm processor
        self.mlx_model_vlm_flags = {}  # {path: bool} cache for VLM detection

        # vLLM backend variables
        self.vllm_model = None  # Will hold loaded vLLM model
        self.available_vllm_models = []  # List of available vLLM models
        self.selected_vllm_path = StringVar()  # Currently selected vLLM model path

        # Transformers backend variables
        self.transformers_model = None  # Will hold loaded transformers model
        self.transformers_tokenizer = None  # Will hold transformers tokenizer
        self.available_transformers_models = []  # List of available transformers models
        self.selected_transformers_path = StringVar()  # Currently selected transformers model path


        # Claude backend variables
        self.available_claude_models = []  # List of available Claude models
        self.claude_model_var = StringVar()  # Currently selected Claude model
        
        # RAG state variables
        self.rag_enabled = BooleanVar(value=False)
        self.current_collection = StringVar(value="")
        self.current_persist_dir = ""
        self.default_persist_dir = os.path.join(os.path.expanduser("~"), ".chromadb")
        
        # Image handling variables
        self.current_image = None  # Store the currently attached image
        self.image_display = None  # Reference to displayed image
        self.image_file_path = None  # Store path to selected image
        self.image_tk = None  # Store the Tkinter image object

        # Response timer variables
        self.response_start_time = None  # When the current response started
        self.timer_update_id = None  # ID for the timer update callback

        # Token counting variables
        self.last_input_tokens = 0   # Input tokens for last send
        self.last_output_tokens = 0  # Output tokens for last response
        self.last_output_speed = 0.0 # Tokens per second for last response
        self.total_input_tokens = 0  # Running total input tokens (resets on /restart)
        self.total_output_tokens = 0 # Running total output tokens (resets on /restart)
        
        # Game instruction prompts
        self.game_prompts = {
            "-- Select Game Preset --": "", # Placeholder
            "Space Invaders": """Create a Space Invaders game faithful to the 1978 Taito original.

Key features:
- Player controls laser cannon at bottom, moves horizontally, fires straight up (one bullet max)
- 5 rows of 11 aliens (55 total): squid-shaped (top 2), bug-shaped (middle 2), octopus-shaped (bottom)
- Aliens move in formation, shift horizontally then drop when hitting edges
- Aliens accelerate and drop bombs as their numbers decrease
- 4 protective bunkers that erode when hit by player or alien fire
- Mystery UFO appears occasionally for bonus points
- Player has 3 lives; game ends when aliens reach bottom or all lives lost
- Score display at top with classic arcade font

GREAT GRAPHICS & ANIMATION:
- Black background with bright colored sprites
- Aliens in 3 distinct designs with GREAT smooth animation
- Green player cannon, green destructible bunkers with GREAT effects
- White laser bullets, alien bombs with GREAT particle effects""",
            "Asteroids": """Create an Asteroids game faithful to the 1979 Atari original.

Key features:
- Player controls triangular spaceship with 360° rotation and momentum physics
- Thrust propels ship in direction it's facing, ship has inertia
- Large asteroids (4 at start) split into medium, then small when shot
- Two UFO types: large (slow, random shots) and small (fast, precise shots)
- Screen wraparound on all edges
- Player starts with 3 lives, earns extra life at 10,000 points
- Ship shows flame animation when thrusting

GREAT GRAPHICS & ANIMATION:
- Pure black background with bright white vector lines
- Triangular player ship with GREAT thrust flame animation
- Irregular jagged asteroid shapes as wireframe outlines with GREAT rotation
- UFOs as classic flying saucer wireframes with GREAT movement
- All graphics are unfilled wireframe outlines with GREAT visual effects""",
            "Breakout": """Create a Breakout game faithful to the 1976 Atari original.

Key features:
- Player controls white paddle at bottom that moves horizontally
- 8 rows of colored bricks: red (top), orange, green, yellow (bottom)
- Ball speed increases as bricks are cleared
- Ball angle changes based on paddle hit location
- Paddle shrinks after reaching certain brick rows
- Three balls (lives) per game
- Ball can break through multiple bricks on single bounce
- Solid walls and ceiling as boundaries

GREAT GRAPHICS & ANIMATION:
- Black background with bright saturated colors
- White paddle and ball with GREAT smooth movement and effects
- Colored brick rows: red, orange, green, yellow with GREAT destruction animation
- White borders for walls and ceiling
- Blocky arcade font for score with GREAT visual feedback""",
            "Pac-Man": """Create a Pac-Man game capturing the 1980 Namco original gameplay.

Key features:
- Yellow Pac-Man moves through maze eating 240 white dots
- 4 colored ghosts with distinct AI: red chases, pink ambushes, cyan flanks, orange erratic
- 4 power pellets make ghosts vulnerable (blue) and edible for bonus points
- Fruit appears periodically for extra points
- Side tunnels allow screen wrapping
- Ghosts return to central pen when eaten
- Grid-based movement with precise alignment
- Lives counter and level progression

GREAT GRAPHICS & ANIMATION:
- Black background with blue maze walls
- Yellow Pac-Man with GREAT smooth mouth animation and movement
- 4 distinctly colored ghosts with GREAT unique shapes and animation
- White dots and larger power pellets with GREAT visual effects
- Central ghost house and side tunnels with GREAT detail
- Classic arcade font for score with GREAT visual feedback""",
            "Frogger": """Create a Frogger game based on the 1981 Konami original.

Key features:
- Frog starts at bottom, must reach 5 home slots at top safely
- Bottom half: 5 lanes of traffic moving left/right at different speeds
- Top half: river with 5 rows of moving logs, turtles, and alligators
- Frog jumps on logs/turtles to cross river (water = death)
- Some turtles dive underwater periodically
- Crocodiles appear on logs with snapping jaws
- Timer counts down - reaching zero = death
- Bonus points for remaining time
- Grid-based movement, 3 lives, multiple difficulty levels

GREAT GRAPHICS & ANIMATION:
- Purple road, blue river, green grass background with GREAT detail
- Bright green frog with GREAT smooth 4-leg animation and movement
- Various colored vehicles with GREAT realistic movement and effects
- Brown logs, green turtles with GREAT diving animation, green alligators
- 5 home slots at top, timer bar, score display with GREAT visual feedback
- Chunky 1980s arcade style graphics with GREAT visual polish""",
            "Centipede": """Create a Centipede game faithful to the 1981 Atari original.

Key features:
- Player controls small gun at bottom, moves in lower portion only
- Centipede starts at top, moves horizontally then drops when hitting obstacles
- Centipede consists of 10-12 connected segments following head in chain
- Player shoots upward, destroying segments and mushrooms (4 hits to destroy mushrooms)
- When middle segment hit, centipede splits into two independent centipedes
- Additional enemies: fleas drop mushrooms, spiders zigzag diagonally, scorpions poison mushrooms
- Each wave increases difficulty and adds more centipedes
- 3 lives, extra life every 12,000 points, classic arcade scoring

GREAT GRAPHICS & ANIMATION:
- Black background with ultra-bright vivid colors
- Bright green/yellow segmented centipede with GREAT smooth movement and distinct head
- Colorful mushrooms with GREAT damage animation and visual effects
- Various colored enemies with GREAT movement patterns and effects
- Small player character at bottom with GREAT visual feedback
- Grid-based layout for mushrooms and movement with GREAT polish""",
            "Defender": """Defend humanoids from alien abduction.

Key features:
- Fly spaceship horizontally, rescue walking humanoids from landers
- Landers turn into mutants if they successfully abduct humanoids
- Use laser and smart bombs against enemies

GREAT GRAPHICS & ANIMATION:
- Black background, vector-style thin lines with GREAT detail
- Colorful enemies with GREAT smooth movement and effects
- Scrolling terrain with GREAT visual polish and animation""",
                "Floppy Birds": """Simple Flappy Bird clone.

Key features:
- Bird falls with gravity, spacebar makes it flap up
- Fly through pipe gaps, score increases per pipe passed
- Game over on collision, 'R' to restart

GREAT GRAPHICS & ANIMATION:
- Yellow bird with GREAT smooth flapping animation and physics
- Green pipes with gaps and GREAT scrolling movement
- Brown ground, blue sky with GREAT visual detail
- Simple shapes with GREAT visual polish and effects""",

            "Doom-Style FPS": """Simple raycasting FPS like Doom.

Key features:
- ARROW KEYS movement, mouse look, raycasting for 3D walls from 2D grid
- Weapons: pistol, shotgun, chaingun, rocket launcher with number key switching
- Enemies: simple colored shapes that chase player and attack

GREAT GRAPHICS & ANIMATION:
- Gray walls with GREAT texture detail and lighting effects
- Dark floor/ceiling with GREAT depth and perspective
- Enemy sprites with GREAT scaling animation and visual effects""",

            "Mario Kart": """Top-down kart racing game.

Key features:
- Arrow keys control (accelerate, brake, turn), 4-8 AI karts, 3 laps, position tracking
- Power-ups: item boxes give shells, bananas, mushrooms, lightning, stars with spacebar usage

GREAT GRAPHICS & ANIMATION:
- Colored karts with GREAT smooth movement and realistic physics
- Tracks with grass off-road areas and GREAT visual detail
- Item boxes with GREAT pickup animation and effects
- Mini-map with GREAT real-time updates and visual feedback""",
            "Mr. Do!": """Digging action game.

Key features:
- Control clown character that digs tunnels through dirt
- Push apples to crush monsters, collect cherries or kill all monsters to win level
- Throwable powerball that bounces and returns, alpha monsters drop letters for extra life

GREAT GRAPHICS & ANIMATION:
- Colorful clown character with GREAT movement and facial expressions
- Brown dirt with GREAT digging animation and particle effects
- Red apples with GREAT physics and crushing effects
- Round monsters with GREAT movement patterns and animations
- Underground setting with GREAT atmospheric detail and lighting""",
            "Minecraft": """Create a fully functional Minecraft-style voxel building game using Three.js.

Key features:
- First-person 3D voxel world with block-based terrain generation
- ARROW KEYS movement, mouse look, space to jump, shift to sneak/crouch
- Left click to destroy blocks, right click to place selected block
- Number keys 1-9 to select different block types from inventory
- Procedural terrain generation using 3D noise (Simplex or Perlin)
- Multiple biomes: grassy plains, deserts, forests, mountains, water bodies
- Day/night cycle with moving sun and changing sky colors
- Physics: gravity, collision detection, falling blocks
- Inventory system with hotbar showing selected block type
- Minimap in top-right corner showing explored chunks
- Creative mode: unlimited blocks, fly mode (double-tap space)
- Block types: grass, dirt, stone, cobblestone, wood, leaves, sand, water, bedrock
- Water blocks with transparency and flowing physics
- Trees generated naturally in forests
- Crosshair in center of screen for aiming
- Performance optimization: chunk-based rendering, frustum culling

GREAT GRAPHICS & ANIMATION:
- Smooth first-person camera with mouselook controls
- Blocky pixel-perfect voxel aesthetics with clean edges
- Dynamic lighting with shadows from the sun
- Sky gradient that changes from day to night (blue to purple/black)
- Water with realistic transparency and reflection effects
- Trees with trunk and leaf blocks, natural placement
- Smooth terrain transitions between biomes
- Particle effects when breaking/placing blocks
- Atmospheric fog that limits render distance
- High-performance rendering with 60+ FPS target""",
            "Super Mario Bros": """Create a Super Mario Bros game faithful to the 1985 Nintendo original.

Key features:
- Side-scrolling platformer with smooth camera that follows Mario
- Mario runs, jumps, and collects coins while avoiding enemies and obstacles
- Power-up system: Super Mushroom makes Mario big, Fire Flower gives fireball power
- Enemies: Goombas (walking mushrooms), Koopas (turtles in shells), Piranha Plants
- Pipes for level navigation and secret areas
- Flagpole at end of each level for bonus points based on height
- Classic level design with platforms, gaps, and destructible bricks
- Underground and underwater levels with different physics
- Lives system (3 lives), continue after game over
- Score system with points for enemies defeated and items collected
- Warp zones and hidden areas accessible through pipes or secret bricks

GREAT GRAPHICS & ANIMATION:
- Bright, colorful side-scrolling world with detailed sprites
- Smooth Mario animations: walking, jumping, power-up transformations
- Realistic physics with momentum, gravity, and bouncy jumps
- Animated background elements like clouds, bushes, and moving platforms
- Particle effects for coin collection, brick destruction, and enemy defeats
- Classic 8-bit pixel art style with sharp, clean graphics
- Scrolling parallax backgrounds with depth layers
- Animated water effects for underwater levels
- Detailed level design with foreground and background elements""",
            "Missile Command": """Create a Missile Command game faithful to the 1980 Atari original.

Key features:
- Player controls anti-missile batteries at bottom of screen
- Defend 6 cities from incoming nuclear missiles
- Missiles split into multiple warheads that spread out
- Crosshair targeting system with mouse or cursor keys
- Exploding missile effects that can destroy incoming threats
- Bonus cities awarded at certain score intervals
- Increasing difficulty with faster, more numerous missiles
- Realistic physics for missile trajectories and explosions
- Score system based on missiles destroyed and cities saved

GREAT GRAPHICS & ANIMATION:
- Dark night sky background with stars and horizon
- Bright white missiles streaking across screen
- Colorful explosion effects when missiles are destroyed
- Green cities and missile batteries with damage states
- Smooth missile trails and particle effects
- Radar display showing incoming threats
- Dramatic sound effects for launches and explosions
- Retro 1980s arcade aesthetic with glowing effects""",
            "Q*bert": """Create a Q*bert game faithful to the 1982 Gottlieb original.

Key features:
- Q*bert jumps on a pyramid of cubes, changing their color
- Jump in any of 4 diagonal directions to adjacent cubes
- All cubes must be changed to target color to complete level
- Avoid enemies: Coily (snake), Slick and Sam (green balls), Ugg and Wrongway (purple creatures)
- Collect items for bonus points: green balls, flying discs
- Falling off pyramid results in life loss
- Progressive difficulty with more enemies and faster movement
- 8 different enemy types with unique behaviors
- Score multipliers and bonus rounds

GREAT GRAPHICS & ANIMATION:
- Colorful isometric pyramid made of cubes
- Q*bert's smooth jumping animations and sound effects
- Enemies with distinct colors and movement patterns
- Cube color-changing effects with satisfying sound
- Particle effects when Q*bert lands on cubes
- Animated background with floating platforms
- Vibrant 1980s arcade color palette
- Smooth character movements and physics""",
        }
        self.selected_game_prompt = StringVar(value="-- Select Game Preset --")
        
        # Load available models
        if OLLAMA_AVAILABLE:
            try:
                self.available_models = get_available_models()
                self.ollama_available = True
            except Exception as e:
                self.available_models = [model]
                self.ollama_available = False
                print(f"Error initializing Ollama: {str(e)}")
        else:
            self.available_models = [model]
            self.ollama_available = False
            print("Ollama not available - using default model")
            
        # Load available GGUF models if llama-cpp-python is available
        if LLAMA_CPP_AVAILABLE:
            try:
                self.available_gguf_models = find_gguf_models()
                if self.available_gguf_models:
                    self.selected_gguf_path.set(self.available_gguf_models[0])
            except Exception as e:
                print(f"Error finding GGUF models: {str(e)}")
                self.available_gguf_models = []

        # Load available MLX models if MLX is available
        if MLX_AVAILABLE:
            try:
                self.available_mlx_models, self.mlx_model_vlm_flags = get_available_mlx_models()
                if self.available_mlx_models:
                    self.selected_mlx_path.set(self.available_mlx_models[0])
            except Exception as e:
                print(f"Error finding MLX models: {str(e)}")
                self.available_mlx_models = []
                self.mlx_model_vlm_flags = {}

        # Load available Claude models
        try:
            self.available_claude_models = get_claude_models()
            if self.available_claude_models:
                # Set default to claude-3-5-sonnet-latest if available, otherwise first model
                default_claude = "claude-3-5-sonnet-20241022"
                if default_claude in self.available_claude_models:
                    self.claude_model_var.set(default_claude)
                else:
                    self.claude_model_var.set(self.available_claude_models[0])
        except Exception as e:
            print(f"Error finding Claude models: {str(e)}")
            self.available_claude_models = []


        # Load available Transformers models
        if TRANSFORMERS_AVAILABLE:
            try:
                self.available_transformers_models = get_available_transformers_models()
                if self.available_transformers_models:
                    self.selected_transformers_path.set(self.available_transformers_models[0])
            except Exception as e:
                print(f"Error finding Transformers models: {str(e)}")
                self.available_transformers_models = []

        # Create main interface
        self.create_widgets()

        # Initialize IDE window
        self.ide_window = IDEWindow(self)

        # Track current IDE content for context
        self.ide_current_content = None
        self.ide_current_filename = None

        # Auto-propose is enabled — the _auto_fix_in_progress flag guards
        # the fix workflow, and Move to IDE handles first-shot.
        self.disable_auto_propose = False

        # Command list
        self.commands = {
            '/exit': 'Exit the chat',
            '/restart': 'Clear chat history and start over',
            '/think': 'Toggle hiding thinking from chat history',
            '/save': 'Save current chat to a file',
            '/load': 'Load a chat from a file',
            '/models': 'List available models (current backend)',
            '/backend': 'Switch between Ollama and GGUF backends',
            '/help': 'Show available commands',
            '/image': 'Add image from file',
            '/cleari': 'Clear attached image',
            '/rag': 'Toggle RAG functionality on/off',
            '/testrouting': 'Test system message routing (debug)',
            '/indexdir': 'Index a folder into ChromaDB',
            '/loaddb': 'Load an existing ChromaDB collection',
            '/clearrag': 'Unload the current RAG collection',
            '/testrag': 'Test a query against the current RAG collection',
            '/startollama': 'Attempt to start Ollama if not running',
            '/search': 'Search the web (e.g., /search python tutorials)'
        }
        
        # Display backend info
        if LLAMA_CPP_AVAILABLE:
            self.display_status_message("Multiple backends available: Ollama and GGUF. Use radio buttons to switch.")
            gguf_count = len(self.available_gguf_models)
            if gguf_count > 0:
                self.display_status_message(f"Found {gguf_count} GGUF models in your collection.")
        else:
            self.display_status_message("Using Ollama backend. Install llama-cpp-python for GGUF model support.")
            
        self.display_command_help()
        
        # Display RAG info
        if CHROMADB_AVAILABLE:
            self.display_status_message("RAG functionality available! Use /indexdir to index a folder of documents, or /loaddb to load an existing ChromaDB collection.")
        else:
            self.display_status_message("RAG functionality not available - ChromaDB not installed. Install with: pip install chromadb")
        
        # Set up copy/paste shortcuts
        self.setup_shortcuts()

        # Display welcome message (after widgets are created)
        self.display_status_message("Chat started. Type '/help' to see available commands.")
        self.display_status_message("Current mode: Helpful Assistant. Use radio buttons to change assistant mode.")

        # Set up clipboard monitoring for images
        self.setup_image_paste()

        self.use_ollama = BooleanVar(value=True)
        self.use_ollama_webui = BooleanVar(value=False)
        self.use_programmer_mode = BooleanVar(value=False)
        self.use_ollama_webui_web = BooleanVar(value=False)

        # Initialize IDE context tracking
        self.last_ide_context_included = False
        self.ide_original_code = None

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for copy/paste and chat editing"""
        # Standard copy/paste bindings for macOS and Windows/Linux
        self.root.bind("<Command-c>", self.copy_selected)
        self.root.bind("<Command-v>", self.paste_clipboard)
        self.root.bind("<Control-c>", self.copy_selected)
        self.root.bind("<Control-v>", self.paste_clipboard)
        
        # Chat editing shortcut - Cmd/Ctrl+S to sync chat edits
        self.root.bind("<Command-s>", lambda e: self.sync_chat_edits())
        self.root.bind("<Control-s>", lambda e: self.sync_chat_edits())
        
        # Right-click context menu
        self.create_context_menu()
        
        # Show a hint for Mac users
        def set_status_hint():
            try:
                self.status_var.set("Mac users: Control+click or two-finger tap for context menu")
            except AttributeError:
                pass  # status_var not created yet

        def set_status_ready():
            try:
                self.status_var.set("Ready")
            except AttributeError:
                pass  # status_var not created yet

        self.root.after(1000, set_status_hint)
        self.root.after(5000, set_status_ready)

    def create_context_menu(self):
        """Create right-click context menus for text widgets"""
        # Context menu for chat display
        self.chat_context_menu = Menu(self.root, tearoff=0)
        self.chat_context_menu.add_command(label="Copy Selected", command=self.copy_selected)
        self.chat_context_menu.add_command(label="Copy Last Message", command=self.copy_last_message)
        self.chat_context_menu.add_command(label="Copy Entire Conversation", command=self.copy_entire_chat)
        self.chat_context_menu.add_separator()
        self.chat_context_menu.add_command(label="Sync Chat Edits (Cmd+S)", command=self.sync_chat_edits)
        
        # Bind both right-click and Control+click for Mac
        try:
            self.chat_display.bind("<Button-2>", lambda e: self.show_context_menu(e, self.chat_context_menu))
            self.chat_display.bind("<Button-3>", lambda e: self.show_context_menu(e, self.chat_context_menu))
            self.chat_display.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.chat_context_menu))
        except AttributeError:
            # Widgets not created yet, skip binding
            pass
        
        # Context menu for user input
        self.input_context_menu = Menu(self.root, tearoff=0)
        self.input_context_menu.add_command(label="Copy", command=self.copy_selected)
        self.input_context_menu.add_command(label="Paste", command=self.paste_clipboard)
        self.input_context_menu.add_command(label="Cut", command=self.cut_selected)
        # Bind user input context menu
        try:
            self.user_input.bind("<Button-2>", lambda e: self.show_context_menu(e, self.input_context_menu))
            self.user_input.bind("<Button-3>", lambda e: self.show_context_menu(e, self.input_context_menu))
            self.user_input.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.input_context_menu))
        except AttributeError:
            # Widgets not created yet, skip binding
            pass
        
        # Context menu for system prompt
        self.prompt_context_menu = Menu(self.root, tearoff=0)
        self.prompt_context_menu.add_command(label="Copy", command=self.copy_from_prompt)
        self.prompt_context_menu.add_command(label="Paste", command=self.paste_to_prompt)
        self.prompt_context_menu.add_command(label="Cut", command=self.cut_from_prompt)
        # Bind prompt editor context menu
        try:
            self.prompt_editor.bind("<Button-2>", lambda e: self.show_context_menu(e, self.prompt_context_menu))
            self.prompt_editor.bind("<Button-3>", lambda e: self.show_context_menu(e, self.prompt_context_menu))
            self.prompt_editor.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.prompt_context_menu))
        except AttributeError:
            # Widgets not created yet, skip binding
            pass

        # Context menu for system console
        self.system_context_menu = Menu(self.root, tearoff=0)
        self.system_context_menu.add_command(label="Select All", command=self.select_all_system)
        self.system_context_menu.add_command(label="Copy", command=self.copy_system_selected)
        # Bind system console context menu
        try:
            self.system_console.bind("<Button-2>", lambda e: self.show_context_menu(e, self.system_context_menu))
            self.system_console.bind("<Button-3>", lambda e: self.show_context_menu(e, self.system_context_menu))
            self.system_console.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.system_context_menu))
        except AttributeError:
            # Widgets not created yet, skip binding
            pass

        # Context menu for debug console
        self.debug_context_menu = Menu(self.root, tearoff=0)
        self.debug_context_menu.add_command(label="Select All", command=self.select_all_debug)
        self.debug_context_menu.add_command(label="Copy", command=self.copy_debug_selected)
        # Bind debug console context menu
        try:
            self.debug_console.bind("<Button-2>", lambda e: self.show_context_menu(e, self.debug_context_menu))
            self.debug_console.bind("<Button-3>", lambda e: self.show_context_menu(e, self.debug_context_menu))
            self.debug_console.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.debug_context_menu))
        except AttributeError:
            # Widgets not created yet, skip binding
            pass

        # Context menu for search results console
        self.search_context_menu = Menu(self.root, tearoff=0)
        self.search_context_menu.add_command(label="Select All", command=self.select_all_search)
        self.search_context_menu.add_command(label="Copy", command=self.copy_search_selected)
        # Bind search results console context menu
        try:
            self.search_results_console.bind("<Button-2>", lambda e: self.show_context_menu(e, self.search_context_menu))
            self.search_results_console.bind("<Button-3>", lambda e: self.show_context_menu(e, self.search_context_menu))
            self.search_results_console.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.search_context_menu))
        except AttributeError:
            # Widgets not created yet, skip binding
            pass

    def show_context_menu(self, event, menu):
        """Show the context menu at the cursor position"""
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
            
    def copy_selected(self, event=None):
        """Copy selected text to clipboard"""
        try:
            if self.root.focus_get() == self.chat_display:
                selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
                self.root.clipboard_clear()
                self.root.clipboard_append(selected_text)
            elif self.root.focus_get() == self.user_input:
                selected_text = self.user_input.get(tk.SEL_FIRST, tk.SEL_LAST)
                self.root.clipboard_clear()
                self.root.clipboard_append(selected_text)
        except tk.TclError:
            # No selection
            pass
        return "break"
            
    def paste_clipboard(self, event=None):
        """Paste clipboard text to focused widget"""
        if self.root.focus_get() == self.user_input:
            clipboard_text = self.root.clipboard_get()
            self.user_input.insert(tk.INSERT, clipboard_text)
        return "break"
    
    def cut_selected(self, event=None):
        """Cut selected text to clipboard"""
        if self.root.focus_get() == self.user_input:
            self.copy_selected()
            self.user_input.delete(tk.SEL_FIRST, tk.SEL_LAST)
        return "break"
            
    def copy_from_prompt(self):
        """Copy selected text from system prompt editor"""
        try:
            selected_text = self.prompt_editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass
            
    def paste_to_prompt(self):
        """Paste clipboard text to system prompt editor"""
        clipboard_text = self.root.clipboard_get()
        self.prompt_editor.insert(tk.INSERT, clipboard_text)
            
    def cut_from_prompt(self):
        """Cut selected text from system prompt editor"""
        self.copy_from_prompt()
        try:
            self.prompt_editor.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass

    def select_all_system(self):
        """Select all text in system console"""
        self.system_console.tag_add(tk.SEL, "1.0", tk.END)
        self.system_console.focus_set()
        return "break"

    def copy_system_selected(self):
        """Copy selected text from system console"""
        try:
            selected_text = self.system_console.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass
        return "break"

    def select_all_debug(self):
        """Select all text in debug console"""
        self.debug_console.tag_add(tk.SEL, "1.0", tk.END)
        self.debug_console.focus_set()
        return "break"

    def copy_debug_selected(self):
        """Copy selected text from debug console"""
        try:
            selected_text = self.debug_console.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass
        return "break"

    def select_all_search(self):
        """Select all text in search results console"""
        self.search_results_console.tag_add(tk.SEL, "1.0", tk.END)
        self.search_results_console.focus_set()
        return "break"

    def copy_search_selected(self):
        """Copy selected text from search results console"""
        try:
            selected_text = self.search_results_console.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass
        return "break"

    def copy_last_message(self):
        """Copy the last message in the chat"""
        if not self.messages or len(self.messages) < 2:
            return  # No messages to copy

        last_msg = self.messages[-1]
        mode = self.system_mode.get()

        if mode == "programmer":
            sender_label = "Py Assistant"
        elif mode == "html_programmer":
            sender_label = "HTML Assistant"
        else:
            sender_label = "Assistant"

        if last_msg['role'] == 'assistant':
            content = f"{sender_label}: {last_msg['content']}"
        elif last_msg['role'] == 'user':
            content = f"You: {last_msg['content']}"
        else:
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.show_copy_status("Last message copied to clipboard")

    def copy_entire_chat(self):
        """Copy the entire chat conversation"""
        if not self.messages:
            return  # No messages to copy

        content = []
        mode = self.system_mode.get()

        if mode == "programmer":
            sender_label = "Py Assistant"
        elif mode == "html_programmer":
            sender_label = "HTML Assistant"
        else:
            sender_label = "Assistant"
        
        for msg in self.messages:
            if msg['role'] == 'assistant':
                content.append(f"{sender_label}: {msg['content']}")
            elif msg['role'] == 'user':
                content.append(f"You: {msg['content']}")
            # Skip system messages
            
        if content:
            full_content = "\n\n".join(content)
            self.root.clipboard_clear()
            self.root.clipboard_append(full_content)
            self.show_copy_status("Entire conversation copied to clipboard")
            
    def show_copy_status(self, message, duration=2000):
        """Show a status message when something is copied"""
        try:
            original_status = self.status_var.get()
            self.status_var.set(message)

            # Reset the status after a delay
            def reset_status():
                try:
                    self.status_var.set(original_status)
                except AttributeError:
                    pass  # status_var might be gone

            self.root.after(duration, reset_status)
        except AttributeError:
            pass  # status_var not available, skip status update

    def sync_chat_edits(self):
        """Sync the edited chat display back to the message history"""
        try:
            # Get the current content of the chat display
            chat_content = self.chat_display.get("1.0", tk.END).strip()
            
            if not chat_content:
                self.display_status_message("Chat is empty - nothing to sync")
                return
            
            # Parse the chat content back into messages
            new_messages = self.parse_chat_display_to_messages(chat_content)
            
            # Keep the system message at the beginning
            system_msg = None
            for msg in self.messages:
                if msg['role'] == 'system':
                    system_msg = msg
                    break
            
            # Replace the message history (except system message)
            if system_msg:
                self.messages = [system_msg] + new_messages
            else:
                self.messages = new_messages
            
            self.display_status_message(f"Chat history synced - {len(new_messages)} messages")
            
        except Exception as e:
            self.display_status_message(f"Error syncing chat edits: {str(e)}")

    def parse_chat_display_to_messages(self, chat_content):
        """Parse the chat display content back into message format"""
        messages = []
        lines = chat_content.split('\n')
        
        current_sender = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new message
            if line.startswith("You: "):
                # Save previous message
                if current_sender and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:  # Only add non-empty messages
                        if current_sender == "user":
                            messages.append({'role': 'user', 'content': content})
                        elif current_sender == "assistant":
                            messages.append({'role': 'assistant', 'content': content})
                
                # Start new user message
                current_sender = "user"
                current_content = [line[5:]]  # Remove "You: " prefix
                
            elif (line.startswith("Assistant: ") or
                  line.startswith("Helpful Assistant: ") or
                  line.startswith("Py Assistant: ") or
                  line.startswith("Therapist: ")):
                
                # Save previous message
                if current_sender and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:  # Only add non-empty messages
                        if current_sender == "user":
                            messages.append({'role': 'user', 'content': content})
                        elif current_sender == "assistant":
                            messages.append({'role': 'assistant', 'content': content})
                
                # Start new assistant message
                current_sender = "assistant"
                if line.startswith("Assistant: "):
                    current_content = [line[11:]]  # Remove "Assistant: " prefix
                elif line.startswith("Helpful Assistant: "):
                    current_content = [line[19:]]  # Remove "Helpful Assistant: " prefix
                elif line.startswith("Py Assistant: "):
                    current_content = [line[14:]]  # Remove "Py Assistant: " prefix
                elif line.startswith("Therapist: "):
                    current_content = [line[10:]]  # Remove "Therapist: " prefix
                    
            elif line.startswith("System: "):
                # Skip system messages for now - they're not part of conversation history
                current_sender = None
                current_content = []
                
            else:
                # This is a continuation of the current message
                if current_sender:
                    current_content.append(line)
        
        # Don't forget the last message
        if current_sender and current_content:
            content = '\n'.join(current_content).strip()
            if content:  # Only add non-empty messages
                if current_sender == "user":
                    messages.append({'role': 'user', 'content': content})
                elif current_sender == "assistant":
                    messages.append({'role': 'assistant', 'content': content})
        
        return messages

    def create_widgets(self):
        """Create all widgets for the interface"""
        # Main container with padding
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Backend and Model selection bar
        model_frame = Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Backend selection
        backend_frame = Frame(model_frame)
        backend_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        Label(backend_frame, text="Backend:").pack(side=tk.TOP, anchor=tk.W)
        backend_select_frame = Frame(backend_frame)
        backend_select_frame.pack(side=tk.TOP, fill=tk.X)
        
        from tkinter import Radiobutton
        if OLLAMA_AVAILABLE:
            Radiobutton(backend_select_frame, text="Ollama", variable=self.backend_var, value="ollama", command=self.change_backend).pack(side=tk.LEFT, padx=(0, 5))
        else:
            Label(backend_select_frame, text="(Ollama not installed)", fg="gray").pack(side=tk.LEFT, padx=(0, 5))
        if LLAMA_CPP_AVAILABLE:
            Radiobutton(backend_select_frame, text="GGUF", variable=self.backend_var, value="llama_cpp", command=self.change_backend).pack(side=tk.LEFT, padx=5)
        else:
            Label(backend_select_frame, text="(GGUF not available)", fg="gray").pack(side=tk.LEFT, padx=5)
        if MLX_AVAILABLE:
            Radiobutton(backend_select_frame, text="MLX", variable=self.backend_var, value="mlx", command=self.change_backend).pack(side=tk.LEFT, padx=5)
        else:
            Label(backend_select_frame, text="(MLX not available)", fg="gray").pack(side=tk.LEFT, padx=5)
        # vLLM backend - works on x86 and ARM (e.g. DGX Spark Blackwell) with CUDA
        if VLLM_AVAILABLE:
            Radiobutton(backend_select_frame, text="vLLM", variable=self.backend_var, value="vllm", command=self.change_backend).pack(side=tk.LEFT, padx=5)
        else:
            Label(backend_select_frame, text="(vLLM not available - requires CUDA)", fg="gray").pack(side=tk.LEFT, padx=5)
        if TRANSFORMERS_AVAILABLE:
            Radiobutton(backend_select_frame, text="Transformers", variable=self.backend_var, value="transformers", command=self.change_backend).pack(side=tk.LEFT, padx=5)
        else:
            Label(backend_select_frame, text="(Transformers not available)", fg="gray").pack(side=tk.LEFT, padx=5)
        Radiobutton(backend_select_frame, text="Claude", variable=self.backend_var, value="claude", command=self.change_backend).pack(side=tk.LEFT, padx=5)
        Radiobutton(backend_select_frame, text="OpenAI", variable=self.backend_var, value="openai", command=self.change_backend).pack(side=tk.LEFT, padx=5)
        
        # Model selection
        model_select_frame = Frame(model_frame)
        model_select_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(15, 0))
        
        Label(model_select_frame, text="Model:").pack(side=tk.TOP, anchor=tk.W)
        model_controls_frame = Frame(model_select_frame)
        model_controls_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Unified model dropdown for all backends
        self.model_dropdown = ttk.Combobox(model_controls_frame, textvariable=self.model_var, values=self.available_models, width=25)
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 5))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.change_model)
        
        # Backend-specific buttons frame
        self.backend_buttons_frame = Frame(model_controls_frame)
        self.backend_buttons_frame.pack(side=tk.LEFT, padx=2)
        
        # Load model button (text changes based on backend)
        self.load_model_btn = Button(self.backend_buttons_frame, text="Load Model", command=self.load_current_backend_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=2)
        
        # Refresh models button
        self.refresh_model_btn = Button(self.backend_buttons_frame, text="↻", command=self.refresh_current_backend_models, width=2)
        self.refresh_model_btn.pack(side=tk.LEFT, padx=2)

        # Clear GPU memory button
        self.clear_gpu_btn = Button(self.backend_buttons_frame, text="🧹", command=self.clear_gpu_memory_gui, width=2)
        self.clear_gpu_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.clear_gpu_btn, "Clear GPU memory\nFrees VRAM for new models")
        
        # Initialize OpenAI models
        self.openai_models = get_openai_models()

        # Model status label
        self.model_status_label = Label(model_select_frame, text="", fg="green")
        self.model_status_label.pack(side=tk.TOP, anchor=tk.W)
        
        # Code mode radio buttons (code-only: Python or HTML)
        mode_frame = Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 5))

        Label(mode_frame, text="Code Mode:").pack(side=tk.LEFT, padx=(0, 5))

        from tkinter import Radiobutton, Checkbutton
        Radiobutton(mode_frame, text="Python / Pygame", variable=self.system_mode, value="programmer", command=self.toggle_system_prompt).pack(side=tk.LEFT, padx=5)
        Radiobutton(mode_frame, text="HTML / JavaScript", variable=self.system_mode, value="html_programmer", command=self.toggle_system_prompt).pack(side=tk.LEFT, padx=5)
        
        # Search mode toggle frame
        self.search_frame = Frame(main_frame)
        self.search_frame.pack(fill=tk.X, pady=(5, 5))

        # Initialize search mode variable
        self.search_mode = tk.BooleanVar(value=False)

        # System prompt editor checkbox
        self.system_prompt_visible = tk.BooleanVar(value=False)
        system_prompt_checkbox = Checkbutton(self.search_frame, text="System Prompt Editor",
                                           variable=self.system_prompt_visible,
                                           command=self.toggle_system_prompt_editor)
        system_prompt_checkbox.pack(side=tk.LEFT, padx=5)

        # Enable Web Search checkbox
        search_checkbox = Checkbutton(self.search_frame, text="Enable Web Search (Tool Calls)",
                                    variable=self.search_mode,
                                    command=self.toggle_search_mode)
        search_checkbox.pack(side=tk.LEFT, padx=5)
        
        # "Fast edits" checkbox removed — fix prompt always asks for complete code now
        # Keep the variable for backward compat with update_system_message_for_targeted_changes
        self.request_functions_only = tk.BooleanVar(value=False)

        # Checkbox to show/hide sampling controls (temperature, top_p, top_k)
        self.show_sampling_controls = tk.BooleanVar(value=False)
        sampling_checkbox = Checkbutton(
            self.search_frame,
            text="Show Sampling Controls",
            variable=self.show_sampling_controls,
            command=self.toggle_sampling_controls
        )
        sampling_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Temperature control frame (shown when "Show Sampling Controls" is checked)
        self.temp_frame = Frame(main_frame)
        
        # Temperature label with value display
        temp_label_frame = Frame(self.temp_frame)
        temp_label_frame.pack(side=tk.LEFT)

        Label(temp_label_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temp_value_label = Label(temp_label_frame, text="0.1")
        self.temp_value_label.pack(side=tk.LEFT, padx=5)

        # Temperature slider
        from tkinter import Scale
        self.temp_slider = Scale(
            self.temp_frame, 
            from_=0.0, 
            to=1.0, 
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.temperature,
            command=self.update_temp_label,
            length=200
        )
        self.temp_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Temperature presets
        preset_frame = Frame(self.temp_frame)
        preset_frame.pack(side=tk.RIGHT)
        
        Button(preset_frame, text="Code Exact (0.15)", command=lambda: self.set_temperature(0.15)).pack(side=tk.LEFT, padx=2)
        Button(preset_frame, text="Precise (0.2)", command=lambda: self.set_temperature(0.2)).pack(side=tk.LEFT, padx=2)
        Button(preset_frame, text="Game (0.35)", command=lambda: self.set_temperature(0.35)).pack(side=tk.LEFT, padx=2)
        Button(preset_frame, text="Balanced (0.5)", command=lambda: self.set_temperature(0.5)).pack(side=tk.LEFT, padx=2)
        Button(preset_frame, text="Creative (0.8)", command=lambda: self.set_temperature(0.8)).pack(side=tk.LEFT, padx=2)

        # Top-p (nucleus sampling) control
        self.topp_frame = Frame(main_frame)

        Label(self.topp_frame, text="Top-p:").pack(side=tk.LEFT, padx=(0, 5))
        self.topp_slider = Scale(
            self.topp_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.top_p,
            length=200
        )
        self.topp_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Top-p presets
        topp_preset_frame = Frame(self.topp_frame)
        topp_preset_frame.pack(side=tk.RIGHT)
        Button(topp_preset_frame, text="Precise (0.5)", command=lambda: self.top_p.set(0.5)).pack(side=tk.LEFT, padx=2)
        Button(topp_preset_frame, text="Balanced (0.9)", command=lambda: self.top_p.set(0.9)).pack(side=tk.LEFT, padx=2)
        Button(topp_preset_frame, text="Creative (0.95)", command=lambda: self.top_p.set(0.95)).pack(side=tk.LEFT, padx=2)

        # Top-k control
        self.topk_frame = Frame(main_frame)

        Label(self.topk_frame, text="Top-k:").pack(side=tk.LEFT, padx=(0, 5))
        self.topk_slider = Scale(
            self.topk_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.top_k,
            length=200
        )
        self.topk_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Top-k presets
        topk_preset_frame = Frame(self.topk_frame)
        topk_preset_frame.pack(side=tk.RIGHT)
        Button(topk_preset_frame, text="Precise (10)", command=lambda: self.top_k.set(10)).pack(side=tk.LEFT, padx=2)
        Button(topk_preset_frame, text="Balanced (40)", command=lambda: self.top_k.set(40)).pack(side=tk.LEFT, padx=2)
        Button(topk_preset_frame, text="Creative (80)", command=lambda: self.top_k.set(80)).pack(side=tk.LEFT, padx=2)

        # Max tokens control for GGUF (initially hidden)
        self.max_tokens_frame = Frame(main_frame)
        self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Max tokens label with value display
        max_tokens_label_frame = Frame(self.max_tokens_frame)
        max_tokens_label_frame.pack(side=tk.LEFT)
        
        Label(max_tokens_label_frame, text="Max Tokens:").pack(side=tk.LEFT)
        self.max_tokens_value_label = Label(max_tokens_label_frame, text="16000")
        self.max_tokens_value_label.pack(side=tk.LEFT, padx=5)
        
        # Max tokens slider - increased range for modern models like Qwen3 (245k context)
        self.max_tokens_slider = Scale(
            self.max_tokens_frame, 
            from_=1024, 
            to=250000,  # Increased to support Qwen3's ~245k context window
            resolution=1024,
            orient=tk.HORIZONTAL,
            variable=self.max_tokens_var,
            command=self.update_max_tokens_label,
            length=200
        )
        self.max_tokens_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        

        
        # Hide max tokens frame initially
        self.max_tokens_frame.pack_forget()

        # Button bar
        button_frame = Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # Restart button (removed System Prompt button - now a checkbox)
        restart_btn = Button(button_frame, text="Restart Chat", command=self.restart_chat)
        restart_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_btn = Button(button_frame, text="Save Chat", command=self.save_chat)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Load button
        load_btn = Button(button_frame, text="Load Chat", command=self.load_chat)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Image button
        image_btn = Button(button_frame, text="Add Image", command=self.select_image)
        image_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear image button
        clear_image_btn = Button(button_frame, text="Clear Image", command=self.clear_image)
        clear_image_btn.pack(side=tk.LEFT, padx=5)
        
        # Hide thinking checkbox
        think_check = Checkbutton(button_frame, text="Hide Thinking", variable=self.hide_thinking, command=self.sync_thinking_state)
        think_check.pack(side=tk.LEFT, padx=5)

        # DO thinking checkbox (request thinking from model)
        do_think_check = Checkbutton(button_frame, text="Request Thinking", variable=self.do_thinking)
        do_think_check.pack(side=tk.LEFT, padx=5)
        
        # System prompt editor frame (hidden by default)
        self.prompt_frame = Frame(main_frame)
        self.prompt_frame.pack(fill=tk.X, pady=(0, 10))
        self.prompt_frame.pack_forget()  # Hide initially
        
        # System prompt text area
        Label(self.prompt_frame, text="System Prompt:").pack(anchor=tk.W)
        self.prompt_editor = scrolledtext.ScrolledText(self.prompt_frame, wrap=tk.WORD, height=5)
        self.prompt_editor.pack(fill=tk.X, pady=(0, 5))
        self.prompt_editor.insert(tk.END, self.system_message['content'])
        
        # Bind context menu to prompt editor
        self.prompt_editor.bind("<Button-2>", lambda e: self.show_context_menu(e, self.prompt_context_menu))
        self.prompt_editor.bind("<Button-3>", lambda e: self.show_context_menu(e, self.prompt_context_menu))
        self.prompt_editor.bind("<Control-Button-1>", lambda e: self.show_context_menu(e, self.prompt_context_menu))
        
        # Save and Close buttons for prompt editor
        prompt_btn_frame = Frame(self.prompt_frame)
        prompt_btn_frame.pack(fill=tk.X)
        
        Button(prompt_btn_frame, text="Save", command=self.save_system_prompt).pack(side=tk.RIGHT, padx=5)
        Button(prompt_btn_frame, text="Close", command=self.hide_system_prompt).pack(side=tk.RIGHT, padx=5)
        
        # ---- Chat header: [Move to IDE] [Open IDE]  timer  token counters ----
        chat_frame = Frame(main_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        chat_header_frame = Frame(chat_frame)
        chat_header_frame.pack(fill=tk.X, pady=(0, 5))

        # IDE buttons
        Button(chat_header_frame, text="Move Code to IDE", command=self.move_code_to_ide, bg="lightgreen").pack(side=tk.LEFT, padx=2)
        Button(chat_header_frame, text="Open IDE", command=self.open_ide_window, bg="lightblue").pack(side=tk.LEFT, padx=2)

        Label(chat_header_frame, text="(1st time: loads code | after: shows diff)", font=("TkDefaultFont", 8), fg="gray50").pack(side=tk.LEFT, padx=(4, 0))
        Label(chat_header_frame, text="LLM Chat:").pack(side=tk.LEFT, padx=(10, 0))

        # Timer display for response elapsed time
        self.timer_label = Label(chat_header_frame, text="00:00", font=("Arial", 10, "bold"), fg="blue")
        self.timer_label.pack(side=tk.LEFT, padx=(10, 0))

        # Token counters (per-message and running totals)
        self.token_label = Label(chat_header_frame, text="In:0 Out:0 | Total In:0 Out:0",
                                 font=("Arial", 9), fg="gray40")
        self.token_label.pack(side=tk.LEFT, padx=(10, 0))

        self.speed_label = Label(chat_header_frame, text="",
                                 font=("Arial", 9), fg="gray40")
        self.speed_label.pack(side=tk.LEFT, padx=(4, 0))
        
        # Chat display with scrolling - NOW EDITABLE (reduced height by 50%)
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=10)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        # Remove the DISABLED state to make it editable
        
        # Image display area
        self.image_frame = Frame(main_frame)
        self.image_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.image_label = Label(self.image_frame, text="No image attached")
        self.image_label.pack(fill=tk.X, pady=5)
        
        self.image_display_label = Label(self.image_frame)
        self.image_display_label.pack(fill=tk.X, pady=5)
        
        # Hide image frame initially
        self.image_frame.pack_forget()
        
        # System messages area - NEW THIRD WINDOW
        # This window displays non-critical system messages to reduce chat clutter
        # Critical messages (code loading, errors, etc.) still appear in main chat
        system_frame = Frame(main_frame)
        system_frame.pack(fill=tk.X, pady=(5, 5))
        
        system_header_frame = Frame(system_frame)
        system_header_frame.pack(fill=tk.X, pady=(0, 2))
        
        Label(system_header_frame, text="System Messages:").pack(side=tk.LEFT)
        Button(system_header_frame, text="Clear", command=self.clear_system_console).pack(side=tk.RIGHT, padx=2)
        
        # System messages console with scrolling
        # Shows timestamped system messages like model status, backend switches, etc.
        self.system_console = scrolledtext.ScrolledText(system_frame, wrap=tk.WORD, height=6)
        self.system_console.pack(fill=tk.X)
        self.system_console.config(state=tk.DISABLED)  # Make it read-only
        
        # Debug console area
        debug_frame = Frame(main_frame)
        debug_frame.pack(fill=tk.X, pady=(5, 5))
        
        debug_header_frame = Frame(debug_frame)
        debug_header_frame.pack(fill=tk.X, pady=(0, 2))
        
        Label(debug_header_frame, text="Debug Console:").pack(side=tk.LEFT)
        # Removed dedicated COPY button - use context menu instead
        Button(debug_header_frame, text="Clear", command=self.clear_debug_console).pack(side=tk.RIGHT, padx=2)
        # Toggle for capturing browser errors back to debug console (non-invasive)
        self.capture_browser_errors = BooleanVar(value=True)
        Checkbutton(
            debug_header_frame,
            text="Capture Browser Errors",
            variable=self.capture_browser_errors,
            command=self._on_toggle_capture_browser_errors
        ).pack(side=tk.LEFT, padx=8)
        
        # Debug console with scrolling (doubled size)
        self.debug_console = scrolledtext.ScrolledText(debug_frame, wrap=tk.WORD, height=10)
        self.debug_console.pack(fill=tk.X)
        self.debug_console.config(state=tk.DISABLED)  # Make it read-only
        
        # Search results console (hidden placeholder for compat — RAG removed)
        search_results_frame = Frame(main_frame)
        # Not packed — RAG removed in code-only version
        self.search_results_console = scrolledtext.ScrolledText(search_results_frame, wrap=tk.WORD, height=4)
        self.search_results_console.config(state=tk.DISABLED)

        # Game preset selection
        preset_frame = Frame(main_frame)
        preset_frame.pack(fill=tk.X, pady=(5, 0))
        Label(preset_frame, text="Game Preset:").pack(side=tk.LEFT, padx=(0, 5))
        self.game_preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.selected_game_prompt,
            values=list(self.game_prompts.keys()),
            state="readonly" # Prevent typing custom values
        )
        self.game_preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.game_preset_combo.bind("<<ComboboxSelected>>", self.on_game_preset_selected)

        # RAG removed in code-only version — keep variables for backward compat
        self.rag_toggle = None
        
        # User input area - for entering messages
        Label(main_frame, text="Your Message (describe code to write, or what to fix):").pack(anchor=tk.W, pady=(5,0))
        self.user_input = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=4)
        self.user_input.pack(fill=tk.X, pady=(0, 10))
        self.user_input.focus_set()
        
        # Bind Enter key to send message (Shift+Enter for new line)
        self.user_input.bind("<Return>", self.on_enter_pressed)
        self.user_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
        
        # Send/Stop button frame
        send_frame = Frame(main_frame)
        send_frame.pack(side=tk.RIGHT)
        
        # Create Send button
        self.send_button = Button(send_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
        
        # Add 'Send + Search' button to force-include recent search results with this message
        self.send_with_search_button = Button(send_frame, text="Send + Search", command=self.send_message_with_search)
        self.send_with_search_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Create LLM Fix button (always visible — grabs code from IDE and sends fix request)
        self.fix_button = Button(send_frame, text="LLM Fix", command=self.fix_from_chat, bg="lightyellow",
                                 font=("TkDefaultFont", 9, "bold"))
        self.fix_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Create Stop button (hidden initially)
        self.stop_button = Button(send_frame, text="Stop", command=self.stop_message, bg="red", fg="white")
        # Stop button is initially hidden
        
        # Status bar
        status_frame = Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = Label(status_frame, textvariable=self.status_var, bd=0, anchor=tk.W)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Ollama status indicator
        self.ollama_status_var = tk.StringVar()
        self.ollama_status_var.set("⚪ Ollama: Unknown")
        self.ollama_status = Label(status_frame, textvariable=self.ollama_status_var, bd=0, padx=5)
        self.ollama_status.pack(side=tk.RIGHT)
        
        # Set up periodic Ollama status check
        self.schedule_ollama_status_check()
        
        # Initialize backend UI state
        self.change_backend()

    def clear_gpu_memory_gui(self):
        """Clear GPU memory from GUI button"""
        try:
            # Clear any previously loaded models to actually free GPU memory
            if hasattr(self, 'transformers_model') and self.transformers_model:
                self.display_status_message("Unloading Transformers model...")
                self.transformers_model = None
            if hasattr(self, 'transformers_tokenizer') and self.transformers_tokenizer:
                self.transformers_tokenizer = None
            if hasattr(self, 'llama_cpp_model') and self.llama_cpp_model:
                self.display_status_message("Unloading llama-cpp model...")
                self.llama_cpp_model = None
            if hasattr(self, 'mlx_model') and self.mlx_model:
                self.display_status_message("Unloading MLX model...")
                self.mlx_model = None
            if hasattr(self, 'mlx_tokenizer') and self.mlx_tokenizer:
                self.mlx_tokenizer = None
            if hasattr(self, 'vllm_model') and self.vllm_model:
                self.display_status_message("Unloading vLLM model...")
                self.vllm_model = None
            if hasattr(self, 'model') and self.model:
                self.display_status_message(f"Unloading Ollama model: {self.model}")
                self.model = None

            # Now clear GPU memory
            clear_gpu_memory()
            self.display_status_message("GPU memory cleared successfully")
        except Exception as e:
            self.display_status_message(f"Failed to clear GPU memory: {str(e)}")

    def schedule_ollama_status_check(self):
        """Schedule periodic Ollama status checks"""
        self.update_ollama_status()
        self.root.after(10000, self.schedule_ollama_status_check)  # Check every 10 seconds
        
    def update_ollama_status(self):
        """Update the Ollama status indicator"""
        current_backend = self.backend_var.get()

        # Only show Ollama status when Ollama backend is selected
        if current_backend != "ollama":
            self.ollama_status_var.set("○ Not Applicable")
            self.ollama_status.config(foreground="gray")
            return

        if not OLLAMA_AVAILABLE:
            self.ollama_status_var.set("⚫ Ollama: Not Installed")
            self.ollama_status.config(foreground="gray")
            return

        previously_available = self.ollama_available
        currently_available = self.check_ollama_available()

        if currently_available:
            self.ollama_status_var.set("🟢 Ollama: Running")
            self.ollama_status.config(foreground="green")

            # If ollama just became available, refresh models
            if not previously_available:
                self.refresh_models()
                self.display_status_message("Ollama is now available. Models refreshed.")
        else:
            self.ollama_status_var.set("🔴 Ollama: Not Running")
            self.ollama_status.config(foreground="red")

    def toggle_system_prompt_editor(self):
        """Toggle the visibility of the system prompt editor"""
        if self.system_prompt_visible.get():
            # Show the prompt editor
            if not self.prompt_frame.winfo_ismapped():
                self.prompt_editor.delete("1.0", tk.END)
                self.prompt_editor.insert(tk.END, self.system_message['content'])
                # Find a suitable parent frame to pack after
                if hasattr(self, 'search_frame'):
                    self.prompt_frame.pack(after=self.search_frame, fill=tk.X, pady=(0, 10))
                else:
                    # Fallback - pack after the main frame's last visible child
                    self.prompt_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            # Hide the prompt editor
            self.hide_system_prompt()
    
    def hide_system_prompt(self):
        """Hide the system prompt editor"""
        self.prompt_frame.pack_forget()
    
    def save_system_prompt(self):
        """Save the edited system prompt"""
        new_prompt = self.prompt_editor.get("1.0", tk.END).strip()
        if new_prompt:
            self.system_message['content'] = new_prompt
            # Update the first message in the history if it's the system message
            if self.messages and self.messages[0]['role'] == 'system':
                self.messages[0]['content'] = new_prompt
            self.display_status_message("System prompt updated.")
        self.hide_system_prompt()
        
    def save_chat(self):
        """Save the current chat conversation to a file"""
        # Ask for file location
        chat_data = {
            "model": self.model_var.get(),
            "backend": self.backend_var.get(),  # Save which backend was used
            "system_message": self.system_message,
            "messages": self.messages,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Save additional state for complete restoration
            "gguf_model_path": getattr(self, 'selected_gguf_path', StringVar()).get() if hasattr(self, 'selected_gguf_path') else "",
            "claude_model": getattr(self, 'claude_model_var', StringVar()).get() if hasattr(self, 'claude_model_var') else "",
            "max_tokens": getattr(self, 'max_tokens_var', DoubleVar()).get() if hasattr(self, 'max_tokens_var') else 65536,
            "temperature": getattr(self, 'temperature_var', DoubleVar()).get() if hasattr(self, 'temperature_var') else 0.7,
        }
        
        # Use current file path if exists, otherwise ask for new one
        filepath = self.current_file
        if not filepath:
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=filename
            )
            
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(chat_data, f, indent=2)
                self.current_file = filepath
                self.display_chat_system_message(f"Chat saved to {os.path.basename(filepath)}")
                self.root.title(f"CodeRunner IDE - {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error Saving Chat", f"Could not save chat: {str(e)}")
    
    def load_chat(self):
        """Load a chat conversation from a file"""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    chat_data = json.load(f)
                
                # Extract data
                loaded_model = chat_data.get("model", model)
                loaded_backend = chat_data.get("backend", "ollama")  # Restore backend
                self.system_message = chat_data.get("system_message", {'role': 'system', 'content': python_system_message})
                loaded_messages = chat_data.get("messages", [self.system_message])

                # Update UI and state
                self.messages = loaded_messages
                self.current_file = filepath
                self.root.title(f"CodeRunner IDE - {os.path.basename(filepath)}")

                # Restore backend if it exists in saved data
                if loaded_backend and loaded_backend in ["ollama", "llama_cpp", "claude", "openai"]:
                    self.backend_var.set(loaded_backend)
                    self.change_backend()  # Apply the backend change

                # Update model if available
                if loaded_model in self.available_models:
                    self.model_var.set(loaded_model)
                    self.model = loaded_model

                # Restore additional state variables if they exist in saved data
                if "gguf_model_path" in chat_data and chat_data["gguf_model_path"]:
                    if hasattr(self, 'selected_gguf_path'):
                        self.selected_gguf_path.set(chat_data["gguf_model_path"])

                if "claude_model" in chat_data and chat_data["claude_model"]:
                    if hasattr(self, 'claude_model_var'):
                        self.claude_model_var.set(chat_data["claude_model"])

                if "max_tokens" in chat_data:
                    if hasattr(self, 'max_tokens_var'):
                        self.max_tokens_var.set(chat_data["max_tokens"])

                if "temperature" in chat_data:
                    if hasattr(self, 'temperature_var'):
                        self.temperature_var.set(chat_data["temperature"])
                
                # Update system prompt mode radio buttons (code-only: Python or HTML)
                system_content = self.system_message['content']
                if system_content == html_system_message:
                    self.system_mode.set("html_programmer")
                else:
                    self.system_mode.set("programmer")
                
                # Clear and reload the chat display - chat is now always editable
                self.chat_display.delete("1.0", tk.END)
                
                # Reset thinking state tracker
                self.previous_thinking_state = not self.hide_thinking.get()  # Inverted logic
                # Reset first message flag since this is a "new" conversation
                self.first_message = True
                
                # Display messages in the chat
                for msg in self.messages:
                    if msg['role'] == 'user':
                        self.display_message("You", msg['content'])
                    elif msg['role'] == 'assistant':
                        # Let display_message handle sender label based on current mode
                        self.display_message("Assistant", msg['content'])
                    # System messages are not displayed to keep the chat clean
                
                # Show what was restored
                restored_info = f"Chat loaded from {os.path.basename(filepath)}"
                if loaded_backend != "ollama":
                    restored_info += f" | Backend: {loaded_backend}"
                if "gguf_model_path" in chat_data and chat_data["gguf_model_path"]:
                    restored_info += f" | GGUF: {os.path.basename(chat_data['gguf_model_path'])}"
                if "claude_model" in chat_data and chat_data["claude_model"]:
                    restored_info += f" | Claude: {chat_data['claude_model']}"

                self.display_status_message(restored_info)
            except Exception as e:
                messagebox.showerror("Error Loading Chat", f"Could not load chat: {str(e)}")
                
    def load_code_to_chat(self):
        """Load code from a file and add it to the chat with proper formatting"""
        filepath = filedialog.askopenfilename(
            defaultextension=".py",
            filetypes=[
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("TypeScript files", "*.ts"),
                ("HTML files", "*.html"),
                ("CSS files", "*.css"),
                ("Java files", "*.java"),
                ("C++ files", "*.cpp"),
                ("C files", "*.c"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # Get file extension to determine language
                file_extension = os.path.splitext(filepath)[1].lower()
                language_map = {
                    '.py': 'python',
                    '.js': 'javascript', 
                    '.ts': 'typescript',
                    '.html': 'html',
                    '.css': 'css',
                    '.java': 'java',
                    '.cpp': 'cpp',
                    '.c': 'c',
                    '.txt': 'text'
                }
                language = language_map.get(file_extension, 'text')
                
                # Format the code for chat
                filename = os.path.basename(filepath)
                formatted_message = f"Here is the code from {filename}:\n\n```{language}\n{code_content}\n```\n\nPlease help me understand or improve this code."
                
                # Add to user input
                self.user_input.delete("1.0", tk.END)
                self.user_input.insert("1.0", formatted_message)
                
                self.display_chat_system_message(f"Code loaded from {filename}. You can edit the message before sending.")
                
            except Exception as e:
                messagebox.showerror("Error Loading Code", f"Could not load code file: {str(e)}")

    def on_enter_pressed(self, event):
        """Handle Enter key press in the input box"""
        if not event.state & 0x1:  # Shift key not pressed
            self.send_message()
            return "break"  # Prevents the newline from being inserted

    def send_message(self):
        """Send the user message and get a response from the model"""
        input_text = self.user_input.get("1.0", tk.END).strip()
        image_to_use = self.current_image
        
        # Don't send empty messages (unless we have an image)
        if not input_text and not image_to_use:
            return
        
        # Check backend availability
        backend = self.backend_var.get()
        
        # Warn about image usage with llama-cpp
        if backend == "llama_cpp" and image_to_use:
            response = messagebox.askyesno(
                "Image with llama-cpp Backend", 
                "You're using llama-cpp backend with an image attached.\n\n"
                "Most llama-cpp models don't support images and may hang or give poor results.\n\n"
                "Would you like to:\n"
                "• Yes: Switch to Ollama backend (recommended for images)\n"
                "• No: Continue with llama-cpp (image will be ignored)"
            )
            if response:  # User chose Yes - switch to Ollama
                self.backend_var.set("ollama")
                self.change_backend()
                backend = "ollama"
        
        if backend == "llama_cpp":
            # Check if GGUF model is loaded
            if not self.llama_cpp_model:
                self.display_status_message("No GGUF model loaded. Please load a model first.")
                return
        elif backend == "ollama":
            # Check if Ollama is available - don't auto-start
            if not self.check_ollama_available():
                self.display_status_message("Ollama service is not running. Please start Ollama manually using '/startollama' command or start it externally.")
                return
        
        # Clear input box
        self.user_input.delete("1.0", tk.END)
        
        # Process commands (only if no image attached)
        if input_text.startswith('/') and not image_to_use:
            self.process_command(input_text)
            return
        
        # Rebuild message history from chat display so user edits (cut/paste) take effect
        self._rebuild_messages_from_chat()

        # Display user message (and image if attached)
        image_msg = " [with image]" if image_to_use else ""
        self.display_message("You", f"{input_text}{image_msg}")
        
        # Add thinking toggle command if the state changed or it's the first message
        current_thinking = not self.hide_thinking.get()  # Inverted logic
        if current_thinking != self.previous_thinking_state or self.first_message:
            if current_thinking:
                input_text += " /think"
            else:
                input_text += " /no_think"
            # Update the previous state
            self.previous_thinking_state = current_thinking
        
        # Mark that we've sent the first message
        self.first_message = False
        
        # If we have an image, save it temporarily and prepare for Ollama
        image_path = None
        image_base64 = None
        if image_to_use:
            try:
                # Save image to a temp file with unique name
                if not self.image_file_path:  # If pasted from clipboard
                    temp_filename = f"temp_image_{uuid.uuid4().hex}.png"
                    temp_dir = tempfile.gettempdir()
                    image_path = os.path.join(temp_dir, temp_filename)
                    image_to_use.save(image_path, "PNG")
                else:
                    image_path = self.image_file_path
                
                # Convert to base64 for Ollama and Claude
                buffered = io.BytesIO()
                image_to_use.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_media_type = "image/png"  # Since we're saving as PNG
                
                # Show image in debug console
                self.add_to_debug_console(f"Image attached: {os.path.basename(image_path) if image_path else 'from clipboard'}")
                
            except Exception as e:
                self.display_status_message(f"Error processing image: {str(e)}")
                self.add_to_debug_console(f"Error processing image: {str(e)}")
                # Continue without image if error
                image_path = None
                image_base64 = None
                image_media_type = None
        else:
            image_base64 = None
            image_media_type = None
        
        # Check if RAG is enabled - get relevant context
        rag_context = None
        if self.rag_enabled.get() and self.current_collection.get():
            try:
                rag_context = self.get_relevant_context(input_text)
                if rag_context:
                    # Log to debug console
                    self.add_to_debug_console("Using RAG to enhance prompt with relevant context")
                    
                    # Modify user's input to include context
                    enhanced_input = f"""I have a question that needs context from my documents.

Relevant context from my knowledge base:
{rag_context}

Based on the above context, please answer: {input_text}"""
                    
                    # Use enhanced input instead
                    display_message = input_text  # For display - keep original
                    input_text = enhanced_input   # For API - use enhanced
            except Exception as e:
                self.add_to_debug_console(f"RAG error: {str(e)}")
                # Continue without RAG if there's an error
                display_message = input_text
        else:
            display_message = input_text
        
        # Add IDE context if code is loaded
        if hasattr(self, 'ide_current_content') and self.ide_current_content and not image_base64:
            should_include_ide_context = False
            
            # Always include in Python Programmer mode
            if self.system_mode.get() == "programmer":
                should_include_ide_context = True
                context_reason = "(Python Programmer mode)"
            else:
                # In other modes, only include if user is asking about code
                code_keywords = ['code', 'function', 'class', 'method', 'variable', 'fix', 'change', 'modify', 'update', 'improve', 'refactor', 'error', 'bug']
                if any(keyword in input_text.lower() for keyword in code_keywords):
                    should_include_ide_context = True
                    context_reason = "(code-related query detected)"
            
            if should_include_ide_context:
                ide_context = f"\n\n[IMPORTANT: Current code already loaded in IDE"
                if self.ide_current_filename:
                    ide_context += f" - {os.path.basename(self.ide_current_filename)}"
                ide_context += f":\n```python\n{self.ide_current_content}\n```\n]"
                
                # Add context to input
                input_text = input_text + ide_context
                
                # Set flag to track that we included IDE context
                self.last_ide_context_included = True
                
                # Notify user that IDE context was included
                self.add_to_debug_console(f"✓ IDE context automatically included in prompt {context_reason}")
        
        # Add search results if available and search mode is enabled
        if hasattr(self, 'search_mode') and self.search_mode.get():
            search_content = self.get_search_results_content()
            if search_content:
                search_context = f"\n\n[SEARCH RESULTS from previous tool calls:\n{search_content}\n\nEND SEARCH RESULTS]"
                input_text = input_text + search_context
                # Previous search results included silently
        
        # Store the original message for history
        if image_base64:
            # For history - using structured format
            history_message = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': input_text},
                    {'type': 'image', 'image': image_base64}
                ]
            }
            
            # For API - using the direct API format
            display_message = input_text
        else:
            # Text-only message - same for both history and display
            history_message = {'role': 'user', 'content': input_text}
            display_message = input_text
        
        # Add user message to history (for future context)
        self.messages.append(history_message)
        
        # Clear image after sending
        if image_to_use:
            self.clear_image()
        
        # Disable input while processing
        self.user_input.config(state=tk.DISABLED)
        self.send_button.pack_forget()
        self.fix_button.pack_forget()
        self.stop_button.pack(side=tk.RIGHT)
        self.status_var.set("Thinking...")
        
        # Reset stop flag
        self.stop_generation = False
        self.generation_active = True
        
        # Start response timer
        self.start_response_timer()

        # Start a thread to get the model's response
        threading.Thread(target=self.get_model_response, args=(image_base64 is not None, image_base64, display_message, image_media_type)).start()

    def send_message_with_search(self):
        """Send message, explicitly including current search results in chat and prompt"""
        # Take current input and append a directive plus the search results content
        input_text = self.user_input.get("1.0", tk.END).strip()
        if not input_text:
            return
        search_content = self.get_search_results_content()
        if search_content:
            directive = "\n\n[USE THE FOLLOWING SEARCH RESULTS IN YOUR ANALYSIS]\n"
            directive += f"{search_content}\n[END SEARCH RESULTS]"
            # Insert directive into the visible chat message so the LLM has context in history
            self.user_input.insert(tk.END, directive)
        # Reuse the normal send flow
        self.send_message()

    def stop_message(self):
        """Stop the current message generation"""
        if self.generation_active:
            self.stop_generation = True
            self.status_var.set("Stopping generation...")

    # ---------- Central dispatch: routes to the active backend's stream method ----------

    def get_model_response(self, has_image=False, image_data=None, display_message=None, image_media_type=None):
        """Get response from the model in a separate thread.
        Dispatches to stream_*_response() based on backend_var, then records token stats."""
        try:
            # Store current restart counter to check if chat was restarted during processing - FIX for restart not clearing
            current_restart_counter = getattr(self, 'restart_counter', 0)
            
            # Determine assistant label based on current radio-button mode ONLY
            mode = self.system_mode.get()
            if mode == "programmer":
                assistant_label = "Py Assistant"
            elif mode == "html_programmer":
                assistant_label = "HTML Assistant"
            else:
                assistant_label = "Assistant"

            # Check if chat was restarted before displaying
            if getattr(self, 'restart_counter', 0) != current_restart_counter:
                return  # Chat was restarted, abandon this response
                
            # Prepare display for assistant's response
            self.display_message(assistant_label, "", end=False)
            
            # Choose backend and get response — with agent loop for tool calls
            backend = self.backend_var.get()
            model_to_use = None  # Initialize to avoid NameError in exception handler
            search_enabled = hasattr(self, 'search_mode') and self.search_mode.get()

            for _agent_turn in range(MAX_AGENT_TURNS):
                # Clear pending tool results before each turn
                self._pending_tool_results = []

                if backend == "llama_cpp":
                    if not self.llama_cpp_model:
                        raise Exception("No GGUF model loaded. Please load a model first.")
                    full_response = self.stream_llama_cpp_response(
                        has_image and _agent_turn == 0, image_data if _agent_turn == 0 else None, display_message)

                elif backend == "claude":
                    if not self.claude_model_var.get():
                        raise Exception("No Claude model selected. Please select a model first.")
                    full_response = self.stream_claude_response(
                        has_image and _agent_turn == 0, image_data if _agent_turn == 0 else None,
                        display_message, image_media_type)

                elif backend == "openai":
                    if not self.model_var.get():
                        raise Exception("No OpenAI model selected. Please select a model first.")
                    full_response = self.stream_openai_response(display_message)

                elif backend == "mlx":
                    if not self.mlx_model and not self.mlx_vlm_model:
                        raise Exception("No MLX model loaded. Please load a model first.")
                    full_response = self.stream_mlx_response(
                        display_message, has_image=has_image and _agent_turn == 0,
                        image_data=image_data if _agent_turn == 0 else None)

                elif backend == "vllm":
                    if not hasattr(self, 'vllm_model') or not self.vllm_model:
                        raise Exception("No vLLM model loaded. Please load a model first.")
                    full_response = self.stream_vllm_response(display_message)

                elif backend == "transformers":
                    if not self.transformers_model or not self.transformers_tokenizer:
                        raise Exception("No Transformers model loaded. Please load a model first.")
                    full_response = self.stream_transformers_response(display_message)

                else:
                    # Ollama backend (default)
                    model_to_use = self.model_var.get()
                    if has_image and image_data and _agent_turn == 0:
                        full_response = self.stream_multimodal_response(model_to_use, image_data, display_message)
                    else:
                        full_response = self.stream_text_response(model_to_use)

                # --- Agent loop: if no tool calls or search off, break ---
                if not self._pending_tool_results or not search_enabled or self.stop_generation:
                    break

                # Execute tool calls and build tool-result messages
                self.display_chat_system_message(f"--- Agent turn {_agent_turn + 1}: executing {len(self._pending_tool_results)} tool call(s) ---")

                # Record assistant response so far
                self.messages.append({'role': 'assistant', 'content': full_response or ''})

                if backend == "claude":
                    # Claude format: assistant content has tool_use blocks, user sends tool_result
                    assistant_content = []
                    if full_response:
                        assistant_content.append({"type": "text", "text": full_response})
                    tool_result_content = []
                    for tc in self._pending_tool_results:
                        assistant_content.append({
                            "type": "tool_use", "id": tc['id'],
                            "name": tc['name'], "input": tc['arguments']
                        })
                        result_text = self._execute_tool_call(tc['name'], tc['arguments'])
                        self.add_to_search_results(f"Tool: {tc['name']}\n\n{result_text[:2000]}")
                        tool_result_content.append({
                            "type": "tool_result", "tool_use_id": tc['id'],
                            "content": result_text[:8000]
                        })
                    # Replace last assistant message with structured content
                    self.messages[-1] = {"role": "assistant", "content": assistant_content}
                    self.messages.append({"role": "user", "content": tool_result_content})
                else:
                    # OpenAI / Ollama format: assistant has tool_calls, then role=tool messages
                    tool_calls_list = []
                    for tc in self._pending_tool_results:
                        tool_calls_list.append({
                            "id": tc['id'],
                            "type": "function",
                            "function": {"name": tc['name'], "arguments": json.dumps(tc['arguments'])}
                        })
                    # Replace last assistant message with tool_calls
                    self.messages[-1] = {
                        "role": "assistant",
                        "content": full_response or '',
                        "tool_calls": tool_calls_list
                    }
                    for tc in self._pending_tool_results:
                        result_text = self._execute_tool_call(tc['name'], tc['arguments'])
                        self.add_to_search_results(f"Tool: {tc['name']}\n\n{result_text[:2000]}")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc['id'],
                            "content": result_text[:8000]
                        })

                # Show continuation in chat
                self.append_to_chat(f"\n\n")
                self.display_message(assistant_label, "", end=False)

            # If we stopped generation early
            if self.stop_generation:
                # Chat is now always editable
                self.chat_display.insert(tk.END, "\n\n[Generation stopped]")
                self.chat_display.see(tk.END)
                
                # Still add what we got to the history
                if full_response:
                    self.messages.append({'role': 'assistant', 'content': full_response})
                # Restore full history if this was a fix request
                if hasattr(self, '_pre_fix_messages') and self._pre_fix_messages is not None:
                    fix_request = self.messages[1]  # The fix prompt
                    self.messages = self._pre_fix_messages
                    self.messages.append(fix_request)
                    if full_response:
                        self.messages.append({'role': 'assistant', 'content': full_response})
                    self._pre_fix_messages = None
            else:
                # Add complete response to history
                self.messages.append({'role': 'assistant', 'content': full_response})
                # Restore full history if this was a fix request
                if hasattr(self, '_pre_fix_messages') and self._pre_fix_messages is not None:
                    fix_request = self.messages[1]  # The fix prompt
                    self.messages = self._pre_fix_messages
                    self.messages.append(fix_request)
                    self.messages.append({'role': 'assistant', 'content': full_response})
                    self._pre_fix_messages = None
                
                # Complete the message display - chat is now always editable
                self.chat_display.insert(tk.END, "\n\n")
                self.chat_display.see(tk.END)
            
        except Exception as e:
            # Handle any errors
            error_msg = str(e)
            # Send error to system console and debug console
            self.display_status_message(f"Model error: {error_msg[:100]}")
            
            # Log full error to debug console for troubleshooting
            self.add_to_debug_console("="*60)
            self.add_to_debug_console("MODEL RESPONSE ERROR:")
            self.add_to_debug_console(f"Error: {error_msg}")
            self.add_to_debug_console(f"Backend: {backend}, Has image: {has_image}")
            
            # Check for specific multimodal errors
            if has_image and ("does not support images" in error_msg.lower() or 
                              "invalid content type" in error_msg.lower() or
                              "unsupported" in error_msg.lower() or
                              "not a vision model" in error_msg.lower() or
                              "content field" in error_msg.lower() or
                              "validation error" in error_msg.lower()):
                # Check if chat was restarted before displaying error - FIX for restart not clearing
                if getattr(self, 'restart_counter', 0) == current_restart_counter:
                    model_name = model_to_use if model_to_use else backend
                    self.display_chat_system_message(f"Model '{model_name}' doesn't support images. Try a multimodal model like llava.")
        
        finally:
            # Record token stats before stopping timer (needs elapsed time)
            try:
                resp = full_response if full_response else ""
            except NameError:
                resp = ""
            self.record_token_stats(resp)

            # Stop response timer
            self.stop_response_timer()

            # Re-enable input
            self.generation_active = False
            self.stop_generation = False
            self.root.after(0, self.enable_input)

    # ---------- Streaming backend methods (one per LLM backend) ----------
    # Each returns the full response text. Called from get_model_response().

    def stream_openai_response(self, prompt_text):
        """Get response from OpenAI chat models, handling GPT-5 vs others.
        - GPT-5: non-streaming, use max_completion_tokens, omit temperature
        - Others: streaming with max_tokens and temperature
        """
        import os, requests
        full_response = ""
        model_name = self.model_var.get()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            key_path = os.path.join(os.path.dirname(__file__), "openai_key.txt")
            if os.path.exists(key_path):
                with open(key_path, "r") as f:
                    api_key = f.read().strip()
        if not api_key:
            raise Exception("Missing OPENAI_API_KEY. Set env var or add openai_key.txt next to this script.")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        is_gpt5 = model_name.startswith("gpt-5")
        payload = {
            "model": model_name,
            "messages": self.messages,
        }
        if is_gpt5:
            payload["max_completion_tokens"] = int(self.max_tokens_var.get())
            # omit temperature
            stream = False  # GPT-5 models don't support streaming (organization verification required)
        else:
            payload["max_tokens"] = int(min(max(self.max_tokens_var.get(), 1), 100000))
            payload["temperature"] = float(self.temperature.get())
            stream = True
        payload["stream"] = stream

        url = "https://api.openai.com/v1/chat/completions"
        using_responses_api = False
        if stream:
            # Streaming path
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    # Check if this is the specific error about model only supported in v1/responses
                    if resp.status_code == 404 and "only supported in v1/responses" in resp.text:
                        # Retry with responses endpoint - adapt payload for Responses API
                        url = "https://api.openai.com/v1/responses"
                        using_responses_api = True
                        responses_payload = payload.copy()
                        responses_payload["input"] = responses_payload.pop("messages")  # Change 'messages' to 'input'
                        # Change parameter names for Responses API
                        if "max_completion_tokens" in responses_payload:
                            responses_payload["max_output_tokens"] = responses_payload.pop("max_completion_tokens")
                        if "max_tokens" in responses_payload:
                            responses_payload["max_output_tokens"] = responses_payload.pop("max_tokens")
                        # Remove unsupported parameters for Responses API
                        responses_payload.pop("temperature", None)
                        responses_payload.pop("stream", None)
                        self.display_status_message(f"🔄 Retrying {model_name} with Responses API")
                        with requests.post(url, headers=headers, json=responses_payload, timeout=180) as retry_resp:
                            self.display_status_message(f"📡 Responses API status: {retry_resp.status_code}")
                            if retry_resp.status_code != 200:
                                self.display_status_message(f"❌ Responses API error: {retry_resp.text[:200]}...")
                                raise Exception(f"OpenAI error: {retry_resp.status_code} - {retry_resp.text}")
                            resp = retry_resp
                    else:
                        raise Exception(f"OpenAI error: {resp.status_code} - {resp.text}")

                if using_responses_api:
                    # For Responses API, parse as non-streaming even in streaming path
                    data = resp.json()
                    if "content" in data and len(data["content"]) > 0:
                        content_item = data["content"][0]
                        if content_item.get("type") == "text":
                            full_response = content_item["text"]["value"]
                            if full_response:
                                self.append_to_chat(full_response)
                else:
                    # Original streaming logic for chat completions
                    buffer = ""
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                import json as _json
                                obj = _json.loads(data)
                                delta = obj["choices"][0]["delta"].get("content")
                                if delta:
                                    full_response += delta
                                    self.append_to_chat(delta)
                            except Exception:
                                continue
        else:
            # Non-streaming path
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                # Check if this is the specific error about model only supported in v1/responses
                if resp.status_code == 404 and "only supported in v1/responses" in resp.text:
                    # Retry with responses endpoint - adapt payload for Responses API
                    url = "https://api.openai.com/v1/responses"
                    using_responses_api = True
                    responses_payload = payload.copy()
                    responses_payload["input"] = responses_payload.pop("messages")  # Change 'messages' to 'input'
                    # Change parameter names for Responses API
                    if "max_completion_tokens" in responses_payload:
                        responses_payload["max_output_tokens"] = responses_payload.pop("max_completion_tokens")
                    if "max_tokens" in responses_payload:
                        responses_payload["max_output_tokens"] = responses_payload.pop("max_tokens")
                    # Remove unsupported parameters for Responses API
                    responses_payload.pop("temperature", None)
                    responses_payload.pop("stream", None)
                    self.display_status_message(f"🔄 Retrying {model_name} with Responses API")
                    resp = requests.post(url, headers=headers, json=responses_payload, timeout=180)
                    self.display_status_message(f"📡 Responses API status: {resp.status_code}")
                    if resp.status_code != 200:
                        self.display_status_message(f"❌ Responses API error: {resp.text[:200]}...")
                        raise Exception(f"OpenAI error: {resp.status_code} - {resp.text}")
                else:
                    raise Exception(f"OpenAI error: {resp.status_code} - {resp.text}")
            data = resp.json()
            if using_responses_api:
                # Responses API format
                self.display_status_message(f"📄 Processing Responses API data (keys: {list(data.keys())})")
                if "content" in data and len(data["content"]) > 0:
                    content_item = data["content"][0]
                    if content_item.get("type") == "text":
                        full_response = content_item["text"]["value"]
                    else:
                        full_response = ""
                else:
                    full_response = ""
            else:
                # Chat completions format
                full_response = data["choices"][0]["message"].get("content", "")
            if full_response:
                self.append_to_chat(full_response)

        return full_response

    def stream_text_response(self, model_to_use):
        """Stream a text-only response using the Python client"""
        full_response = ""
        last_visible_update = 0

        # Check if ollama is available
        if not OLLAMA_AVAILABLE:
            self.display_chat_system_message("❌ Ollama is not installed. Please install ollama or switch to another backend.")
            return

        # Get the message display widget position for appending text - chat is now always editable
        end_position = self.chat_display.index(tk.END)

        try:
            # Verify Ollama is available
            if not self.check_ollama_available():
                raise Exception("Ollama service is not available")
                
            # Prepare API call parameters
            api_params = {
                "model": model_to_use,
                "messages": self.messages,
                "stream": True,
                "options": {"temperature": self.temperature.get()}
            }

            # Add think parameter if DO thinking is enabled
            if self.do_thinking.get():
                api_params["think"] = True
            
            # Add tools if search mode is enabled
            if hasattr(self, 'search_mode') and self.search_mode.get():
                api_params["tools"] = get_mcp_tool_definitions("openai", self.mcp_client)
                self.display_chat_system_message("🔍 Web search tools enabled")

            # Start streaming with ollama Python client
            try:
                stream = ollama.chat(**api_params)
            except Exception as e:
                # If the model doesn't support tools, try without them
                if "does not support tools" in str(e) and hasattr(self, 'search_mode') and self.search_mode.get():
                    self.display_chat_system_message("⚠️ This model doesn't support tool calling")
                    api_params.pop("tools", None)
                    stream = ollama.chat(**api_params)
                else:
                    raise

            # Initialize thinking state tracking
            in_thinking_block = False
            pending_buffer = ""

            # Process each chunk as it arrives
            for chunk in stream:
                # Check if we should stop generation
                if self.stop_generation:
                    try:
                        stream.close()
                    except:
                        pass
                    break

                # Check for tool calls in the chunk — collect for agent loop
                if 'message' in chunk and 'tool_calls' in chunk['message'] and chunk['message']['tool_calls']:
                    tool_calls = chunk['message']['tool_calls']
                    for tool_call in tool_calls:
                        func_name = tool_call.get('function', {}).get('name', '')
                        arguments = tool_call.get('function', {}).get('arguments', {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        self.display_chat_system_message(f"🔍 Tool call: {func_name}({json.dumps(arguments)[:100]})")
                        self._pending_tool_results.append({
                            'name': func_name,
                            'arguments': arguments,
                            'id': tool_call.get('id', str(uuid.uuid4())),
                            'backend': 'ollama'
                        })

                chunk_text = chunk['message'].get('content', '')
                if chunk_text:  # Only add if there's actual content
                    full_response += chunk_text

                    # Track thinking tags for post-processing
                    if '<think>' in chunk_text or '<thinking>' in chunk_text:
                        has_thinking_tags = True

                # Handle thinking filtering with state tracking
                if chunk_text and self.hide_thinking.get():
                    # Process chunk with thinking state
                    display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                        chunk_text, in_thinking_block, pending_buffer
                    )

                    # Display filtered content
                    if display_text:
                        self.append_to_chat(display_text)
                elif chunk_text:
                    # No filtering needed - display everything
                    self.append_to_chat(chunk_text)

                last_visible_update = len(full_response)
        except Exception as e:
            error_msg = f"\n[Error during generation: {str(e)}]"
            self.append_to_chat(error_msg)
            return error_msg

        # Clean up response for storage
        clean_response = full_response.strip()

        # Thinking content is now filtered in real-time during streaming

        return clean_response

    def stream_multimodal_response(self, model_to_use, image_data, prompt_text):
        """Stream a multimodal response using direct API calls"""
        full_response = ""
        
        try:
            # Prepare the request payload
            import requests
            import json
            
            # Format the context from previous messages
            context = []
            for msg in self.messages[:-1]:  # Exclude the last message which has the image
                if msg['role'] == 'user':
                    context.append({"role": "user", "content": msg['content'] if isinstance(msg['content'], str) else "..."})
                elif msg['role'] == 'assistant':
                    context.append({"role": "assistant", "content": msg['content']})
                # Skip system messages for the API call
            
            # Create the API payload - using the documented format for vision models
            payload = {
                "model": model_to_use,
                "stream": True,
                "options": {"temperature": self.temperature.get()},
                "messages": context + [{
                    "role": "user",
                    "content": prompt_text,
                    "images": [image_data]
                }]
            }

            # Add think parameter if DO thinking is enabled
            if self.do_thinking.get():
                payload["think"] = True
            
            # Make the API request
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                stream=True,
                timeout=60
            )
            
            # Initialize thinking state tracking
            in_thinking_block = False
            pending_buffer = ""

            # Handle the streaming response
            if response.status_code == 200:
                for line in response.iter_lines():
                    if self.stop_generation:
                        break

                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                chunk_text = chunk['message']['content']
                                full_response += chunk_text

                                # Track thinking tags for post-processing
                                if '<think>' in chunk_text or '<thinking>' in chunk_text:
                                    has_thinking_tags = True

                                # Handle thinking filtering with state tracking
                                if self.hide_thinking.get():
                                    # Process chunk with thinking state
                                    display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                                        chunk_text, in_thinking_block, pending_buffer
                                    )

                                    # Display filtered content
                                    if display_text:
                                        self.append_to_chat(display_text)
                                else:
                                    # No filtering needed - display everything
                                    self.append_to_chat(chunk_text)

                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
            else:
                error_text = response.text
                self.append_to_chat(f"\n[API Error: {response.status_code} - {error_text}]")
                raise Exception(f"API error: {response.status_code} - {error_text}")

        except Exception as e:
            error_msg = f"\n[Error during multimodal generation: {str(e)}]"
            self.append_to_chat(error_msg)
            return error_msg

        # Clean up response for storage
        clean_response = full_response.strip()

        # Thinking content is now filtered in real-time during streaming

        return clean_response

    def stream_llama_cpp_response(self, has_image=False, image_data=None, display_message=None):
        """Stream response using GGUF backend"""
        full_response = ""
        last_visible_update = 0
        
        try:
            if has_image and image_data:
                # Check if this model supports images - most llama-cpp models don't
                self.display_chat_system_message("⚠️ Warning: Most llama-cpp models don't support images. Consider switching to Ollama backend with a multimodal model (like llava) for image analysis.", end=True)
                # For now, ignore the image and process as text-only
                has_image = False
                image_data = None
                
            if has_image and image_data:
                # For multimodal with GGUF, we need to prepare the message format
                # Create messages in ChatML format with image
                messages = []
                for msg in self.messages[:-1]:  # Exclude the last message with image
                    if msg['role'] == 'user':
                        content = msg['content'] if isinstance(msg['content'], str) else "..."
                        messages.append({"role": "user", "content": content})
                    elif msg['role'] == 'assistant':
                        messages.append({"role": "assistant", "content": msg['content']})
                
                # Add the current message with image
                messages.append({
                    "role": "user", 
                    "content": display_message,
                    "images": [image_data]
                })
                
                # Use create_chat_completion with images
                max_tokens = int(self.max_tokens_var.get())
                # Ensure temperature is not too low to prevent infinite loops
                # For code: use 0.0-0.3 for deterministic, 0.4-0.7 for creative solutions
                temp_value = max(self.temperature.get(), 0.05)  # Allow near-0 but prevent infinite loops

                api_params = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temp_value,  # Prevent infinite loops with minimum temp
                    "top_p": 0.8,  # Qwen recommended setting
                    "top_k": 20,  # Qwen recommended setting
                    "min_p": 0.01,  # Qwen recommended setting (optional)
                    "repeat_penalty": 1.05,  # Qwen recommended setting
                    "stream": True,
                    "stop": ["</s>", "<|endoftext|>", "<|eot_id|>", "\n\nUser:", "\n\nAssistant:"]  # Add proper stop sequences
                }

                # Add think parameter if DO thinking is enabled (llama-cpp may ignore this)
                if self.do_thinking.get():
                    api_params["think"] = True

                # Add tools if search mode is enabled
                if hasattr(self, 'search_mode') and self.search_mode.get():
                    api_params["tools"] = get_mcp_tool_definitions("openai", self.mcp_client)
                    self.display_chat_system_message("🔍 Web search tools enabled")

                try:
                    response = self.llama_cpp_model.create_chat_completion(**api_params)
                except (RuntimeError, MemoryError, Exception) as e:
                    error_msg = str(e).lower()
                    if "memory" in error_msg or "metal" in error_msg or "gpu" in error_msg or "insufficient" in error_msg:
                        self.display_chat_system_message("❌ This model doesn't support images. Please switch to Ollama backend with a multimodal model (like llava) or use a text-only model.")
                        return ""
                    else:
                        self.display_chat_system_message(f"❌ Model error: {str(e)[:100]}...")
                        return ""

                # Process streaming response
                for chunk in response:
                    if self.stop_generation:
                        break

                    # Check if the response is finished
                    if chunk.get('choices') and chunk['choices'][0].get('finish_reason'):
                        finish_reason = chunk['choices'][0]['finish_reason']
                        if finish_reason == 'length':
                            self.append_to_chat(f"\n[Response truncated - reached {max_tokens} token limit]")
                        break

                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            chunk_text = delta['content']
                            full_response += chunk_text

                            # Always stream everything during generation
                            self.append_to_chat(chunk_text)
                            last_visible_update = len(full_response)

            else:
                # Text-only response using create_chat_completion
                max_tokens = int(self.max_tokens_var.get())

                # Ensure temperature is not too low to prevent infinite loops
                # For code: use 0.0-0.3 for deterministic, 0.4-0.7 for creative solutions
                temp_value = max(self.temperature.get(), 0.05)  # Allow near-0 but prevent infinite loops

                api_params = {
                    "messages": self.messages,
                    "max_tokens": max_tokens,
                    "temperature": temp_value,  # Prevent infinite loops with minimum temp
                    "top_p": 0.8,  # Qwen recommended setting
                    "top_k": 20,  # Qwen recommended setting
                    "min_p": 0.01,  # Qwen recommended setting (optional)
                    "repeat_penalty": 1.05,  # Qwen recommended setting
                    "stream": True,
                    "stop": ["</s>", "<|endoftext|>", "<|eot_id|>", "\n\nUser:", "\n\nAssistant:"]  # Add proper stop sequences
                }

                # Add think parameter if DO thinking is enabled (llama-cpp may ignore this)
                if self.do_thinking.get():
                    api_params["think"] = True

                # Add tools if search mode is enabled
                if hasattr(self, 'search_mode') and self.search_mode.get():
                    api_params["tools"] = get_mcp_tool_definitions("openai", self.mcp_client)
                    self.display_chat_system_message("🔍 Web search tools enabled")

                try:
                    response = self.llama_cpp_model.create_chat_completion(**api_params)
                except (RuntimeError, MemoryError, Exception) as e:
                    error_msg = str(e).lower()
                    if "memory" in error_msg or "metal" in error_msg or "gpu" in error_msg or "insufficient" in error_msg:
                        self.display_chat_system_message("❌ This model doesn't support images. Please switch to Ollama backend with a multimodal model (like llava) or use a text-only model.")
                        return ""
                    else:
                        self.display_chat_system_message(f"❌ Model error: {str(e)[:100]}...")
                        return ""

                # Process streaming response
                for chunk in response:
                    if self.stop_generation:
                        break

                    # Check if the response is finished
                    if chunk.get('choices') and chunk['choices'][0].get('finish_reason'):
                        finish_reason = chunk['choices'][0]['finish_reason']
                        if finish_reason == 'length':
                            self.append_to_chat(f"\n[Response truncated - reached {max_tokens} token limit]")
                        break

                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})

                        # Check for tool calls in llama-cpp response — collect for agent loop
                        if 'tool_calls' in delta and delta['tool_calls']:
                            for tool_call in delta['tool_calls']:
                                func_name = tool_call.get('function', {}).get('name', '')
                                arguments = tool_call.get('function', {}).get('arguments', {})
                                if isinstance(arguments, str):
                                    try:
                                        arguments = json.loads(arguments)
                                    except json.JSONDecodeError:
                                        arguments = {}
                                self.display_chat_system_message(f"🔍 Tool call: {func_name}({json.dumps(arguments)[:100]})")
                                self._pending_tool_results.append({
                                    'name': func_name,
                                    'arguments': arguments,
                                    'id': tool_call.get('id', str(uuid.uuid4())),
                                    'backend': 'llama_cpp'
                                })

                        if 'content' in delta:
                            chunk_text = delta['content']
                            full_response += chunk_text

                            # Always stream everything during generation
                            self.append_to_chat(chunk_text)
                            last_visible_update = len(full_response)
        
        except Exception as e:
            error_msg = f"\n[Error during llama-cpp generation: {str(e)}]"
            self.append_to_chat(error_msg)
            return error_msg

        # Clean up response for storage
        clean_response = full_response.strip()

        # For thinking models, hide thinking if checkbox is off or user asks to hide it
        display_message = getattr(self, 'current_user_message', '')  # Get the current user message for phrase checking
        hide_thinking = self.hide_thinking.get()

        has_thinking_tags = ("<think>" in clean_response and "</think>" in clean_response) or ("<thinking>" in clean_response and "</thinking>" in clean_response)

        if hide_thinking and has_thinking_tags:
            # User wants to hide thinking - extract only the final answer
            if "</think>" in clean_response:
                think_end = clean_response.find("</think>") + len("</think>")
                final_answer = clean_response[think_end:].strip()
            elif "</thinking>" in clean_response:
                think_end = clean_response.find("</thinking>") + len("</thinking>")
                final_answer = clean_response[think_end:].strip()
            else:
                final_answer = clean_response  # Fallback

            if final_answer:
                clean_response = final_answer

        return clean_response

    def set_mlx_sampling(self, temp=0.15, top_p=0.95, top_k=40, repetition_penalty=1.08, repetition_context_size=64):
        """Set up MLX sampling parameters"""
        sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
        procs = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            logit_bias={}
        )
        return sampler, procs

    def stream_mlx_response(self, display_message=None, has_image=False, image_data=None):
        """Stream response using MLX backend

        Supports all MLX models including MiniMax models (mlx-community/MiniMax-M2-mlx-8bit-gs32).
        When has_image=True and a VLM model is loaded, uses mlx-vlm for vision processing.
        """
        # ---------- VLM image path ----------
        if has_image and image_data and self.mlx_is_vlm and self.mlx_vlm_model and MLX_VLM_AVAILABLE:
            try:
                return self._stream_mlx_vlm_response(display_message, image_data)
            except Exception as vlm_err:
                self.display_status_message(
                    f"Image processing failed ({vlm_err}) — retrying without image...")
                # Fall through to text-only generation below

        if has_image and image_data and not self.mlx_is_vlm:
            self.display_status_message(
                "This model doesn't support images (text-only). Image will be ignored.")

        # ---------- VLM text-only fallback ----------
        # When a VLM-only model is loaded and there's no image, use the VLM
        # model for text generation (VLM models handle text-only prompts fine).
        if (not self.mlx_model or not self.mlx_tokenizer) and self.mlx_vlm_model and MLX_VLM_AVAILABLE:
            return self._stream_mlx_vlm_text_only(display_message)

        # ---------- Standard text-only path ----------
        if not self.mlx_model or not self.mlx_tokenizer:
            error_msg = "Text-only MLX model not available. Please load a model first."
            self.display_status_message(error_msg)
            return error_msg

        full_response = ""
        assistant_response_start = None  # Track where assistant response starts in chat display
        has_thinking_tags = False  # Track if response contains thinking tags
        try:
            # Format messages for MLX model
            # Check if tokenizer has a chat_template (e.g., Devstral-2-123B-Instruct)
            mlx_tools_enabled = hasattr(self, 'search_mode') and self.search_mode.get()

            if hasattr(self.mlx_tokenizer, 'chat_template') and self.mlx_tokenizer.chat_template is not None:
                # Build messages list for chat template
                messages = []

                # Add system message if available
                if self.system_message and self.system_message.get('content'):
                    messages.append({"role": "system", "content": self.system_message['content']})

                # Add conversation history (including tool results as user messages for MLX)
                for msg in self.messages:
                    if msg['role'] in ['user', 'assistant']:
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            messages.append({"role": msg['role'], "content": content})
                    elif msg.get('role') == 'tool':
                        # MLX doesn't have a tool role — inject as user message
                        messages.append({"role": "user", "content": f"[Tool result]: {msg.get('content', '')}"})

                # Add current user message
                if display_message:
                    messages.append({"role": "user", "content": display_message})
                elif self.messages and self.messages[-1]['role'] == 'user':
                    # Already in messages, don't duplicate
                    pass

                # Build tool definitions for Qwen3 chat template (if search enabled)
                mlx_tool_defs = None
                if mlx_tools_enabled:
                    mlx_tool_defs = []
                    for td in get_mcp_tool_definitions("openai", self.mcp_client):
                        mlx_tool_defs.append(td)
                    self.display_chat_system_message("🔍 Web search tools enabled (MLX text parsing)")

                # Apply chat template — pass tools if available (Qwen3 templates support this)
                try:
                    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
                    if mlx_tool_defs:
                        template_kwargs["tools"] = mlx_tool_defs
                    prompt = self.mlx_tokenizer.apply_chat_template(
                        messages, **template_kwargs
                    )
                except TypeError:
                    # Fallback: tokenizer doesn't accept tools kwarg
                    prompt = self.mlx_tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
            else:
                # Use standard formatting for models without chat_template
                prompt = self.format_messages_for_mlx(display_message)
                
                if not prompt:
                    # First message - use simple format
                    if display_message:
                        prompt = f"<|im_start|>user\n{display_message}<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        prompt = f"<|im_start|>user\n{self.messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"

            # Log prompt length to system console (not debug — debug is for runtime errors only)
            self.display_system_message(f"MLX prompt: {len(prompt)} chars")

            # Stream the response in real-time with live display
            response = ""

            # Set up sampling parameters
            sampler, procs = self.set_mlx_sampling(
                temp=self.temperature.get(),
                top_p=self.top_p.get(),
                top_k=self.top_k.get(),
                repetition_penalty=1.08,
                repetition_context_size=64
            )

            # Initialize thinking state tracking
            in_thinking_block = False
            pending_buffer = ""
            
            # MLX LONG INFERENCE FIX: Track tokens and clear Metal cache periodically
            # to prevent memory fragmentation that causes abort() on long generations
            token_count = 0

            for response_chunk in stream_generate(
                self.mlx_model,
                self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=int(self.max_tokens_var.get()),
                sampler=sampler,              # controls temp / top-p / top-k
                logits_processors=procs,      # controls repetition penalty
            ):
                if self.stop_generation:
                    break
                
                token_count += 1
                
                # MLX LONG INFERENCE FIX: Clear Metal cache every 500 tokens
                # to prevent memory accumulation that triggers abort() in libmlx
                if token_count % 500 == 0:
                    mx.metal.clear_cache()

                # Accumulate the response
                chunk_text = response_chunk.text
                response += chunk_text
                full_response += chunk_text

                # Track thinking tags for post-processing
                if '<think>' in chunk_text or '<thinking>' in chunk_text:
                    has_thinking_tags = True

                # Handle thinking filtering with state tracking
                if self.hide_thinking.get():
                    # Process chunk with thinking state
                    display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                        chunk_text, in_thinking_block, pending_buffer
                    )

                    # Track start of assistant response
                    if assistant_response_start is None and display_text.strip():
                        assistant_response_start = self.chat_display.index(tk.END)

                    # Display filtered content
                    if display_text:
                        self.append_to_chat(display_text)
                else:
                    # No filtering needed - display everything
                    # Track start of assistant response
                    if assistant_response_start is None and chunk_text.strip():
                        assistant_response_start = self.chat_display.index(tk.END)

                    # Display the chunk
                    self.append_to_chat(chunk_text)

            # Clean up response - remove <|im_end|> if present
            if "<|im_end|>" in response:
                clean_response = response.split("<|im_end|>")[0].strip()
            else:
                clean_response = response.strip()

            # Parse tool calls from MLX text output
            if mlx_tools_enabled:
                clean_response = self._parse_mlx_tool_calls(clean_response)

            # Thinking content is now filtered in real-time during streaming

            return clean_response

        except Exception as e:
            error_msg = f"Error during MLX generation: {str(e)}"
            self.append_to_chat(f"\n[{error_msg}]")
            return error_msg

    def _stream_mlx_vlm_response(self, display_message, image_data):
        """Stream a VLM response using mlx-vlm with an attached image.

        Converts base64 image_data to a PIL Image, builds a chat-template prompt
        with the image placeholder, and streams the generation.
        Returns the full response text.
        """
        import base64
        import tempfile
        from PIL import Image

        # Decode base64 image data and save to a temp file —
        # vlm_stream_generate expects a file path string, not a PIL Image
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_data)

        temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_img.write(raw_bytes)
        temp_img.close()
        image_path = temp_img.name

        # Log dimensions for the user
        pil_image = Image.open(image_path)
        self.display_status_message(f"Image attached: {pil_image.size[0]}x{pil_image.size[1]} — sending to VLM...")
        pil_image.close()

        # Build messages list — vlm_apply_chat_template expects plain string content
        # and inserts image tokens automatically via get_message_json for the first user message
        messages = []
        if self.system_message and self.system_message.get('content'):
            messages.append({"role": "system", "content": self.system_message['content']})

        # Add conversation history (text only — prior images are not resent)
        for msg in self.messages:
            if msg['role'] in ['user', 'assistant']:
                messages.append({"role": msg['role'], "content": msg['content']})

        # Current user message (plain text — image token inserted by apply_chat_template)
        if display_message:
            messages.append({"role": "user", "content": display_message})

        # Apply the VLM chat template — num_images=1 tells it to insert image placeholder
        prompt = vlm_apply_chat_template(
            self.mlx_vlm_processor, config=self.mlx_vlm_model.config,
            prompt=messages, num_images=1
        )

        self.display_status_message(f"VLM prompt: {len(prompt)} chars — streaming response with image...")

        # Stream generation
        full_response = ""
        assistant_response_start = None
        in_thinking_block = False
        pending_buffer = ""
        token_count = 0

        for response_chunk in vlm_stream_generate(
            self.mlx_vlm_model,
            self.mlx_vlm_processor,
            prompt=prompt,
            image=image_path,
            max_tokens=int(self.max_tokens_var.get()),
            temperature=self.temperature.get(),
        ):
            if self.stop_generation:
                break

            token_count += 1
            if token_count % 500 == 0:
                mx.metal.clear_cache()

            chunk_text = response_chunk.text
            full_response += chunk_text

            # Handle thinking filtering
            if self.hide_thinking.get():
                display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                    chunk_text, in_thinking_block, pending_buffer
                )
                if assistant_response_start is None and display_text.strip():
                    assistant_response_start = self.chat_display.index(tk.END)
                if display_text:
                    self.append_to_chat(display_text)
            else:
                if assistant_response_start is None and chunk_text.strip():
                    assistant_response_start = self.chat_display.index(tk.END)
                self.append_to_chat(chunk_text)

        # Clean up temp image file
        try:
            os.unlink(image_path)
        except OSError:
            pass

        # Clean up
        if "<|im_end|>" in full_response:
            clean_response = full_response.split("<|im_end|>")[0].strip()
        else:
            clean_response = full_response.strip()

        return clean_response

    def _stream_mlx_vlm_text_only(self, display_message):
        """Stream a text-only response using the VLM model (no image).

        VLM models can generate text without an image — we just build the
        prompt with num_images=0 and skip the image argument.
        """
        # Build messages list
        messages = []
        if self.system_message and self.system_message.get('content'):
            messages.append({"role": "system", "content": self.system_message['content']})

        for msg in self.messages:
            if msg['role'] in ['user', 'assistant']:
                content = msg.get('content', '')
                if isinstance(content, str):
                    messages.append({"role": msg['role'], "content": content})
            elif msg.get('role') == 'tool':
                messages.append({"role": "user", "content": f"[Tool result]: {msg.get('content', '')}"})

        if display_message:
            messages.append({"role": "user", "content": display_message})

        # Build tool definitions for Qwen3 chat template (if search enabled)
        mlx_tools_enabled = hasattr(self, 'search_mode') and self.search_mode.get()
        mlx_tool_defs = None
        if mlx_tools_enabled:
            mlx_tool_defs = get_mcp_tool_definitions("openai", self.mcp_client)
            self.display_chat_system_message("🔍 Web search tools enabled (MLX/VLM text parsing)")

        # Apply chat template — num_images=0 for text-only
        try:
            template_kwargs = {"num_images": 0}
            if mlx_tool_defs:
                template_kwargs["tools"] = mlx_tool_defs
            prompt = vlm_apply_chat_template(
                self.mlx_vlm_processor, config=self.mlx_vlm_model.config,
                prompt=messages, **template_kwargs
            )
        except TypeError:
            # Fallback: some VLM templates may not accept num_images=0
            prompt = vlm_apply_chat_template(
                self.mlx_vlm_processor, config=self.mlx_vlm_model.config,
                prompt=messages, num_images=1
            )

        self.display_status_message(f"VLM text-only prompt: {len(prompt)} chars")

        # Stream generation (no image argument)
        full_response = ""
        in_thinking_block = False
        pending_buffer = ""
        token_count = 0

        for response_chunk in vlm_stream_generate(
            self.mlx_vlm_model,
            self.mlx_vlm_processor,
            prompt=prompt,
            max_tokens=int(self.max_tokens_var.get()),
            temperature=self.temperature.get(),
        ):
            if self.stop_generation:
                break

            token_count += 1
            if token_count % 500 == 0:
                mx.metal.clear_cache()

            chunk_text = response_chunk.text
            full_response += chunk_text

            # Handle thinking filtering
            if self.hide_thinking.get():
                display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                    chunk_text, in_thinking_block, pending_buffer
                )
                if display_text:
                    self.append_to_chat(display_text)
            else:
                self.append_to_chat(chunk_text)

        # Clean up
        if "<|im_end|>" in full_response:
            clean_response = full_response.split("<|im_end|>")[0].strip()
        else:
            clean_response = full_response.strip()

        # Parse tool calls from text output
        if mlx_tools_enabled:
            clean_response = self._parse_mlx_tool_calls(clean_response)

        return clean_response

    def _process_chunk_with_thinking(self, chunk_text, in_thinking_block, pending_buffer):
        """Process a chunk of text with proper thinking tag handling.

        Returns: (display_text, new_in_thinking_block, new_pending_buffer)
        """
        # Add chunk to pending buffer
        pending_buffer += chunk_text

        display_parts = []
        remaining_buffer = pending_buffer

        while remaining_buffer:
            if not in_thinking_block:
                # Look for start of thinking block
                think_start = remaining_buffer.find('<think>')
                thinking_start = remaining_buffer.find('<thinking>')

                if think_start != -1 and (thinking_start == -1 or think_start <= thinking_start):
                    # Found <think> tag first
                    display_parts.append(remaining_buffer[:think_start])
                    in_thinking_block = True
                    remaining_buffer = remaining_buffer[think_start + 7:]  # Skip "<think>"
                elif thinking_start != -1:
                    # Found <thinking> tag first
                    display_parts.append(remaining_buffer[:thinking_start])
                    in_thinking_block = True
                    remaining_buffer = remaining_buffer[thinking_start + 10:]  # Skip "<thinking>"
                else:
                    # No thinking tags found, display everything
                    display_parts.append(remaining_buffer)
                    remaining_buffer = ""
            else:
                # We're inside a thinking block, look for end
                think_end = remaining_buffer.find('</think>')
                thinking_end = remaining_buffer.find('</thinking>')

                if think_end != -1 and (thinking_end == -1 or think_end <= thinking_end):
                    # Found </think> tag first
                    in_thinking_block = False
                    remaining_buffer = remaining_buffer[think_end + 8:]  # Skip "</think>"
                elif thinking_end != -1:
                    # Found </thinking> tag first
                    in_thinking_block = False
                    remaining_buffer = remaining_buffer[thinking_end + 11:]  # Skip "</thinking>"
                else:
                    # End tag not found yet, buffer for next chunk
                    break

        # Update pending buffer with unprocessed content
        pending_buffer = remaining_buffer

        # Combine display parts
        display_text = ''.join(display_parts)

        return display_text, in_thinking_block, pending_buffer

    def format_messages_for_mlx(self, display_message=None):
        """Format messages for MLX model input"""
        formatted_messages = []

        # Add the current system message (which changes based on mode: Python, HTML, Helpful, etc.)
        if self.system_message and self.system_message.get('content'):
            system_content = self.system_message['content']

            formatted_messages.append(f"<|im_start|>system\n{system_content}<|im_end|>")

        for msg in self.messages:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == 'assistant':
                formatted_messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add the current user message if provided
        if display_message:
            formatted_messages.append(f"<|im_start|>user\n{display_message}<|im_end|>")
            return "\n".join(formatted_messages) + "\n<|im_start|>assistant\n"

        # Add the current user message from messages if not provided
        elif self.messages:
            return "\n".join(formatted_messages) + "\n<|im_start|>assistant\n"
        else:
            # Fallback to helpful system message if no system message is set
            fallback_system = self.system_message['content'] if self.system_message else python_system_message
            return f"<|im_start|>system\n{fallback_system}<|im_end|>\n<|im_start|>assistant\n"

    def stream_vllm_response(self, display_message=None):
        """Stream response using vLLM backend"""
        full_response = ""
        assistant_response_start = None  # Track where assistant response starts in chat display
        has_thinking_tags = False  # Track if response contains thinking tags
        try:
            # Format messages for vLLM (similar to Ollama format)
            messages = []

            # Add system message if available
            if self.system_message and self.system_message.get('content'):
                messages.append({"role": "system", "content": self.system_message['content']})

            # Add conversation history
            for msg in self.messages:
                if msg['role'] in ['user', 'assistant', 'system']:
                    if isinstance(msg['content'], str):
                        messages.append({"role": msg['role'], "content": msg['content']})
                    elif isinstance(msg['content'], list):
                        # Handle multimodal content (though vLLM may not support images directly)
                        # For now, extract text only
                        text_content = ""
                        for item in msg['content']:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_content += item.get('text', '')
                        if text_content:
                            messages.append({"role": msg['role'], "content": text_content})

            # Convert messages to string format for vLLM
            prompt = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "  # Start the assistant's response

            # DEBUG: Show final prompt being sent to vLLM
            self.add_to_debug_console("="*30)
            self.add_to_debug_console("FINAL PROMPT SENT TO VLLM:")
            self.add_to_debug_console("="*30)
            if len(prompt) > 500:
                self.add_to_debug_console(f"Length: {len(prompt)} chars")
                self.add_to_debug_console("First 500 chars:")
                self.add_to_debug_console(prompt[:500])
                self.add_to_debug_console("...")
            else:
                self.add_to_debug_console(prompt)
            self.add_to_debug_console("="*30)

            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=self.temperature.get(),
                top_p=self.top_p.get(),
                max_tokens=int(self.max_tokens_var.get()),
                stop=["</s>", "<|endoftext|>", "<|eot_id|>", "\n\nUser:", "\n\nAssistant:"]  # Common stop sequences
            )

            # Generate response using vLLM
            outputs = self.vllm_model.generate([prompt], sampling_params, use_tqdm=False)

            # Initialize thinking state tracking
            in_thinking_block = False
            pending_buffer = ""

            # Stream the response
            for output in outputs:
                for chunk in output.outputs[0].text:
                    if self.stop_generation:
                        break

                    # Add chunk to full response
                    full_response += chunk

                    # Track thinking tags for post-processing
                    if '<think>' in chunk or '<thinking>' in chunk:
                        has_thinking_tags = True

                    # Handle thinking filtering with state tracking
                    if self.hide_thinking.get():
                        # Process chunk with thinking state
                        display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                            chunk, in_thinking_block, pending_buffer
                        )

                        # Track start of assistant response
                        if assistant_response_start is None and display_text.strip():
                            assistant_response_start = len(self.chat_display.get("1.0", tk.END)) - 1

                        # Display filtered content
                        if display_text:
                            self.append_to_chat(display_text)
                    else:
                        # No filtering needed - display everything
                        # Track start of assistant response
                        if assistant_response_start is None and chunk.strip():
                            assistant_response_start = len(self.chat_display.get("1.0", tk.END)) - 1

                        # Display the chunk
                        self.append_to_chat(chunk)

            # Clean up response
            clean_response = full_response.strip()

            # Thinking content is now filtered in real-time during streaming

            return clean_response

        except Exception as e:
            error_msg = f"Error during vLLM generation: {str(e)}"
            self.append_to_chat(f"\n[{error_msg}]")
            return error_msg

    def stream_transformers_response(self, display_message=None):
        """Stream response using Transformers backend"""
        full_response = ""
        assistant_response_start = None  # Track where assistant response starts in chat display
        has_thinking_tags = False  # Track if response contains thinking tags

        # Get model path for use throughout the function
        model_path = getattr(self, 'selected_transformers_path', StringVar()).get()



        # Check if model is loaded
        if not self.transformers_model or not self.transformers_tokenizer:
            error_msg = "Transformers model not loaded. Please load a model first."
            self.append_to_chat(f"\n[{error_msg}]")
            return error_msg

        try:
            # Format messages for transformers (similar to chat format)
            messages = []

            # Add system message if available
            if self.system_message and self.system_message.get('content'):
                messages.append({"role": "system", "content": self.system_message['content']})

            # Add conversation history
            for msg in self.messages:
                if msg['role'] in ['user', 'assistant', 'system']:
                    if isinstance(msg['content'], str):
                        messages.append({"role": msg['role'], "content": msg['content']})
                    elif isinstance(msg['content'], list):
                        # Handle multimodal content - extract text only for now
                        text_content = ""
                        for item in msg['content']:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_content += item.get('text', '')
                        if text_content:
                            messages.append({"role": msg['role'], "content": text_content})

            # Handle tokenization differently for MistralCommonBackend
            if MISTRAL_COMMON_AVAILABLE and isinstance(self.transformers_tokenizer, MistralCommonBackend):
                # MistralCommonBackend should use apply_chat_template exactly like the HF example
                try:
                    # Format messages for MistralCommonBackend (conversation format)
                    conversation = messages  # messages is already in the right format

                    tokenized = self.transformers_tokenizer.apply_chat_template(
                        conversation=conversation,
                        tools=[],  # Empty tools list if no tools available
                        return_tensors="pt",
                        return_dict=True,
                    )
                    inputs = tokenized
                    # Skip the string prompt creation and go directly to device placement
                except Exception as e:
                    self.add_to_debug_console(f"MistralCommonBackend tokenization failed: {e}")
                    # Fallback to manual formatting and regular tokenization
                    prompt = ""
                    for msg in messages:
                        role = msg['role']
                        content = msg['content']
                        if role == 'system':
                            prompt += f"System: {content}\n\n"
                        elif role == 'user':
                            prompt += f"User: {content}\n\n"
                        elif role == 'assistant':
                            prompt += f"Assistant: {content}\n\n"
                    prompt += "Assistant: "
                    inputs = self.transformers_tokenizer(prompt, return_tensors="pt")
            else:
                # Apply chat template if tokenizer supports it
                if hasattr(self.transformers_tokenizer, 'apply_chat_template') and hasattr(self.transformers_tokenizer, 'chat_template') and self.transformers_tokenizer.chat_template:
                    try:
                        prompt = self.transformers_tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except:
                        # Fallback to manual formatting
                        prompt = ""
                        for msg in messages:
                            role = msg['role']
                            content = msg['content']
                            if role == 'system':
                                prompt += f"System: {content}\n\n"
                            elif role == 'user':
                                prompt += f"User: {content}\n\n"
                            elif role == 'assistant':
                                prompt += f"Assistant: {content}\n\n"
                        prompt += "Assistant: "
                else:
                    # Manual formatting
                    prompt = ""
                    for msg in messages:
                        role = msg['role']
                        content = msg['content']
                        if role == 'system':
                            prompt += f"System: {content}\n\n"
                        elif role == 'user':
                            prompt += f"User: {content}\n\n"
                        elif role == 'assistant':
                            prompt += f"Assistant: {content}\n\n"
                    prompt += "Assistant: "

                # GPT-OSS SPECIAL TOKENIZATION: Use exact same method as working standalone script
                if "gpt-oss" in model_path.lower():
                    self.add_to_debug_console("🎯 GPT-OSS detected - using exact tokenization from standalone script")
                    try:
                        # Use EXACT same tokenization as gptoss.py - this is critical for GPT-OSS to work
                        inputs = self.transformers_tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,  # Add system prompt for chat
                            tokenize=True,
                            return_tensors="pt",
                            return_dict=True,
                        ).to(self.transformers_model.device)
                        self.add_to_debug_console("✅ GPT-OSS tokenization successful")
                    except Exception as e:
                        self.add_to_debug_console(f"❌ GPT-OSS tokenization failed: {e}")
                        # Fallback to standard tokenization if chat template fails
                        inputs = self.transformers_tokenizer(prompt, return_tensors="pt")
                else:
                    # Standard transformers generation
                    inputs = self.transformers_tokenizer(prompt, return_tensors="pt")

            # DEBUG: Show final prompt being sent to transformers (only for non-MistralCommonBackend)
            if not (MISTRAL_COMMON_AVAILABLE and isinstance(self.transformers_tokenizer, MistralCommonBackend)):
                self.add_to_debug_console("="*30)
                self.add_to_debug_console("FINAL PROMPT SENT TO TRANSFORMERS:")
                self.add_to_debug_console("="*30)
                if len(prompt) > 500:
                    self.add_to_debug_console(f"Length: {len(prompt)} chars")
                    self.add_to_debug_console("First 500 chars:")
                    self.add_to_debug_console(prompt[:500])
                    self.add_to_debug_console("...")
                else:
                    self.add_to_debug_console(prompt)
                self.add_to_debug_console("="*30)

                # Standard transformers generation
                inputs = self.transformers_tokenizer(prompt, return_tensors="pt")
            else:
                # For MistralCommonBackend, inputs were already created above
                self.add_to_debug_console("="*30)
                self.add_to_debug_console("MISTRALCOMMONBACKEND TOKENS CREATED DIRECTLY")
                self.add_to_debug_console("="*30)

            # Ensure inputs are on GPU (standard pattern)
            device = next(self.transformers_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get max tokens from UI
            max_new_tokens = int(self.max_tokens_var.get())

            # Initialize thinking state tracking
            in_thinking_block = False
            pending_buffer = ""

            # Use different generation approach for MistralCommonBackend vs regular models
            with torch.no_grad():
                if MISTRAL_COMMON_AVAILABLE and isinstance(self.transformers_tokenizer, MistralCommonBackend):
                    # Try the exact HF example first - no EOS token specified
                    input_ids = inputs["input_ids"]
                    self.add_to_debug_console(f"🔷 Starting MistralCommonBackend generation with max_new_tokens={max_new_tokens}")

                    try:
                        output = self.transformers_model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                        )[0]
                        self.add_to_debug_console(f"🔷 Generation completed, output shape: {output.shape}")
                    except Exception as e:
                        self.add_to_debug_console(f"❌ Generation failed: {e}")
                        # Fallback with EOS token
                        eos_token_id = getattr(self.transformers_tokenizer, 'eos_token_id', None) or 2
                        # Nanbeige4 model requires specific EOS token ID
                        if "nanbeige4" in model_path.lower():
                            eos_token_id = 166101
                        self.add_to_debug_console(f"🔷 Retrying with eos_token_id={eos_token_id}")
                        output = self.transformers_model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            eos_token_id=eos_token_id,
                        )[0]
                    outputs = output.unsqueeze(0)  # Add batch dimension back
                else:
                    # Set generation parameters with safeguards for regular models
                    temp_value = self.temperature.get()
                    do_sample_flag = temp_value > 0
                    pad_token_id = getattr(self.transformers_tokenizer, 'eos_token_id', None)
                    eos_token_id = getattr(self.transformers_tokenizer, 'eos_token_id', None)

                    # Nanbeige4 model requires specific EOS token ID
                    if "nanbeige4" in model_path.lower():
                        eos_token_id = 166101
                        self.add_to_debug_console("🔷 Using Nanbeige4 specific eos_token_id=166101")

                    if "gpt-oss" in model_path.lower():
                        # GPT-OSS specific generation (exact settings from working standalone script)
                        self.add_to_debug_console("🎯 GPT-OSS detected - using specialized generation settings")
                        outputs = self.transformers_model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            max_new_tokens=max_new_tokens,
                            do_sample=True,  # Enable sampling for variety
                            temperature=0.7,  # Fixed temperature like standalone script
                            use_cache=True,
                        )
                    else:
                        # Standard generation for other models
                        # Add CUDA assert protection by monitoring for numerical issues
                        try:
                            outputs = self.transformers_model.generate(
                                inputs["input_ids"],
                                attention_mask=inputs.get("attention_mask"),
                                max_new_tokens=max_new_tokens,
                                temperature=temp_value if do_sample_flag else None,
                                do_sample=do_sample_flag,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                use_cache=True,
                            )
                        except RuntimeError as cuda_error:
                            if "device-side assert" in str(cuda_error) or "probability tensor" in str(cuda_error):
                                self.add_to_debug_console(f"⚠️ CUDA assert detected: {str(cuda_error)}")
                                self.add_to_debug_console("🔧 Attempting to fix numerical issues...")

                                # Try generation with clamped inputs and safer parameters
                                with torch.no_grad():
                                    # Clamp input logits to prevent extreme values
                                    if hasattr(self.transformers_model, 'lm_head'):
                                        original_forward = self.transformers_model.lm_head.forward

                                        def clamped_forward(input):
                                            output = original_forward(input)
                                            # First replace NaN/inf, then clamp
                                            output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
                                            return torch.clamp(output, min=-50.0, max=50.0)

                                        self.transformers_model.lm_head.forward = clamped_forward

                                    try:
                                        outputs = self.transformers_model.generate(
                                            inputs["input_ids"],
                                            attention_mask=inputs.get("attention_mask"),
                                            max_new_tokens=max_new_tokens,
                                            temperature=max(temp_value, 0.1),  # Ensure minimum temperature
                                            do_sample=True,  # Force sampling to avoid greedy issues
                                            pad_token_id=pad_token_id,
                                            eos_token_id=eos_token_id,
                                            use_cache=False,  # Disable cache to avoid cached bad values
                                        )

                                        # Restore original forward if we patched it
                                        if hasattr(self.transformers_model, 'lm_head'):
                                            self.transformers_model.lm_head.forward = original_forward

                                        self.add_to_debug_console("✅ CUDA assert issue mitigated")
                                    except Exception as retry_error:
                                        self.add_to_debug_console(f"❌ Retry also failed: {str(retry_error)}")
                                        raise cuda_error
                            else:
                                raise cuda_error

            # Decode the full response
            if MISTRAL_COMMON_AVAILABLE and isinstance(self.transformers_tokenizer, MistralCommonBackend):
                # Use HF example decoding approach for MistralCommonBackend
                full_generated_text = self.transformers_tokenizer.decode(
                    output[len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                )
            else:
                # Standard decoding for regular models
                full_generated_text = self.transformers_tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                )

            # DEBUG: Show raw decoded response before any parsing
            self.add_to_debug_console("="*50)
            self.add_to_debug_console("RAW DECODED RESPONSE:")
            self.add_to_debug_console("="*50)
            self.add_to_debug_console(repr(full_generated_text))  # Use repr to show exact characters
            self.add_to_debug_console("="*50)

            # Nanbeige4 SPECIAL PARSING: Extract final answer from thinking model output
            if "nanbeige4" in model_path.lower():
                self.add_to_debug_console("🎯 Nanbeige4 thinking model detected - extracting final answer")
                import re

                # Look for the transition from thinking to final answer
                # Pattern: thinking text followed by newlines then the actual response starting with "Hello!"
                thinking_pattern = r'^(.*?)\n\n+(Hello!.*)$'
                match = re.search(thinking_pattern, full_generated_text, re.DOTALL)

                if match:
                    thinking_text = match.group(1).strip()
                    final_answer = match.group(2).strip()

                    # DEBUG: Show what we extracted
                    self.add_to_debug_console(f"EXTRACTED THINKING ({len(thinking_text)} chars): {thinking_text[:100]}...")
                    self.add_to_debug_console(f"EXTRACTED ANSWER ({len(final_answer)} chars): {final_answer[:100]}...")
                    self.add_to_debug_console(f"Hide thinking setting: {self.hide_thinking.get()}")

                    # Replace with just the final answer (optionally keep thinking if user wants it)
                    if self.hide_thinking.get():
                        full_generated_text = final_answer
                        self.add_to_debug_console("✅ Nanbeige4 thinking hidden - showing only final answer")
                    else:
                        # For UI compatibility, keep it shorter - just show the final answer with a note
                        full_generated_text = f"[AI Thinking Process]\n\n{final_answer}"
                        self.add_to_debug_console("✅ Nanbeige4 showing condensed response for UI")
                else:
                    # Fallback: if parsing fails, try to extract just the final answer by finding the last coherent response
                    self.add_to_debug_console("⚠️ Nanbeige4 parsing failed - trying fallback extraction")
                    # Look for the last occurrence of a complete sentence starting with common patterns
                    fallback_patterns = [
                        r'.*?(Hello!.*?\n\n.*?\))',  # Try to capture Hello! response with note
                        r'.*?(Hello!.*)',  # Just capture from Hello! onward
                    ]
                    for pattern in fallback_patterns:
                        fallback_match = re.search(pattern, full_generated_text, re.DOTALL)
                        if fallback_match:
                            full_generated_text = fallback_match.group(1).strip()
                            self.add_to_debug_console(f"✅ Fallback parsing succeeded: {len(full_generated_text)} chars")
                            break
                    else:
                        self.add_to_debug_console("❌ All parsing failed - using raw response")

            # GPT-OSS SPECIAL PARSING: Handle Harmony format with analysis/final channels
            elif "gpt-oss" in model_path.lower():
                self.add_to_debug_console("🎯 GPT-OSS detected - parsing Harmony format")
                import re

                # Extract analysis (reasoning) and final answer from GPT-OSS Harmony format
                analysis_match = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", full_generated_text, re.DOTALL)
                final_match = re.search(r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", full_generated_text, re.DOTALL)

                if analysis_match and final_match:
                    analysis_text = analysis_match.group(1).strip()
                    final_text = final_match.group(1).strip()

                    # Replace the raw response with formatted analysis + final answer
                    full_generated_text = f"🔍 Analysis: {analysis_text}\n🤖 Assistant: {final_text}"
                    self.add_to_debug_console("✅ GPT-OSS Harmony format parsed successfully")
                else:
                    self.add_to_debug_console("⚠️ GPT-OSS parsing failed - using raw response")

            # GPU cleanup and synchronization before GUI updates (prevents segfaults) DISABLED
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure all CUDA operations complete
                    self.add_to_debug_console("🧹 GPU synchronized")
            except Exception as e:
                self.add_to_debug_console(f"⚠️ GPU cleanup failed: {e}")

            # Simulate streaming by yielding chunks (standard pattern)
            chunk_size = 5  # Characters per chunk for smooth streaming
            for i in range(0, len(full_generated_text), chunk_size):
                if self.stop_generation:
                    break

                chunk = full_generated_text[i:i+chunk_size]
                full_response += chunk

                # Track thinking tags for post-processing
                if '<think>' in chunk or '<thinking>' in chunk:
                    has_thinking_tags = True

                # Handle thinking filtering with state tracking - INSIDE the loop to display each chunk
                if self.hide_thinking.get():
                    # Process chunk with thinking state
                    display_text, in_thinking_block, pending_buffer = self._process_chunk_with_thinking(
                        chunk, in_thinking_block, pending_buffer
                    )

                    # Track start of assistant response
                    if assistant_response_start is None and display_text.strip():
                        assistant_response_start = len(self.chat_display.get("1.0", tk.END)) - 1

                    # Display filtered content
                    if display_text:
                        self.append_to_chat(display_text)
                else:
                    # No filtering needed - display everything
                    # Track start of assistant response
                    if assistant_response_start is None and chunk.strip():
                        assistant_response_start = len(self.chat_display.get("1.0", tk.END)) - 1

                    # Display the chunk
                    self.append_to_chat(chunk)

                # Small delay for streaming effect
                time.sleep(0.01)

            # Clean up response
            clean_response = full_response.strip()

            # DEBUG: Show final clean response
            self.add_to_debug_console("="*30)
            self.add_to_debug_console("FINAL CLEAN RESPONSE:")
            self.add_to_debug_console("="*30)
            self.add_to_debug_console(repr(clean_response))  # Use repr to show exact characters
            self.add_to_debug_console("="*30)

            # Thinking content is now filtered in real-time during streaming

            return clean_response

        except Exception as e:
            error_msg = str(e)
            # Check for Triton/MXFP4 issues during generation
            if "triton" in error_msg.lower() and ("mxfp4" in error_msg.lower() or "quantization" in error_msg.lower() or "ptxas" in error_msg.lower()):
                error_msg = "MXFP4 kernel error during generation. The model loaded but MXFP4 quantization failed on ARM. Performance will be degraded. Consider converting the model to a different quantization format."
                self.append_to_chat(f"\n[{error_msg}]")
                self.add_to_debug_console(f"⚠️ MXFP4 runtime error (model loaded but kernels failed): {str(e)[:300]}...")
            elif "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                error_msg = "GPU memory error. The model may be too large for your GPU. Try reducing max tokens or using a smaller model."
                self.append_to_chat(f"\n[{error_msg}]")
                self.add_to_debug_console(f"⚠️ GPU memory error: {str(e)}")
            else:
                error_msg = f"Error during Transformers generation: {error_msg}"
                self.append_to_chat(f"\n[{error_msg}]")
            return error_msg

    def stream_response(self, model_to_use, has_image=False):
        """Legacy method for backward compatibility"""
        # Just delegate to the appropriate method
        if has_image:
            return self.stream_multimodal_response(model_to_use, None, "")
        else:
            return self.stream_text_response(model_to_use)

    def stream_claude_response(self, has_image=False, image_data=None, display_message=None, image_media_type=None):
        """Stream response using Claude API"""
        full_response = ""
        buffer = ""
        in_thinking = False
        last_visible_update = 0
        assistant_response_start = None  # Track where assistant response starts in chat display
        has_thinking_tags = False  # Track if response contains thinking tags
        
        try:
            # Prepare messages for Claude API
            messages = []
            system_message = ""
            
            # Extract system message and convert messages
            for msg in self.messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                elif msg['role'] in ['user', 'assistant']:
                    # Handle image in current message
                    if msg == self.messages[-1] and has_image and image_data:
                        # For the current message with image
                        content = [
                            {"type": "text", "text": display_message},
                            {"type": "image", "source": {"type": "base64", "media_type": image_media_type or "image/png", "data": image_data}}
                        ]
                        messages.append({"role": "user", "content": content})
                    else:
                        # Regular text message
                        content = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
                        messages.append({"role": msg['role'], "content": content})
            
            # Prepare API request
            headers = {
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.claude_model_var.get(),
                "max_tokens": int(self.max_tokens_var.get()),
                "temperature": self.temperature.get(),
                "messages": messages,
                "stream": True
            }
            
            if system_message:
                data["system"] = system_message
            
            # Add tools if search mode is enabled
            if hasattr(self, 'search_mode') and self.search_mode.get():
                data["tools"] = get_mcp_tool_definitions("claude", self.mcp_client)
                self.display_chat_system_message("🔍 Web search tools enabled")

            # Make streaming request
            response = requests.post("https://api.anthropic.com/v1/messages",
                                   headers=headers, json=data, stream=True)

            if response.status_code != 200:
                raise Exception(f"Claude API error: {response.status_code} - {response.text}")

            # Process streaming response
            for line in response.iter_lines():
                if self.stop_generation:
                    break

                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == '[DONE]':
                                break

                            data_json = json.loads(data_str)

                            # Handle tool calls in Claude — collect for agent loop
                            if data_json.get('type') == 'content_block_start':
                                content_block = data_json.get('content_block', {})
                                if content_block.get('type') == 'tool_use':
                                    tool_name = content_block.get('name')
                                    tool_id = content_block.get('id', '')
                                    # Store tool info for when we get the complete input
                                    self._claude_pending_tool = {
                                        'name': tool_name,
                                        'id': tool_id,
                                        'input': ''
                                    }

                            # Handle tool input chunks
                            if data_json.get('type') == 'content_block_delta':
                                delta = data_json.get('delta', {})
                                if delta.get('type') == 'input_json_delta':
                                    if hasattr(self, '_claude_pending_tool') and self._claude_pending_tool:
                                        self._claude_pending_tool['input'] += delta.get('partial_json', '')

                            # Handle tool completion — collect for agent loop
                            if data_json.get('type') == 'content_block_stop':
                                if hasattr(self, '_claude_pending_tool') and self._claude_pending_tool:
                                    try:
                                        tool_input = json.loads(self._claude_pending_tool['input']) if self._claude_pending_tool['input'] else {}
                                        tool_name = self._claude_pending_tool['name']
                                        tool_id = self._claude_pending_tool['id']
                                        self.display_chat_system_message(f"🔍 Tool call: {tool_name}({json.dumps(tool_input)[:100]})")
                                        self._pending_tool_results.append({
                                            'name': tool_name,
                                            'arguments': tool_input,
                                            'id': tool_id,
                                            'backend': 'claude'
                                        })
                                    except Exception as e:
                                        self.add_to_debug_console(f"❌ Error parsing tool input: {e}")
                                    self._claude_pending_tool = None
                            
                            if data_json.get('type') == 'content_block_delta':
                                delta = data_json.get('delta', {})
                                if delta.get('type') == 'text_delta':
                                    chunk_text = delta.get('text', '')
                                    if chunk_text:
                                        full_response += chunk_text
                                        buffer += chunk_text

                                        # Track start of assistant response for potential replacement
                                        if assistant_response_start is None:
                                            assistant_response_start = len(self.chat_display.get("1.0", tk.END)) - 1

                                        # Handle thinking tags and display logic
                                        if '<thinking>' in buffer:
                                            in_thinking = True
                                            has_thinking_tags = True
                                            pre_thinking = buffer.split('<thinking>')[0]
                                            if pre_thinking:
                                                self.chat_display.insert(tk.END, pre_thinking)
                                                self.chat_display.see(tk.END)
                                                self.chat_display.update()
                                            buffer = buffer.split('<thinking>', 1)[1]

                                        if '</thinking>' in buffer and in_thinking:
                                            in_thinking = False
                                            buffer = buffer.split('</thinking>', 1)[1]

                                        # Always display content during streaming (thinking will be filtered later if needed)
                                        current_time = time.time()
                                        if current_time - last_visible_update > 0.05:  # 50ms throttle
                                            self.chat_display.insert(tk.END, buffer)
                                            self.chat_display.see(tk.END)
                                            self.chat_display.update()
                                            buffer = ""
                                            last_visible_update = current_time
                        except json.JSONDecodeError:
                            continue
            
            # Display any remaining buffer
            if buffer:
                self.chat_display.insert(tk.END, buffer)
                self.chat_display.see(tk.END)
                self.chat_display.update()

            # Thinking content is now filtered in real-time during streaming
            return filter_thinking(full_response, not self.hide_thinking.get())
            
        except Exception as e:
            error_msg = f"Claude API error: {str(e)}"
            self.display_status_message(error_msg)
            return f"Error: {error_msg}"


    def enable_input(self):
        """Re-enable the input controls after processing is complete"""
        self.user_input.config(state=tk.NORMAL)
        self.stop_button.pack_forget()
        self.fix_button.pack_forget()
        self.send_button.pack(side=tk.RIGHT)
        self.fix_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.status_var.set("Ready")
        self.user_input.focus_set()

    def fix_from_chat(self):
        """LLM Fix triggered from main chat button — same as IDE's LLM Fix."""
        if hasattr(self, 'ide_window') and self.ide_window and self.ide_window.root.winfo_exists():
            content = self.ide_window.get_content()
            if content and content.strip():
                self.fix_code_from_ide(content)
                return
        self.display_chat_system_message("No code in IDE to fix")

    def _rebuild_messages_from_chat(self):
        """Rebuild self.messages from chat_display so user edits (cut/paste) take effect.

        Parses visible chat text for 'You: ' and '*Assistant: ' prefixes,
        maps them to user/assistant roles, and rebuilds the message list.
        System messages in chat are skipped (not sent to LLM).
        The system_message (role='system') is always kept as messages[0].
        """
        import re
        chat_text = self.chat_display.get("1.0", tk.END).strip()
        if not chat_text:
            self.messages = [self.system_message]
            return

        # Known sender prefixes → roles
        sender_map = {
            'You': 'user',
            'HTML Assistant': 'assistant',
            'Py Assistant': 'assistant',
            'Assistant': 'assistant',
        }
        # Build regex: match sender label at start of text or after \n\n
        labels = sorted(sender_map.keys(), key=len, reverse=True)  # longest first
        label_pattern = '|'.join(re.escape(l) for l in labels)
        pattern = re.compile(rf'(?:^|\n\n)({label_pattern}): ', re.MULTILINE)

        matches = list(pattern.finditer(chat_text))
        if not matches:
            # No parseable messages — keep current history
            return

        rebuilt = [self.system_message]
        for i, m in enumerate(matches):
            sender = m.group(1)
            role = sender_map.get(sender)
            if role is None:
                continue  # skip System or unknown
            content_start = m.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(chat_text)
            content = chat_text[content_start:content_end].strip()
            if content:
                rebuilt.append({'role': role, 'content': content})

        self.messages = rebuilt

    def append_to_chat(self, text):
        """Append text to the chat display and update the GUI
        
        THREAD-SAFE: May be called from background inference thread.
        Schedules GUI work on main thread using root.after(0, ...).
        
        This is used for streaming responses. It now checks if the text
        contains system-type messages (errors, truncation notices) and
        routes them appropriately.
        """
        # Schedule GUI work on main thread for thread safety
        self.root.after(0, lambda: self._append_to_chat_impl(text))
    
    def _append_to_chat_impl(self, text):
        """Internal implementation of append_to_chat - runs on main thread only"""
        # Check if this is a system-type message that should be routed
        system_indicators = [
            "[API Error:",
            "[Error during",
            "[Response truncated",
            "[Timeout:",
            "[Connection error:",
            "[Execution error:",
            "[Code execution",
            "[Program error:",
            "❌ ERROR:",
        ]
        
        is_system_message = any(indicator in text for indicator in system_indicators)
        
        if is_system_message:
            # Route to system console instead of chat (already thread-safe)
            self._display_system_message_impl(text.strip(), True)
            # For PROGRAM EXECUTION errors, DON'T add notice to chat - they should only go to debug console
            # Only add brief notice for critical streaming/generation errors
            if ("[Error" in text or "[API Error" in text) and not any(err in text for err in [
                "ModuleNotFoundError", "SyntaxError", "IndentationError", "NameError",
                "TypeError", "ValueError", "ImportError", "AttributeError", "❌ ERROR:",
                "[Execution error:", "[Code execution", "[Program error:"
            ]):
                # Chat is now always editable, no need to change state
                self.chat_display.insert(tk.END, "\n[Error occurred - see system console for details]\n")
                self.chat_display.see(tk.END)
        else:
            # Normal streaming text goes to chat - now always editable
            self.chat_display.insert(tk.END, text)
            self.chat_display.see(tk.END)
        
        self.root.update_idletasks()  # Update the GUI immediately

    def display_message(self, sender, message, end=True, to_chat=None):
        """Display a message in the chat display or system console based on type
        
        THREAD-SAFE: May be called from background threads.
        Schedules GUI work on main thread using root.after(0, ...).
        
        Args:
            sender: The sender of the message (e.g., "System", "You", "Assistant")
            message: The message content
            end: Whether to add newlines after the message
            to_chat: Explicitly control routing for system messages:
                     - True: Force to chat history
                     - False: Force to system console
                     - None: Use auto-routing for backward compatibility
        
        THREE-WINDOW SYSTEM EXPLANATION:
        1. Main Chat Window: User messages, assistant responses, and critical system messages
        2. System Console: Non-critical system messages (status updates, mode switches, etc.)
        3. Debug Console: Backend debugging information and detailed logs
        """
        # Schedule GUI work on main thread for thread safety
        self.root.after(0, lambda: self._display_message_impl(sender, message, end, to_chat))
    
    def _display_message_impl(self, sender, message, end=True, to_chat=None):
        """Internal implementation of display_message - runs on main thread only"""
        # Check for IDE proposals before displaying
        self.check_and_handle_ide_proposals(message)
        
        # Define important system messages that should stay in chat history
        # These messages provide important context for the LLM and user
        important_system_keywords = [
            "📝 Code loaded",  # Code loading notifications - LLM needs to know what code is loaded
            "Running code block",  # Code execution notifications
            "--- Execution Output ---",  # Code output headers
            "Chat loaded from",  # Chat loading notifications
            "Chat saved to",  # Chat saving notifications
            "Successfully loaded",  # Model loading confirmations (for current session's model)
            "Error:",  # Error messages - critical for debugging
            "Error running code",  # Code execution errors
            "Failed to",  # Failure notifications
            "Chat restarted",  # Important state changes
            "Fix request sent",  # IDE interaction notifications
            "Code loaded from",  # When loading code files
            "No code block found",  # Important for debugging
            "⚠️ Warning:",  # Important warnings
            "IMPORTANT:",  # Critical notices
        ]
        
        # Check if this is a system message
        if sender == "System":
            # Check if system console is available (might not be during early initialization)
            if not hasattr(self, 'system_console') or self.system_console is None:
                # During initialization, all system messages go to chat - but check if chat_display exists
                if hasattr(self, 'chat_display') and self.chat_display is not None:
                    self.chat_display.insert(tk.END, f"{sender}: {message}")
                    if end:
                        self.chat_display.insert(tk.END, "\n\n")
                    self.chat_display.see(tk.END)
                else:
                    # Chat display not ready yet, print to console
                    print(f"{sender}: {message}")
                return
                
            # Use explicit routing if specified
            if to_chat is not None:
                if to_chat:
                    # Explicitly send to chat - chat is now always editable
                    self.chat_display.insert(tk.END, f"{sender}: {message}")
                    if end:
                        self.chat_display.insert(tk.END, "\n\n")
                    self.chat_display.see(tk.END)
                else:
                    # Explicitly send to system console
                    self._display_system_message_impl(message, end)
            else:
                # Backward compatibility: use keyword matching for old calls
                # This should be phased out as we update all system message calls
                is_important = any(keyword in message for keyword in important_system_keywords)
                
                if is_important:
                    # Chat is now always editable
                    self.chat_display.insert(tk.END, f"{sender}: {message}")
                    if end:
                        self.chat_display.insert(tk.END, "\n\n")
                    self.chat_display.see(tk.END)
                else:
                    self._display_system_message_impl(message, end)
            return
        
        # Determine the actual sender label based on current mode (code-only)
        actual_sender = sender
        if sender in ("Therapist", "Py Assistant", "Helpful Assistant", "HTML Assistant", "Assistant"):
            mode = self.system_mode.get()
            if mode == "programmer":
                actual_sender = "Py Assistant"
            elif mode == "html_programmer":
                actual_sender = "HTML Assistant"
            else:
                actual_sender = "Assistant"
            
        # Chat is now always editable - no need to change state
        self.chat_display.insert(tk.END, f"{actual_sender}: {message}")
        if end:
            self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.see(tk.END)

    def start_response_timer(self):
        """Start the response timer when an assistant response begins"""
        import time
        # Reset timer display to 00:00 before starting new timer
        self.timer_label.config(text="00:00")
        self.response_start_time = time.time()
        self.update_timer_display()

    def stop_response_timer(self):
        """Stop the response timer but keep final elapsed time displayed"""
        if self.timer_update_id:
            self.root.after_cancel(self.timer_update_id)
            self.timer_update_id = None
        # Keep response_start_time set so timer shows final time
        # Only reset when next message is sent (in start_response_timer)

    def update_timer_display(self):
        """Update the timer display with current elapsed time"""
        if self.response_start_time:
            import time
            elapsed = time.time() - self.response_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")

            # Schedule next update in 1 second
            self.timer_update_id = self.root.after(1000, self.update_timer_display)

    # ---------- Token counting & speed metrics ----------

    def estimate_token_count(self, text):
        """Estimate token count from text (~4 chars per token for English).
        Works across all backends without requiring a tokenizer."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def count_messages_input_tokens(self):
        """Estimate the total input tokens for the current message list"""
        total = 0
        for msg in self.messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                total += self.estimate_token_count(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        total += self.estimate_token_count(item.get('text', ''))
        return total

    def record_token_stats(self, full_response):
        """Record token stats after a response completes and update display"""
        import time
        elapsed = 0
        if self.response_start_time:
            elapsed = time.time() - self.response_start_time

        # Input tokens = everything sent to the model
        self.last_input_tokens = self.count_messages_input_tokens()
        # Output tokens = the response
        self.last_output_tokens = self.estimate_token_count(full_response) if full_response else 0
        # Speed (output tokens per second)
        self.last_output_speed = self.last_output_tokens / elapsed if elapsed > 0 else 0

        # Accumulate running totals
        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += self.last_output_tokens

        # Update UI on the main thread
        self.root.after(0, self.update_token_display)

    def update_token_display(self):
        """Update the token counter labels in the chat header"""
        speed_str = f" | {self.last_output_speed:.1f} tok/s" if self.last_output_speed > 0 else ""
        self.token_label.config(
            text=f"In:{self.last_input_tokens:,} Out:{self.last_output_tokens:,}{speed_str} | Total In:{self.total_input_tokens:,} Out:{self.total_output_tokens:,}"
        )
        if self.last_output_speed > 0:
            self.speed_label.config(text=f"({self.last_output_speed:.1f} tok/s)")
        else:
            self.speed_label.config(text="")

    def handle_code_update_response(self, response_text):
        """Handle code update responses - just extract code and show as diff

        Much simpler than FIM - just show what the LLM suggests!
        """
        # This is now handled by check_and_handle_ide_proposals
        # which extracts code blocks and shows them as diffs
        return False
    
    # ---------- Response -> IDE routing ----------
    # After the LLM finishes, this extracts code from the response and routes
    # it to the IDE as a diff.  Tries SEARCH/REPLACE blocks first (fast edits),
    # then falls back to extracting full code blocks.

    def check_and_handle_ide_proposals(self, message):
        """Route LLM response to IDE: try SEARCH/REPLACE blocks, then code blocks."""
        # Auto-propose is always enabled; the _auto_fix_in_progress flag
        # ensures the fix workflow flows through correctly.
        import re
        
        # Look for IDE_PROPOSE markers first
        ide_pattern = r'<IDE_PROPOSE>(.*?)</IDE_PROPOSE>'
        matches = re.findall(ide_pattern, message, re.DOTALL)
        
        if matches:
            # Take the last/most recent proposal
            proposed_code = matches[-1].strip()
            
            # If IDE window exists, propose the changes
            if hasattr(self, 'ide_window'):
                self.propose_code_changes(proposed_code)
                self._auto_fix_in_progress = False

                # Show notification
                self.show_copy_status("💡 Code changes proposed to IDE - check IDE window to accept/reject", 4000)
                return

        # Determine language pattern based on mode
        is_html_mode = (self.system_mode.get() == "html_programmer")

        # If we have original code from a fix request, try to apply the response
        if hasattr(self, 'ide_original_code') and self.ide_original_code:

            # METHOD 1: Try SEARCH/REPLACE blocks first (fast, targeted edits)
            sr_result = self._apply_search_replace_blocks(self.ide_original_code, message)
            if sr_result and sr_result.strip() != self.ide_original_code.strip():
                if hasattr(self, 'ide_window'):
                    self.propose_code_changes(sr_result)
                    self._auto_fix_in_progress = False
                    self.show_copy_status("💡 Edits applied to IDE - check IDE window to accept/reject", 4000)
                self.ide_original_code = None
                return

            # METHOD 2: Extract code block and try partial merge first
            if is_html_mode:
                code_pattern = r'```html\s*\n(.*?)```'
            else:
                code_pattern = r'```python\s*\n(.*?)```'
            code_matches = re.findall(code_pattern, message, re.DOTALL)

            if code_matches:
                proposed_code = code_matches[-1].strip()

                if proposed_code != self.ide_original_code.strip():
                    # Try partial merge first — if the LLM returned only
                    # changed functions, merge them into the original rather
                    # than showing a raw diff that deletes everything else.
                    merged = self._merge_partial_fix(self.ide_original_code, proposed_code)
                    final_code = merged if merged else proposed_code

                    if hasattr(self, 'ide_window'):
                        self.propose_code_changes(final_code)
                        self._auto_fix_in_progress = False
                        self.show_copy_status("💡 Fixed code proposed to IDE - check IDE window to accept/reject", 4000)

                    self.ide_original_code = None
                    return

        # If IDE has content and response contains code that seems to be a modification
        if hasattr(self, 'ide_current_content') and self.ide_current_content:
            # Look for code blocks matching current mode
            if is_html_mode:
                code_pattern = r'```html\s*\n(.*?)```'
            else:
                code_pattern = r'```python\s*\n(.*?)```'
            code_matches = re.findall(code_pattern, message, re.DOTALL)
            
            if code_matches:
                # Take the last/most recent code block
                proposed_code = code_matches[-1].strip()
                
                # If it's different from current IDE content, check if we should propose it
                if proposed_code != self.ide_current_content.strip():
                    should_propose = False
                    
                    # Method 1: Check if the response seems to be about modifying the IDE code
                    modify_keywords = ['here\'s the updated', 'here\'s the modified', 'here\'s the fixed', 
                                     'here is the updated', 'here is the modified', 'here is the fixed',
                                     'updated code', 'modified code', 'fixed code', 'corrected code',
                                     'with the changes', 'after the changes', 'complete code', 'full code']
                    
                    if any(keyword in message.lower() for keyword in modify_keywords):
                        should_propose = True
                    
                    # Method 2: Check if we recently included IDE context (indicating user was asking about current code)
                    elif hasattr(self, 'last_ide_context_included') and self.last_ide_context_included:
                        # Check if the code seems to be an evolution of the current IDE code
                        # Simple heuristic: if the proposed code contains significant portions of the current code
                        current_lines = set(line.strip() for line in self.ide_current_content.split('\n') if line.strip())
                        proposed_lines = set(line.strip() for line in proposed_code.split('\n') if line.strip())
                        
                        # If at least 30% of current code lines are preserved, it's likely an update
                        if current_lines and len(current_lines & proposed_lines) / len(current_lines) >= 0.3:
                            should_propose = True
                    
                    # Method 3: If in Python Programmer mode and code looks substantial (not just a snippet)
                    elif self.system_mode.get() == "programmer" and len(proposed_code.split('\n')) >= 5:
                        # Check if it looks like a complete function/class rather than just a snippet
                        if ('def ' in proposed_code or 'class ' in proposed_code or 
                            proposed_code.count('\n') >= len(self.ide_current_content.split('\n')) * 0.5):
                            should_propose = True
                    
                    if should_propose:
                        if hasattr(self, 'ide_window'):
                            self.propose_code_changes(proposed_code)
                            self._auto_fix_in_progress = False

                            # Show notification
                            self.show_copy_status("💡 Updated code proposed to IDE - check IDE window to accept/reject", 4000)
                            
                        # Reset the context included flag
                        if hasattr(self, 'last_ide_context_included'):
                            self.last_ide_context_included = False

    def process_command(self, command):
        """Process special commands starting with /"""
        cmd_parts = command.lower().split()
        cmd = cmd_parts[0]
        
        if cmd == '/exit':
            self.root.destroy()
            
        elif cmd == '/restart':
            self.restart_chat()
            
        elif cmd == '/think':
            self.hide_thinking.set(not self.hide_thinking.get())
            status = "enabled" if self.hide_thinking.get() else "disabled"
            self.previous_thinking_state = not self.hide_thinking.get()  # Update the tracking state (inverted)
            self.display_status_message(f"Thinking hiding {status}")
            
        elif cmd == '/save':
            self.save_chat()
            
        elif cmd == '/load':
            self.load_chat()
            
        elif cmd == '/models':
            # Display available models for current backend
            backend = self.backend_var.get()
            if backend == "llama_cpp":
                if self.available_gguf_models:
                    model_list = "\n".join([os.path.basename(p) for p in self.available_gguf_models])
                    self.display_status_message(f"Available GGUF models:\n{model_list}")
                else:
                    self.display_status_message("No GGUF models found. Check your GGUF_Models directory.")
            else:
                # Refresh and display Ollama models
                self.refresh_models()
                model_list = "\n".join(self.available_models)
                self.display_status_message(f"Available Ollama models:\n{model_list}\n\nNote: Any model can be tried with images.")
                
        elif cmd == '/backend':
            # Switch backend
            current = self.backend_var.get()
            if current == "ollama":
                if LLAMA_CPP_AVAILABLE:
                    self.backend_var.set("llama_cpp")
                    self.change_backend()
                else:
                    self.display_status_message("GGUF not available. Install with: pip install llama-cpp-python")
            else:
                self.backend_var.set("ollama")
                self.change_backend()
            
        elif cmd == '/help':
            self.display_command_help()
            
        elif cmd == '/image':
            # Select an image
            self.select_image()
            
        elif cmd == '/cleari':
            # Clear the current image
            self.clear_image()
            
        elif cmd == '/rag':
            # Toggle RAG functionality
            if self.current_collection.get():
                self.rag_enabled.set(not self.rag_enabled.get())
                self.display_status_message(f"RAG functionality {'enabled' if self.rag_enabled.get() else 'disabled'}")
            else:
                self.display_status_message("No RAG collection loaded. Use /indexdir or /loaddb first.")
                
        elif cmd == '/indexdir':
            # Index a folder into ChromaDB
            self.index_new_folder()
            
        elif cmd == '/loaddb':
            # Load an existing ChromaDB collection
            self.load_existing_collection()
            
        elif cmd == '/clearrag':
            # Clear RAG collection
            self.clear_rag_collection()
            
        elif cmd == '/testrag':
            # Test RAG query
            self.test_rag_query()
            
        elif cmd == '/startollama':
            # Try to start Ollama
            if self.check_ollama_available():
                self.display_status_message("Ollama is already running.")
            else:
                self.try_start_ollama()
                
        elif cmd == '/testrouting':
            # Test command to verify system message routing
            self.test_system_message_routing()
            
        elif cmd == '/search':
            # Web search command
            if len(cmd_parts) < 2:
                self.display_message("System", "Usage: /search <query>\nExample: /search python tutorials")
                return
            
            query = ' '.join(cmd_parts[1:])
            self.display_status_message(f"Searching for: {query}")
            
            # Perform web search
            try:
                results = safe_web_search(query)
                self.display_message("Web Search", results)
                self.add_to_debug_console(f"Web search completed for: {query}")
            except Exception as e:
                self.display_status_message(f"Search failed: {str(e)}")
                self.add_to_debug_console(f"Web search error: {str(e)}")
            
        else:
            self.display_status_message(f"Unknown command: {command}")

    def test_system_message_routing(self):
        """Test method to verify system message routing is working correctly"""
        self.display_message("System", "=== TESTING SYSTEM MESSAGE ROUTING ===")
        
        # Test messages that should go to system console
        test_system_msgs = [
            "Model list refreshed",
            "Found 5 GGUF models",
            "Switched to ollama backend",
            "Please select a model first.",
            "RAG functionality enabled",
            "Ollama is already running.",
        ]
        
        # Test messages that should stay in chat
        test_chat_msgs = [
            "Error: Something went wrong",
            "📝 Code loaded in IDE: test.py",
            "Running code block...",
            "Chat saved to file.json",
            "Failed to load model",
            "⚠️ Warning: Important notice",
        ]
        
        self.display_message("System", "Testing messages for SYSTEM CONSOLE:")
        for msg in test_system_msgs:
            self.display_message("System", f"  TEST: {msg}")
            
        self.display_message("System", "Testing messages for CHAT HISTORY:")
        for msg in test_chat_msgs:
            self.display_message("System", f"  TEST: {msg}")
            
        self.display_message("System", "=== END OF ROUTING TEST ===")
        self.display_message("System", "Check system console and chat history to verify routing!")
    
    def restart_chat(self):
        """Clear chat history and start over"""
        # Increment restart counter to invalidate any running threads - FIX for restart not clearing
        if not hasattr(self, 'restart_counter'):
            self.restart_counter = 0
        self.restart_counter += 1
        
        # Get the current system message based on the radio button selection (code-only)
        mode = self.system_mode.get()
        if mode == "html_programmer":
            content = html_system_message
        else:
            content = python_system_message
        self.system_message = {'role': 'system', 'content': content}

        self.messages = [self.system_message]
        self.current_file = None  # Clear the current file reference
        # Clear IDE context that persists after restart - FIX for restart showing old info
        self.ide_current_content = None
        self.ide_current_filename = None
        if hasattr(self, 'last_ide_context_included'):
            self.last_ide_context_included = False
        self.root.title("CodeRunner IDE")  # Reset title
        # Reset token counters
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_output_speed = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.update_token_display()
        # Chat is now always editable
        self.chat_display.delete("1.0", tk.END)
        # Reset thinking state tracker
        self.previous_thinking_state = not self.hide_thinking.get()  # Inverted logic
        # Reset first message flag
        self.first_message = True
        self.display_system_message("Chat restarted")  # Route to system console, not chat history
        self.display_command_help()

    def display_command_help(self):
        """Display the available commands"""
        # ROUTING FIX: Send available commands to System Console (not chat history)
        help_lines = ["Available commands:"]
        for cmd, desc in self.commands.items():
            help_lines.append(f"  {cmd}: {desc}")
        help_text = "\n".join(help_lines) + "\n"
        self.display_message("System", help_text, to_chat=False)

    def toggle_system_prompt(self):
        """Toggle between code modes (Python or HTML)"""
        mode = self.system_mode.get()

        if mode == "programmer":
            new_content = python_system_message
            self.display_message("System", "Switched to Python / Pygame mode")
            self.set_temperature(0.35)
            self.top_p.set(0.9)
            self.top_k.set(20)
        elif mode == "html_programmer":
            new_content = html_system_message
            self.display_message("System", "Switched to HTML / JavaScript mode")
            self.set_temperature(0.35)
            self.top_p.set(0.9)
            self.top_k.set(20)
        else:
            # Fallback to Python programmer
            new_content = python_system_message
            self.system_mode.set("programmer")
            self.display_message("System", "Switched to Python / Pygame mode")
            self.set_temperature(0.35)
            self.top_p.set(0.9)
            self.top_k.set(20)
            
        # Update the system message (this will also handle targeted changes instruction)
        self.update_system_message_for_targeted_changes()
        
        # Clean legacy system messages: keep ONLY the primary system message
        # Ensures mode switch uses a clean, single system instruction with no leftovers
        if self.messages:
            if self.messages[0].get('role') != 'system':
                self.messages = [self.system_message] + [m for m in self.messages if m.get('role') != 'system']
            else:
                primary = self.messages[0]
                rest = [m for m in self.messages[1:] if m.get('role') != 'system']
                self.messages = [primary] + rest
            
        # Update the prompt editor if it's open
        if hasattr(self, 'prompt_editor') and self.prompt_frame.winfo_ismapped():
            self.prompt_editor.delete("1.0", tk.END)
            self.prompt_editor.insert(tk.END, new_content)
    
    def _ensure_mcp_initialized(self):
        """Lazy-init MCP client on first use of web search toggle."""
        if self._mcp_initialized:
            return
        self._mcp_initialized = True

        if not MCP_AVAILABLE:
            self.display_status_message("MCP not available — using DuckDuckGo fallback")
            return

        config_path = os.path.join(os.path.dirname(__file__), "config", "mcp_coderunner.json")
        if not os.path.exists(config_path):
            self.display_status_message(f"MCP config not found at {config_path} — using DuckDuckGo fallback")
            return

        try:
            self.display_status_message("Starting MCP servers...")
            self.mcp_client = MCPClient(config_path)
            self.mcp_client.start_servers()
            tools = self.mcp_client.list_tools()
            n_servers = len(self.mcp_client.servers)
            self.display_status_message(f"MCP initialized: {len(tools)} tools from {n_servers} servers")
            for t in tools:
                self.add_to_debug_console(f"  MCP tool: {t.server}/{t.name}")
        except Exception as e:
            self.display_status_message(f"MCP init failed: {e} — using DuckDuckGo fallback")
            self.add_to_debug_console(f"MCP init error: {e}")
            self.mcp_client = None

    def toggle_search_mode(self):
        """Toggle web search mode on/off"""
        is_enabled = self.search_mode.get()
        status = "enabled" if is_enabled else "disabled"
        self.display_status_message(f"Web search mode {status}")
        self.add_to_debug_console(f"Web search mode: {status}")

        if is_enabled:
            self._ensure_mcp_initialized()
            # In HTML mode, tell the LLM it can search for and embed images
            if self.system_mode.get() == "html_programmer":
                addendum = ("\n\nWhen web search is enabled you have access to tools: "
                            "brave_web_search (search the web), fetch (fetch a URL), "
                            "and fetch_image (download an image and save it locally). "
                            "To use real images in HTML: first search for image URLs, then call "
                            "fetch_image for each URL. It returns a relative path like 'assets/sprite_abc.png' "
                            "that you can use directly in <img src='assets/sprite_abc.png'>.")
                if addendum not in self.system_message.get('content', ''):
                    self.system_message['content'] += addendum
                    if self.messages and self.messages[0].get('role') == 'system':
                        self.messages[0]['content'] = self.system_message['content']
        else:
            # Strip the search addendum when toggling off
            content = self.system_message.get('content', '')
            marker = "\n\nWhen web search is enabled you have access to tools:"
            idx = content.find(marker)
            if idx != -1:
                self.system_message['content'] = content[:idx]
                if self.messages and self.messages[0].get('role') == 'system':
                    self.messages[0]['content'] = self.system_message['content']

    def _parse_mlx_tool_calls(self, response_text):
        """Parse tool calls from MLX text output, handling multiple formats.

        Supports:
        - Qwen3 JSON: <tool_call>{"name":..., "arguments":...}</tool_call>
        - XML format: <tool><name>...</name><arguments><key>k</key><value>v</value>...</arguments></tool>

        Populates self._pending_tool_results and returns response with tool blocks stripped.
        """
        import re as _re

        # --- Format 1: Qwen3 JSON <tool_call> blocks ---
        json_pattern = _re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', _re.DOTALL)
        for match in json_pattern.finditer(response_text):
            try:
                tc_data = json.loads(match.group(1))
                tc_name = tc_data.get('name', '')
                tc_args = tc_data.get('arguments', {})
                if isinstance(tc_args, str):
                    tc_args = json.loads(tc_args)
                self.display_chat_system_message(f"🔍 Tool call: {tc_name}({json.dumps(tc_args)[:100]})")
                self._pending_tool_results.append({
                    'name': tc_name, 'arguments': tc_args,
                    'id': str(uuid.uuid4()), 'backend': 'mlx'
                })
            except (json.JSONDecodeError, KeyError) as e:
                self.add_to_debug_console(f"⚠️ Could not parse <tool_call> JSON: {e}")
        response_text = json_pattern.sub('', response_text)

        # --- Format 2: XML <tool> blocks (some VLM models) ---
        # Matches: <tool><name>tool_name</name><arguments><key>k</key><value>v</value>...</arguments></tool>
        xml_tool_pattern = _re.compile(r'<tools?>\s*(.*?)\s*</tools?>', _re.DOTALL)
        xml_single = _re.compile(r'<tool>\s*(.*?)\s*</tool>', _re.DOTALL)
        xml_name = _re.compile(r'<name>\s*(.*?)\s*</name>', _re.DOTALL)
        xml_kv = _re.compile(r'<key>\s*(.*?)\s*</key>\s*<value>\s*(.*?)\s*</value>', _re.DOTALL)

        for tools_match in xml_tool_pattern.finditer(response_text):
            tools_block = tools_match.group(1)
            # Could contain multiple <tool> blocks or be a single tool
            tool_blocks = xml_single.findall(tools_block)
            if not tool_blocks:
                tool_blocks = [tools_block]  # Treat the whole thing as one tool

            for tool_block in tool_blocks:
                name_match = xml_name.search(tool_block)
                if not name_match:
                    continue
                tc_name = name_match.group(1).strip()
                tc_args = {}
                for kv_match in xml_kv.finditer(tool_block):
                    tc_args[kv_match.group(1).strip()] = kv_match.group(2).strip()
                self.display_chat_system_message(f"🔍 Tool call: {tc_name}({json.dumps(tc_args)[:100]})")
                self._pending_tool_results.append({
                    'name': tc_name, 'arguments': tc_args,
                    'id': str(uuid.uuid4()), 'backend': 'mlx'
                })
        response_text = xml_tool_pattern.sub('', response_text)

        return response_text.strip()

    def _execute_tool_call(self, name, arguments):
        """Execute a tool call and return the result as text.

        Routes to MCP servers, local helpers, or DuckDuckGo fallback.
        Handles both prefixed (mcp_brave-search_brave_web_search) and
        bare tool names (brave_web_search) from different model formats.
        """
        try:
            # Local helper: fetch_image → saves to Generated_games/assets/
            if name == "fetch_image":
                url = arguments.get("url", "")
                self.display_status_message(f"Fetching image: {url[:80]}...")
                result = fetch_image_as_file(url)
                if not result.startswith("Error"):
                    self.display_status_message(f"Saved: {result}")
                return result

            # Legacy web_search → route to brave-search MCP or DuckDuckGo
            if name == "web_search":
                query = arguments.get("query", "")
                if self.mcp_client and "brave-search" in self.mcp_client.servers:
                    self.display_status_message(f"Brave search: {query[:60]}...")
                    return self.mcp_client.call_tool("brave-search", "brave_web_search", {"query": query})
                else:
                    self.display_status_message(f"DuckDuckGo search: {query[:60]}...")
                    return safe_web_search(query)

            # MCP tool: mcp_<server>_<toolname>
            if name.startswith("mcp_") and self.mcp_client:
                parts = name.split("_", 2)
                if len(parts) >= 3:
                    server_name = parts[1]
                    tool_name = parts[2]
                    self.display_status_message(f"MCP {server_name}/{tool_name}...")
                    return self.mcp_client.call_tool(server_name, tool_name, arguments)

            # Bare tool name (model didn't use mcp_ prefix) → search MCP servers
            if self.mcp_client:
                for server in self.mcp_client.servers.values():
                    for tool in server.tools:
                        if tool.name == name:
                            self.display_status_message(f"MCP {server.name}/{name}...")
                            return self.mcp_client.call_tool(server.name, name, arguments)

            return f"Unknown tool: {name}"
        except Exception as e:
            error_msg = f"Tool error ({name}): {e}"
            self.add_to_debug_console(error_msg)
            return error_msg

    def toggle_sampling_controls(self):
        """Toggle visibility of sampling controls (temperature, top_p, top_k)"""
        if self.show_sampling_controls.get():
            # Show sampling controls
            self.temp_frame.pack(fill=tk.X, pady=(0, 10))
            self.topp_frame.pack(fill=tk.X, pady=(0, 5))
            self.topk_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            # Hide sampling controls
            self.temp_frame.pack_forget()
            self.topp_frame.pack_forget()
            self.topk_frame.pack_forget()

    def update_system_message_for_targeted_changes(self):
        """Update the system message based on current code mode.
        (SEARCH/REPLACE instruction removed — fix prompt handles format.)"""
        mode = self.system_mode.get()
        if mode == "html_programmer":
            base_content = html_system_message
        else:
            base_content = python_system_message

        self.system_message['content'] = base_content

        if self.messages and self.messages[0]['role'] == 'system':
            self.messages[0]['content'] = base_content

    def change_model(self, *args):
        """Handle unified model selection change for all backends"""
        selected_model = self.model_var.get()
        backend = self.backend_var.get()
        
        if selected_model:
            if backend == "ollama":
                self.model_status_label.config(text=f"Model selected: {selected_model} - click 'Load Model' to load")
            elif backend == "llama_cpp":
                # For GGUF models, update the selected_gguf_path
                for path in self.available_gguf_models:
                    if os.path.basename(path) == selected_model:
                        self.selected_gguf_path.set(path)
                        break
                self.model_status_label.config(text=f"Model selected: {selected_model} - click 'Load Model' to load")
            elif backend == "mlx":
                # For MLX models, strip [VL]/[TX] tag and update selected_mlx_path
                stripped_name = selected_model
                if stripped_name.startswith("[VL] ") or stripped_name.startswith("[TX] "):
                    stripped_name = stripped_name[5:]
                for path in self.available_mlx_models:
                    if os.path.basename(path) == stripped_name:
                        self.selected_mlx_path.set(path)
                        break
                self.model_status_label.config(text=f"Model selected: {selected_model} - click 'Load Model' to load")
            elif backend == "vllm":
                # For vLLM models, update the selected_vllm_path
                for path in self.available_vllm_models:
                    if os.path.basename(path) == selected_model:
                        self.selected_vllm_path.set(path)
                        break
                self.model_status_label.config(text=f"Model selected: {selected_model} - click 'Load Model' to load")
            elif backend == "transformers":
                # For transformers models, update the selected_transformers_path
                for path in self.available_transformers_models:
                    if os.path.basename(path) == selected_model:
                        self.selected_transformers_path.set(path)
                        break
                self.model_status_label.config(text=f"Model selected: {selected_model} - click 'Load Model' to load")
            elif backend == "claude":
                # Sync with claude_model_var
                self.claude_model_var.set(selected_model)
                self.model_status_label.config(text=f"Selected: {selected_model}", fg="green")
            elif backend == "openai":
                # OpenAI models are ready to use immediately
                self.model_status_label.config(text=f"Selected: {selected_model}", fg="green")
        else:
            self.model_status_label.config(text="No model selected")
             
    def load_ollama_model(self):
        """Load the selected Ollama model"""
        selected_model = self.model_var.get()
        if not selected_model:
            self.display_status_message("Please select a model first.")
            return
            
        # Clear any previously loaded local models before loading new one
        if hasattr(self, 'model') and self.model:
            self.display_status_message(f"Unloading previous Ollama model: {self.model}")
            self.model = None
        if self.llama_cpp_model:
            self.display_status_message("Unloading llama-cpp model...")
            self.llama_cpp_model = None
            
        # Check if Ollama is available - don't auto-start
        if not self.check_ollama_available():
            self.display_status_message("Ollama not running. Please start Ollama manually using '/startollama' command before loading models.")
            return
            
        # If Ollama is available, complete the model loading
        self._complete_ollama_model_load()
        
    def _complete_ollama_model_load(self):
        """Complete the Ollama model loading process"""
        selected_model = self.model_var.get()
        
        # Check if Ollama is now available
        if not self.check_ollama_available():
            self.display_status_message("Error: Could not start Ollama service. Please start it manually.")
            return
            
        # Set the current model
        self.model = selected_model
        self.model_status_label.config(text=f"Loaded: {selected_model}", fg="green")
        # ROUTING FIX: Send Ollama load success to System Console (not chat history)
        self.display_message("System", f"✓ Successfully loaded Ollama model: {selected_model}", to_chat=False)
        
        # Now that a model is loaded, enable the Ollama backend if not already active
        if self.backend_var.get() != "ollama":
            self.backend_var.set("ollama")
            self.change_backend()
            
    def refresh_models(self):
        """Refresh the list of available models"""
        # Check if Ollama is available
        if not self.check_ollama_available():
            self.display_status_message("Error: Ollama service is not available. Please start Ollama first.")
            return
            
        prev_model = self.model_var.get()
        
        self.available_models = get_available_models()
        
        # Update the dropdown with new models
        self.model_dropdown['values'] = self.available_models
        
        # If the current model is still available, keep it selected
        if prev_model in self.available_models:
            self.model_var.set(prev_model)
        else:
            self.model_var.set(self.available_models[0] if self.available_models else model)
            
        self.display_status_message("Model list refreshed")
        
    def refresh_gguf_models(self):
        """Refresh the list of available GGUF models"""
        if not LLAMA_CPP_AVAILABLE:
            self.display_status_message("GGUF not available.")
            return
            
        try:
            self.available_gguf_models = find_gguf_models()
            # Update dropdown with just filenames for display
            self.gguf_dropdown['values'] = [os.path.basename(p) for p in self.available_gguf_models]
            
            if self.available_gguf_models:
                # If current selection is still valid, keep it
                current_path = self.selected_gguf_path.get()
                if current_path not in self.available_gguf_models:
                    self.selected_gguf_path.set(self.available_gguf_models[0])
                    self.gguf_dropdown.set(os.path.basename(self.available_gguf_models[0]))
            
            self.display_status_message(f"Found {len(self.available_gguf_models)} GGUF models")
        except Exception as e:
            self.display_status_message(f"Error refreshing GGUF models: {str(e)}")

    def refresh_mlx_models(self):
        """Refresh the list of available MLX models"""
        if not MLX_AVAILABLE:
            self.display_status_message("MLX not available.")
            return

        try:
            self.available_mlx_models, self.mlx_model_vlm_flags = get_available_mlx_models()

            if self.available_mlx_models:
                # If current selection is still valid, keep it
                current_path = self.selected_mlx_path.get()
                if current_path not in self.available_mlx_models:
                    self.selected_mlx_path.set(self.available_mlx_models[0])

            vlm_count = sum(1 for v in self.mlx_model_vlm_flags.values() if v)
            self.display_status_message(f"Found {len(self.available_mlx_models)} MLX models ({vlm_count} VLM, {len(self.available_mlx_models) - vlm_count} text-only)")
        except Exception as e:
            self.display_status_message(f"Error refreshing MLX models: {str(e)}")

    def _refresh_mlx_dropdown_tags(self):
        """Rebuild MLX dropdown values to match current vlm_flags (e.g. after a VLM load failure)."""
        if self.backend_var.get() != "mlx":
            return
        mlx_display_names = []
        for p in self.available_mlx_models:
            tag = "[VL] " if self.mlx_model_vlm_flags.get(p, False) else "[TX] "
            mlx_display_names.append(tag + os.path.basename(p))
        self.model_dropdown['values'] = mlx_display_names

    def change_claude_model(self, event=None):
        """Handle Claude model selection change"""
        selected_model = self.claude_dropdown.get()
        self.claude_model_var.set(selected_model)
        if selected_model:
            self.model_status_label.config(text=f"Selected: {selected_model}", fg="green")
    
    def load_claude_model(self):
        """Load/select the Claude model (no actual loading needed)"""
        selected_model = self.claude_model_var.get()
        if not selected_model:
            self.display_status_message("Please select a Claude model first")
            return
        
        self.model_status_label.config(text=f"Ready: {selected_model}", fg="green")
        self.display_status_message(f"Claude model ready: {selected_model}")
    
    def refresh_claude_models(self):
        """Refresh the list of available Claude models"""
        try:
            self.available_claude_models = get_claude_models()
            self.claude_dropdown['values'] = self.available_claude_models
            
            if self.available_claude_models:
                # If current selection is still valid, keep it
                current_model = self.claude_model_var.get()
                if current_model not in self.available_claude_models:
                    # Set default to claude-3-5-sonnet if available
                    default_claude = "claude-3-5-sonnet-20241022"
                    if default_claude in self.available_claude_models:
                        self.claude_model_var.set(default_claude)
                    else:
                        self.claude_model_var.set(self.available_claude_models[0])
            
            self.display_status_message(f"Found {len(self.available_claude_models)} Claude models")
        except Exception as e:
            self.display_status_message(f"Error refreshing Claude models: {str(e)}")
            
    def refresh_openai_models(self):
        """Refresh the list of available OpenAI chat models"""
        try:
            self.openai_models = get_openai_models()
            # If currently on OpenAI backend, update the dropdown
            if self.backend_var.get() == "openai":
                self.model_dropdown['values'] = self.openai_models
                # Keep current if still valid, otherwise pick a sensible default
                current = self.model_var.get()
                if current not in self.openai_models:
                    preferred = next((m for m in self.openai_models if m.startswith("gpt-5")), None)
                    if not preferred and "gpt-4o" in self.openai_models:
                        preferred = "gpt-4o"
                    self.model_var.set(preferred or (self.openai_models[0] if self.openai_models else ""))
            self.display_status_message(f"Found {len(self.openai_models)} OpenAI models")
        except Exception as e:
            self.display_status_message(f"Error refreshing OpenAI models: {str(e)}")

    def refresh_vllm_models(self):
        """Refresh available vLLM models"""
        try:
            self.available_vllm_models = get_available_vllm_models()

            if self.available_vllm_models:
                self.display_status_message(f"Found {len(self.available_vllm_models)} vLLM models")
                # If current selection is still valid, keep it
                current_path = self.selected_vllm_path.get()
                if current_path and current_path not in self.available_vllm_models:
                    # Current selection no longer valid, select first available
                    self.selected_vllm_path.set(self.available_vllm_models[0])
            else:
                self.display_status_message("No vLLM models found in Models_Transformer directory")

            # Update dropdown if we're on vLLM backend
            if self.backend_var.get() == "vllm":
                if self.available_vllm_models:
                    vllm_display_names = [os.path.basename(p) for p in self.available_vllm_models]
                    self.model_dropdown['values'] = vllm_display_names
                else:
                    self.model_dropdown['values'] = []

        except Exception as e:
            self.display_status_message(f"Error refreshing vLLM models: {str(e)}")
            self.available_vllm_models = []


    def change_backend(self):
        """Handle backend selection change"""
        backend = self.backend_var.get()
        
        if backend == "ollama":
            # Populate dropdown with Ollama models
            self.model_dropdown['values'] = self.available_models
            # Check if a model is already loaded
            if hasattr(self, 'model') and self.model and self.model in self.available_models:
                self.model_var.set(self.model)
                self.model_status_label.config(text=f"Loaded: {self.model}", fg="green")
            else:
                self.model_var.set(self.available_models[0] if self.available_models else "")
                self.model_status_label.config(text="Select and load a model to use Ollama backend", fg="orange")
            # Hide max tokens controls for Ollama
            self.max_tokens_frame.pack_forget()
            # Unload llama-cpp model if loaded
            if self.llama_cpp_model:
                self.llama_cpp_model = None
                
        elif backend == "llama_cpp":
            if not LLAMA_CPP_AVAILABLE:
                self.display_status_message("GGUF not available. Please install it: pip install llama-cpp-python")
                self.backend_var.set("ollama")  # Revert to Ollama
                return
                
            # Populate dropdown with GGUF models (display names only)
            gguf_display_names = [os.path.basename(p) for p in self.available_gguf_models]
            self.model_dropdown['values'] = gguf_display_names
            # Show max tokens controls for GGUF
            self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))
            
            if self.llama_cpp_model:
                model_name = os.path.basename(self.selected_gguf_path.get())
                self.model_var.set(model_name)
                self.model_status_label.config(text=f"Loaded: {model_name}")
            else:
                self.model_var.set(gguf_display_names[0] if gguf_display_names else "")
                self.model_status_label.config(text="No model loaded - click 'Load Model'")

        elif backend == "mlx":
            if not MLX_AVAILABLE:
                self.display_status_message("MLX not available. Please install it: pip install mlx mlx-lm")
                self.backend_var.set("ollama")  # Revert to Ollama
                return

            # Populate dropdown with MLX models — tag [VL] or [TX]
            mlx_display_names = []
            for p in self.available_mlx_models:
                tag = "[VL] " if self.mlx_model_vlm_flags.get(p, False) else "[TX] "
                mlx_display_names.append(tag + os.path.basename(p))
            self.model_dropdown['values'] = mlx_display_names
            # Show max tokens controls for MLX
            self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))

            if self.mlx_model:
                model_path = self.selected_mlx_path.get()
                tag = "[VL] " if self.mlx_model_vlm_flags.get(model_path, False) else "[TX] "
                model_name = tag + os.path.basename(model_path)
                self.model_var.set(model_name)
                self.model_status_label.config(text=f"Loaded: {model_name}", fg="green")
            else:
                self.model_var.set(mlx_display_names[0] if mlx_display_names else "")
                self.model_status_label.config(text="No model loaded - click 'Load Model'", fg="orange")
            # Unload llama-cpp model if loaded
            if self.llama_cpp_model:
                self.llama_cpp_model = None

        elif backend == "vllm":
            # vLLM works on x86 and ARM (e.g. DGX Spark Blackwell) with CUDA
            if not VLLM_AVAILABLE:
                self.display_status_message("vLLM not available. Please install vLLM and ensure CUDA is available: pip install vllm torch")
                self.backend_var.set("ollama")  # Revert to Ollama
                return

            # Populate dropdown with vLLM models (display names only)
            if self.available_vllm_models:
                vllm_display_names = [os.path.basename(p) for p in self.available_vllm_models]
                self.model_dropdown['values'] = vllm_display_names
            else:
                # Fallback to refresh models if none available
                self.refresh_vllm_models()
                if self.available_vllm_models:
                    vllm_display_names = [os.path.basename(p) for p in self.available_vllm_models]
                    self.model_dropdown['values'] = vllm_display_names
                else:
                    self.model_dropdown['values'] = []
            # Show max tokens controls for vLLM
            self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))

            if hasattr(self, 'vllm_model') and self.vllm_model:
                # Show the basename of the loaded model
                current_path = self.selected_vllm_path.get()
                if current_path:
                    model_name = os.path.basename(current_path)
                    self.model_var.set(model_name)
                    self.model_status_label.config(text=f"Loaded: {model_name}", fg="green")
                else:
                    self.model_var.set("")
                    self.model_status_label.config(text="No model loaded", fg="orange")
            else:
                # Set default selection to first available model
                if self.available_vllm_models:
                    first_model_path = self.available_vllm_models[0]
                    first_model_name = os.path.basename(first_model_path)
                    self.model_var.set(first_model_name)
                    self.selected_vllm_path.set(first_model_path)
                    self.model_status_label.config(text="vLLM model ready - click 'Load Model'", fg="orange")
                else:
                    self.model_var.set("")
                    self.model_status_label.config(text="No vLLM models found", fg="red")
            # Unload llama-cpp model if loaded
            if self.llama_cpp_model:
                self.llama_cpp_model = None

        elif backend == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                self.display_status_message("Transformers not available. Please install with: pip install transformers torch")
                self.backend_var.set("ollama")  # Revert to Ollama
                return

            # Populate dropdown with transformers models (display names only)
            if self.available_transformers_models:
                transformers_display_names = [os.path.basename(p) for p in self.available_transformers_models]
                self.model_dropdown['values'] = transformers_display_names
            else:
                # Fallback to refresh models if none available
                self.refresh_transformers_models()
                if self.available_transformers_models:
                    transformers_display_names = [os.path.basename(p) for p in self.available_transformers_models]
                    self.model_dropdown['values'] = transformers_display_names
                else:
                    self.model_dropdown['values'] = []
            # Show max tokens controls for transformers
            self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))

            if self.transformers_model:
                # Show the basename of the loaded model
                current_path = self.selected_transformers_path.get()
                if current_path:
                    model_name = os.path.basename(current_path)
                    self.model_var.set(model_name)
                    self.model_status_label.config(text=f"Loaded: {model_name}", fg="green")
                else:
                    self.model_var.set("")
                    self.model_status_label.config(text="No model loaded", fg="orange")
            else:
                # Set default selection to first available model
                if self.available_transformers_models:
                    first_model_path = self.available_transformers_models[0]
                    first_model_name = os.path.basename(first_model_path)
                    self.model_var.set(first_model_name)
                    self.selected_transformers_path.set(first_model_path)
                    self.model_status_label.config(text="Transformers model ready - click 'Load Model'", fg="orange")
                else:
                    self.model_var.set("")
                    self.model_status_label.config(text="No Transformers models found", fg="red")
            # Unload llama-cpp model if loaded
            if self.llama_cpp_model:
                self.llama_cpp_model = None


        elif backend == "claude":
            # Populate dropdown with Claude models
            self.model_dropdown['values'] = self.available_claude_models
            # Show max tokens controls for Claude
            self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))
            
            if self.claude_model_var.get():
                self.model_var.set(self.claude_model_var.get())
                self.model_status_label.config(text=f"Selected: {self.claude_model_var.get()}", fg="green")
            else:
                self.model_var.set(self.available_claude_models[0] if self.available_claude_models else "")
                self.model_status_label.config(text="Select a Claude model", fg="orange")
            # Unload llama-cpp model if loaded  
            if self.llama_cpp_model:
                self.llama_cpp_model = None
        
        elif backend == "openai":
            # Populate dropdown with OpenAI models
            self.model_dropdown['values'] = self.openai_models
            # Show max tokens controls for OpenAI (parity with Claude)
            self.max_tokens_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Set default to GPT-5 or GPT-4o
            current_openai_model = next((m for m in self.openai_models if m.startswith("gpt-5")), None) or ("gpt-4o" if "gpt-4o" in self.openai_models else (self.openai_models[0] if self.openai_models else ""))
            self.model_var.set(current_openai_model)
            if current_openai_model:
                self.model_status_label.config(text=f"Selected: {current_openai_model}", fg="green")
            else:
                self.model_status_label.config(text="Select an OpenAI model", fg="orange")
                
        self.display_status_message(f"Switched to {backend} backend")
    
    def load_current_backend_model(self):
        """Load model for the currently selected backend"""
        backend = self.backend_var.get()
        
        if backend == "ollama":
            self.load_ollama_model()
        elif backend == "llama_cpp":
            self.load_gguf_model()
        elif backend == "mlx":
            self.load_mlx_model()
        elif backend == "vllm":
            # vLLM works on x86 and ARM (e.g. DGX Spark Blackwell) with CUDA
            self.load_vllm_model()
        elif backend == "transformers":
            self.load_transformers_model()
        elif backend == "claude":
            self.load_claude_model()
        elif backend == "openai":
            # OpenAI models don't need loading, just selection
            selected_model = self.model_var.get()
            if selected_model:
                self.model_status_label.config(text=f"Selected: {selected_model}", fg="green")
            else:
                self.model_status_label.config(text="Select an OpenAI model", fg="orange")

        # Update Ollama status indicator based on current backend
        self.update_ollama_status()
    
    def refresh_current_backend_models(self):
        """Refresh models for the currently selected backend"""
        backend = self.backend_var.get()
        
        if backend == "ollama":
            self.refresh_models()
        elif backend == "llama_cpp":
            self.refresh_gguf_models()
        elif backend == "mlx":
            self.refresh_mlx_models()
        elif backend == "vllm":
            # vLLM works on x86 and ARM (e.g. DGX Spark Blackwell) with CUDA
            self.refresh_vllm_models()
        elif backend == "transformers":
            self.refresh_transformers_models()
        elif backend == "claude":
            self.refresh_claude_models()
        elif backend == "openai":
            self.refresh_openai_models()
        
        # Update dropdown after refresh
        self.change_backend()
        
    def change_gguf_model(self, event=None):
        """Handle GGUF model selection change"""
        selected_name = self.gguf_dropdown.get()
        # Find the full path for the selected model
        for path in self.available_gguf_models:
            if os.path.basename(path) == selected_name:
                self.selected_gguf_path.set(path)
                break
                
        # Update status if model is not yet loaded
        if not self.llama_cpp_model:
            self.model_status_label.config(text="Model selected - click 'Load Model' to load")

    def change_vllm_model(self, event=None):
        """Handle vLLM model selection change"""
        selected_name = self.model_var.get()
        # Find the full path for the selected model
        for path in self.available_vllm_models:
            if os.path.basename(path) == selected_name:
                self.selected_vllm_path.set(path)
                break

        # Update status if model is not yet loaded
        if not self.vllm_model:
            self.model_status_label.config(text="Model selected - click 'Load Model' to load")

    def load_gguf_model(self):
        """Load the selected GGUF model"""
        if not LLAMA_CPP_AVAILABLE:
            self.display_status_message("GGUF not available.")
            return
            
        model_path = self.selected_gguf_path.get()
        if not model_path or not os.path.exists(model_path):
            self.display_status_message("Please select a valid GGUF model first.")
            return
            
        # Clear any previously loaded local models before loading new one
        if self.llama_cpp_model:
            self.display_status_message("Unloading previous llama-cpp model...")
            self.llama_cpp_model = None
        if hasattr(self, 'model') and self.model:
            self.display_status_message(f"Unloading Ollama model: {self.model}")
            self.model = None
            
        # Disable UI during loading
        self.load_model_btn.config(state=tk.DISABLED, text="Loading...")
        self.model_status_label.config(text="Loading model...")
        
        # Start loading in background thread
        threading.Thread(target=self._load_gguf_model_thread, args=(model_path,)).start()
        
    def _load_gguf_model_thread(self, model_path):
        """Load GGUF model in background thread"""
        try:
            # Clean up any existing model first to avoid conflicts
            if hasattr(self, 'llama_cpp_model') and self.llama_cpp_model:
                try:
                    del self.llama_cpp_model
                except:
                    pass  # Ignore cleanup errors
                self.llama_cpp_model = None
            
            # Detect model type from filename to set appropriate parameters
            model_name = os.path.basename(model_path).lower()
            
            # Check for Unsloth UD-Q8_K_XL format which requires special handling
            is_unsloth_xl = "q8_k_xl" in model_name or "q8_xl" in model_name

            # Define GPU layer calculation function (outside conditional blocks)
            def get_optimal_gpu_layers(model_path, model_name):
                """Get optimal GPU layers - aggressive approach for RTX 4090"""
                try:
                    # Get model file size to make informed decisions
                    model_size_gb = os.path.getsize(model_path) / (1024**3)

                    # RTX 4090 with 24GB VRAM can handle much more than 24 layers
                    # Let llama-cpp-python auto-manage GPU/CPU split

                    if model_size_gb < 5:  # Small models (< 5GB)
                        gpu_layers = 60  # Very aggressive - can fit most/all layers
                    elif model_size_gb < 15:  # Medium models (5-15GB)
                        gpu_layers = 50  # Aggressive - GPU handles most work
                    elif model_size_gb < 30:  # Large models (15-30GB)
                        gpu_layers = 40  # Moderate - good GPU/CPU balance
                    else:  # Very large models (>30GB)
                        gpu_layers = 35  # Conservative but still much better than 24

                    # Debug logging
                    print(f"🤖 GPU Layer Calculation:")
                    print(f"   Model: {os.path.basename(model_path)}")
                    print(f"   Size: {model_size_gb:.1f}GB")
                    print(f"   GPU Layers: {gpu_layers}")
                    print(f"   VRAM Estimate: ~{gpu_layers * model_size_gb / 50:.1f}GB used")

                    return gpu_layers

                except Exception as e:
                    # If we can't get file size, be aggressive anyway
                    print(f"⚠️  Could not calculate model size: {e}")
                    print(f"   Using default GPU layers: 50")
                    return 50  # Better to try more layers than fewer

            # Check if CUDA is actually available before trying GPU layers
            cuda_supported = False
            try:
                import llama_cpp
                cuda_supported = llama_cpp.llama_supports_gpu_offload()
            except:
                cuda_supported = False

            if not cuda_supported:
                print("❌ CUDA not available - forcing CPU-only mode")
                n_gpu_layers = 0
            else:
                # CUDA is available - use aggressive GPU layer allocation
                n_gpu_layers = get_optimal_gpu_layers(model_path, model_name)

            # Set context size and GPU layers
            if "glm" in model_name:
                # GLM models - use conservative settings
                n_ctx = 32768  # 32k context for GLM models
                chat_format = None  # Let llama-cpp-python auto-detect
                n_gpu_layers = get_optimal_gpu_layers(model_path, model_name)
            elif "qwen" in model_name:
                # Qwen models support large context
                n_ctx = 262144  # 256k context for Qwen models
                chat_format = "chatml"
                n_gpu_layers = get_optimal_gpu_layers(model_path, model_name)
            else:
                # Default for other models - be aggressive!
                n_ctx = 32768  # Safe default
                chat_format = None  # Auto-detect
                n_gpu_layers = get_optimal_gpu_layers(model_path, model_name)

            # Override GPU layers for Unsloth XL formats - be aggressive with dual RTX 4090s
            if is_unsloth_xl:
                # With 48GB total VRAM (2x RTX 4090), we can handle much more than 20 layers
                # Q8_K_XL format should work fine with modern CUDA builds
                original_layers = n_gpu_layers
                n_gpu_layers = min(n_gpu_layers, 45)  # Very aggressive - you have 48GB VRAM!
                print(f"⚠️  Q8_K_XL format detected - using {n_gpu_layers} GPU layers (reduced from {original_layers} for Q8_K_XL compatibility)")
                print(f"   With 48GB VRAM (2x RTX 4090), this should fit comfortably in GPU memory")
                self.root.after(0, lambda: self.display_message("System", f"Q8_K_XL format: Using {n_gpu_layers} GPU layers with dual RTX 4090s.", to_chat=False))
            
            # Debug: Check CUDA support before loading
            print(f"🔍 Pre-load CUDA check:")
            try:
                import llama_cpp
                cuda_supported = llama_cpp.llama_supports_gpu_offload()
                print(f"   CUDA available: {cuda_supported}")

                # Try to get GPU count - handle missing function gracefully
                try:
                    gpu_count = getattr(llama_cpp, 'llama_get_gpu_count', lambda: 'N/A')()
                    print(f"   GPU count: {gpu_count}")
                except:
                    print(f"   GPU count: Unable to detect")

                if not cuda_supported:
                    print(f"   ⚠️  CUDA not supported - model will run on CPU only")

            except Exception as e:
                print(f"   CUDA check failed: {e}")
                cuda_supported = False

            # Debug: Show what we're trying to load
            print(f"🚀 Initializing llama-cpp model:")
            print(f"   Model: {os.path.basename(model_path)}")
            print(f"   Context: {n_ctx}")
            print(f"   GPU Layers: {n_gpu_layers}")
            print(f"   Chat Format: {chat_format}")

            # Initialize the model with optimized settings for high RAM system
            self.llama_cpp_model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,  # Adaptive context window based on model type
                n_threads=16,  # More CPU threads for better performance
                n_threads_batch=8,  # Batch processing threads
                n_batch=2048,  # Larger batch size for efficiency
                use_mlock=True,  # Lock memory to prevent swapping
                use_mmap=True,  # Memory-map the model file
                rope_freq_base=1000000.0,  # RoPE frequency base for modern models
                rope_freq_scale=1.0,  # RoPE frequency scaling
                n_gpu_layers=n_gpu_layers,  # Adaptive GPU layers based on model type
                chat_format=chat_format,  # Adaptive chat format based on model type
                verbose=True  # Enable verbose to see CUDA initialization
            )

            # Verify GPU usage after loading
            print(f"✅ Model loaded successfully!")
            try:
                # Check available attributes
                gpu_layers = getattr(self.llama_cpp_model, 'n_gpu_layers', 0)
                total_layers = getattr(self.llama_cpp_model, 'n_layers', lambda: 0)()

                print(f"   GPU offload supported: {gpu_layers > 0}")
                print(f"   Layers on GPU: {gpu_layers}")
                print(f"   Total layers: {total_layers}")
                if total_layers > 0:
                    print(f"   GPU memory usage: ~{(gpu_layers / total_layers) * 100:.1f}%")
                else:
                    print(f"   GPU memory usage: Unknown (could not get total layers)")

                # Show available methods for debugging
                available_attrs = [attr for attr in dir(self.llama_cpp_model) if not attr.startswith('_')]
                print(f"   Available attributes: {available_attrs[:10]}...")  # Show first 10

            except Exception as e:
                print(f"   GPU verification failed: {e}")
                print(f"   Model type: {type(self.llama_cpp_model)}")
            
            # Update UI in main thread
            model_name = os.path.basename(model_path)
            self.root.after(0, lambda: self.model_status_label.config(text=f"Loaded: {model_name}", fg="green"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            # ROUTING FIX: Force GGUF load success to System Console (not chat history)
            self.root.after(0, lambda: self.display_message("System", f"Successfully loaded GGUF model: {model_name}", to_chat=False))
            
        except Exception as e:
            error_msg = str(e)
            model_name = os.path.basename(model_path).lower()
            is_unsloth_xl = "q8_k_xl" in model_name or "q8_xl" in model_name
            
            # Provide specific guidance for different model issues
            if "n_ctx_per_seq" in error_msg and "n_ctx_train" in error_msg:
                error_msg += "\nTip: Model context size has been automatically adjusted. Try loading again."
            elif "sampler" in error_msg:
                if is_unsloth_xl:
                    error_msg += "\nTip: Unsloth UD-Q8_K_XL models require BF16 support. Your llama-cpp-python build may not support this format."
                    error_msg += "\nSolution: Try a regular Q8_0 quantized version of this model instead."
                else:
                    error_msg += "\nTip: This may be a llama-cpp-python version issue. Consider updating: pip install --upgrade llama-cpp-python"
            elif is_unsloth_xl and "bf16" in error_msg.lower():
                error_msg += "\nTip: Unsloth UD-Q8_K_XL models use BF16 precision which requires specific Metal/CUDA support."
                error_msg += "\nSolution: Use a regular Q8_0 quantized version or rebuild llama-cpp-python with full BF16 support."
            
            self.root.after(0, lambda: self.model_status_label.config(text=f"Load failed: {error_msg[:50]}...", fg="red"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            self.root.after(0, lambda: self.display_status_message(f"Failed to load GGUF model: {error_msg}"))

    def load_mlx_model(self):
        """Load the selected MLX model

        Supports all MLX model formats including MiniMax models.
        MiniMax models (mlx-community/MiniMax-M2-mlx-8bit-gs32) are now fully supported.
        """
        if not MLX_AVAILABLE:
            self.display_status_message("MLX not available. Please install it: pip install mlx mlx-lm")
            return

        model_path = self.selected_mlx_path.get()
        if not model_path or not os.path.exists(model_path):
            self.display_status_message("Please select a valid MLX model first.")
            return

        # Clear any previously loaded local models before loading new one
        if self.mlx_model or self.mlx_vlm_model:
            self.display_status_message("Unloading previous MLX model...")
            self.mlx_model = None
            self.mlx_tokenizer = None
            self.mlx_vlm_model = None
            self.mlx_vlm_processor = None
            self.mlx_is_vlm = False
        if self.llama_cpp_model:
            self.display_status_message("Unloading llama-cpp model...")
            self.llama_cpp_model = None
        if hasattr(self, 'model') and self.model:
            self.display_status_message(f"Unloading Ollama model: {self.model}")
            self.model = None

        # Clear GPU memory when switching models (even though MLX is CPU-based, good practice)
        clear_gpu_memory()

        # Disable UI during loading
        self.load_model_btn.config(state=tk.DISABLED, text="Loading...")
        self.model_status_label.config(text="Loading MLX model...")

        # Start loading in background thread
        threading.Thread(target=self._load_mlx_model_thread, args=(model_path,)).start()

    def _load_mlx_model_thread(self, model_path):
        """Load MLX model in background thread.

        If the model is detected as a VLM and mlx-vlm is available, attempts
        vlm_load first. Falls back to text-only mlx_lm.load on any failure.
        """
        try:
            # Clean up any existing model first to avoid conflicts
            if hasattr(self, 'mlx_model') and self.mlx_model:
                try:
                    del self.mlx_model
                    del self.mlx_tokenizer
                except:
                    pass  # Ignore cleanup errors
                self.mlx_model = None
                self.mlx_tokenizer = None
            if hasattr(self, 'mlx_vlm_model') and self.mlx_vlm_model:
                try:
                    del self.mlx_vlm_model
                    del self.mlx_vlm_processor
                except:
                    pass
                self.mlx_vlm_model = None
                self.mlx_vlm_processor = None
            self.mlx_is_vlm = False

            model_name = os.path.basename(model_path)
            is_vlm = self.mlx_model_vlm_flags.get(model_path, False)

            # Attempt VLM loading if model is flagged as VLM and mlx-vlm is available
            if is_vlm and MLX_VLM_AVAILABLE:
                try:
                    self.root.after(0, lambda: self.display_status_message(
                        f"Loading VLM model: {model_name} (vision-language)..."))
                    self.mlx_vlm_model, self.mlx_vlm_processor = vlm_load(model_path)
                    self.mlx_is_vlm = True

                    # Also load as text-only mlx_lm for non-image queries (faster)
                    try:
                        self.mlx_model, self.mlx_tokenizer = load(model_path)
                    except Exception:
                        # If text-only load fails, we'll use VLM path for everything
                        self.mlx_model = None
                        self.mlx_tokenizer = None

                    tag = "[VL] "
                    self.root.after(0, lambda: self.model_status_label.config(
                        text=f"Loaded: {tag}{model_name}", fg="green"))
                    self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
                    self.root.after(0, lambda: self.display_status_message(
                        f"Successfully loaded VLM model: {model_name} (vision + text)"))
                    return
                except Exception as vlm_error:
                    # VLM load failed — fall back to text-only
                    self.mlx_vlm_model = None
                    self.mlx_vlm_processor = None
                    self.mlx_is_vlm = False
                    # Correct the flags cache so dropdown matches reality
                    self.mlx_model_vlm_flags[model_path] = False
                    self.root.after(0, self._refresh_mlx_dropdown_tags)
                    vlm_err_msg = f"VLM load failed for {model_name} ({vlm_error}), loading as text-only..."
                    self.root.after(0, lambda msg=vlm_err_msg: self.display_status_message(msg))

            # Standard text-only loading (original logic)
            try:
                self.mlx_model, self.mlx_tokenizer = load(model_path)
            except Exception as first_error:
                error_str = str(first_error)
                # Only use trust_remote_code if we get a TokenizersBackend error
                if "TokenizersBackend" in error_str or "tokenizer" in error_str.lower():
                    self.mlx_model, self.mlx_tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
                else:
                    raise

            # Update UI on successful load
            tag = "[TX] "
            self.root.after(0, lambda: self.model_status_label.config(text=f"Loaded: {tag}{model_name}", fg="green"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            self.root.after(0, lambda: self.display_status_message(f"Successfully loaded MLX model: {model_name} (text-only)"))

        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error message for tokenizer backend issues
            if "TokenizersBackend" in error_msg or "tokenizer" in error_msg.lower():
                helpful_msg = f"{error_msg}\n\nTip: This may be a tokenizer compatibility issue. Try updating mlx-lm: pip install --upgrade mlx-lm"
            else:
                helpful_msg = error_msg
            self.root.after(0, lambda: self.model_status_label.config(text=f"Load failed: {error_msg[:50]}...", fg="red"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            self.root.after(0, lambda: self.display_status_message(f"Failed to load MLX model: {helpful_msg}"))

    def load_vllm_model(self):
        """Load the selected vLLM model - works on x86 and ARM (e.g. DGX Spark Blackwell) with CUDA"""
        if not VLLM_AVAILABLE:
            self.display_status_message("vLLM not available. Please install vLLM and ensure CUDA is available: pip install vllm torch")
            return

        # Use the selected model path
        model_path = self.selected_vllm_path.get()

        if not model_path or not os.path.exists(model_path):
            self.display_status_message("Selected vLLM model path not found. Please select a valid model.")
            return

        # Clear any previously loaded models
        if hasattr(self, 'vllm_model') and self.vllm_model:
            self.display_status_message("Unloading previous vLLM model...")
            self.vllm_model = None
        if self.llama_cpp_model:
            self.display_status_message("Unloading llama-cpp model...")
            self.llama_cpp_model = None
        if self.mlx_model:
            self.display_status_message("Unloading MLX model...")
            self.mlx_model = None
            self.mlx_tokenizer = None
        if hasattr(self, 'model') and self.model:
            self.display_status_message(f"Unloading Ollama model: {self.model}")
            self.model = None

        # Clear GPU memory when switching models
        clear_gpu_memory()

        # Disable UI during loading
        self.load_model_btn.config(state=tk.DISABLED, text="Loading...")
        self.model_status_label.config(text="Loading vLLM model...")

        # Start loading in background thread
        threading.Thread(target=self._load_vllm_model_thread, args=(model_path,)).start()

    def _load_vllm_model_thread(self, model_path):
        """Load vLLM model in background thread"""
        try:
            # Clean up any existing vLLM model first
            if hasattr(self, 'vllm_model') and self.vllm_model:
                try:
                    del self.vllm_model
                except:
                    pass  # Ignore cleanup errors
                self.vllm_model = None

            # Get user's preferred max tokens from slider
            user_max_tokens = int(self.max_tokens_var.get())

            # Read model's config.json to determine model capabilities
            config_path = os.path.join(model_path, "config.json")
            model_max_len = 32768  # Default fallback

            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    # Try different possible keys for max position embeddings
                    if 'max_position_embeddings' in config:
                        model_max_len = config['max_position_embeddings']
                    elif 'n_positions' in config:
                        model_max_len = config['n_positions']
                    elif 'max_seq_len' in config:
                        model_max_len = config['max_seq_len']

                    # For models with rope_scaling, we might be able to use longer context
                    if 'rope_scaling' in config and config['rope_scaling']:
                        # Some models support longer context with rope scaling
                        if model_max_len < 131072:
                            model_max_len = min(131072, model_max_len * 2)

                except Exception as e:
                    print(f"Warning: Could not read model config, using default max_model_len: {e}")

            # Use the smaller of user preference and model capability
            # This prevents errors while respecting user choice
            max_model_len = min(user_max_tokens, model_max_len)

            # Log the configuration
            model_name = os.path.basename(model_path)
            print(f"Loading vLLM model '{model_name}':")
            print(f"  Model capability: {model_max_len} tokens")
            print(f"  User requested: {user_max_tokens} tokens")
            print(f"  Using: {max_model_len} tokens")

            # Check if model has custom architecture that might need special handling
            model_architecture = None
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'architectures' in config:
                            model_architecture = config['architectures'][0] if config['architectures'] else None
                            print(f"  Architecture: {model_architecture}")
                except:
                    pass

            # Warn about potentially unsupported architectures
            supported_architectures = ['Qwen3NextForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM', 'Mistral3ForCausalLM']
            # Also allow any architecture containing 'mistral' (case insensitive)
            is_mistral_variant = 'mistral' in model_architecture.lower() if model_architecture else False
            if model_architecture and model_architecture not in supported_architectures and not is_mistral_variant:
                print(f"  WARNING: Architecture '{model_architecture}' may not be fully supported by vLLM")
                print(f"  If loading fails, the model may require custom implementation")

            # Determine tensor parallel size based on available GPUs
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            # Use min of available GPUs and 4 (optimal for most large models)
            tensor_parallel_size = min(gpu_count, 4)

            self.add_to_debug_console(f"Loading vLLM model with {tensor_parallel_size} GPU(s) (available: {gpu_count})")

            # Load vLLM model with model-specific settings
            if not os.path.isdir(model_path):
                raise ValueError(f"Model path is not a valid directory: {model_path}")

            # Debug: Log the model path being used
            print(f"vLLM loading from local path: {model_path}")
            print(f"Directory exists: {os.path.exists(model_path)}")
            print(f"Contents: {os.listdir(model_path)[:5] if os.path.exists(model_path) else 'N/A'}")

            # Set PyTorch memory configuration and CUDA debugging
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')  # Keep async for performance but we'll add manual checks

            try:
                self.vllm_model = LLM(
                    model=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=max_model_len,
                    trust_remote_code=True,
                    dtype="auto",
                    gpu_memory_utilization=0.8,  # Reduced from 0.9 to leave more memory for warmup
                    max_num_seqs=8  # Limit concurrent sequences to reduce memory pressure during warmup
                )
            except Exception as vllm_error:
                print(f"vLLM direct loading failed: {vllm_error}")
                # Try alternative approach - sometimes vLLM needs specific parameters
                raise vllm_error

            # Update UI on successful load
            model_name = os.path.basename(model_path)
            self.root.after(0, lambda: self.model_status_label.config(text=f"Loaded: {model_name}", fg="green"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            self.root.after(0, lambda: self.display_status_message(f"Successfully loaded vLLM model: {model_name}"))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.model_status_label.config(text=f"Load failed: {error_msg[:50]}...", fg="red"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            self.root.after(0, lambda: self.display_status_message(f"Failed to load vLLM model: {error_msg}"))

    def load_transformers_model(self):
        """Load the selected Transformers model"""
        if not TRANSFORMERS_AVAILABLE:
            self.display_status_message("Transformers not available. Please install with: pip install transformers torch")
            return

        # Use the selected model path
        model_path = self.selected_transformers_path.get()

        if not model_path or not os.path.exists(model_path):
            self.display_status_message("Selected Transformers model path not found. Please select a valid model.")
            return

        # Clear any previously loaded models
        if self.transformers_model:
            self.display_status_message("Unloading previous Transformers model...")
            self.transformers_model = None
            self.transformers_tokenizer = None
        if self.llama_cpp_model:
            self.display_status_message("Unloading llama-cpp model...")
            self.llama_cpp_model = None
        if self.mlx_model:
            self.display_status_message("Unloading MLX model...")
            self.mlx_model = None
            self.mlx_tokenizer = None
        if hasattr(self, 'vllm_model') and self.vllm_model:
            self.display_status_message("Unloading vLLM model...")
            self.vllm_model = None
        if hasattr(self, 'model') and self.model:
            self.display_status_message(f"Unloading Ollama model: {self.model}")
            self.model = None

        # Clear GPU memory when switching models
        clear_gpu_memory()

        # Disable UI during loading
        self.load_model_btn.config(state=tk.DISABLED, text="Loading...")
        self.model_status_label.config(text="Loading Transformers model...")

        # Start loading in background thread
        threading.Thread(target=self._load_transformers_model_thread, args=(model_path,)).start()

    def _load_transformers_model_thread(self, model_path):
        """Load Transformers model in background thread"""
        try:
            # Clean up any existing transformers model first
            if self.transformers_model:
                try:
                    del self.transformers_model
                    del self.transformers_tokenizer
                except:
                    pass  # Ignore cleanup errors
                self.transformers_model = None
                self.transformers_tokenizer = None

            model_name = os.path.basename(model_path)
            self.add_to_debug_console(f"Loading Transformers model: {model_name}")

            # Load tokenizer - use MistralCommonBackend for Devstral models if available
            if MISTRAL_COMMON_AVAILABLE and "devstral" in model_path.lower():
                self.transformers_tokenizer = MistralCommonBackend.from_pretrained(model_path)
                self.add_to_debug_console("🔷 Using MistralCommonBackend tokenizer")
            elif "nanbeige4" in model_path.lower():
                self.transformers_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
                self.add_to_debug_console("🔷 Using slow tokenizer for Nanbeige4 model")
            else:
                self.transformers_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.add_to_debug_console("🔷 Using standard transformers tokenizer")

            # Standard transformers loading - let transformers handle quantization automatically
            self.add_to_debug_console("Loading model with transformers...")

            import platform
            is_arm = platform.machine() == 'aarch64'

            # Detect model types for specialized loading
            is_gpt_oss_model = "gpt-oss" in model_path.lower()
            is_devstral_model = "devstral" in model_path.lower()

            # Standard transformers loading
            # (environment variables already set globally at import time)
            if is_devstral_model:
                # Devstral models work with MistralForCausalLM
                # FIXED: Removed torch_dtype - let transformers auto-detect (matches gptoss.py)
                # device_map={"": "cuda:0"} ensures explicit GPU placement (not "auto")
                # use_cache=True enables KV caching for faster inference
                self.add_to_debug_console("🔷 Detected Devstral model - using MistralForCausalLM")
                try:
                    self.transformers_model = MistralForCausalLM.from_pretrained(
                        model_path,
                        device_map={"": "cuda:0"},
                        trust_remote_code=True,
                        use_cache=True,  # Enable KV caching for faster inference (matches gptoss.py)
                        low_cpu_mem_usage=True,
                    )
                    self.add_to_debug_console("✅ Mistral model loading successful!")
                except Exception as e:
                    error_msg = str(e)
                    self.add_to_debug_console(f"❌ Mistral model loading failed: {error_msg}")
                    raise
            else:
                # Standard transformers loading
                self.transformers_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": "cuda:0"},
                    dtype="auto",
                    trust_remote_code=True,
                    use_cache=True,
                    local_files_only=True
                )
                self.add_to_debug_console("✅ Model loading successful")



            # Set to eval mode
            self.transformers_model.eval()

            # Verify model is on GPU (standard transformers check)
            device = next(self.transformers_model.parameters()).device
            if device.type != 'cuda':
                self.add_to_debug_console(f"⚠️ WARNING: Model loaded on {device.type}, not GPU!")
                if torch.cuda.is_available():
                    self.add_to_debug_console("Attempting to move model to GPU...")
                    try:
                        self.transformers_model = self.transformers_model.to("cuda")
                        device = next(self.transformers_model.parameters()).device
                        self.add_to_debug_console(f"✅ Model moved to GPU: {device}")
                    except Exception as e:
                        self.add_to_debug_console(f"❌ Failed to move to GPU: {e}")
                        raise Exception(f"Model could not be loaded on GPU: {e}")
                else:
                    raise Exception("CUDA not available - cannot load model on GPU")
            else:
                self.add_to_debug_console(f"✅ Model confirmed on GPU: {device}")

            # Update UI on successful load
            model_name = os.path.basename(model_path)
            gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
            memory_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            self.root.after(0, lambda: self.model_status_label.config(text=f"Loaded: {model_name}", fg="green"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            # Validate model numerics to catch CUDA assert issues early DISABLED

            # Validate model numerics to catch CUDA assert issues early DISABLED

            self.root.after(0, lambda: self.display_status_message(f"Loaded Transformers model: {model_name} on {gpu_name} ({memory_gb:.1f}GB)"))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.model_status_label.config(text=f"Load failed: {error_msg[:50]}...", fg="red"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL, text="Load Model"))
            self.root.after(0, lambda: self.display_status_message(f"Failed to load Transformers model: {error_msg}"))

    def refresh_transformers_models(self):
        """Refresh the list of available Transformers models"""
        if not TRANSFORMERS_AVAILABLE:
            self.display_status_message("Transformers not available.")
            return

        try:
            self.available_transformers_models = get_available_transformers_models()

            if self.available_transformers_models:
                # If current selection is still valid, keep it
                current_path = getattr(self, 'selected_transformers_path', StringVar()).get()
                if current_path and current_path not in self.available_transformers_models:
                    # Set to first available model
                    if hasattr(self, 'selected_transformers_path'):
                        self.selected_transformers_path.set(self.available_transformers_models[0])

            self.display_status_message(f"Found {len(self.available_transformers_models)} Transformers models")
        except Exception as e:
            self.display_status_message(f"Error refreshing Transformers models: {str(e)}")

    def check_ollama_available(self):
        """Check if Ollama service is available"""
        try:
            import requests
            response = requests.get('http://localhost:11434/api/version', timeout=2)
            self.ollama_available = response.status_code == 200
            return self.ollama_available
        except Exception:
            self.ollama_available = False
            return False
            
    def try_start_ollama(self):
        """Attempt to start Ollama if not running"""
        if not OLLAMA_AVAILABLE:
            self.display_status_message("❌ Ollama is not installed. Cannot start Ollama service.")
            return False

        if self.check_ollama_available():
            # Already running
            return True

        self.add_to_debug_console("Ollama not detected. Attempting to start it...")
        self.display_status_message("Ollama not running. Attempting to start it...")
        
        try:
            # Check platform and try to start Ollama
            import platform
            import subprocess
            import time
            
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                # Try to start Ollama using open command
                subprocess.Popen(["open", "-a", "Ollama"])
                self.display_status_message("Attempting to start Ollama application...")
            elif system == "linux":
                # Try systemctl or direct command
                try:
                    subprocess.Popen(["systemctl", "start", "ollama"])
                except:
                    subprocess.Popen(["ollama", "serve"])
            elif system == "windows":
                # Try to start Ollama from Program Files
                try:
                    import os
                    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
                    ollama_path = os.path.join(program_files, "Ollama", "Ollama.exe")
                    if os.path.exists(ollama_path):
                        subprocess.Popen([ollama_path])
                    else:
                        # Try Windows Run
                        subprocess.Popen(["cmd", "/c", "start", "ollama"])
                except Exception as e:
                    self.add_to_debug_console(f"Error starting Ollama on Windows: {str(e)}")
            
            # Wait for Ollama to start (up to 20 seconds)
            self.add_to_debug_console("Waiting for Ollama to start...")
            self.status_var.set("Starting Ollama...")
            
            for _ in range(10):
                time.sleep(2)
                if self.check_ollama_available():
                    self.display_status_message("Ollama started successfully!")
                    self.status_var.set("Ollama started")
                    return True
                    
            # If we get here, Ollama didn't start
            self.display_status_message("Could not start Ollama automatically. Please start it manually.")
            self.status_var.set("Ollama not available")
            return False
            
        except Exception as e:
            self.add_to_debug_console(f"Error trying to start Ollama: {str(e)}")
            self.display_status_message(f"Error trying to start Ollama: {str(e)}")
            return False

    def update_temp_label(self, value):
        """Update the temperature label when slider is moved"""
        self.temp_value_label.config(text=f"{float(value):.2f}")

    def set_temperature(self, value):
        """Set temperature to a preset value"""
        self.temperature.set(value)
        self.update_temp_label(value)
        
    def update_max_tokens_label(self, value):
        """Update the max tokens label when slider is moved"""
        self.max_tokens_value_label.config(text=f"{int(float(value))}")



    def copy_last_code_block(self):
        """Copy the last code block from the most recent assistant message."""
        code = self._find_last_code_block() # Use the helper
        if code:
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            self.show_copy_status("Code block copied to clipboard")
        else:
            self.show_copy_status("No Python code blocks found in recent messages")
            
    def save_last_code_block(self, code_content=None):
        """Save the last code block to a file."""
        code = code_content if code_content is not None else self._find_last_code_block()
        if code:
            # Ask user for filename
            def_ext, file_types = self._get_mode_default_ext_and_types()
            filename = filedialog.asksaveasfilename(
                defaultextension=def_ext,
                filetypes=file_types,
                title="Save Code Block"
            )
            if filename:
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(code)
                    self.show_copy_status(f"Code saved to {os.path.basename(filename)}")
                    self.add_to_debug_console(f"Code block saved to: {filename}")
                except Exception as e:
                    self.show_copy_status(f"Error saving code: {str(e)}")
                    self.add_to_debug_console(f"Error saving code: {str(e)}")
        else:
            self.show_copy_status("No Python code blocks found in recent messages")
    
    def _find_last_code_block(self):
        """Find the last code block in assistant messages.

        Search order (most specific to least):
        1. Fenced code blocks with language tag (```python or ```html)
        2. Any fenced code block (``` ... ```)
        3. Raw <html>...</html> blocks (HTML mode fallback)
        4. SEARCH/REPLACE blocks (returns the raw block text so
           move_code_to_ide() can apply them to the existing IDE code)
        """
        if not self.messages:
            return None

        is_html_mode = self.system_mode.get() == "html_programmer"

        def _normalize_html(text):
            """Convert tabs to spaces for consistent indentation."""
            if is_html_mode:
                return '\n'.join(line.replace('\t', '    ') for line in text.split('\n'))
            return text

        for msg in reversed(self.messages):
            if msg['role'] == 'assistant':
                content = msg['content']

                # 1. Fenced code blocks with explicit language tags
                code_blocks = re.findall(r'```(?:python|py|html|javascript|js)\s*\n([\s\S]*?)```', content, re.DOTALL)
                if code_blocks:
                    if len(code_blocks) == 1:
                        return _normalize_html(code_blocks[0]).strip('\n\r')
                    # Multiple code blocks — combine them all so partial
                    # fixes spread across blocks aren't lost.
                    combined = '\n\n'.join(b.strip('\n\r') for b in code_blocks)
                    return _normalize_html(combined).strip('\n\r')

                # 2. Any fenced code block (with or without language tag)
                code_blocks = re.findall(r'```\w*\s*\n([\s\S]*?)```', content, re.DOTALL)
                if code_blocks:
                    if len(code_blocks) == 1:
                        return _normalize_html(code_blocks[0]).strip('\n\r')
                    combined = '\n\n'.join(b.strip('\n\r') for b in code_blocks)
                    return _normalize_html(combined).strip('\n\r')

                # 3. Raw HTML document
                html_blocks = re.findall(r'<html[\s\S]*?</html>', content, re.IGNORECASE)
                if html_blocks:
                    return _normalize_html(html_blocks[-1]).strip()

                # 4. SEARCH/REPLACE blocks (return raw text for move_code_to_ide to apply)
                if re.search(r'<{6,7}\s*SEARCH', content):
                    return content.strip()

        return None


    def _execute_html_code(self, html_code):
        """Execute HTML code by saving to a temporary file and opening in default browser"""
        try:
            # Optional, non-invasive browser error capture: append small loader script only if enabled
            if hasattr(self, 'capture_browser_errors') and self.capture_browser_errors.get():
                # Insert an inline, minimal error hook as early as possible
                inline_hook = (
                    "<script>" 
                    "window.addEventListener('error',function(e){try{var x=new XMLHttpRequest();x.open('POST','http://localhost:8765/report_error',true);x.setRequestHeader('Content-Type','application/x-www-form-urlencoded');var d='type='+encodeURIComponent('JavaScript Error')+'&message='+encodeURIComponent(e.message)+'&source='+encodeURIComponent(e.filename||'')+'&line='+encodeURIComponent(e.lineno||'')+'&column='+encodeURIComponent(e.colno||'')+'&stack='+encodeURIComponent(e.error&&e.error.stack||'');x.send(d);}catch(_){}});"
                    "window.addEventListener('unhandledrejection',function(e){try{var x=new XMLHttpRequest();x.open('POST','http://localhost:8765/report_error',true);x.setRequestHeader('Content-Type','application/x-www-form-urlencoded');var d='type='+encodeURIComponent('Unhandled Promise Rejection')+'&message='+encodeURIComponent((e.reason&&e.reason.toString())||'')+'&stack='+encodeURIComponent((e.reason&&e.reason.stack)||'');x.send(d);}catch(_){}});"
                    "(function(){var oe=console.error;console.error=function(){try{var m=Array.prototype.map.call(arguments,function(a){return (typeof a==='object')?JSON.stringify(a):String(a)}).join(' ');var x=new XMLHttpRequest();x.open('POST','http://localhost:8765/report_error',true);x.setRequestHeader('Content-Type','application/x-www-form-urlencoded');x.send('type='+encodeURIComponent('Console Error')+'&message='+encodeURIComponent(m));}catch(_){ } oe&&oe.apply(console,arguments);};})();"
                    "</script>"
                )
                # Prefer injecting right after <head ...> to catch early parse/runtime errors
                try:
                    import re as _re
                    m = _re.search(r'<head[^>]*>', html_code, _re.IGNORECASE)
                    if m:
                        insert_at = m.end()
                        html_code = html_code[:insert_at] + "\n" + inline_hook + "\n" + html_code[insert_at:]
                    else:
                        # Fallback: append at end; will still capture runtime errors
                        html_code = html_code + "\n" + inline_hook
                except Exception:
                    html_code = html_code + "\n" + inline_hook

            # Create a temporary HTML file in user's home directory for better browser access
            import os
            temp_dir = os.path.expanduser("~/Code_Runner/temp_html")
            os.makedirs(temp_dir, exist_ok=True)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8', dir=temp_dir) as temp_file:
                temp_file.write(html_code)
                html_path = temp_file.name

            # Ensure the file has proper permissions
            os.chmod(html_path, 0o644)

            # Show file info in system console
            self.display_system_message(f"HTML file created: {html_path}")
            self.display_system_message(f"File size: {len(html_code)} characters")

            # Open in default browser with multiple fallback methods
            import webbrowser
            import subprocess

            # Check system browser configuration (user-facing diagnostics)
            try:
                result = subprocess.run(['which', 'xdg-open'], capture_output=True, text=True)
                if result.returncode != 0:
                    self.display_system_message("xdg-open not found (run: sudo apt install xdg-utils)")

                chromium_found = False
                for browser in ['chromium-browser', 'chromium']:
                    if subprocess.run(['which', browser], capture_output=True).returncode == 0:
                        chromium_found = True
                        break
                if not chromium_found:
                    self.display_system_message("No Chromium browser found (run: sudo apt install chromium-browser)")

                try:
                    browser_result = subprocess.run(['xdg-settings', 'get', 'default-web-browser'], capture_output=True, text=True)
                    if browser_result.returncode != 0:
                        self.display_system_message("No default browser set (common on fresh ARM Ubuntu)")
                except:
                    pass
            except Exception:
                pass

            browser_opened = False

            # Method 1: Try opening directly with xdg-open (Linux standard)
            try:
                subprocess.run(['xdg-open', html_path], check=True, capture_output=True, timeout=10)
                browser_opened = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                self.display_system_message(f"xdg-open failed: {e}")

            # Method 2: Try standard webbrowser.open with file:// URL
            if not browser_opened:
                try:
                    webbrowser.open(f'file://{html_path}')
                    browser_opened = True
                    pass  # opened successfully
                except Exception as e:
                    self.display_system_message(f"webbrowser.open failed: {e}")

            # Method 3: Try with different browsers directly (ARM-compatible)
            if not browser_opened:
                browsers = ['chromium-browser', 'chromium', 'firefox', 'google-chrome', 'opera', 'vivaldi', 'brave-browser']
                for browser in browsers:
                    try:
                        subprocess.run([browser, html_path], check=True, capture_output=True)
                        browser_opened = True
                        break  # opened successfully
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue

            # Method 4: Last resort - show file path for manual opening
            if not browser_opened:
                self.display_system_message(f"Could not auto-open browser. Open file://{html_path} manually")
                self.display_system_message("Tip: sudo apt install chromium-browser xdg-utils")

            # Display success/status to user via system messages
            line_count = html_code.count('\n') + 1
            if browser_opened:
                self.display_status_message(f"HTML opened in browser ({line_count} lines)")
            else:
                self.display_status_message("HTML saved but browser not opened")

            # Clear debug console — keep it clean for browser runtime errors only
            self.clear_debug_console()

        except Exception as e:
            error_msg = f"Error executing HTML: {str(e)}"
            self.display_status_message("HTML execution error")
            self.clear_debug_console()
            self.add_to_debug_console(f"HTML EXECUTION ERROR:\n{error_msg}\n{str(e)}")

    def run_last_code_block(self, code_to_run=None, use_timeout=True):
        """Find the last code block (Python or HTML), execute it, and display output."""
        using_fallback = False
        
        # If no code provided, find the last code block
        if code_to_run is None:
            code_to_run = self._find_last_code_block()
            
            # If no code block found in message history, try to use selected text as fallback
            if not code_to_run:
                try:
                    selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
                    if selected_text.strip():
                        code_to_run = selected_text.strip()
                        using_fallback = True
                except tk.TclError:
                    # No selection, continue with error
                    pass
                
        # If still no code found, show error and return
        if not code_to_run:
            self.show_copy_status("No code block found in the last message.")
            return

        # Check current mode to determine execution type
        mode = self.system_mode.get()
        is_html_mode = (mode == "html_programmer")
        # Auto-detect HTML content to avoid running HTML as Python
        try:
            preview = code_to_run.strip().lower()
            if not is_html_mode and (
                "<html" in preview or "<!doctype" in preview or "<head" in preview or "<body" in preview or "<canvas" in preview
            ):
                is_html_mode = True
        except Exception:
            pass
        
        self.last_run_code = code_to_run
        self.last_run_stdout = None
        self.last_run_stderr = None

        if using_fallback:
            code_type = "HTML" if is_html_mode else "Python"
            self.display_message("System", f"No code block found, falling back to selected {code_type} code...")
            self.status_var.set(f"Running selected {code_type} code...")
        else:
            code_type = "HTML" if is_html_mode else "Python"
            if is_html_mode:
                self.display_system_message(f"Running {code_type} code block...")  # Redirect HTML run notice to system console per user request
            else:
                self.display_status_message(f"Running {code_type} code block...")
            
        self.root.update_idletasks() # Show message immediately

        # Execute HTML or Python based on current mode or detection
        if is_html_mode:
            if hasattr(self, 'capture_browser_errors'):
                cap = 'ON' if self.capture_browser_errors.get() else 'OFF'
                self.display_system_message(f"Capture Browser Errors: {cap}")
                if cap == 'ON':
                    self.display_system_message("Note: Browser line numbers include +3 line offset due to capture hook")
            self._execute_html_code(code_to_run)
            return

        try:
            # Create a temporary file to store the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_script:
                temp_script.write(code_to_run)
                script_path = temp_script.name

            # Execute the script using subprocess
            run_kwargs = {
                'args': ['python', script_path],
                'capture_output': True,
                'text': True
            }
            
            # Add timeout only if requested
            if use_timeout:
                run_kwargs['timeout'] = 60  # 60 seconds timeout when enabled
                
            process = subprocess.run(**run_kwargs)

            self.last_run_stdout = process.stdout
            self.last_run_stderr = process.stderr

            # Status to user via system messages
            line_count = code_to_run.count('\n') + 1
            if process.returncode == 0:
                self.display_status_message(f"Code executed successfully ({line_count} lines)")
            elif self.last_run_stderr and self.last_run_stderr.strip():
                self.display_status_message("Code execution failed — see debug console for errors")
            else:
                self.display_status_message(f"Code completed with exit code {process.returncode}")

            # Debug console: ONLY runtime errors and output (what helps debug the generated code)
            self.clear_debug_console()
            if self.last_run_stderr:
                self.add_to_debug_console("ERROR OUTPUT:")
                self.add_to_debug_console(self.last_run_stderr.strip())
            if self.last_run_stdout:
                self.add_to_debug_console("STDOUT:")
                self.add_to_debug_console(self.last_run_stdout.strip())

        except FileNotFoundError:
             # Add error to debug console instead of chat
             self.add_to_debug_console("\n❌ ERROR: 'python' command not found.")
             self.add_to_debug_console("Is Python installed and in your PATH?")
             self.display_status_message("Python not found - check installation")
             self.last_run_code = None # Clear code if python isn't found
        except subprocess.TimeoutExpired:
            # Route timeout to debug console
            self.add_to_debug_console("\n⏱️ TIMEOUT: Code execution exceeded time limit")
            self.display_status_message("Code execution timed out")
            self.last_run_stderr = "Execution timed out." # Store timeout as error
        except Exception as e:
            # Add error to debug console
            self.add_to_debug_console(f"\n❌ ERROR: {str(e)}")
            self.display_status_message(f"Execution error - see debug console")
            self.last_run_stderr = f"Error running code: {str(e)}" # Store exception as error
            self.last_run_code = None # Clear code if execution failed unexpectedly
        finally:
            # Clean up the temporary file
            if 'script_path' in locals() and os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {script_path}: {e}") # Log cleanup error

    def fix_last_code_block(self, code_content=None):
        """Send the last run code and its errors to the LLM for fixing."""
        # If code_content is provided (from IDE), use it directly for fixing
        if code_content is not None:
            code_to_fix = code_content
            # Create a prompt for fixing the provided code
            user_message = self.user_input.get("1.0", tk.END).strip()
            
            # Determine language based on current system mode
            mode = self.system_mode.get()
            is_html_mode = (mode == "html_programmer")
            language = "html" if is_html_mode else "python"
            code_type = "HTML" if is_html_mode else "Python"
            
            # Check if user wants full code back
            want_full = getattr(self.ide_window, 'return_full_code', None)
            if want_full and want_full.get():
                fix_instruction = f"Return the COMPLETE fixed program in a single ```{language} code block."
            else:
                fix_instruction = f"RULES — follow ALL strictly:\n1. Say what's wrong in 1-2 sentences.\n2. Show ONLY the fixed function(s) in ONE ```{language} code block.\n3. Do NOT return the entire program. Maximum 50 lines of code.\n4. No comments like 'ADD THIS' or 'rest of code unchanged'.\n5. No explanations inside the code block."

            prompt = f"""Please review and fix any issues in the following {code_type} code:
```{language}
{code_to_fix}
```

{f"Additional context: {user_message}" if user_message else ""}

{fix_instruction}"""
            
            # Clear input and send the fix request
            self.user_input.delete("1.0", tk.END)
            self.user_input.insert("1.0", prompt)
            return
            
    def simple_code_update(self, original_code, updated_code, description="Code update"):
        """Simple method to show code updates as diffs in the IDE
        
        This is much more practical than complex parsing or FIM tokens.
        Just show what changed!
        """
        if self.ide_window and self.ide_window.root.winfo_exists():
            # Show the updated code as a diff
            self.ide_window.show_diff(updated_code)
            self.display_chat_system_message(f"{description} - review changes in IDE")
        else:
            self.display_chat_system_message(f"{description} - open IDE to see changes")
    
    # ---------- Fix workflow: IDE -> LLM -> diff -> Accept/Reject ----------
    # Called by both "Ask LLM to Fix" and Run & Fix (F6) after error detection.

    def fix_code_from_ide(self, code_content):
        """Fix code from IDE.

        TWO MODES (controlled by 'Return full code' checkbox in IDE toolbar):
        - OFF (default): Ask LLM to explain the bug and show only the
          corrected lines. Fast, cheap, works for trivial errors.
        - ON: Ask LLM to return the complete fixed program. The diff
          view shows what changed. Use for big changes / rewrites.
        """
        # Get any text from the user input box for additional context
        user_message = self.user_input.get("1.0", tk.END).strip()

        # Get debug console content for context
        self.debug_console.config(state=tk.NORMAL)
        debug_content = self.debug_console.get("1.0", tk.END).strip()
        self.debug_console.config(state=tk.DISABLED)

        # Determine language based on current system mode
        mode = self.system_mode.get()
        is_html_mode = (mode == "html_programmer")
        language = "html" if is_html_mode else "python"
        code_type = "HTML" if is_html_mode else "Python"

        # Store the original code for the accept/reject workflow
        self.ide_original_code = code_content

        # Build error context
        error_section = ""
        if self.last_run_stderr and self.last_run_stderr.strip():
            error_section = f"\nError:\n```\n{self.last_run_stderr.strip()}\n```"
        if self.last_run_stdout and self.last_run_stdout.strip():
            error_section += f"\nOutput:\n```\n{self.last_run_stdout.strip()}\n```"

        problem_desc = f"Problem: {user_message}" if user_message else "Find and fix the bugs."

        # Check if user wants full code back
        want_full_code = getattr(self.ide_window, 'return_full_code', None)
        full_code_mode = want_full_code and want_full_code.get()

        if full_code_mode:
            # FULL CODE MODE: send program, get complete fixed program back
            prompt = f"""Fix this {code_type} program.

```{language}
{code_content}
```
{error_section}
{problem_desc}

Return the COMPLETE fixed program in a single ```{language} code block. Do not skip any part of the code."""
            mode_label = "full code"
        else:
            # TARGETED FIX MODE (default): return only the fixed functions
            prompt = f"""Fix this {code_type} program:

```{language}
{code_content}
```
{error_section}
{problem_desc}

One sentence: what is wrong.
Then ONE ```{language} code block with ONLY the fixed functions. Not the whole program."""
            mode_label = "targeted fix"

        # Display what we're sending
        if self.last_run_stderr and self.last_run_stderr.strip():
            error_summary = self.last_run_stderr.strip().split("\n")[-1]
            display_message = f"[Fix code — error: {error_summary}]"
            if user_message:
                display_message = f"{display_message}\n{user_message}"
        elif user_message:
            display_message = f"[Ask LLM to fix code]\n{user_message}"
        else:
            display_message = "[Ask LLM to review and fix code]"

        self.display_message("You", display_message)

        # Surface fix status in system console
        if self.last_run_stderr and self.last_run_stderr.strip():
            self.display_system_message(f"--- {mode_label} mode — error: {self.last_run_stderr.strip().split(chr(10))[-1]}")
        else:
            self.display_system_message(f"Fix request sent ({mode_label} mode)")

        # Clear input box
        self.user_input.delete("1.0", tk.END)

        # Save full history, send only focused context for fix
        # The fix prompt already contains full current code + errors + rules,
        # so old conversation history just confuses weaker LLMs.
        # Use a fix-specific system message (the normal one says "include all features"
        # which contradicts fix mode's "only return changed functions").
        fix_system_message = {'role': 'system', 'content':
            "You are a code debugger. Return ONLY the fixed functions in a code block. "
            "Not the whole program."}
        self._pre_fix_messages = list(self.messages)
        self.messages = [
            fix_system_message,
            {'role': 'user', 'content': prompt}
        ]

        # Log what we're sending (user-facing status in system console)
        self.display_system_message(f"FIX MODE: sending {len(self.messages)} messages (system + fix prompt), {len(prompt)} chars")

        # Disable input while processing
        self.user_input.config(state=tk.DISABLED)
        self.send_button.pack_forget()
        self.fix_button.pack_forget()
        self.stop_button.pack(side=tk.RIGHT)
        self.status_var.set("Requesting fix from LLM...")

        # Reset stop flag
        self.stop_generation = False
        self.generation_active = True

        # Start response timer
        self.start_response_timer()

        # Start a thread to get the model's response
        threading.Thread(target=self.get_model_response).start()

    def on_game_preset_selected(self, event=None):
        """Handle selection of a game preset from the combobox."""
        selected_game = self.selected_game_prompt.get()
        prompt_text = self.game_prompts.get(selected_game, "")
        # HTML/Python language control should come ONLY from the radio button selection.
        # If HTML mode is selected, sanitize legacy Pygame/Python phrasing in presets
        # so the LLM doesn't receive conflicting language instructions. (Minimal, targeted replacements.)
        if prompt_text:
            mode = self.system_mode.get()
            if mode == "html_programmer":
                # Targeted replacements to keep presets language-agnostic when in HTML mode
                replacements = [
                    ("using Python and Pygame", "using HTML5 Canvas and JavaScript"),
                    ("made by pygames NO EXTERNAL SOUND FILES", "NO EXTERNAL SOUND FILES"),
                    ("made with pygame NO EXTERNAL SOUND FILES", "NO EXTERNAL SOUND FILES"),
                    ("Pygame-generated audio", "Generated audio (no external files)"),
                    ("Pygame-generated sounds", "Generated sounds (no external files)"),
                    ("ALL graphics created with Pygame drawing functions only", "ALL graphics created with Canvas drawing functions only"),
                    ("ALL graphics drawn with Pygame primitives", "ALL graphics drawn with Canvas primitives"),
                ]
                for old, new in replacements:
                    prompt_text = prompt_text.replace(old, new)
        if prompt_text: # Avoid clearing if placeholder is selected
            self.user_input.delete("1.0", tk.END)
            self.user_input.insert("1.0", prompt_text)
            self.user_input.focus_set()
        # Reset to placeholder if a real game was selected, otherwise keep selection
        # This allows re-selecting the same game to paste again
        if selected_game != "-- Select Game Preset --":
             self.root.after(100, lambda: self.selected_game_prompt.set("-- Select Game Preset --"))

    def run_selected_code(self):
        """Run the currently selected text in the chat display."""
        try:
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            if not selected_text.strip():
                self.show_copy_status("No text selected. Please select code to run.")
                return
            
            # Run the selected code directly
            self.run_last_code_block(code_to_run=selected_text.strip())
            
        except tk.TclError:
            # No selection, show error
            self.show_copy_status("No text selected. Please select code to run.")
            return

    def notify_ide_content_loaded(self, content, filename=None):
        """Called when content is loaded in the IDE"""
        self.ide_current_content = content
        self.ide_current_filename = filename
        
        # Don't show "code loaded" message when content is empty (NEW button clicked)
        if not content.strip():
            return
            
        # Show user-friendly notification in chat (without instructions)
        if filename:
            base_filename = os.path.basename(filename)
            self.display_status_message(f"Code loaded in IDE: {base_filename}")
        else:
            self.display_status_message("Code loaded in IDE")

        # Show code analysis in system messages window
        self.analyze_code_for_debugging(content, filename)
        # Sync Save button state with loaded filename
        if hasattr(self, 'save_button'):
            self.save_button.config(state=(tk.NORMAL if filename else tk.DISABLED))
    
    def open_ide_window(self):
        """Open the IDE window"""
        self.ide_window.show_window()

        # Show help message if it's the first time
        if not hasattr(self, '_ide_help_shown'):
            self._ide_help_shown = True
            self.display_status_message(
                "IDE opened. Workflow: "
                "1) Ask LLM to write code -> 'Move to IDE' (first time loads, after shows diff). "
                "2) F5 = Run. "
                "3) Type problem in chat -> 'Ask LLM to Fix' -> Accept/Reject diff."
            )

    def _merge_partial_fix(self, full_code, fragment):
        """Merge a partial code fragment back into the full program.

        Layered approach — tries the most reliable method first:
          1. Function-name matching: find functions by name in both the
             original and fragment, swap matched ones in-place.
          2. SequenceMatcher fallback: for non-function code or when
             function names don't match, align via longest common
             subsequences and splice in changes.

        Returns the merged code string, or None if merging failed.
        """
        full_lines = full_code.splitlines()
        frag_lines = fragment.splitlines()

        # If the fragment is nearly the full program, let caller diff directly
        if len(frag_lines) > len(full_lines) * 0.95:
            return None
        if len(frag_lines) < 3:
            return None

        # --- LAYER 1: function-name replacement (deterministic) ---
        is_js = '<script' in full_code or 'function ' in full_code
        if is_js:
            merged = self._merge_by_function_name(full_lines, frag_lines, lang='js')
        else:
            merged = self._merge_by_function_name(full_lines, frag_lines, lang='python')

        if merged is not None:
            result = '\n'.join(merged)
            if full_code.endswith('\n'):
                result += '\n'
            return result

        # --- LAYER 2: SequenceMatcher with edge protection ---
        merged = self._merge_by_sequence_match(full_lines, frag_lines)
        if merged is not None:
            result = '\n'.join(merged)
            if full_code.endswith('\n'):
                result += '\n'
            return result

        return None

    # ---- Layer 1: function-name merge ----

    def _find_functions(self, lines, lang):
        """Find function boundaries in code. Returns dict of name -> (start, end).

        JS patterns: function name(, async function name(, name(args) {,
                     const/let/var name = function(, name: function(,
                     name = (args) =>
        Python patterns: def name(  at any indent (includes decorators above)
        """
        functions = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if lang == 'js':
                name = self._match_js_func_start(stripped)
                if name:
                    end = self._find_brace_end(lines, i)
                    functions[name] = (i, end)
                    i = end
                    continue
            else:
                m = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
                if m:
                    indent = len(m.group(1))
                    name = m.group(2)
                    start = i
                    while start > 0 and lines[start - 1].strip().startswith('@'):
                        start -= 1
                    end = i + 1
                    while end < len(lines):
                        s = lines[end].strip()
                        if s == '':
                            end += 1
                            continue
                        if (len(lines[end]) - len(lines[end].lstrip())) <= indent:
                            break
                        end += 1
                    functions[name] = (start, end)
                    i = end
                    continue
            i += 1
        return functions

    def _match_js_func_start(self, stripped):
        """Return function name if this line starts a JS function, else None."""
        # function name(  /  async function name(
        m = re.match(r'(?:async\s+)?function\s+(\w+)\s*\(', stripped)
        if m:
            return m.group(1)
        # const/let/var name = function(  or  = async function(  or  = (...) =>
        m = re.match(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>)', stripped)
        if m:
            return m.group(1)
        # name: function(  (object literal / class)
        m = re.match(r'(\w+)\s*:\s*(?:async\s+)?function\s*\(', stripped)
        if m:
            return m.group(1)
        # name(args) {  (class method shorthand) — but not if/for/while/switch
        m = re.match(r'(\w+)\s*\([^)]*\)\s*\{', stripped)
        if m and m.group(1) not in ('if', 'for', 'while', 'switch', 'catch', 'else'):
            return m.group(1)
        return None

    def _find_brace_end(self, lines, start):
        """Find closing brace for a JS function starting at `start`. Returns end line index."""
        depth = 0
        for j in range(start, len(lines)):
            depth += lines[j].count('{') - lines[j].count('}')
            if depth <= 0 and j > start:
                return j + 1
        return len(lines)

    def _merge_by_function_name(self, full_lines, frag_lines, lang):
        """Replace functions in full_lines with same-named functions from frag_lines."""
        orig_funcs = self._find_functions(full_lines, lang)
        fix_funcs = self._find_functions(frag_lines, lang)

        if not fix_funcs:
            return None

        matched = set(fix_funcs.keys()) & set(orig_funcs.keys())
        if not matched:
            return None

        result = list(full_lines)
        # Replace in reverse order so indices stay valid
        for name in sorted(matched, key=lambda n: orig_funcs[n][0], reverse=True):
            orig_start, orig_end = orig_funcs[name]
            fix_start, fix_end = fix_funcs[name]
            result[orig_start:orig_end] = frag_lines[fix_start:fix_end]

        if len(result) < len(full_lines) * 0.5:
            return None
        return result

    # ---- Layer 2: SequenceMatcher merge with edge protection ----

    def _merge_by_sequence_match(self, full_lines, frag_lines):
        """Align fragment into full code using SequenceMatcher opcodes.

        Key rules:
          - 'equal' regions anchor the alignment
          - 'delete' = lines only in full code -> KEEP (fragment is partial)
          - 'replace' inside anchored region -> use fragment (the fix)
          - 'replace' outside anchored region -> keep original (HTML header etc)
          - 'insert' inside anchored region -> include (new code from fix)
        """
        sm = difflib.SequenceMatcher(None, full_lines, frag_lines, autojunk=False)
        opcodes = sm.get_opcodes()
        total_equal = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag == 'equal')

        if total_equal < 3:
            # Try stripped comparison for indentation mismatches
            full_s = [l.strip() for l in full_lines]
            frag_s = [l.strip() for l in frag_lines]
            sm = difflib.SequenceMatcher(None, full_s, frag_s, autojunk=False)
            opcodes = sm.get_opcodes()
            total_equal = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag == 'equal')
            if total_equal < 3:
                return None

        # Find anchored region (between first and last equal blocks)
        first_eq = last_eq = None
        for idx, (tag, *_) in enumerate(opcodes):
            if tag == 'equal':
                if first_eq is None:
                    first_eq = idx
                last_eq = idx
        if first_eq is None:
            return None

        merged = []
        for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
            inside = first_eq <= idx <= last_eq
            if tag == 'equal':
                merged.extend(full_lines[i1:i2])
            elif tag == 'replace':
                if inside:
                    merged.extend(frag_lines[j1:j2])  # the fix
                else:
                    merged.extend(full_lines[i1:i2])  # preserve edge
            elif tag == 'delete':
                merged.extend(full_lines[i1:i2])  # keep — fragment is partial
            elif tag == 'insert':
                if inside:
                    merged.extend(frag_lines[j1:j2])  # new lines from fix

        if len(merged) < len(full_lines) * 0.85:
            return None
        return merged

    def move_code_to_ide(self):
        """Move code from chat to IDE.

        WORKFLOW:
        - IDE is EMPTY  -> Load code directly (first time setup).
        - IDE has code   -> Show a DIFF so you can Accept/Reject without
                            losing your existing program.

        Handles three kinds of LLM output:
        1. SEARCH/REPLACE blocks  -> apply as targeted edits, show diff.
        2. Partial fix fragment   -> merge into existing code, show diff.
        3. Full code block        -> show diff against current IDE code.
        4. IDE is empty           -> load directly (no diff needed).
        """
        code_to_move = None

        # First try to get selected text from chat
        try:
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected_text.strip():
                code_to_move = selected_text.strip()
        except tk.TclError:
            pass

        # If no selection, get the last code block from the conversation
        if not code_to_move:
            code_to_move = self._find_last_code_block()

        if not code_to_move:
            self.show_copy_status("No code found to move to IDE")
            return

        current_ide = self.ide_window.get_content() if self.ide_window else ""
        ide_has_code = bool(current_ide.strip())

        # --- CASE 1: SEARCH/REPLACE blocks (targeted edits) ---
        import re
        has_sr = re.search(r'<{6,7}\s*SEARCH', code_to_move)
        if has_sr:
            if ide_has_code:
                result = self._apply_search_replace_blocks(current_ide, code_to_move)
                if result and result.strip() != current_ide.strip():
                    self.propose_code_changes(result)
                    self.show_copy_status("Edits applied as diff — Accept or Reject in IDE")
                    return
                self.show_copy_status("Could not apply SEARCH/REPLACE blocks — no matching code in IDE")
            else:
                self.show_copy_status("Got edit blocks but IDE is empty — ask for full code first")
            return

        # --- CASE 2: IDE already has code -> show diff (never clobber) ---
        if ide_has_code:
            if code_to_move.strip() == current_ide.strip():
                self.show_copy_status("Code is identical to what's already in the IDE")
                self.ide_window.show_window()
                return

            # If the new code is a small fragment (partial fix), merge it
            # into the full program first so the diff only highlights the
            # actual changes — not 680 "deleted" lines.
            merged = self._merge_partial_fix(current_ide, code_to_move)
            if merged is not None:
                self.propose_code_changes(merged)
                self.show_copy_status("Partial fix merged — Accept or Reject in IDE")
            else:
                # Merge failed or full replacement — show diff as-is.
                # For partial fragments this will be noisy, but per-hunk
                # review lets the user accept only the real fixes.
                self.propose_code_changes(code_to_move)
                if len(code_to_move.splitlines()) < len(current_ide.splitlines()) * 0.5:
                    self.show_copy_status("Partial fix (merge failed) — use hunk nav to pick changes")
                    self.add_to_debug_console("⚠ Auto-merge failed for partial fragment. Showing raw diff.")
                else:
                    self.show_copy_status("New code shown as diff — Accept or Reject in IDE")
            return

        # --- CASE 3: IDE is empty -> load directly (first time) ---
        self.ide_window.set_content(code_to_move, None)
        self.ide_window.show_window()
        self.show_copy_status("Code loaded into IDE — press F5 to Run")
            
    def propose_code_changes(self, proposed_code):
        """Propose code changes to the IDE window for user review"""
        if hasattr(self, 'ide_window'):
            self.ide_window.show_diff(proposed_code)
            self.ide_window.show_window()

    # ---------- SEARCH/REPLACE block parser ----------
    # LLM returns <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks.
    # This method finds each block, matches it in the original code, and applies the replacement.

    def _apply_search_replace_blocks(self, original_code, message):
        """Parse SEARCH/REPLACE blocks from LLM response and apply them to original code.

        Returns the modified code, or None if no SEARCH/REPLACE blocks were found.
        """
        import re
        # Strip markdown code fences so blocks inside ```html ... ``` are found
        cleaned = re.sub(r'```(?:python|py|html|javascript|js)?\s*\n', '', message)
        cleaned = cleaned.replace('```', '')

        # Match <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks
        pattern = r'<{6,7}\s*SEARCH\s*\n(.*?)\n={6,7}\s*\n(.*?)\n>{6,7}\s*REPLACE'
        blocks = re.findall(pattern, cleaned, re.DOTALL)

        if not blocks:
            return None

        result = original_code
        applied = 0
        for search_text, replace_text in blocks:
            # Try exact match first
            if search_text in result:
                result = result.replace(search_text, replace_text, 1)
                applied += 1
            else:
                # Try with stripped trailing whitespace per line (LLM whitespace drift)
                search_lines = [line.rstrip() for line in search_text.split('\n')]
                result_lines = result.split('\n')
                result_stripped = [line.rstrip() for line in result_lines]

                # Sliding window search
                search_len = len(search_lines)
                found = False
                for i in range(len(result_stripped) - search_len + 1):
                    if result_stripped[i:i + search_len] == search_lines:
                        # Replace the original lines (preserving original line endings)
                        replace_lines = replace_text.split('\n')
                        result_lines[i:i + search_len] = replace_lines
                        result = '\n'.join(result_lines)
                        applied += 1
                        found = True
                        break

                if not found:
                    self.add_to_debug_console(f"SEARCH/REPLACE: could not match block ({search_text[:50]}...)")

        if applied > 0:
            self.add_to_debug_console(f"Applied {applied}/{len(blocks)} SEARCH/REPLACE edits")
            return result

        return None

    def clear_debug_console(self):
        """Clear the contents of the debug console"""
        self.debug_console.config(state=tk.NORMAL)
        self.debug_console.delete("1.0", tk.END)
        self.debug_console.config(state=tk.DISABLED)
    
    def copy_debug_console(self):
        """Copy all text from the debug console to clipboard"""
        try:
            text = self.debug_console.get("1.0", tk.END).strip()
            if not text:
                self.show_copy_status("Debug console is empty")
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.show_copy_status("Debug console copied to clipboard")
        except Exception as e:
            self.show_copy_status(f"Copy failed: {str(e)}")

    def _on_toggle_capture_browser_errors(self):
        """Handle toggling the capture of browser errors (no-op hook)."""
        enabled = False
        try:
            enabled = self.capture_browser_errors.get()
        except Exception:
            pass
        status = "ON" if enabled else "OFF"
        self.display_system_message(f"Capture Browser Errors: {status}")
        # Start/stop the error server on demand
        try:
            if enabled:
                if getattr(self, 'error_server', None) is not None:
                    self.error_server.start()
            else:
                if getattr(self, 'error_server', None) is not None:
                    self.error_server.stop()
        except Exception as e:
            self.add_to_debug_console(f"Capture server toggle error: {e}")
    
    def clear_system_console(self):
        """Clear the contents of the system console"""
        self.system_console.config(state=tk.NORMAL)
        self.system_console.delete("1.0", tk.END)
        self.system_console.config(state=tk.DISABLED)
    
    def clear_search_results(self):
        """Clear the contents of the search results console"""
        self.search_results_console.config(state=tk.NORMAL)
        self.search_results_console.delete("1.0", tk.END)
        self.search_results_console.config(state=tk.DISABLED)
        self.add_to_debug_console("Search results cleared")
    
    def add_to_search_results(self, text):
        """Add text to the search results console"""
        self.search_results_console.config(state=tk.NORMAL)
        current_content = self.search_results_console.get("1.0", tk.END).strip()
        if current_content:
            self.search_results_console.insert(tk.END, "\n\n" + "="*50 + "\n\n")
        self.search_results_console.insert(tk.END, text)
        self.search_results_console.see(tk.END)
        self.search_results_console.config(state=tk.DISABLED)
    
    def get_search_results_content(self):
        """Get the current content of the search results console"""
        return self.search_results_console.get("1.0", tk.END).strip()
    
    def display_chat_system_message(self, message, end=True):
        """Display a system message in the chat history (for important context)
        
        USE THIS FOR:
        - Error messages that need debugging
        - Code loading notifications
        - Execution results and outputs
        - Chat save/load confirmations
        - Any message the LLM needs to see for context
        
        Examples:
        - "📝 Code loaded in IDE: filename.py"
        - "Running code block..."
        - "Error: Module not found"
        - "--- Execution Output ---"
        """
        self.display_message("System", message, end=end, to_chat=True)
    
    def display_status_message(self, message, end=True):
        """Display a status message in the system console (non-critical info)
        
        USE THIS FOR:
        - Model status updates
        - Backend switches
        - General information messages
        - Progress updates
        - Available options/lists
        
        Examples:
        - "Model list refreshed"
        - "Found 5 GGUF models"
        - "Switched to ollama backend"
        - "Chat started. Type '/help' to see available commands."
        """
        self.display_message("System", message, end=end, to_chat=False)
    
    def display_system_message(self, message, end=True):
        """Display a message in the system console with timestamp
        
        This is for non-critical system messages that don't need to be in chat history.
        Examples:
        - "Model list refreshed"
        - "Found X GGUF models"
        - "Switched to X backend"
        - "RAG functionality enabled/disabled"
        - General status updates
        
        These messages are informative but don't provide context needed by the LLM.
        
        THREAD-SAFE:
        - May be called from background inference thread via append_to_chat error routing
        - Schedule all GUI work on main thread using root.after(0, ...)
        """
        # Schedule GUI work on main thread for thread safety
        self.root.after(0, lambda: self._display_system_message_impl(message, end))
    
    def _display_system_message_impl(self, message, end=True):
        """Internal implementation of display_system_message - runs on main thread only"""
        # Ensure system console exists
        if not hasattr(self, 'system_console') or self.system_console is None:
            # Fallback to debug console if system console not available
            if hasattr(self, 'add_to_debug_console'):
                self.add_to_debug_console(f"[System Console Not Ready] {message}")
            return
            
        try:
            self.system_console.config(state=tk.NORMAL)
            # Add timestamp for system messages
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.system_console.insert(tk.END, f"[{timestamp}] {message}")
            if end:
                self.system_console.insert(tk.END, "\n")
            self.system_console.config(state=tk.DISABLED)
            self.system_console.see(tk.END)
            self.root.update_idletasks()  # Update the GUI immediately
        except Exception as e:
            # If any error occurs, log it
            if hasattr(self, 'add_to_debug_console'):
                self.add_to_debug_console(f"Error displaying system message: {str(e)}")

    def select_image(self):
        """Open a file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                self.set_current_image(image, file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def set_current_image(self, image, file_path=None):
        """Set the current image for the chat"""
        # Store the image and path
        self.current_image = image
        self.image_file_path = file_path
        
        # Resize the image for display while maintaining aspect ratio
        max_width = 400
        max_height = 300
        width, height = image.size
        
        # Calculate new dimensions
        if width > max_width or height > max_height:
            ratio = min(max_width/width, max_height/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            display_image = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            display_image = image
            
        # Convert to PhotoImage for Tkinter
        self.image_tk = ImageTk.PhotoImage(display_image)
        
        # Update UI
        self.image_label.config(text=f"Image attached: {os.path.basename(file_path) if file_path else 'from clipboard'}")
        self.image_display_label.config(image=self.image_tk)
        
        # Show the image frame
        self.image_frame.pack(after=self.chat_display.master, fill=tk.X, pady=(5, 10))
        
        # Update status
        image_size = f"{width}x{height}"
        self.status_var.set(f"Image attached: {image_size}")

    def clear_image(self):
        """Remove the current image"""
        if self.current_image:
            self.current_image = None
            self.image_file_path = None
            self.image_tk = None
            self.image_display_label.config(image="")
            self.image_frame.pack_forget()
            self.status_var.set("Image cleared")

    def setup_image_paste(self):
        """Set up image pasting from clipboard"""
        self.root.bind("<Control-v>", self.paste_image_or_text)
        self.root.bind("<Command-v>", self.paste_image_or_text)
        
    def paste_image_or_text(self, event=None):
        """Handle paste event - check if it's an image or text"""
        # If we're not focused on the chat input, don't handle paste
        if self.root.focus_get() != self.user_input:
            return
            
        try:
            # Try to get image from clipboard
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                # It's an image, handle it
                self.set_current_image(image)
                return "break"  # Prevent default paste behavior
        except Exception as e:
            print(f"Error getting image from clipboard: {e}")
            # Fall back to regular text paste
            pass
            
        # If no image or error, fall back to text paste
        return self.paste_clipboard(event)

    def sync_thinking_state(self):
        """Update the previous thinking state to match current state when checkbox is clicked"""
        self.previous_thinking_state = not self.hide_thinking.get()  # Inverted logic

        # Optional: Show status message when changed
        status = "enabled" if self.hide_thinking.get() else "disabled"
        self.status_var.set(f"Hide thinking {status}")

    # RAG Implementation Methods
    def index_new_folder(self):
        """Index a folder of documents into ChromaDB"""
        # Ask user for folder to index
        folder_path = filedialog.askdirectory(
            title="Select Folder to Index"
        )
        
        if not folder_path:
            return  # User cancelled
            
        # Ask for collection name
        collection_name = simpledialog.askstring(
            "Collection Name", 
            "Enter a name for this collection:",
            initialvalue=os.path.basename(folder_path)
        )
        
        if not collection_name:
            return  # User cancelled
            
        # Ask for persistent directory (optional - use default if not specified)
        persist_dir = self.default_persist_dir
        use_custom_dir = messagebox.askyesno(
            "Persistent Directory",
            f"Use default ChromaDB directory?\n({persist_dir})\n\nSelect 'No' to choose a custom location."
        )
        
        if not use_custom_dir:
            custom_dir = filedialog.askdirectory(
                title="Select ChromaDB Storage Location"
            )
            if custom_dir:  # If user didn't cancel
                persist_dir = custom_dir
        
        # Create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Start indexing in a separate thread
        threading.Thread(
            target=self._run_indexing,
            args=(folder_path, collection_name, persist_dir)
        ).start()
    
    def _run_indexing(self, folder_path, collection_name, persist_dir):
        """Run the indexing process in a background thread"""
        # Disable buttons during indexing
        self.root.after(0, lambda: self.index_folder_btn.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.load_db_btn.config(state=tk.DISABLED))
        
        # Clear the debug console for status updates
        self.root.after(0, self.clear_debug_console)
        
        # Update status
        self.root.after(0, lambda: self.status_var.set(f"Indexing folder: {os.path.basename(folder_path)}..."))
        self.root.after(0, lambda: self.add_to_debug_console(f"Starting indexing for folder: {folder_path}"))
        
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            # Do a QUICK initial scan - limit depth and file count to avoid hanging
            file_count = 0
            dir_count = 0
            try:
                max_dirs_to_scan = 100  # Limit directories to scan for initial count
                max_depth = 5           # Limit directory depth for initial scan
                
                self.root.after(0, lambda: self.add_to_debug_console(f"Quick scan in progress (limited to {max_dirs_to_scan} dirs and depth {max_depth})..."))
                
                for root, dirs, files in os.walk(folder_path, topdown=True):
                    # Check depth
                    depth = root.count(os.sep) - folder_path.count(os.sep)
                    if depth > max_depth:
                        dirs[:] = []  # Don't traverse deeper
                        continue
                    
                    # Skip hidden dirs
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    # Count non-hidden files
                    file_count += len([f for f in files if not f.startswith('.')])
                    dir_count += 1
                    
                    # Update status occasionally to keep UI responsive
                    if dir_count % 10 == 0:
                        self.root.after(0, lambda c=dir_count, f=file_count: 
                                        self.status_var.set(f"Quick scan: {c} dirs, {f} files found..."))
                        # Allow UI to update
                        time.sleep(0.01)
                    
                    # Stop if we hit the limits
                    if dir_count >= max_dirs_to_scan or file_count > 5000:
                        self.root.after(0, lambda: self.add_to_debug_console(f"Quick scan: Found over {file_count} files in {dir_count} directories (sampling only)"))
                        break
                
                self.root.after(0, lambda: self.add_to_debug_console(f"Quick scan complete: Found {file_count} files in {dir_count} directories"))
                self.root.after(0, lambda: self.status_var.set(f"Found ~{file_count} files to process. Starting indexing..."))
                
                # Process UI events before continuing
                self.root.update_idletasks()
                
            except Exception as e:
                self.root.after(0, lambda: self.add_to_debug_console(f"Error during initial scan: {str(e)}"))
            
            # Run the indexing with a timeout
            indexing_timeout = 900  # 15 minutes max
            
            # Define a function to run indexing in a separate thread
            indexing_complete = [False]  # Use list for mutable reference
            indexing_error = [None]      # Track error in thread
            
            def update_progress():
                # Update progress while indexing is running
                elapsed = 0
                update_interval = 5  # seconds
                while not indexing_complete[0] and elapsed < indexing_timeout:
                    time.sleep(update_interval)
                    elapsed = time.time() - start_time
                    self.root.after(0, lambda e=elapsed: self.status_var.set(f"Indexing in progress... ({e:.1f}s elapsed)"))
                    self.root.after(0, lambda e=elapsed: self.add_to_debug_console(f"Still working... {e:.1f}s elapsed"))
                
                if not indexing_complete[0] and elapsed >= indexing_timeout:
                    # If still running after timeout, signal timeout
                    indexing_error[0] = "Indexing timed out after 15 minutes"
                    self.root.after(0, lambda: self.add_to_debug_console("Indexing timed out! Stopping..."))
            
            def run_indexing():
                try:
                    # Inform status monitor that it can track updates
                    result = index_folder_to_chromadb(
                        folder_path=folder_path,
                        collection_name=collection_name,
                        persist_dir=persist_dir,
                        status_callback=lambda msg: self.root.after(0, lambda: self.add_to_debug_console(msg))
                    )
                    # Set result atomically
                    nonlocal success
                    success = result
                except Exception as e:
                    # Set error atomically
                    indexing_error[0] = str(e)
                    self.root.after(0, lambda: self.add_to_debug_console(f"Indexing error: {str(e)}"))
                finally:
                    # Signal completion
                    indexing_complete[0] = True
            
            # Start indexing in a thread
            indexing_thread = threading.Thread(target=run_indexing)
            indexing_thread.daemon = True
            indexing_thread.start()
            
            # Start progress monitoring in another thread
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Wait for indexing to complete or timeout (with periodic UI updates)
            while not indexing_complete[0] and indexing_thread.is_alive():
                # Update UI periodically
                self.root.update_idletasks()
                time.sleep(0.1)
            
            # If there was an error in the thread, get it
            if indexing_error[0]:
                error_message = indexing_error[0]
                success = False
                
            # Handle results
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Update UI with results
            if success:
                self.root.after(0, lambda: self.status_var.set(f"Indexing complete in {elapsed:.1f}s"))
                self.root.after(0, lambda: self.current_collection.set(collection_name))
                self.root.after(0, lambda: setattr(self, 'current_persist_dir', persist_dir))
                if self.rag_toggle:
                    self.root.after(0, lambda: self.rag_toggle.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.rag_enabled.set(True))
                self.root.after(0, lambda: self.display_status_message(f"Indexed folder '{os.path.basename(folder_path)}' as collection '{collection_name}'"))
            else:
                error_text = error_message if error_message else "Unknown error"
                self.root.after(0, lambda: self.status_var.set(f"Indexing failed: {error_text}"))
                self.root.after(0, lambda: self.display_status_message(f"Failed to index folder. Error: {error_text}"))
                
        except Exception as e:
            # Handle any exceptions in the main thread
            end_time = time.time()
            elapsed = end_time - start_time
            error_text = str(e)
            self.root.after(0, lambda: self.status_var.set(f"Indexing error in {elapsed:.1f}s: {error_text}"))
            self.root.after(0, lambda: self.add_to_debug_console(f"Main thread error: {error_text}"))
            self.root.after(0, lambda: self.display_status_message(f"Error: {error_text}"))
        
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.index_folder_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_db_btn.config(state=tk.NORMAL))
    
    def load_existing_collection(self):
        """Load an existing ChromaDB collection"""
        # Ask for persistent directory
        persist_dir = filedialog.askdirectory(
            title="Select ChromaDB Directory",
            initialdir=self.default_persist_dir if os.path.exists(self.default_persist_dir) else None
        )
        
        if not persist_dir:
            return  # User cancelled
            
        # Try to list available collections
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            if not collection_names:
                messagebox.showinfo("No Collections", "No collections found in the selected directory.")
                return
                
            # Ask user to select a collection
            collection_name = None
            if len(collection_names) == 1:
                # If only one collection, use it directly
                if messagebox.askyesno("Load Collection", f"Load collection '{collection_names[0]}'?"):
                    collection_name = collection_names[0]
            else:
                # Create a proper selection dialog for multiple collections
                selection_window = tk.Toplevel(self.root)
                selection_window.title("Select Collection")
                selection_window.geometry("400x300")
                selection_window.transient(self.root)  # Set as transient to main window
                selection_window.grab_set()  # Make modal
                selection_window.focus_set()  # Set focus
                
                # Add a label
                tk.Label(selection_window, text="Select a collection to load:", pady=10).pack(fill=tk.X)
                
                # Add list with scrollbar
                list_frame = tk.Frame(selection_window)
                list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                
                scrollbar = tk.Scrollbar(list_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("TkDefaultFont", 12), 
                                     selectmode=tk.SINGLE)
                listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                scrollbar.config(command=listbox.yview)
                
                # Populate the listbox
                for name in collection_names:
                    listbox.insert(tk.END, name)
                    
                # Pre-select the first item
                if collection_names:
                    listbox.selection_set(0)
                    listbox.activate(0)
                    listbox.see(0)
                
                # Create a variable to store the result
                result = [None]
                
                # Define selection functions
                def on_select():
                    selected_indices = listbox.curselection()
                    if selected_indices:
                        result[0] = collection_names[selected_indices[0]]
                        selection_window.destroy()
                    else:
                        messagebox.showwarning("Selection Required", "Please select a collection.")
                
                def on_double_click(event):
                    selected_indices = listbox.curselection()
                    if selected_indices:
                        result[0] = collection_names[selected_indices[0]]
                        selection_window.destroy()
                
                # Add double-click binding
                listbox.bind("<Double-1>", on_double_click)
                
                # Add buttons at the bottom
                button_frame = tk.Frame(selection_window)
                button_frame.pack(fill=tk.X, padx=10, pady=10)
                
                cancel_btn = tk.Button(button_frame, text="Cancel", command=selection_window.destroy)
                cancel_btn.pack(side=tk.RIGHT, padx=5)
                
                select_btn = tk.Button(button_frame, text="Select", command=on_select)
                select_btn.pack(side=tk.RIGHT, padx=5)
                select_btn.focus_set()  # Set initial focus to Select button
                
                # Make Enter press the Select button
                selection_window.bind("<Return>", lambda event: on_select())
                selection_window.bind("<Escape>", lambda event: selection_window.destroy())
                
                # Center the dialog over the main window
                selection_window.update_idletasks()
                window_width = selection_window.winfo_width()
                window_height = selection_window.winfo_height()
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                
                x = self.root.winfo_x() + (self.root.winfo_width() - window_width) // 2
                y = self.root.winfo_y() + (self.root.winfo_height() - window_height) // 2
                
                # Ensure the window is visible on screen
                x = max(0, min(x, screen_width - window_width))
                y = max(0, min(y, screen_height - window_height))
                
                selection_window.geometry(f"+{x}+{y}")
                
                # Wait for the window to be destroyed
                self.root.wait_window(selection_window)
                
                # Get the result
                collection_name = result[0]
            
            # If a collection was selected, load it
            if collection_name:
                self.current_collection.set(collection_name)
                self.current_persist_dir = persist_dir
                self.rag_toggle.config(state=tk.NORMAL)
                self.rag_enabled.set(True)
                self.display_status_message(f"Loaded collection: {collection_name}")
                
                # Show summary of collection
                try:
                    collection = client.get_collection(name=collection_name)
                    count = collection.count()
                    self.add_to_debug_console(f"Collection '{collection_name}' contains {count} chunks")
                except Exception as e:
                    self.add_to_debug_console(f"Error getting collection info: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load collections: {str(e)}")
            self.add_to_debug_console(f"Error loading collections: {str(e)}")
    
    def get_relevant_context(self, query, n_results=3):
        """Get relevant context from ChromaDB for the given query"""
        if not self.rag_enabled.get() or not self.current_collection.get():
            return None
            
        # Query ChromaDB
        results, error = query_chromadb(
            query=query,
            collection_name=self.current_collection.get(),
            persist_dir=self.current_persist_dir,
            n_results=n_results
        )
        
        if error or not results:
            self.add_to_debug_console(f"RAG query error: {error or 'No results'}")
            return None
            
        # Format results for the prompt
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Build context string
        context_parts = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Truncate doc if too long
            if len(doc) > 1500:
                doc = doc[:1500] + "..."
                
            source = meta.get('source', 'unknown')
            context_parts.append(f"[Document {i+1}] From: {source}\n{doc}\n")
            
        context = "\n".join(context_parts)
        
        # Log to debug console
        self.add_to_debug_console(f"Found {len(documents)} relevant documents for: {query[:50]}...")
        
        return context
        
    def clear_rag_collection(self):
        """Clear the current RAG collection from memory"""
        if not self.current_collection.get():
            self.display_status_message("No RAG collection loaded.")
            return
            
        if messagebox.askyesno("Clear RAG", f"Are you sure you want to unload the '{self.current_collection.get()}' collection?"):
            self.current_collection.set("")
            self.current_persist_dir = ""
            self.rag_enabled.set(False)
            self.rag_toggle.config(state=tk.DISABLED)
            self.display_status_message("RAG collection unloaded.")
            
    def test_rag_query(self):
        """Run a test query against the current RAG collection"""
        if not self.current_collection.get():
            self.display_status_message("No RAG collection loaded.")
            return
            
        # Ask for query
        query = simpledialog.askstring(
            "Test RAG Query", 
            "Enter a test query:",
            initialvalue="What is this document about?"
        )
        
        if not query:
            return  # User cancelled
            
        # Clear debug console for results
        self.clear_debug_console()
        self.add_to_debug_console(f"Testing RAG query: {query}")
        
        # Get results
        try:
            results, error = query_chromadb(
                query=query,
                collection_name=self.current_collection.get(),
                persist_dir=self.current_persist_dir,
                n_results=3
            )
            
            if error:
                self.add_to_debug_console(f"Error: {error}")
                return
                
            if not results or not results['documents'] or not results['documents'][0]:
                self.add_to_debug_console("No results found.")
                return
                
            # Display results
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] if 'distances' in results else [0] * len(documents)
            
            self.add_to_debug_console(f"Found {len(documents)} relevant documents:")
            
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Truncate doc if too long
                if len(doc) > 200:
                    doc = doc[:200] + "..."
                    
                source = meta.get('source', 'unknown')
                self.add_to_debug_console(f"\n[{i+1}] From: {source}")
                self.add_to_debug_console(f"Relevance: {1 - dist:.4f}")
                self.add_to_debug_console(f"Content: {doc}")
                
        except Exception as e:
            self.add_to_debug_console(f"Error testing RAG query: {str(e)}")
            
    def add_to_debug_console(self, text):
        """Add text to debug console with clickable line numbers
        
        THREAD-SAFE: May be called from background inference thread.
        Schedules GUI work on main thread using root.after(0, ...).
        """
        # Schedule GUI work on main thread for thread safety
        self.root.after(0, lambda: self._add_to_debug_console_impl(text))
    
    def _add_to_debug_console_impl(self, text):
        """Internal implementation of add_to_debug_console - runs on main thread only"""
        import re
        
        self.debug_console.config(state=tk.NORMAL)
        
        # Look for "line X" patterns in error messages
        line_pattern = r'(line \d+)'
        
        if re.search(line_pattern, text, re.IGNORECASE):
            # Split text to handle line number links
            parts = re.split(line_pattern, text, flags=re.IGNORECASE)
            
            for part in parts:
                if re.match(line_pattern, part, re.IGNORECASE):
                    # Extract line number
                    line_num = re.findall(r'\d+', part)[0]
                    
                    # Insert as clickable link
                    start_pos = self.debug_console.index(tk.INSERT)
                    self.debug_console.insert(tk.END, part)
                    end_pos = self.debug_console.index(tk.INSERT)
                    
                    # Create unique tag for this link
                    tag_name = f"line_link_{line_num}_{start_pos}"
                    self.debug_console.tag_add(tag_name, start_pos, end_pos)
                    self.debug_console.tag_configure(tag_name, foreground="blue", underline=True)
                    
                    # Bind click event
                    self.debug_console.tag_bind(tag_name, "<Button-1>", 
                        lambda e, ln=line_num: self.jump_to_line_from_debug(ln))
                    
                    # Change cursor on hover
                    self.debug_console.tag_bind(tag_name, "<Enter>", 
                        lambda e: self.debug_console.config(cursor="hand2"))
                    self.debug_console.tag_bind(tag_name, "<Leave>", 
                        lambda e: self.debug_console.config(cursor=""))
                else:
                    # Regular text
                    self.debug_console.insert(tk.END, part)
        else:
            # No line numbers found, insert normally
            self.debug_console.insert(tk.END, text)
        
        self.debug_console.insert(tk.END, "\n")
        self.debug_console.see(tk.END)
        self.debug_console.config(state=tk.DISABLED)
    
    def analyze_code_for_debugging(self, content, filename=None):
        """Analyze loaded code and add useful debugging information to debug console
        
        This provides automatic context that helps with debugging and code updates:
        - Code structure analysis
        - Import statements
        - Function/class definitions
        - Potential issues or patterns
        - File metadata
        """
        # Show analysis in system messages window (user-facing, not debug)
        msg = self.display_system_message

        if filename:
            line_count = content.count('\n') + 1
            file_size = len(content.encode('utf-8'))
            msg(f"CODE ANALYSIS: {os.path.basename(filename)}")
            msg(f"  {line_count:,} lines, {file_size:,} bytes")
        else:
            msg("CODE ANALYSIS")

        # Imports
        import_lines = re.findall(r'^(?:from|import)\s+.+$', content, re.MULTILINE)
        if import_lines:
            msg(f"  Imports: {len(import_lines)}")

        # Classes
        classes = re.findall(r'^class\s+(\w+).*?:', content, re.MULTILINE)
        if classes:
            msg(f"  Classes: {', '.join(classes)}")

        # Functions
        functions = re.findall(r'^(?:\s{0,4})?def\s+(\w+)\s*\(.*?\):', content, re.MULTILINE)
        if functions:
            unique_funcs = list(dict.fromkeys(functions))
            if len(unique_funcs) <= 10:
                msg(f"  Functions: {', '.join(f + '()' for f in unique_funcs)}")
            else:
                msg(f"  Functions: {len(unique_funcs)} total")

        # Patterns
        patterns = []
        if re.findall(r'^[A-Z_]+\s*=\s*.+$', content, re.MULTILINE):
            patterns.append("globals")
        if len(re.findall(r'\btry\s*:', content)):
            patterns.append("try/except")
        if '__main__' in content:
            patterns.append("__main__")
        todos = re.findall(r'#\s*(?:TODO|FIXME|XXX|HACK|NOTE).*$', content, re.MULTILINE | re.IGNORECASE)
        if todos:
            patterns.append(f"{len(todos)} TODOs")
        if patterns:
            msg(f"  Patterns: {', '.join(patterns)}")
    
    def jump_to_line_from_debug(self, line_num):
        """Jump to line in IDE from debug console click"""
        try:
            line_number = int(line_num)
            
            # Check if IDE window exists and is available
            if hasattr(self, 'ide_window') and self.ide_window.root.winfo_exists():
                self.ide_window.go_to_line_direct(line_number)
                self.ide_window.show_window()
                
                # Show confirmation in debug console
                self.debug_console.config(state=tk.NORMAL)
                self.debug_console.insert(tk.END, f"→ Jumped to line {line_number} in IDE\n")
                self.debug_console.see(tk.END)
                self.debug_console.config(state=tk.DISABLED)
            else:
                # IDE not open, show message
                self.debug_console.config(state=tk.NORMAL)
                self.debug_console.insert(tk.END, f"→ IDE not open. Line {line_number} noted.\n")
                self.debug_console.see(tk.END)
                self.debug_console.config(state=tk.DISABLED)
                
        except ValueError:
            # Invalid line number
            self.debug_console.config(state=tk.NORMAL)
            self.debug_console.insert(tk.END, f"→ Invalid line number: {line_num}\n")
            self.debug_console.see(tk.END)
            self.debug_console.config(state=tk.DISABLED)

    def _validate_model_numerics(self):
        """Validate model for numerical issues that cause CUDA asserts"""
        if not hasattr(self, 'transformers_model') or self.transformers_model is None:
            return

        try:
            self.add_to_debug_console("🔍 Validating model numerics...")

            # Check model parameters for NaN/inf values
            nan_params = []
            inf_params = []
            for name, param in self.transformers_model.named_parameters():
                if param.numel() > 0:  # Skip empty parameters
                    if torch.isnan(param).any():
                        nan_params.append(name)
                    if torch.isinf(param).any():
                        inf_params.append(name)

            if nan_params:
                self.add_to_debug_console(f"⚠️ WARNING: Found NaN values in parameters: {nan_params[:3]}...")
                # Try to fix NaN values
                for name, param in self.transformers_model.named_parameters():
                    if torch.isnan(param).any():
                        self.add_to_debug_console(f"🔧 Fixing NaN in {name}")
                        param.data = torch.nan_to_num(param.data, nan=0.0)
            if inf_params:
                self.add_to_debug_console(f"⚠️ WARNING: Found Inf values in parameters: {inf_params[:3]}...")
                # Try to fix Inf values
                for name, param in self.transformers_model.named_parameters():
                    if torch.isinf(param).any():
                        self.add_to_debug_console(f"🔧 Fixing Inf in {name}")
                        param.data = torch.clamp(param.data, min=-1e6, max=1e6)

            # Try a small forward pass to catch CUDA assert issues early
            with torch.no_grad():
                # Create dummy input
                dummy_input = torch.randint(0, 1000, (1, 10), dtype=torch.long)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    self.transformers_model = self.transformers_model.cuda()

                try:
                    outputs = self.transformers_model(dummy_input, output_hidden_states=False, output_attentions=False)
                    logits = outputs.logits

                    # Check for NaN/inf in logits
                    if torch.isnan(logits).any():
                        self.add_to_debug_console("⚠️ WARNING: Model produces NaN logits - clamping")
                        logits = torch.clamp(logits, min=-1e6, max=1e6)
                    if torch.isinf(logits).any():
                        self.add_to_debug_console("⚠️ WARNING: Model produces Inf logits - clamping")
                        logits = torch.clamp(logits, min=-1e6, max=1e6)

                    # Check logit ranges
                    logit_min, logit_max = logits.min().item(), logits.max().item()
                    if abs(logit_min) > 1e6 or abs(logit_max) > 1e6:
                        self.add_to_debug_console(f"⚠️ WARNING: Extreme logit values detected: min={logit_min:.2e}, max={logit_max:.2e}")

                    # Test softmax computation to catch CUDA assert issues
                    try:
                        probs = torch.softmax(logits.float(), dim=-1)
                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            self.add_to_debug_console("⚠️ CRITICAL: Softmax produces NaN/Inf probabilities!")
                        elif (probs < 0).any():
                            self.add_to_debug_console("⚠️ CRITICAL: Softmax produces negative probabilities!")
                        else:
                            self.add_to_debug_console("✅ Softmax computation OK")
                    except Exception as softmax_e:
                        self.add_to_debug_console(f"⚠️ CRITICAL: Softmax computation failed: {str(softmax_e)}")

                except Exception as e:
                    self.add_to_debug_console(f"⚠️ WARNING: Dummy forward pass failed: {str(e)}")

            self.add_to_debug_console("✅ Model validation completed")

        except Exception as e:
            self.add_to_debug_console(f"⚠️ Model validation failed: {str(e)}")

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            # Create tooltip window
            tooltip = Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            # Add text
            label = Label(tooltip, text=text, background="lightyellow",
                         relief="solid", borderwidth=1, font=("TkDefaultFont", 9))
            label.pack()

            # Store reference
            widget.tooltip = tooltip

        def on_leave(event):
            # Destroy tooltip if it exists
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

def filter_thinking(text, show_thinking):
    """Filter thinking sections based on the show_thinking flag"""
    if show_thinking:
        return text
    # Remove both <think> and <thinking> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return text.strip()

# Main entry point
if __name__ == "__main__":
    # Set up Ctrl+C cleanup handler
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    root = tk.Tk()
    app = OllamaGUI(root)
    root.mainloop()