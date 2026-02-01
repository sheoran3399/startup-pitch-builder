# Multi-Agent Demo Setup Guide 🤖

This guide walks you through setting up the CrewAI Multi-Agent Demo with both **Ollama (free, local)** and **OpenAI (paid, cloud)** providers.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Ollama Setup (Free, Local)](#ollama-setup-free-local)
4. [OpenAI Setup (Paid, Cloud)](#openai-setup-paid-cloud)
5. [Running the Demo](#running-the-demo)
6. [Troubleshooting](#troubleshooting)
7. [CLI Usage](#cli-usage)

---

## Overview

The Multi-Agent Demo showcases three AI agents collaborating on a research task:

| Agent | Role | What They Do |
|-------|------|--------------|
| 🔍 **Researcher** | Research Analyst | Gathers facts, statistics, and insights |
| ✍️ **Writer** | Content Writer | Transforms research into clear prose |
| 📝 **Editor** | Editor | Polishes for clarity and accuracy |

You can run this with:
- **Ollama** — Free, runs entirely on your machine, no API key needed
- **OpenAI** — Faster, higher quality, requires API key (~$0.01 per run)

---

## Quick Start

### Fastest Path (Ollama - Free)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh   # Linux/Mac
# Or download from https://ollama.ai for Windows

# 2. Pull a model
ollama pull llama3.2

# 3. Start Ollama
ollama serve

# 4. Install Python dependencies
pip install crewai langchain-community

# 5. Run the Streamlit app
streamlit run Home.py
```

Then navigate to the "Multi-Agent Demo" page in the sidebar.

---

## Ollama Setup (Free, Local)

### What is Ollama?

Ollama lets you run large language models locally on your machine. No API keys, no costs, no data leaving your computer.

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
1. Download from [ollama.ai](https://ollama.ai)
2. Run the installer
3. Ollama will be available in your terminal

### Step 2: Pull a Model

Ollama needs to download a model before you can use it:

```bash
# Recommended: Llama 3.2 (good balance of speed/quality)
ollama pull llama3.2

# Alternatives:
ollama pull mistral      # Fast, good for simple tasks
ollama pull llama3.1     # More capable, slower
ollama pull phi3         # Microsoft's small model, very fast
ollama pull gemma2       # Google's model
```

**Model Comparison:**

| Model | Size | Speed | Quality | RAM Needed |
|-------|------|-------|---------|------------|
| phi3 | 2.3GB | ⚡⚡⚡ | ★★☆ | 4GB |
| mistral | 4.1GB | ⚡⚡ | ★★★ | 8GB |
| llama3.2 | 4.7GB | ⚡⚡ | ★★★ | 8GB |
| llama3.1 | 8.5GB | ⚡ | ★★★★ | 16GB |

### Step 3: Start Ollama

```bash
ollama serve
```

This starts the Ollama server on `http://localhost:11434`. Keep this terminal open.

### Step 4: Verify Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Or use the built-in check
python -m crews.research_crew --check
```

You should see a list of your installed models.

### Step 5: Install Python Dependencies

```bash
pip install crewai langchain-community
```

---

## OpenAI Setup (Paid, Cloud)

### When to Use OpenAI

- **Faster responses** — Cloud GPUs are more powerful
- **Higher quality** — GPT-4o-mini is excellent for most tasks
- **No local resources** — Doesn't use your computer's RAM/CPU

### Step 1: Get an API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to **API Keys** in the sidebar
4. Click **Create new secret key**
5. Copy the key (starts with `sk-`)

### Step 2: Add Credits

OpenAI requires prepaid credits:
1. Go to **Settings** → **Billing**
2. Add a payment method
3. Add credits ($5 minimum)

**Cost Estimate:**
- GPT-4o-mini: ~$0.01 per demo run
- GPT-4o: ~$0.10 per demo run
- $5 credit = ~500 demo runs with GPT-4o-mini

### Step 3: Set Your API Key

**Option A: Environment Variable (Recommended)**

```bash
# Linux/Mac - add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-your-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=sk-your-key-here
```

**Option B: Enter in App**

The Streamlit app has a text input for the API key in the sidebar.

### Step 4: Install Python Dependencies

```bash
pip install crewai langchain-openai
```

---

## Running the Demo

### Option 1: Streamlit (Recommended)

```bash
# From the repo root
streamlit run Home.py
```

Then click **"Multi-Agent Demo"** in the sidebar.

### Option 2: Docker

```bash
# Build with CrewAI dependencies
docker build -t agenticai-foundry .

# Run with Ollama (must have Ollama running on host)
docker run -p 8501:8501 --add-host=host.docker.internal:host-gateway agenticai-foundry

# Run with OpenAI
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-key agenticai-foundry
```

### Option 3: CLI

```bash
# With Ollama
python -m crews.research_crew --provider ollama --task "Research AI in healthcare"

# With OpenAI
python -m crews.research_crew --provider openai --task "Research AI in healthcare"

# See all options
python -m crews.research_crew --help
```

---

## Troubleshooting

### Ollama Issues

**"Ollama not running"**
```bash
# Start the server
ollama serve

# Or check if it's already running
curl http://localhost:11434/api/tags
```

**"Model not found"**
```bash
# Pull the model
ollama pull llama3.2

# List installed models
ollama list
```

**"Out of memory"**
- Try a smaller model: `ollama pull phi3`
- Close other applications
- Check if you have enough RAM (see model table above)

**Slow responses**
- This is normal for local models
- Try a smaller/faster model like `phi3` or `mistral`
- GPU acceleration helps significantly if available

### OpenAI Issues

**"API key not found"**
```bash
# Check if it's set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-your-key-here"
```

**"Insufficient credits"**
- Add credits at [platform.openai.com/settings/billing](https://platform.openai.com/settings/billing)

**"Rate limit exceeded"**
- Wait a minute and try again
- Upgrade your OpenAI plan for higher limits

### General Issues

**"CrewAI not installed"**
```bash
pip install crewai langchain-community langchain-openai
```

**"Import error"**
```bash
# Make sure you're in the repo directory
cd AgenticAI_foundry

# Install all dependencies
pip install -r requirements.txt
pip install crewai langchain-community langchain-openai
```

---

## CLI Usage

The crew logic works standalone without Streamlit:

```bash
# Basic usage
python -m crews.research_crew --provider ollama --task "Your topic here"

# Full options
python -m crews.research_crew \
  --provider openai \
  --model gpt-4o \
  --api-key sk-your-key \
  --task "Research the impact of AI on education"

# Check provider availability
python -m crews.research_crew --check

# Quiet mode (less output)
python -m crews.research_crew --provider ollama --task "Topic" --quiet
```

**CLI Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--task`, `-t` | Research topic (required) | — |
| `--provider`, `-p` | `ollama` or `openai` | `ollama` |
| `--model`, `-m` | Model name | Provider default |
| `--api-key`, `-k` | OpenAI API key | From env |
| `--quiet`, `-q` | Suppress verbose output | False |
| `--check` | Check provider status | — |

---

## Architecture

```
AgenticAI_foundry/
├── Home.py                        # Landing page
├── pages/
│   ├── 1_LLM_Cost_Calculator.py   # Cost calculator (Module 1)
│   └── 2_Multi_Agent_Demo.py      # Multi-agent demo (Module 2)
├── crews/
│   ├── __init__.py
│   └── research_crew.py           # Agent logic (CLI + importable)
├── docs/
│   ├── DOCKER_GUIDE.md
│   └── CREWAI_SETUP.md            # This file
├── Dockerfile
├── requirements.txt
└── README.md
```

**Key Design Decisions:**

1. **Separation of concerns** — `crews/research_crew.py` is pure Python with no Streamlit dependency
2. **Dual interface** — Same logic works via CLI or Streamlit import
3. **Provider abstraction** — Easy to add new providers (Anthropic, etc.)
4. **Graceful degradation** — App shows helpful errors if dependencies missing

---

## Next Steps

After running the demo:

1. **Compare providers** — Try the same topic with Ollama and OpenAI
2. **Modify agents** — Edit `crews/research_crew.py` to change agent behavior
3. **Add tools** — CrewAI supports web search, file reading, and custom tools
4. **Build your own crew** — Create new agent configurations for different tasks

---

<p align="center">
  <b>MIT Professional Education | Agentic AI Course | Module 2</b>
</p>
