# Agentic AI Foundry 🤖

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MIT Professional Education: Agentic AI**  
*Interactive demos for understanding AI economics and multi-agent systems*

---

## 🎯 What's Included

| Demo | Module | Description |
|------|--------|-------------|
| **💰 LLM Cost Explorer** | Module 1 | Calculate and compare LLM API costs across providers |
| **🤖 Multi-Agent Demo** | Module 2 | Watch three AI agents collaborate in real-time |

---

## ✨ Features

### 💰 LLM Cost Explorer (Module 1)
- **Real-time Token Counter** — Uses OpenAI's tiktoken
- **Multi-Model Comparison** — 10+ models from OpenAI, Anthropic, Google
- **Scale Analysis** — See costs from 1K to 1M API calls
- **Export Results** — CSV, JSON for assignments

### 🤖 Multi-Agent Demo (Module 2)
- **Three Collaborating Agents** — Researcher → Writer → Editor
- **Dual Provider Support** — Ollama (free, local) or OpenAI (paid, cloud)
- **Live Agent Activity** — Watch agents hand off work in real-time
- **CLI Support** — Run from command line or Streamlit

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/dlwhyte/AgenticAI_foundry.git
cd AgenticAI_foundry

# Build and run
docker build -t agenticai-foundry .
docker run -p 8501:8501 agenticai-foundry
```

Open [http://localhost:8501](http://localhost:8501)

### Option 2: Python

```bash
# Clone and install
git clone https://github.com/dlwhyte/AgenticAI_foundry.git
cd AgenticAI_foundry
pip install -r requirements.txt

# Run
streamlit run Home.py
```

### For Multi-Agent Demo (Additional Setup)

The Multi-Agent Demo requires either Ollama or OpenAI:

**Ollama (Free, Local):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

**OpenAI (Paid):**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

👉 See [docs/CREWAI_SETUP.md](docs/CREWAI_SETUP.md) for detailed setup instructions.

---

## 📁 Project Structure

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
│   ├── DOCKER_GUIDE.md            # Docker setup guide
│   └── CREWAI_SETUP.md            # CrewAI/Ollama/OpenAI setup
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🖥️ CLI Usage

The Multi-Agent Demo also works from the command line:

```bash
# With Ollama (free)
python -m crews.research_crew --provider ollama --task "Research AI in healthcare"

# With OpenAI
python -m crews.research_crew --provider openai --task "Research AI in healthcare"

# Check setup
python -m crews.research_crew --check
```

---

## 📚 Module Connections

### Module 1: LLM Cost Explorer
> **The same AI transaction can cost between $1 and $230** — a 200x variance!

Use this tool to understand token economics and model pricing.

### Module 2: Multi-Agent Demo
> Watch three agents collaborate: **Researcher → Writer → Editor**

See multi-agent orchestration in action with CrewAI.

---

## 🛠️ Technologies

- **[Streamlit](https://streamlit.io/)** — Web app framework
- **[CrewAI](https://github.com/joaomdmoura/crewAI)** — Multi-agent orchestration
- **[Ollama](https://ollama.ai/)** — Local LLM runtime
- **[LangChain](https://langchain.com/)** — LLM integrations
- **[Plotly](https://plotly.com/)** — Interactive charts
- **[Docker](https://www.docker.com/)** — Containerization

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<p align="center">
  <b>MIT Professional Education | Agentic AI Course</b><br>
  <i>Demos work locally — API keys optional (Ollama mode)</i>
</p>
