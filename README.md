# LLM Cost Explorer 💰

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MIT Professional Education: Agentic AI**  
*Understanding the Economics of Large Language Models at Scale*

An interactive tool that helps you calculate and visualize LLM API costs across OpenAI, Anthropic, and Google models.

---

## 🎯 The Key Insight

> **The same AI transaction can cost between $1 and $230** depending on model choice — a 200x variance!

Understanding these economics is essential for any business considering AI implementation.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔤 Real-time Token Counter** | Uses OpenAI's tiktoken to count tokens as you type |
| **💰 Multi-Model Comparison** | Compare 10+ models from OpenAI, Anthropic, and Google |
| **📈 Scale Analysis** | See how costs grow from 1K to 1M API calls |
| **🗺️ Cost Heatmaps** | Visualize cost by prompt length vs response length |
| **📊 Export Results** | Download CSV, JSON, or summary for your assignment |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**New to Docker?** 👉 See our [Docker Guide for Beginners](docs/DOCKER_GUIDE.md)

```bash
# Clone the repository
git clone https://github.com/dlwhyte/AgenticAI_foundry.git
cd AgenticAI_foundry

# Build the image (takes 2-3 minutes first time)
docker build -t agenticai-foundry .

# Run the container
docker run -p 8501:8501 agenticai-foundry
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**To stop:** Press `Ctrl+C` in the terminal.

### Option 2: Python (No Docker)

```bash
# Clone the repository
git clone https://github.com/dlwhyte/AgenticAI_foundry.git
cd AgenticAI_foundry

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
```

---

## 📸 Screenshots

### Token Counter
Real-time token counting with breakdown visualization

### Cost Comparison  
Side-by-side comparison across all major models

### Scale Analysis
See how costs compound at enterprise scale

---

## 📁 Project Structure

```
AgenticAI_foundry/
├── Home.py                      # Landing page
├── pages/
│   └── 1_LLM_Cost_Calculator.py # Main calculator tool
├── docs/
│   └── DOCKER_GUIDE.md          # Docker setup guide
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Models Included

| Provider | Models | Price Range (per 1M tokens) |
|----------|--------|----------------------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo | $0.15 - $30.00 |
| **Anthropic** | Claude Opus 4, Sonnet 4, Haiku 4.5 | $1.00 - $75.00 |
| **Google** | Gemini 1.5 Pro, 1.5 Flash, 2.0 Flash | $0.075 - $5.00 |

*Prices as of January 2025. See provider websites for current rates.*

---

## 💡 Key Concepts

### Tokens ≠ Words
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- "Hello, world!" = 4 tokens

### Output Costs More Than Input
- Output tokens are typically **4x more expensive** than input
- Why? Generation requires sequential computation that can't be parallelized

### The 200x Variance
| Model | Monthly Cost (10K calls) |
|-------|-------------------------|
| Gemini 1.5 Flash | ~$1.50 |
| GPT-4o-mini | ~$3.00 |
| Claude Sonnet 4 | ~$72.00 |
| Claude Opus 4 | ~$360.00 |

Same task. 200x price difference.

---

## 📝 Assignment Connection

This tool supports your course assignment:

1. **Enter your business question** → Get real token counts
2. **Select response length** → Match your expected output
3. **Compare models** → See the cost variance across providers
4. **Scale to 10K and 1M calls** → Understand enterprise costs
5. **Export results** → Download data for your write-up

---

## 🔗 Resources

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Official token counter
- [OpenAI Pricing](https://openai.com/pricing) — Current OpenAI rates
- [Anthropic Pricing](https://www.anthropic.com/pricing) — Current Claude rates
- [Google AI Pricing](https://cloud.google.com/vertex-ai/pricing) — Current Gemini rates

---

## 🛠️ Technologies

- **[Streamlit](https://streamlit.io/)** — Web app framework
- **[Plotly](https://plotly.com/)** — Interactive charts
- **[tiktoken](https://github.com/openai/tiktoken)** — OpenAI's tokenizer
- **[Docker](https://www.docker.com/)** — Containerization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>MIT Professional Education | Agentic AI Course</b><br>
  <i>No API key required — all calculations run locally</i>
</p>
