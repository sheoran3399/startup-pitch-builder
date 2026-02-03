FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by Ollama
RUN apt-get update && apt-get install -y curl zstd && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
COPY requirements-crewai.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-crewai.txt

# Pull the llama3.2 model at build time so it's baked into the image
RUN bash -c 'ollama serve & OLLAMA_PID=$! && sleep 5 && ollama pull llama3.2; kill $OLLAMA_PID 2>/dev/null; true'

COPY . .

EXPOSE 8501

# Start both Ollama and Streamlit
CMD bash -c "ollama serve & sleep 3 && streamlit run Home.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"
