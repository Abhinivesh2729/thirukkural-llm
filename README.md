# Thirukkural Leadership LLM API

A FastAPI-based REST API that serves a fine-tuned language model for generating wisdom and guidance based on Thirukkural, an ancient Tamil text on leadership and ethics.

## Model

This API uses the **[0xAbhi/thirukkural-leadership-merged](https://huggingface.co/0xAbhi/thirukkural-leadership-merged)** model from Hugging Face.

## Features

- RESTful API for generating text based on instructions
- Automatic GPU/CPU device detection
- Configurable response length
- Built with FastAPI for high performance
- Simple and intuitive API design

## Requirements

- Python 3.8 - 3.12 (PyTorch does not support Python 3.13 yet)
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM (optional, but recommended for faster inference)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Abhinivesh2729/thirukkural-llm.git
cd thirukkural-llm
```

2. **Create a virtual environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch:**
```bash
# For macOS
pip install torch torchvision torchaudio

# For Linux/Windows with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Start the server

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:
```bash
uvicorn serve:app --reload
```

### API Endpoints

#### Health Check
```bash
GET /
```

Response:
```json
{
  "status": "running",
  "model": "0xAbhi/thirukkural-leadership-merged",
  "device": "cuda"
}
```

#### Generate Text
```bash
POST /generate
```

Request body:
```json
{
  "instruction": "What is the principle of good leadership?",
  "input": "In times of crisis",
  "max_tokens": 256
}
```

Response:
```json
{
  "instruction": "What is the principle of good leadership?",
  "input": "In times of crisis",
  "answer": "..."
}
```

### Example using cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Explain the importance of wisdom in leadership",
    "input": "",
    "max_tokens": 200
  }'
```

### Example using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "instruction": "What makes a great leader?",
        "input": "During difficult times",
        "max_tokens": 256
    }
)

print(response.json()["answer"])
```

## Configuration

You can change the model by editing the `MODEL_NAME` variable in `serve.py`:

```python
MODEL_NAME = "your-username/your-model-name"
```

## Performance

- **CPU Mode**: 30-120 seconds per query
- **GPU Mode**: 2-5 seconds per query

## License

MIT

## Links

- **Model**: [0xAbhi/thirukkural-leadership-merged](https://huggingface.co/0xAbhi/thirukkural-leadership-merged)
- **Repository**: [github.com/Abhinivesh2729/thirukkural-llm](https://github.com/Abhinivesh2729/thirukkural-llm)
