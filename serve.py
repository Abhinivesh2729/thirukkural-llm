from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------------------------------------
# CONFIG — Change this to YOUR Hugging Face model repo
# ---------------------------------------------------------
MODEL_NAME = "0xAbhi/thirukkural-leadership-merged"
# Example:
# MODEL_NAME = "Abhinivesh2729/thirukkural-leadership-merged"

# ---------------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------------
print("Loading model...")

# Pick best device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully!")

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="Thirukkural Leadership LLM API")

class Query(BaseModel):
    instruction: str
    input: str = ""
    max_tokens: int = 256


def build_prompt(instruction, user_input):
    if user_input.strip():
        return f"Instruction: {instruction}\nInput: {user_input}\nAnswer:"
    return f"Instruction: {instruction}\nAnswer:"


@app.post("/generate")
def generate_text(query: Query):
    prompt = build_prompt(query.instruction, query.input)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=query.max_tokens,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract only answer part
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()

    return {
        "instruction": query.instruction,
        "input": query.input,
        "answer": answer
    }


@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": device
    }
