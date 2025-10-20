# text = get_response("gpt-5", "You are concise.", "Explain transformers in one line.")

from typing import Optional
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_response(model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
    client = OpenAI()
    model_lower = model.lower()
    if model_lower.startswith("gpt-5"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()

    chat = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return (chat.choices[0].message.content or "").strip()

def load_model(model_name="Qwen/Qwen2.5-7B", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device).eval()
    return tokenizer, model, device

def get_log_probs(text: str, tokenizer, model, device):
    """
    Return list of log probabilities per token (ignore first token, as standard perplexity does).
    """
    # tokenize without special tokens
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    N = input_ids.size(1)
    if N == 1:
        return []

    # forward pass
    with torch.no_grad():
        logits = model(input_ids).logits              # [1, N, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

    # shift to get P(x_i | x_<i)
    shifted_log_probs = log_probs[:, :-1, :]          # [1, N-1, vocab]
    next_tokens = input_ids[:, 1:]                    # [1, N-1]

    # gather logP for actual tokens
    token_log_probs = torch.gather(
        shifted_log_probs, 2, next_tokens.unsqueeze(-1)
    ).squeeze(-1)                                     # [1, N-1] → [N-1]

    return token_log_probs[0].tolist()
