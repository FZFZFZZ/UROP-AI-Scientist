# text = get_response("gpt-5", "You are concise.", "Explain transformers in one line.")

from typing import Optional
from openai import OpenAI

client = OpenAI()

def get_response(model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
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
