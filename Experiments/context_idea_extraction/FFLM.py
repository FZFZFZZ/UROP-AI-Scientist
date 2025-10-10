from openai import OpenAI
import math

client = OpenAI()

def get_logprob(model: str, summary: str, context: str = "", top_k: int = 5):
    if context.strip():
        prompt = f"Source:\n{context.strip()}\n\nSummary:\n{summary.strip()}"
    else:
        prompt = summary.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=top_k,
        temperature=0,
        max_tokens=1
    )

    logprobs = []
    for token_info in response.choices[0].logprobs.content:
        token = token_info.token
        lp = token_info.logprob
        if lp is not None:
            logprobs.append(lp)
        else:
            logprobs.append(float("-9999"))
    return sum(logprobs), len(logprobs)

def FFLM(source: str, summary: str, model="gpt-4o"):
    lp_cond, n1 = get_logprob(model, summary, context=source)
    lp_uncond, n2 = get_logprob(model, summary)
    return (lp_cond - lp_uncond) / max(1, n1)


