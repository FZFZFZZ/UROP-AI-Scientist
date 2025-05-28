import requests

def query_ollama(prompt, model="llama3.1:8b"):
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False  # Set to True for streaming
    })
    return response.json()["response"]

# Example
output = query_ollama("""Large language models (LLMs) often exhibit hallucinations, producing incorrect or outdated knowledge. Hence, model editing methods have emerged to
...  enable targeted knowledge updates. To achieve this, a prevailing paradigm is the locating-then-editing approach, which first locates influential p
... arameters and then edits them by introducing a perturbation. While effective, current studies have demonstrated that this perturbation inevitably d
... isrupt the originally preserved knowledge within LLMs, especially in sequential editing scenarios. ”Propose a solution and complete the abstract.""")
print(output)