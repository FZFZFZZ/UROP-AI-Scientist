import requests

def query_ollama(prompt, model="llama3.1:8b"):
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

def main():
    abstract =  """
                Large language models (LLMs) often exhibit hallucinations, producing incorrect or outdated knowledge. Hence, model editing methods have emerged to
...             enable targeted knowledge updates. To achieve this, a prevailing paradigm is the locating-then-editing approach, which first locates influential p
...             arameters and then edits them by introducing a perturbation. While effective, current studies have demonstrated that this perturbation inevitably d
...             isrupt the originally preserved knowledge within LLMs, especially in sequential editing scenarios. ”Propose a solution and complete the abstract.
                """
    original_idea = """
                    To address this, we introduce AlphaEdit, a novel solution that projects perturbation onto the null space of the preserved knowledge before applying it to the parameters. 
                    """
    print(query_ollama(abstract))

if __name__ == "__main__":
    main()
