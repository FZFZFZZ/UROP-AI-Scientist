import requests
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

model = SentenceTransformer('all-mpnet-base-v2')

def compare_proposals(original, generated):
    embeddings = model.encode([original, generated], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

def generate_proposal(abstract, prompt_constructor):
    url = "http://localhost:11434/api/generate"
    prompt = prompt_constructor(abstract)
    try:
        response = requests.post(url, json={
            "model": "llama3.1:8b",
            "temperature": 1.4,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        return response.json()["response"]
    except Exception as e:
        return f"[Error] {str(e)}"

def naive(abstract):
    task = ("""
    You are an experienced NLP/IR researcher. You will be given a problem discovered in the field. 
    Your task is to propose a novel idea to address this problem. Limit the response to around 130 words. 
    Be specific in relavant Mathematic rigor. Do not just propose general ideas. Starting from 'We propose'. 
    Write in academic language.
    """)
    return f"{task}\n\nProblem: {abstract.strip()}\n\nProposed Solution: We propose"

def main():
    log_file = "generation_log.txt"

    limitation = """
    Large language models (LLMs) often exhibit hallucinations, producing incorrect or outdated knowledge. 
    Hence, model editing methods have emerged to enable targeted knowledge updates. 
    To achieve this, a prevailing paradigm is the locating-then-editing approach, 
    which first locates influential parameters and then edits them by introducing a perturbation. 
    While effective, current studies have demonstrated that this perturbation inevitably disrupts 
    the originally preserved knowledge within LLMs, especially in sequential editing scenarios.
    """

    original_idea = """
    We propose AlphaEdit, a novel model editing approach that addresses the persistent challenge of preserving 
    knowledge in large language models (LLMs) during sequential edits. AlphaEdit introduces a new solution by 
    removing the e₀ constraint entirely during optimization, allowing full focus on minimizing e₁. To prevent 
    overfitting and preserve existing knowledge implicitly, we apply a null-space projection: the computed parameter 
    perturbation is projected onto the null space of the preserved knowledge before being applied. This projection ensures 
    that the perturbation does not interfere with the preserved knowledge’s representation space. Our method maintains the 
    distributional structure of hidden activations post-edit, ensuring semantic stability and mitigating the accumulation of 
    harmful effects across edits.
    """

    max_score = 0
    best_response = ""
    err_count = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(generate_proposal, limitation, naive) for _ in range(100)]
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("[Error]"):
                err_count += 1
                continue
            score = compare_proposals(original_idea, result)
            if score > max_score:
                max_score = score
                best_response = result
            #if score >= 0.9:
            #    print("Generated Proposal:\n", result)
            #    print("Score (good):", score)
            #else:
            #    print("Score (bad):", score)
    
    print("Best Proposal:\n", best_response)
    print("Best Score:", max_score)
    print("Error Count:", err_count)

for i in range(8):
    if __name__ == "__main__":
        main()
    print("iteration", i + 1, "completed.")


