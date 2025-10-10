import os
import json
import time
import re
from typing import Tuple
from helper import get_response

MAX_ATTEMPTS = 20
SLEEP_BETWEEN_ATTEMPTS = 1

MODEL = "gpt-4o-mini" # choose from gpt-4o-mini, gpt-4o, gpt-5

IN_PATH  = "Data/context_idea_extraction/data.jsonl"
OUT_DIR  = "Data/context_idea_extraction"
OUT_PATH = os.path.join(OUT_DIR, f"short_{re.sub(r'[^A-Za-z0-9_.-]+', '_', MODEL)}.jsonl")
TEMPERATURE = 0
SYS = "You are a highly-experienced expert in NLP and AI. Do the following task. Output strictly in JSON format only. Ideas should EXCLUDE experiments, results, or impacts."
TASK = """
[Abstract]
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

[Output]
{
  "Context": "Previous sequence transduction models—used for tasks like machine translation—mainly relied on recurrent or convolutional neural networks. These models typically consisted of an encoder and a decoder connected through an attention mechanism. However, such architectures were computationally expensive, difficult to parallelize, and inefficient to train on large datasets.",
  "Idea": "Introduce a new, simpler architecture that relies entirely on attention mechanisms, eliminating recurrence and convolution while supporting efficient parallel training."
}

[Abstract]
Numerous capability and safety techniques of Large Language Models (LLMs), including RLHF, automated red-teaming, prompt engineering, and infilling, can be cast as sampling from an unnormalized target distribution defined by a given reward or potential function over the full sequence. In this work, we leverage the rich toolkit of Sequential Monte Carlo (SMC) for these probabilistic inference problems. In particular, we use learned twist functions to estimate the expected future value of the potential at each timestep, which enables us to focus inference-time computation on promising partial sequences. We propose a novel contrastive method for learning the twist functions, and establish connections with the rich literature of soft reinforcement learning. As a complementary application of our twisted SMC framework, we present methods for evaluating the accuracy of language model inference techniques using novel bidirectional SMC bounds on the log partition function. These bounds can be used to estimate the KL divergence between the inference and target distributions in both directions. We apply our inference evaluation techniques to show that twisted SMC is effective for sampling undesirable outputs from a pretrained model (a useful component of harmlessness training and automated red-teaming), generating reviews with varied sentiment, and performing infilling tasks.

[Output]
{
  "Context": "Recent work on improving the capabilities and safety of large language models often involves sampling or optimization under a reward-based objective. However, existing approaches like RLHF or red-teaming lack a unified probabilistic interpretation that links them to broader inference theory.",
  "Idea": "Reformulate various LLM training and control techniques as probabilistic inference problems and introduce a Sequential Monte Carlo framework with learned twist functions to efficiently approximate target distributions and guide generation."
}

[Abstract]
Multi-Head Attention (MHA) is a key component of Transformer. In MHA, attention heads work independently, causing problems such as low-rank bottleneck of attention score matrices and head redundancy. We propose Dynamically Composable Multi-Head Attention (DCMHA), a parameter and computation efficient attention architecture that tackles the shortcomings of MHA and increases the expressive power of the model by dynamically composing attention heads. At the core of DCMHA is a Compose function that transforms the attention score and weight matrices in an input-dependent way. DCMHA can be used as a drop-in replacement of MHA in any transformer architecture to obtain the corresponding DCFormer. DCFormer significantly outperforms Transformer on different architectures and model scales in language modeling, matching the performance of models with 1.7x-2.0x compute. For example, DCPythia-6.9B outperforms open source Pythia-12B on both pretraining perplexity and downstream task evaluation.

[Output]
{
"Context": "Multi-Head Attention (MHA) is a core Transformer component where heads attend independently. This independence can introduce limitations such as low-rank bottlenecks in attention score matrices and redundancy across heads, potentially constraining expressive capacity and efficiency.",
"Idea": "Introduce Dynamically Composable Multi-Head Attention (DCMHA), an attention architecture that composes heads dynamically based on the input. The approach applies a Compose function to transform attention score and weight matrices in an input-dependent manner."
}

[Abstract] 
{abstract}

[Output]
"""

def make_user_prompt(abstract: str) -> str:
    """
    Fill the abstract into TASK. We only replace the {abstract} placeholder
    to avoid interfering with other braces in the prompt.
    """
    if "{abstract}" in TASK:
        return TASK.replace("{abstract}", abstract)
    return f"{TASK.strip()}\n\n[Abstract] {abstract}\n"

def strict_json_extract(text: str) -> dict:
    """
    Extract and parse a single JSON object from model text output.
    - Finds the first '{' to the last '}' and tries json.loads
    - Validates keys "Context" and "Idea" exist and are strings
    Raises ValueError if invalid.
    """
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in output.")
    candidate = text[start:end+1].strip()
    obj = json.loads(candidate)

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    if "Context" not in obj or "Idea" not in obj:
        raise ValueError('JSON must contain keys "Context" and "Idea".')
    if not isinstance(obj["Context"], str) or not isinstance(obj["Idea"], str):
        raise ValueError('"Context" and "Idea" must be strings.')
    # Optional: enforce no result/experiments words in Idea (light heuristic) – skip for now
    return obj

def enforce_json_only_prompt(base_user_prompt: str) -> str:
    """Append a strict instruction to output only JSON with the two keys."""
    suffix = (
        "\n\nIMPORTANT: Output ONLY a single JSON object with exactly two fields:\n"
        '{ "Context": "...", "Idea": "..." }\n'
        "No extra text, no code fences, no comments."
    )
    return base_user_prompt + suffix

def get_valid_response(system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict:
    """
    Calls the LLM up to MAX_ATTEMPTS times until a valid JSON is returned.
    Returns the parsed dict.
    """
    last_err = None
    prompt = enforce_json_only_prompt(user_prompt)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = get_response(model, system_prompt, prompt, temperature=temperature)
            obj = strict_json_extract(raw)
            return obj
        except Exception as e:
            last_err = e
            prompt = (
                enforce_json_only_prompt(user_prompt)
                + "\nIf your previous output was not valid JSON, regenerate strictly as specified."
            )
            time.sleep(SLEEP_BETWEEN_ATTEMPTS)
    raise RuntimeError(f"Failed to obtain valid JSON after {MAX_ATTEMPTS} attempts: {last_err}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total, written = 0, 0

    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            rec = None  # <-- ensure it's defined for except
            try:
                rec = json.loads(line)
                pid = rec.get("id")
                abstract = rec.get("abstract", "")

                if not pid or not isinstance(abstract, str) or not abstract.strip():
                    raise ValueError("Invalid record: missing id or abstract.")

                user_prompt = make_user_prompt(abstract)
                obj = get_valid_response(SYS, user_prompt, MODEL, TEMPERATURE)

                out_record = {
                    "id": pid,
                    "Context": obj["Context"],
                    "Idea": obj["Idea"],
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                written += 1

                # --- small progress log every 10 successes
                if written % 10 == 0:
                    print(f"{written}/100", flush=True)

            except Exception as e:
                rec_id = rec.get("id") if isinstance(rec, dict) else None
                print(f"⚠️ Error on id={rec_id}: {type(e).__name__}: {e}", flush=True)

    print(f"Done. Processed: {total}, wrote: {written}, output: {OUT_PATH}")


if __name__ == "__main__":
    main()

