from helper import get_log_probs, load_model
import math

def perplexity(text, tokenizer, model, device):
    log_probs = get_log_probs(text, tokenizer, model, device)
    avg_logp = sum(log_probs) / len(log_probs)
    return math.exp(-avg_logp)

