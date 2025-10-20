from helper import get_log_probs, load_model
import math

def perplexity(text):
    tokenizer, model, device = load_model()
    log_probs = get_log_probs(text, tokenizer, model, device)
    avg_logp = sum(log_probs) / len(log_probs)
    print(log_probs)
    return math.exp(-avg_logp)

if __name__ == "__main__":
    print(perplexity("You’ll almost certainly see one or two tokens with very low log-prob (big negative number). Those are the reason your PPL ~173."))

