from helper import get_log_probs, load_model
import math

ALPHA = 0.25
BETA = 0.25
GAMMA = 0.5

def FFLM(source, summary, tokenizer, model, device):

    # Compute delta_Y_prior
    pY_lm = get_log_probs(summary, tokenizer, model, device)
    m = len(pY_lm)
    pY_s2s = get_log_probs(source + "TL;DR" + summary, tokenizer, model, device)[-m:]
    delta_Y_prior = (1 / m) * sum(math.exp(math.exp(pY_s2s[i])) * (pY_s2s[i] - pY_lm[i]) for i in range(m))

    # Compute delta_X_prior
    pX_lm = get_log_probs(source, tokenizer, model, device)
    n = len(pX_lm)
    pX_s2s = get_log_probs(summary + "Reference" + source, tokenizer, model, device)[-n:]
    delta_X_prior = (1 / n) * sum(math.exp(math.exp(pX_s2s[i])) * (pX_s2s[i] - pX_lm[i]) for i in range(n))

    # Compute delta_Y_cond
    pY_pref = get_log_probs(summary + source + "TL;DR" + summary, tokenizer, model, device)[-m:]
    delta_Y_cond = (1 / m) * sum(math.exp(math.exp(pY_s2s[i])) * (pY_s2s[i] - pY_pref[i]) for i in range(m))

    return ALPHA * delta_Y_prior + BETA * delta_X_prior + GAMMA * delta_Y_cond
