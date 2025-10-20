from helper import get_log_probs, load_model
import math

ALPHA = 0.25
BETA = 0.25
GAMMA = 0.5

def FFLM(source, summary):
    tokenizer, model, device = load_model()

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

if __name__ == "__main__":
    abstract = "Exceptional mathematical reasoning ability is one of the key features that demonstrate the power of large language models (LLMs). How to comprehensively define and evaluate the mathematical abilities of LLMs, and even reflect the user experience in real-world scenarios, has emerged as a critical issue. Current benchmarks predominantly concentrate on problem-solving capabilities, presenting a substantial risk of model overfitting and fails to accurately measure the genuine mathematical reasoning abilities. In this paper, we argue that if a model really understands a problem, it should be robustly and readily applied across a diverse array of tasks. To this end, we introduce MathCheck, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently. MathCheck includes multiple mathematical reasoning tasks and robustness tests to facilitate a comprehensive evaluation of both mathematical reasoning ability and behavior testing. Utilizing MathCheck, we develop MathCheck-GSM and MathCheck-GEO to assess mathematical textual reasoning and multi-modal reasoning capabilities, respectively, serving as upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K. We adopt MathCheck-GSM and MathCheck-GEO to evaluate over 26 LLMs and 17 multi-modal LLMs, assessing their comprehensive mathematical reasoning abilities. Our results demonstrate that while frontier LLMs like GPT-4o continue to excel in various abilities on the checklist, many other model families exhibit a significant decline. Further experiments indicate that, compared to traditional math benchmarks, MathCheck better reflects true mathematical abilities and represents mathematical intelligence more linearly, thereby supporting our design. Using MathCheck, we can also efficiently conduct informative behavior analysis to deeply investigate models. Finally, we show that our proposed checklist paradigm can easily extend to other reasoning tasks for their comprehensive evaluation."
    context = "Evaluating large language models’ mathematical abilities requires definitions and tests that capture real-world user experience, not just isolated problem-solving scores. Existing math benchmarks risk overfitting and inadequately reflect genuine reasoning; a model that truly understands a problem should generalize robustly across diverse task formulations."
    idea = "To this end, we introduce MathCheck, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently. MathCheck includes multiple mathematical reasoning tasks and robustness tests to facilitate a comprehensive evaluation of both mathematical reasoning ability and behavior testing. Utilizing MathCheck, we develop MathCheck-GSM and MathCheck-GEO to assess mathematical textual reasoning and multi-modal reasoning capabilities, respectively, serving as upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K."
    print(FFLM(abstract, idea))
