"""
Week 4 — Confidence Scorer (Token Probability + Attention Entropy)

Two signals computed via GPT-2 (124M params, CPU-safe, no MPS issues):

  1. token_log_prob  — mean per-token log-probability of the generation text
                       under GPT-2. Low = model considers text unlikely = potential hallucination.

  2. attention_entropy — mean entropy of GPT-2's attention weight distributions
                         across all layers/heads. High entropy = diffuse, unfocused
                         attention = lower confidence signal.

Combined confidence score:
  conf = sigmoid(mean_log_prob / scale) × (1 − norm_attention_entropy)
  range [0, 1]; higher = more confident (less likely hallucination)

GPT-2 is used as a *scoring* model (not the generator). On an 8GB Mac, it
runs entirely on CPU at ~50 tokens/s — well within budget for this pipeline.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch

# lazy-load GPT-2 to avoid startup cost when module is imported but scorer not used
_model = None
_tokenizer = None

_MAX_TOKENS   = 512   # GPT-2 context window is 1024; cap here for speed
_LOG_PROB_SCALE = 3.0  # divisor to map mean log-prob to a ~[-1, 1] range before sigmoid


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ConfidenceScore:
    text: str
    token_log_prob: float            # mean log P(token|context) under GPT-2
    attention_entropy: float         # mean attention entropy (nats)
    confidence: float                # combined [0, 1] score
    n_tokens: int
    model_used: str = "gpt2"


# ── model loader ─────────────────────────────────────────────────────────────

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print("  [confidence_scorer] Loading GPT-2 on CPU …")
    _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    _tokenizer.pad_token = _tokenizer.eos_token
    _model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        attn_implementation="eager",   # use plain PyTorch attention so weights are exposed
    )
    _model.eval()
    print("  [confidence_scorer] GPT-2 loaded.")
    return _model, _tokenizer


# ── token log-probability ─────────────────────────────────────────────────────

def _token_log_prob(text: str, model, tokenizer) -> tuple[float, int]:
    """
    Return (mean_token_log_prob, n_tokens) for *text* under *model*.
    Uses teacher-forcing: shift logits by 1 to get P(t_i | t_<i).
    """
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=_MAX_TOKENS)
        input_ids = enc["input_ids"]       # (1, T)
        n = input_ids.shape[1]
        if n < 2:
            return 0.0, n

        outputs = model(input_ids=input_ids, output_attentions=False)
        logits  = outputs.logits            # (1, T, vocab)
        # log-softmax over vocab
        log_probs = torch.log_softmax(logits, dim=-1)  # (1, T, vocab)
        # for token t_i, its probability is given by logits at position i-1
        shift_log_probs = log_probs[0, :-1, :]          # (T-1, vocab)
        shift_labels    = input_ids[0, 1:]               # (T-1,)
        token_lp        = shift_log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        mean_lp         = token_lp.mean().item()
        return mean_lp, n


# ── attention entropy ─────────────────────────────────────────────────────────

def _attention_entropy(text: str, model, tokenizer) -> float:
    """
    Return the mean entropy (in nats) of the attention distributions
    across all layers and heads.

    Each attention head produces a distribution over token positions.
    Entropy is high when the head attends diffusely (uncertain/unfocused).
    """
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=_MAX_TOKENS)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0

        outputs    = model(input_ids=input_ids, output_attentions=True)
        attentions = outputs.attentions   # tuple of (1, n_heads, T, T) per layer

        entropies = []
        for layer_attn in attentions:
            # layer_attn: (1, n_heads, T, T)
            attn = layer_attn[0]          # (n_heads, T, T)
            # each row is already a probability distribution (softmax applied inside GPT-2)
            # clamp to avoid log(0)
            attn_clamped = attn.clamp(min=1e-9)
            ent = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)  # (n_heads, T)
            entropies.append(ent.mean().item())

        return float(sum(entropies) / len(entropies)) if entropies else 0.0


# ── combined confidence ───────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _combine(mean_log_prob: float, attn_entropy: float,
             max_entropy_nats: float = 6.0) -> float:
    """
    Map (mean_log_prob, attn_entropy) → confidence in [0, 1].
    - Higher mean_log_prob → higher confidence
    - Lower attn_entropy   → higher confidence
    """
    prob_signal    = _sigmoid(mean_log_prob / _LOG_PROB_SCALE)
    norm_entropy   = min(attn_entropy / max_entropy_nats, 1.0)
    entropy_signal = 1.0 - norm_entropy
    return round(0.5 * prob_signal + 0.5 * entropy_signal, 4)


# ── public API ────────────────────────────────────────────────────────────────

def score(text: str) -> ConfidenceScore:
    """Score a single generation or claim text."""
    model, tokenizer = _load_model()
    mean_lp, n_tok  = _token_log_prob(text, model, tokenizer)
    attn_ent        = _attention_entropy(text, model, tokenizer)
    conf            = _combine(mean_lp, attn_ent)
    return ConfidenceScore(
        text=text,
        token_log_prob=round(mean_lp, 4),
        attention_entropy=round(attn_ent, 4),
        confidence=conf,
        n_tokens=n_tok,
    )


def score_batch(texts: list[str]) -> list[ConfidenceScore]:
    """Score a list of texts, loading the model once."""
    _load_model()
    return [score(t) for t in texts]


def score_generation_file(gen: dict) -> ConfidenceScore:
    """
    Score from a week-2 generation dict (has 'generation' key).
    Uses the generated answer text as the scoring target.
    """
    return score(gen.get("generation", ""))


if __name__ == "__main__":
    test_texts = [
        "Albert Einstein was born in Ulm, Germany in 1879.",
        "The moon is made of green cheese and orbits Venus.",
        "Poverty in Africa has declined significantly in the 21st century.",
        "xyzzy fnord blorb wumbo — complete nonsense fabricated text.",
    ]
    print("Confidence Scorer — smoke test")
    print("=" * 55)
    for t in test_texts:
        s = score(t)
        print(f"\n  Text       : {t[:70]}")
        print(f"  log_prob   : {s.token_log_prob:.4f}")
        print(f"  attn_ent   : {s.attention_entropy:.4f}")
        print(f"  confidence : {s.confidence:.4f}")
