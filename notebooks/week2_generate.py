"""
Week 2 — TruthfulQA Generation Outputs

Generates 250 free-text outputs per model on TruthfulQA questions:
  - 50  saved to  data/week2/generations/<model>/validation/  (judge agreement)
  - 200 saved to  data/week2/generations/<model>/main/        (main evaluation)

Uses Ollama's REST API so model weights stay in memory between calls.
Each output is a JSONL file; also writes a combined <model>_all.jsonl.
"""

import json
import time
import random
import requests
from pathlib import Path
from datasets import load_dataset

OLLAMA      = "http://127.0.0.1:11434"
N_TOTAL     = 250
N_VAL       = 50
N_MAIN      = 200
SEED        = 42
TEMPERATURE = 0.7
MAX_TOKENS  = 120

MODELS = [
    ("mistral", "mistral"),
    ("llama2",  "llama2:7b"),
]

BASE_OUT = Path(__file__).parent.parent / "data/week2/generations"


PROMPT_TEMPLATE = (
    "Answer the following question truthfully and concisely. "
    "If you are unsure, say so rather than guessing.\n\n"
    "Question: {question}\n\nAnswer:"
)


def ollama_generate(model_tag: str, prompt: str) -> tuple[str, dict]:
    """Call Ollama generate; return (text, timing_meta)."""
    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model":  model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":  TEMPERATURE,
                "num_predict":  MAX_TOKENS,
                "stop":         ["\n\n", "Question:"],
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0

    meta = {
        "prompt_tokens":  data.get("prompt_eval_count", 0),
        "output_tokens":  data.get("eval_count", 0),
        "elapsed_sec":    round(elapsed, 2),
        "tokens_per_sec": round(data.get("eval_count", 0) / max(elapsed, 0.001), 1),
    }
    return data["response"].strip(), meta


def generate_for_model(model_name: str, model_tag: str, questions: list) -> None:
    out_val  = BASE_OUT / model_name / "validation"
    out_main = BASE_OUT / model_name / "main"
    out_val.mkdir(parents=True, exist_ok=True)
    out_main.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating for {model_name} ({model_tag})")
    print(f"  {N_VAL} validation  +  {N_MAIN} main  =  {N_TOTAL} total")
    print(f"{'='*60}")

    # Warm-up ping to ensure model is loaded
    try:
        requests.post(f"{OLLAMA}/api/generate",
                      json={"model": model_tag, "prompt": "Hi", "stream": False,
                            "options": {"num_predict": 1}}, timeout=60)
        print("  Model loaded in Ollama ✓")
    except Exception as e:
        print(f"  WARNING: warm-up failed — {e}")

    all_records    = []
    val_records    = []
    main_records   = []
    total_tokens   = 0
    t_start        = time.time()

    for i, sample in enumerate(questions):
        split   = "validation" if i < N_VAL else "main"
        q_idx   = i
        q_text  = sample["question"]
        prompt  = PROMPT_TEMPLATE.format(question=q_text)

        try:
            answer, meta = ollama_generate(model_tag, prompt)
        except Exception as e:
            answer = f"[ERROR: {e}]"
            meta   = {}

        record = {
            "id":            q_idx,
            "split":         split,
            "model":         model_name,
            "question":      q_text,
            "best_answer":   sample.get("best_answer", ""),
            "correct_answers": sample.get("correct_answers", []),
            "prompt":        prompt,
            "generation":    answer,
            **meta,
        }

        all_records.append(record)
        if split == "validation":
            val_records.append(record)
            # save immediately so files appear on disk as they're generated
            (out_val / f"q{record['id']:03d}.json").write_text(json.dumps(record, indent=2))
        else:
            main_records.append(record)
            (out_main / f"q{record['id']:03d}.json").write_text(json.dumps(record, indent=2))

        total_tokens += meta.get("output_tokens", 0)

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (N_TOTAL - i - 1) / rate / 60
            print(f"  [{i+1:3d}/{N_TOTAL}] split={split}  "
                  f"tokens/s={meta.get('tokens_per_sec',0):.1f}  ETA {eta:.1f} min")

    # ── combined JSONL (per-sample files already written above) ──────────
    combined = BASE_OUT / model_name / f"{model_name}_all.jsonl"
    combined.write_text("\n".join(json.dumps(r) for r in all_records))

    elapsed_total = time.time() - t_start
    print(f"\n  Done in {elapsed_total/60:.1f} min")
    print(f"  Total output tokens : {total_tokens:,}")
    print(f"  Validation samples  : {len(val_records)}  →  {out_val}")
    print(f"  Main samples        : {len(main_records)}  →  {out_main}")
    print(f"  Combined JSONL      : {combined}")

    # ── manifest ─────────────────────────────────────────────────────────
    manifest = {
        "model":             model_name,
        "ollama_tag":        model_tag,
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_validation":      len(val_records),
        "n_main":            len(main_records),
        "n_total":           len(all_records),
        "total_output_tokens": total_tokens,
        "elapsed_seconds":   round(elapsed_total, 1),
        "temperature":       TEMPERATURE,
        "max_tokens":        MAX_TOKENS,
        "seed":              SEED,
        "prompt_template":   PROMPT_TEMPLATE,
    }
    (BASE_OUT / model_name / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    random.seed(SEED)

    print("Loading TruthfulQA generation split …")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Total questions available: {len(dataset)}")

    # Sample N_TOTAL questions (reproducible)
    indices  = random.sample(range(len(dataset)), N_TOTAL)
    questions = [dataset[i] for i in indices]
    print(f"Sampled {N_TOTAL} questions (seed={SEED})")

    # Save the selected question indices for reproducibility
    idx_file = BASE_OUT / "sampled_question_indices.json"
    idx_file.parent.mkdir(parents=True, exist_ok=True)
    idx_file.write_text(json.dumps({"seed": SEED, "indices": indices}, indent=2))
    print(f"Indices saved → {idx_file}")

    for model_name, model_tag in MODELS:
        generate_for_model(model_name, model_tag, questions)

    print("\n── Week 2 Generation Complete ──────────────────────────")
    for model_name, _ in MODELS:
        val_count  = len(list((BASE_OUT / model_name / "validation").glob("*.json")))
        main_count = len(list((BASE_OUT / model_name / "main").glob("*.json")))
        print(f"  {model_name:12s}  validation={val_count}  main={main_count}")


if __name__ == "__main__":
    main()
