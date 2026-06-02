"""
Week 3 — Pre-Run Checklist
Verifies all systems before pipeline work continues.
Pass/fail for each check. Exits with error if any critical check fails.
"""

import sys, os, json, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

BASE = Path(__file__).parent.parent
PASS = "  PASS"
FAIL = "  FAIL"

results = {}

def check(name, fn):
    try:
        msg = fn()
        print(f"{PASS}  {name}: {msg}")
        results[name] = {"status": "pass", "detail": msg}
    except Exception as e:
        print(f"{FAIL}  {name}: {e}")
        results[name] = {"status": "fail", "detail": str(e)}

print("=" * 55)
print("  Week 3 Pre-Run Checklist")
print("=" * 55)

# 1. Ollama running
def chk_ollama():
    r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    needed = {"mistral", "llama2"}
    found  = {m.split(":")[0] for m in models}
    missing = needed - found
    if missing:
        raise RuntimeError(f"missing: {missing}")
    return f"mistral + llama2 available"
check("Ollama", chk_ollama)

# 2. OpenRouter key valid
def chk_openrouter():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "openai/gpt-oss-120b:free",
              "messages": [{"role": "user", "content": "Reply OK"}],
              "max_tokens": 3},
        timeout=20
    )
    if "choices" not in r.json():
        raise RuntimeError(r.json())
    return "key valid, gpt-oss-120b reachable"
check("OpenRouter API", chk_openrouter)

# 3. FLAN-T5 loadable
def chk_flant5():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained("google/flan-t5-large")
    return "flan-t5-large tokenizer OK"
check("FLAN-T5", chk_flant5)

# 4. Week 2 generation files present
def chk_week2_gen():
    for model in ["mistral", "llama2"]:
        val  = list((BASE / f"data/week2/generations/{model}/validation").glob("q*.json"))
        main = list((BASE / f"data/week2/generations/{model}/main").glob("q*.json"))
        if len(val) != 50:
            raise RuntimeError(f"{model} validation: {len(val)}/50")
        if len(main) != 200:
            raise RuntimeError(f"{model} main: {len(main)}/200")
    return "50 val + 200 main for both models"
check("Week 2 generations", chk_week2_gen)

# 5. Mistral MC2 baseline saved
def chk_mc2():
    f = BASE / "data/week2/baselines/mistral_mc2.json"
    if not f.exists():
        raise RuntimeError("mistral_mc2.json missing")
    d = json.loads(f.read_text())
    return f"Mistral MC2={d['mc2_mean']:.4f}"
check("MC2 baseline (Mistral)", chk_mc2)

# 6. Week 3 output folders exist
def chk_dirs():
    for d in ["data/week3/judge_outputs", "data/week3/kappa", "data/week3/claims"]:
        (BASE / d).mkdir(parents=True, exist_ok=True)
    return "judge_outputs/ kappa/ claims/"
check("Week 3 folders", chk_dirs)

# ── summary ──────────────────────────────────────────────────────────────
print("=" * 55)
passed = sum(1 for v in results.values() if v["status"] == "pass")
total  = len(results)
print(f"  {passed}/{total} checks passed")

out = BASE / "data/week3/precheck_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"  Results saved → {out}")

if passed < total:
    print("\n  Fix failing checks before running week 3 pipeline.")
    sys.exit(1)
else:
    print("\n  All systems go. Run week3_judge_agreement.py next.")
