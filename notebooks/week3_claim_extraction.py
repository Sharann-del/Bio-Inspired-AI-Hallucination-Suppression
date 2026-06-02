"""
Week 3 — Claim Extraction

Decomposes each free-text answer into atomic verifiable claims using
spaCy sentence segmentation + heuristic filtering.

Architecture decision: spaCy en_core_web_sm (already installed, ~12MB).
FLAN-T5-large was the original plan but hangs on this 8GB Mac due to
MPS initialization issues. spaCy sentence splitting achieves the same
goal — breaking answers into independently verifiable units — at
negligible cost and runs in seconds across all 500 outputs.

Each "claim" = one declarative sentence from the answer that can be
fact-checked as true or false.

Output: data/week3/claims/<model>_claims.jsonl
"""

import json, re
from pathlib import Path
import spacy

BASE    = Path(__file__).parent.parent
OUT_DIR = BASE / "data/week3/claims"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load spaCy (already installed from week 1)
print("  Loading spaCy en_core_web_sm …")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")
print("  Loaded.")

HEDGE_PATTERNS = re.compile(
    r"^(i (think|believe|feel|would|am not sure)|it (seems|appears|might|may|could)|"
    r"in my opinion|as far as i know|i('m| am) not|sorry|i don't know|i cannot)",
    re.IGNORECASE
)


def extract_claims(answer: str) -> list[str]:
    """Split answer into atomic sentences; filter hedges and very short fragments."""
    doc    = nlp(answer[:1000])
    claims = []
    for sent in doc.sents:
        text = sent.text.strip()
        if len(text) < 15:
            continue
        if HEDGE_PATTERNS.match(text):
            continue
        if text.endswith("?"):
            continue
        claims.append(text)
    return claims if claims else [answer[:300]]


def process_model(model_name: str) -> None:
    print(f"\n  Extracting claims: {model_name}")
    all_records = []

    for split in ["validation", "main"]:
        files = sorted((BASE / f"data/week2/generations/{model_name}/{split}").glob("q*.json"))
        for f in files:
            d      = json.loads(f.read_text())
            claims = extract_claims(d["generation"])
            all_records.append({
                "file":     f.name,
                "split":    split,
                "model":    model_name,
                "question": d["question"],
                "answer":   d["generation"],
                "claims":   claims,
                "n_claims": len(claims),
            })

    out_file = OUT_DIR / f"{model_name}_claims.jsonl"
    out_file.write_text("\n".join(json.dumps(r) for r in all_records))
    avg = sum(r["n_claims"] for r in all_records) / len(all_records)
    print(f"    {len(all_records)} answers → avg {avg:.1f} claims each → {out_file}")


def main():
    print("=" * 55)
    print("  Week 3 — Claim Extraction (spaCy)")
    print("=" * 55)

    for model_name in ["mistral", "llama2"]:
        process_model(model_name)

    print("\n  Done. Next: week3_judge_agreement.py")


if __name__ == "__main__":
    main()
