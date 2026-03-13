"""
NeuralVault — RAGAS Evaluation Script
======================================
Runs a two-tier evaluation on the golden dataset:

TIER 1 — RAGAS metrics (factual + cross_chunk questions):
  - Faithfulness:       are answer claims supported by retrieved chunks?
  - Answer Relevancy:   does the answer address the question?
  - Context Precision:  are retrieved chunks relevant to the question?
  - Context Recall:     did we retrieve all chunks needed to answer?

TIER 2 — Refusal detection (insufficient + adversarial questions):
  - Does the system return INSUFFICIENT_EVIDENCE when it should?
  - Scored as simple pass/fail per question, no RAGAS needed.

RAGAS LLM: Groq / llama-3.1-8b-instant (free, separate 500K TPD quota)
           8B is sufficient for RAGAS judgment tasks (yes/no scoring)
           Generation still uses Llama 3.3 70B — users never see 8B output

Uses OpenAI client pointed at Groq's OpenAI-compatible endpoint, because
RAGAS 0.4+ requires InstructorLLM via llm_factory() — the old
LangchainLLMWrapper is no longer supported for collection metrics.

Usage:
  python eval/eval.py                    # run all questions
  python eval/eval.py --threshold 0.7    # fail if any metric < 0.7

Exit codes:
  0 = all RAGAS metrics above threshold AND refusal detection above threshold
  1 = any check fails (CI blocks the PR)
"""

import sys
import json
import os
import argparse
from pathlib import Path

import yaml

# ── Windows encoding fix ──────────────────────────────────────────────────
# Windows cmd/PowerShell defaults to cp1252 which can't print emoji (🔧 etc.)
# Reconfigure stdout to UTF-8 with replacement for unsupported chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Add project root to path ───────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.answer import answer
from datasets import Dataset
from ragas import evaluate, RunConfig

# ── RAGAS metrics: singleton instances pinned to ragas==0.4.3 ─────────────
# ragas.metrics.collections (BaseMetric) does NOT work with evaluate() —
# evaluate() expects ragas.metrics.base.Metric instances.
# The singleton instances (_faithfulness, etc.) are Metric subclasses and
# work correctly with evaluate(). LLM/embeddings are auto-injected by
# evaluate() when passed as parameters.
# WARNING: These private imports (_faithfulness, etc.) are version-specific.
# If upgrading ragas beyond 0.4.3, verify the singleton names still exist.
from ragas.metrics import (
    _faithfulness as faithfulness_metric,
    _answer_relevancy as answer_relevancy_metric,
    _context_precision as context_precision_metric,
    _context_recall as context_recall_metric,
)

# ── Modern RAGAS 0.4+ API: llm_factory + compatible embeddings ───────────
# LangchainLLMWrapper / LangchainEmbeddingsWrapper are deprecated and no
# longer work with collection metrics. Use llm_factory() with an OpenAI
# client pointed at Groq's endpoint.
#
# RAGAS 0.4.3 embedding bug:
#   - HuggingFaceEmbeddings (capital F, modern) has embed_text() but NO embed_query()
#   - HuggingfaceEmbeddings (lowercase f, legacy) is abstract — can't instantiate
#   - The singleton _answer_relevancy metric calls embed_query() / embed_documents()
#   Fix: subclass the modern HuggingFaceEmbeddings and add the two missing methods
#   that delegate to embed_text() / embed_texts().
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as _RagasHFEmbeddings
from openai import OpenAI


class CompatHuggingFaceEmbeddings(_RagasHFEmbeddings):
    """Adds embed_query/embed_documents for backward compat with singleton metrics."""
    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)


# ── RunConfig: sequential to avoid Groq free tier timeouts ────────────────
#
# RAGAS by default runs parallel LLM calls per metric.
# Groq free tier can't handle that — we get TimeoutError floods.
# max_workers=1 forces sequential evaluation: slower but always completes.
# ──────────────────────────────────────────────────────────────────────────
RAGAS_RUN_CONFIG = RunConfig(
    max_workers=1,      # sequential — no parallel Groq calls
    timeout=120,        # 2 min per call
    max_retries=3,      # retry on transient failures
)


# ── Load INSUFFICIENT_EVIDENCE marker from prompts.yaml ───────────────────
# Single source of truth — if the marker changes in prompts.yaml, eval
# picks it up automatically without code changes.
def _load_marker() -> str:
    prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("insufficient_evidence_marker", "INSUFFICIENT_EVIDENCE")

MARKER = _load_marker()


def build_ragas_llm():
    """
    Groq / llama-3.1-8b-instant as the RAGAS judge LLM.

    Why 8B instead of 70B:
      - RAGAS only asks simple judgment questions — doesn't need a large model
      - 8B has a separate 500K token/day quota vs 70B's 100K
      - Much less likely to hit rate limits during eval

    RAGAS 0.4+ requires InstructorLLM created via llm_factory().
    The old LangchainLLMWrapper no longer works with collection metrics
    (Faithfulness, AnswerRelevancy, etc.).

    Groq is OpenAI-compatible, so we use an OpenAI client pointed at
    Groq's base URL. This avoids RAGAS's buggy Groq client handling
    (it tries client.messages.create which doesn't exist on the Groq SDK).
    """
    groq_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    return llm_factory(
        "llama-3.1-8b-instant",
        provider="openai",
        client=groq_client,
    )


def build_ragas_embedder():
    """
    CompatHuggingFaceEmbeddings for answer relevancy scoring.
    Already cached on your machine from NeuralVault — no extra download.
    RAGAS uses this to compute semantic similarity between question and answer.

    Subclasses the modern HuggingFaceEmbeddings and adds embed_query() /
    embed_documents() methods that the singleton _answer_relevancy metric
    requires (it calls these internally, but the modern base class only
    provides embed_text / embed_texts).
    """
    return CompatHuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


# ── Load golden dataset ────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


# ── Run your RAG system on each question ──────────────────────────────────
def run_pipeline_on_dataset(golden: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    For each Q&A pair in the golden dataset:
    1. Call answer() — full hybrid retrieval + reranker + LLM pipeline
    2. Route by question type:
       - factual / cross_chunk → strip marker, collect for RAGAS scoring
       - insufficient / adversarial → check if marker present, collect for pass/fail

    Returns:
      ragas_rows   — list of dicts for RAGAS (question, answer, contexts, ground_truth)
      refusal_rows — list of dicts for refusal detection (question, answer, type, detected)
    """
    ragas_rows = []
    refusal_rows = []
    total = len(golden)

    for i, item in enumerate(golden):
        question = item["question"]
        ground_truth = item["ground_truth"]
        topic = item.get("topic")
        q_type = item.get("type", "factual")  # backward compat: default to factual

        print(f"\n[{i+1}/{total}] ({q_type}) {question[:65]}...")

        result = answer(question, topic=topic)

        # Pull raw chunk texts — RAGAS needs strings, not metadata dicts
        # answer.py now always includes "text" in sources (fix #1)
        contexts = [
            s["text"]
            for s in result.get("sources", [])
            if s.get("text")
        ]
        if not contexts:
            contexts = ["No context retrieved"]

        generated_answer = result["answer"]

        # ── Route by question type ────────────────────────────────────
        if q_type in ("insufficient", "adversarial"):
            # Check if the system correctly refused to answer
            detected = generated_answer.strip().startswith(MARKER)
            refusal_rows.append({
                "question": question,
                "answer": generated_answer,
                "type": q_type,
                "detected": detected,
            })
            status_icon = "✅" if detected else "❌"
            print(f"  {status_icon} Refusal detected: {detected}")
            print(f"  ✓ Answer:   {generated_answer[:80]}{'...' if len(generated_answer) > 80 else ''}")

        else:
            # factual or cross_chunk → send through RAGAS
            # Strip INSUFFICIENT_EVIDENCE prefix (with or without colon/space)
            # so RAGAS scores the actual content, not the marker
            stripped = generated_answer.strip()
            if stripped.startswith(MARKER):
                generated_answer = stripped[len(MARKER):].lstrip(": ").strip()

            ragas_rows.append({
                "question": question,
                "answer": generated_answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })
            print(f"  ✓ Contexts: {len(contexts)} chunks")
            print(f"  ✓ Answer:   {generated_answer[:80]}{'...' if len(generated_answer) > 80 else ''}")

    return ragas_rows, refusal_rows


# ── Run RAGAS scoring ──────────────────────────────────────────────────────
def run_ragas_eval(rows: list[dict], ragas_llm, ragas_embedder):
    """
    Pass singleton Metric instances to evaluate() with LLM/embeddings.

    evaluate() auto-injects ragas_llm into MetricWithLLM instances and
    ragas_embedder into MetricWithEmbeddings instances when their .llm
    or .embeddings is None. This avoids the BaseMetric vs Metric type
    mismatch that occurs with ragas.metrics.collections.

    Faithfulness:      needs LLM only (judges claim support)
    AnswerRelevancy:   needs LLM + embedder (semantic similarity scoring)
    ContextPrecision:  needs LLM only (judges chunk relevance)
    ContextRecall:     needs LLM only (judges completeness of retrieval)
    """
    num_questions = len(rows)
    dataset = Dataset.from_list(rows)

    print(f"\n\n🧪 Running RAGAS evaluation with Groq / llama-3.1-8b-instant...")
    print("  Metrics: faithfulness, answer_relevancy, context_precision, context_recall")
    print("  Sequential mode (max_workers=1) — slower but no Groq timeouts")

    # Estimate: ~30s per question per metric × 4 metrics
    est_min = max(1, (num_questions * 4 * 30) // 60)
    est_max = est_min * 2
    print(f"  Estimated time: {est_min}-{est_max} minutes for {num_questions} questions...\n")

    metrics = [
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
        context_recall_metric,
    ]

    results = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embedder,
        run_config=RAGAS_RUN_CONFIG,
    )

    return results


# ── Print results table ────────────────────────────────────────────────────
def print_results(ragas_results, refusal_rows: list[dict], threshold: float):
    """
    Two-section report:
      1. RAGAS metrics table (factual + cross_chunk questions)
      2. Refusal detection table (insufficient + adversarial questions)

    Returns:
      ragas_pass    — bool: all RAGAS metrics >= threshold
      scores        — dict: {metric_name: float}
      refusal_rate  — float: fraction of refusals correctly detected
      refusal_pass  — bool: refusal_rate >= threshold
    """
    # ── Section 1: RAGAS Metrics ──────────────────────────────────────
    ragas_pass = True
    scores = {}

    print("\n" + "=" * 60)
    print("  NEURALVAULT — RAGAS EVALUATION RESULTS")
    print("=" * 60)

    if ragas_results is not None:
        metric_labels = {
            "faithfulness":      "Faithfulness       (no hallucination)",
            "answer_relevancy":  "Answer Relevancy   (answers the question)",
            "context_precision": "Context Precision  (retrieved chunks relevant)",
            "context_recall":    "Context Recall     (all needed chunks found)",
        }

        # RAGAS 0.4.x returns an EvaluationResult object.
        # .to_pandas() gives a DataFrame — one row per question, one col per metric.
        # We take the mean across all questions for each metric.
        df = ragas_results.to_pandas()

        for key, label in metric_labels.items():
            if key in df.columns:
                raw = df[key].mean()
                # NaN means all jobs failed (rate limit / bad request errors).
                # nan != nan is True in Python — safest NaN check without importing math.
                # Treat as 0.0 so we fail the build cleanly instead of crashing.
                score = 0.0 if (raw != raw) else float(raw)
            else:
                score = 0.0

            scores[key] = score
            status = "✅ PASS" if score >= threshold else "❌ FAIL"
            if score < threshold:
                ragas_pass = False

            filled = int(score * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {status}  {label}")
            print(f"          [{bar}] {score:.3f}")
            print()

        print(f"  Threshold:  {threshold:.2f}")
        print(f"  Questions:  {len(df)}  (factual + cross_chunk)")

        if ragas_pass:
            print("  🎉 ALL RAGAS METRICS PASSED")
        else:
            print("  🚨 ONE OR MORE RAGAS METRICS BELOW THRESHOLD")
    else:
        print("  (no factual/cross_chunk questions — RAGAS skipped)")

    print("=" * 60)

    # ── Section 2: Refusal Detection ──────────────────────────────────
    #
    # Two sub-categories with different gate behavior:
    #
    # INSUFFICIENT (gates the build):
    #   Topics completely absent from seed notes (Docker, React, SQL, etc.)
    #   The system MUST return INSUFFICIENT_EVIDENCE — any answer is wrong.
    #
    # ADVERSARIAL (informational only, does NOT gate):
    #   Questions with false premises about NeuralVault (GPT-4, cloud DB, etc.)
    #   The LLM often correctly REFUTES the premise using the notes, e.g.:
    #     "NeuralVault does NOT use GPT-4, it uses Llama 3.3 70B via Groq"
    #   This is CORRECT behavior — the notes DO contain enough info to respond.
    #   Refusing with INSUFFICIENT_EVIDENCE would actually be wrong here.
    #   We report adversarial detection rates for visibility but don't gate on them.
    #
    print("\n" + "=" * 60)
    print("  NEURALVAULT — INSUFFICIENT EVIDENCE DETECTION")
    print("=" * 60)

    refusal_rate = 0.0
    refusal_pass = True

    if refusal_rows:
        # Count by type
        insufficient = [r for r in refusal_rows if r["type"] == "insufficient"]
        adversarial = [r for r in refusal_rows if r["type"] == "adversarial"]

        insuf_correct = sum(1 for r in insufficient if r["detected"])
        adv_correct = sum(1 for r in adversarial if r["detected"])

        # Gate ONLY on insufficient — adversarial is informational
        insuf_rate = insuf_correct / len(insufficient) if insufficient else 1.0
        refusal_rate = insuf_rate
        refusal_pass = insuf_rate >= threshold

        # Print per-type breakdown
        if insufficient:
            insuf_pct = insuf_rate * 100
            insuf_status = "✅ PASS" if refusal_pass else "❌ FAIL"
            print(f"  insufficient:  {insuf_correct}/{len(insufficient)}  ({insuf_pct:.1f}%)  {insuf_status}  [gates build]")
        if adversarial:
            adv_pct = (adv_correct / len(adversarial)) * 100
            print(f"  adversarial:   {adv_correct}/{len(adversarial)}  ({adv_pct:.1f}%)       [informational]")

        # Print any missed insufficient refusals for debugging
        missed_insuf = [r for r in insufficient if not r["detected"]]
        if missed_insuf:
            print(f"\n  ⚠️  Missed insufficient refusals ({len(missed_insuf)}):")
            for m in missed_insuf:
                print(f"    {m['question'][:60]}...")
                print(f"           Got: {m['answer'][:70]}...")

        # Print adversarial corrections (not failures — just informational)
        adv_corrections = [r for r in adversarial if not r["detected"]]
        if adv_corrections:
            print(f"\n  ℹ️  Adversarial corrections ({len(adv_corrections)}) — LLM refuted false premises:")
            for m in adv_corrections:
                print(f"    {m['question'][:60]}...")
                print(f"           Got: {m['answer'][:70]}...")
    else:
        print("  (no insufficient/adversarial questions — detection skipped)")

    print("=" * 60 + "\n")

    return ragas_pass, scores, refusal_rate, refusal_pass


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NeuralVault RAGAS Evaluation")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum acceptable score for all metrics (default: 0.7). "
             "CI build fails if any metric is below this."
    )
    args = parser.parse_args()

    print("🔧 Initializing RAGAS with Groq / llama-3.1-8b-instant...")
    ragas_llm = build_ragas_llm()
    ragas_embedder = build_ragas_embedder()
    print("  ✓ LLM:      Groq / llama-3.1-8b-instant (500K TPD, free tier)")
    print("  ✓ Embedder: sentence-transformers/all-MiniLM-L6-v2 (local)")
    print("  ✓ Mode:     sequential (max_workers=1, no Groq timeouts)\n")

    print("🔍 Loading golden dataset...")
    golden = load_golden_dataset()

    # Count question types for display
    type_counts = {}
    for item in golden:
        q_type = item.get("type", "factual")
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

    print(f"  ✓ Loaded {len(golden)} question-answer pairs")
    for t, c in sorted(type_counts.items()):
        print(f"    - {t}: {c}")
    print()

    print("🤖 Running NeuralVault pipeline on each question...")
    ragas_rows, refusal_rows = run_pipeline_on_dataset(golden)

    # ── Run RAGAS only on factual + cross_chunk questions ─────────────
    ragas_results = None
    if ragas_rows:
        ragas_results = run_ragas_eval(ragas_rows, ragas_llm, ragas_embedder)

    # ── Print combined results ────────────────────────────────────────
    ragas_pass, scores, refusal_rate, refusal_pass = print_results(
        ragas_results, refusal_rows, threshold=args.threshold
    )

    # ── Combined pass: both tiers must pass ───────────────────────────
    overall_pass = ragas_pass and refusal_pass

    print("=" * 60)
    print("  OVERALL RESULT")
    print("=" * 60)
    print(f"  RAGAS metrics:       {'✅ PASS' if ragas_pass else '❌ FAIL'}")
    print(f"  Refusal detection:   {'✅ PASS' if refusal_pass else '❌ FAIL'}")
    print(f"  ────────────────────────────────")
    if overall_pass:
        print(f"  🎉 BUILD PASSED — all checks green")
    else:
        print(f"  🚨 BUILD FAILED — one or more checks below threshold")
    print("=" * 60 + "\n")

    # ── Save results JSON — uploaded as CI artifact ───────────────────
    output_path = Path(__file__).parent / "eval_results.json"

    # Calculate adversarial rate for reporting (informational only)
    adv_rows = [r for r in refusal_rows if r["type"] == "adversarial"]
    adv_rate = sum(1 for r in adv_rows if r["detected"]) / len(adv_rows) if adv_rows else 0.0

    with open(output_path, "w") as f:
        json.dump({
            "threshold": args.threshold,
            "passed": overall_pass,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "insufficient_detection_rate": round(refusal_rate, 4),
            "insufficient_passed": refusal_pass,
            "adversarial_correction_rate": round(1.0 - adv_rate, 4),
            "question_types": type_counts,
            "num_questions": len(golden),
            "num_ragas_questions": len(ragas_rows),
            "num_refusal_questions": len(refusal_rows),
            "ragas_llm": "groq/llama-3.1-8b-instant",
            "ragas_embedder": "sentence-transformers/all-MiniLM-L6-v2",
        }, f, indent=2)

    print(f"📄 Results saved to {output_path}")
    print("    → Upload this as a CI artifact to track scores over time\n")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
