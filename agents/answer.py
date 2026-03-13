import logging
import sys
from functools import lru_cache
from pathlib import Path
from string import Template

import yaml
from dotenv import load_dotenv
from groq import Groq
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.retrieve import retrieve

load_dotenv()

log = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def _load_prompts_cached(mtime: float) -> dict:
    """Load and cache prompts.yaml. Cache busts when file mtime changes (#22)."""
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def load_prompts() -> dict:
    """Load prompts from config, cached until the file changes on disk."""
    mtime = PROMPTS_PATH.stat().st_mtime
    return _load_prompts_cached(mtime)


def answer(query: str, topic: str | None = None) -> dict:
    """
    Run the full NeuralVault pipeline: retrieve → build prompt → LLM → response.

    Args:
        query: the user's question
        topic: optional topic filter for scoped retrieval

    Returns:
        dict with keys: answer, sources (including chunk text #1),
        prompt_version, insufficient_evidence
    """
    prompts         = load_prompts()
    system_prompt   = prompts["system_prompt"]
    rag_template    = prompts["rag_prompt"]
    no_results      = prompts["no_results_message"]
    insuff_marker   = prompts["insufficient_evidence_marker"]
    prompt_version  = prompts["version"]
    model           = prompts.get("model", "llama-3.3-70b-versatile")
    temperature     = prompts.get("temperature", 0.3)
    max_tokens      = prompts.get("max_tokens", 1024)

    chunks = retrieve(query, topic=topic, n_results=5)

    if not chunks:
        return {
            "answer": no_results,
            "sources": [],
            "prompt_version": prompt_version,
            "insufficient_evidence": False,
        }

    # Build numbered context block from retrieved chunks
    sep = "\n\n---\n\n"
    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[{i}] Topic: {ch['topic']}\n{ch['text']}")
    context = sep.join(parts)

    # Safe substitution via string.Template (#14)
    # Prevents accidental replacement if chunk text contains $-placeholders
    tmpl = Template(rag_template)
    final_prompt = tmpl.safe_substitute(
        context=context,
        query=query,
        topic=topic or "all topics",
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer_text = resp.choices[0].message.content
    except Exception as exc:
        log.error("Groq API error: %s", exc)
        raise RuntimeError(f"LLM generation failed: {exc}") from exc

    is_insuff = answer_text.strip().startswith(insuff_marker)

    return {
        "answer": answer_text,
        # Include chunk text so eval.py can pass real content to RAGAS (#1)
        "sources": [
            {
                "text":       c["text"],
                "source":     c["source"],
                "topic":      c["topic"],
                "similarity": c["similarity"],
            }
            for c in chunks
        ],
        "prompt_version": prompt_version,
        "insufficient_evidence": is_insuff,
    }
