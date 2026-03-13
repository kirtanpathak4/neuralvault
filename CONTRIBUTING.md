# Contributing to NeuralVault

Thank you for your interest in contributing. NeuralVault is a personal RAG
knowledge engine — contributions that improve retrieval quality, evaluation
robustness, or local usability are especially welcome.

---

## Before You Start

Read the [NEURALVAULT_REPORT.md](./NEURALVAULT_REPORT.md) to understand the
system architecture. The pipeline has specific design decisions (hybrid search,
cross-encoder reranking, versioned prompts, INSUFFICIENT_EVIDENCE behavior)
that contributions should respect.

---

## Setup

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/neuralvault.git
cd neuralvault
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Get a Groq API key (free)

The CI evaluation pipeline calls Groq to run RAGAS scoring. Groq has a
generous free tier — no credit card required.

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up and create an API key
3. Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 4. Add the key to your fork's CI secrets

The eval must pass in CI before your PR can merge. Add your key to your fork:

```
Your fork on GitHub
  → Settings
  → Secrets and variables
  → Actions
  → New repository secret
  → Name: GROQ_API_KEY
  → Value: your key
```

Without this, the CI eval will fail and your PR cannot be merged.

---

## Making Changes

Work on a branch — never commit directly to `main`:

```bash
git checkout -b feature/your-change
```

Branch naming:
```
feature/   new capability
fix/       bug fix
eval/      changes to the evaluation system
ci/        CI/CD changes
docs/      documentation only
```

---

## Running the Eval Locally

Before opening a PR, run the full evaluation locally to verify your changes
do not regress any RAGAS metric below 0.7:

```bash
# Step 1 — seed the knowledge base
python eval/seed_knowledge.py

# Step 2 — run the full evaluation (takes 45-90 min)
python eval/eval.py --threshold 0.7
```

All four metrics must pass:

| Metric | Threshold |
|--------|-----------|
| Faithfulness | ≥ 0.7 |
| Answer Relevancy | ≥ 0.7 |
| Context Precision | ≥ 0.7 |
| Context Recall | ≥ 0.7 |

Plus refusal detection:
- Insufficient evidence detection rate ≥ 0.7

---

## Opening a PR

- Target branch: `main`
- CI runs automatically and **must pass** before merge
- You will be automatically added as a reviewer (CODEOWNERS)
- The maintainer (@kirtanpathak4) reviews and approves all PRs

---

## What Makes a Good PR

| ✅ Good | ❌ Avoid |
|---------|---------|
| Improves retrieval precision or recall | Breaking the INSUFFICIENT_EVIDENCE behavior |
| Adds eval questions to the golden dataset | Removing or weakening the RAGAS threshold |
| Fixes a real bug with a test case | Adding external cloud dependencies |
| Improves documentation accuracy | General chatbot mode (defeats grounding) |
| Adds a new ingestion format (PDF, URL, etc.) | Committing chroma_db/, data/, or .env |

---

## Questions

Open a GitHub Issue for bugs, feature requests, or questions about the
architecture. For discussion about the RAG design, reference the relevant
section in [NEURALVAULT_REPORT.md](./NEURALVAULT_REPORT.md).
