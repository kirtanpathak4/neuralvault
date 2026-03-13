import html
import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.ingest import ingest_pdf, ingest_url, ingest_text
from agents.answer import answer
from core.vectorstore import get_collection

logging.basicConfig(level=logging.INFO, format="%(message)s")

def _safe(text: str) -> str:
    """Escape HTML to prevent XSS from LLM output or metadata (#4)."""
    return html.escape(str(text))

# ───────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────
st.set_page_config(
    page_title="NeuralVault — AI Learning Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────
# CSS — tested design system
# ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
  --bg:       #0a0c12;
  --surf:     #111520;
  --surf2:    #161b28;
  --border:   rgba(255,255,255,0.08);
  --bdr-hi:   rgba(99,179,237,0.45);
  --blue:     #63b3ed;
  --teal:     #4fd1c5;
  --purple:   #b794f4;
  --text:     #e2e8f0;
  --text2:    #94a3b8;
  --text3:    #4a5568;
  --success:  #68d391;
  --warn:     #fbbf24;
  --ff:       'DM Sans', sans-serif;
  --mono:     'DM Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: var(--ff) !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 65% 45% at 10% 0%, rgba(99,179,237,0.055) 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 90% 100%, rgba(183,148,244,0.065) 0%, transparent 60%) !important;
  min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surf) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.5rem;
}
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--blue), var(--purple));
}

/* ── ALL label text — fix dark-on-dark ── */
label, .stTextInput label, .stTextArea label,
[data-testid="stWidgetLabel"] > div > label,
[data-testid="stWidgetLabel"] p,
.stFileUploader label, .stRadio > div > p {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text2) !important;
  margin-bottom: 4px !important;
}

/* ── Radio option text specifically ── */
.stRadio > div > label > div:last-child,
.stRadio > div > label > div > p,
.stRadio label span,
.stRadio label div[data-testid] {
  color: var(--text2) !important;
  font-family: var(--ff) !important;
  font-size: 13px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}

/* ── Inputs — dark bg, light text, visible in all states ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: #1a2035 !important;
  background-color: #1a2035 !important;
  border: 1px solid rgba(99,179,237,0.2) !important;
  border-radius: 8px !important;
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
  font-family: var(--mono) !important;
  font-size: 13px !important;
  padding: 9px 12px !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  caret-color: var(--blue);
}
.stTextInput > div > div > input:focus,
.stTextInput > div > div > input:active,
.stTextInput > div > div > input:hover,
.stTextInput > div > div > input:not(:placeholder-shown),
.stTextArea > div > div > textarea:focus,
.stTextArea > div > div > textarea:active,
.stTextArea > div > div > textarea:hover,
.stTextArea > div > div > textarea:not(:placeholder-shown) {
  background: #1a2035 !important;
  background-color: #1a2035 !important;
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
}
.stTextInput > div > div > input:-webkit-autofill,
.stTextInput > div > div > input:-webkit-autofill:hover,
.stTextInput > div > div > input:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0px 1000px #1a2035 inset !important;
  -webkit-text-fill-color: #e2e8f0 !important;
  caret-color: #e2e8f0;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
  color: rgba(148,163,184,0.45) !important;
  -webkit-text-fill-color: rgba(148,163,184,0.45) !important;
  font-style: italic;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--bdr-hi) !important;
  box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
  outline: none !important;
}

/* ── Radio ── */
.stRadio > div {
  display: flex !important;
  flex-direction: column !important;
  gap: 6px !important;
}
.stRadio > div > label {
  display: flex !important;
  align-items: center !important;
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 9px 13px !important;
  font-family: var(--ff) !important;
  font-size: 13px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  color: var(--text2) !important;
  cursor: pointer;
  transition: all 0.18s !important;
  font-weight: 400 !important;
}
.stRadio > div > label:hover {
  border-color: var(--bdr-hi) !important;
  background: rgba(99,179,237,0.05) !important;
  color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: var(--ff) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.02em !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.04) !important;
  color: var(--text) !important;
  transition: all 0.18s ease !important;
  padding: 8px 16px !important;
}
.stButton > button:hover {
  border-color: var(--bdr-hi) !important;
  background: rgba(99,179,237,0.07) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 18px rgba(99,179,237,0.12) !important;
  color: var(--text) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, rgba(99,179,237,0.16), rgba(183,148,244,0.12)) !important;
  border-color: rgba(99,179,237,0.45) !important;
  color: var(--blue) !important;
  font-size: 15px !important;
  padding: 11px 28px !important;
  letter-spacing: 0.03em !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, rgba(99,179,237,0.26), rgba(183,148,244,0.2)) !important;
  box-shadow: 0 0 32px rgba(99,179,237,0.22) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
  background: rgba(99,179,237,0.02) !important;
  border: 1.5px dashed rgba(99,179,237,0.25) !important;
  border-radius: 10px !important;
  transition: all 0.2s !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: rgba(99,179,237,0.5) !important;
  background: rgba(99,179,237,0.04) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
  color: var(--text2) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
}
.stSuccess [data-testid="stAlert"] {
  background: rgba(104,211,145,0.08) !important;
  border: 1px solid rgba(104,211,145,0.25) !important;
}
.stSuccess p, .stSuccess span { color: #9ae6b4 !important; }
.stWarning [data-testid="stAlert"] {
  background: rgba(251,191,36,0.08) !important;
  border: 1px solid rgba(251,191,36,0.25) !important;
}
.stWarning p, .stWarning span { color: #fde68a !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.38); }

/* ══ Custom HTML Components ══ */

.nv-tag {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--blue); background: rgba(99,179,237,0.08);
  border: 1px solid rgba(99,179,237,0.22); border-radius: 100px;
  padding: 4px 13px; margin-bottom: 12px;
}
.nv-tag::before { content:''; width:5px; height:5px; background:var(--blue); border-radius:50%; }

.nv-h1 {
  font-size: clamp(2rem,4vw,3.2rem); font-weight: 700; line-height: 1.1;
  background: linear-gradient(128deg, #fff 15%, var(--blue) 55%, var(--purple) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin: 0 0 8px;
}
.nv-sub {
  font-family: var(--mono); font-size: 12px; color: var(--text2); letter-spacing: 0.03em;
}
.nv-hero {
  padding: 32px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
}

.sec-lbl {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--blue);
  margin-bottom: 14px;
}
.sec-lbl::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(99,179,237,0.22), transparent);
}

.sb-lbl {
  font-family: var(--mono); font-size: 9.5px; font-weight: 500;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--text2); display: block; margin-bottom: 4px;
}
.sb-desc {
  font-family: var(--ff) !important; font-size: 12px !important;
  color: var(--text2) !important; line-height: 1.6; margin-bottom: 14px;
  text-transform: none !important; letter-spacing: 0 !important;
  font-weight: 400 !important;
}

.metric-card {
  background: linear-gradient(135deg, rgba(99,179,237,0.07), rgba(183,148,244,0.07));
  border: 1px solid rgba(99,179,237,0.2); border-radius: 10px;
  padding: 14px 18px; text-align: center; margin-top: 10px;
}
.metric-num {
  font-size: 30px; font-weight: 700;
  background: linear-gradient(135deg, var(--blue), var(--purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  line-height: 1;
}
.metric-lbl {
  font-family: var(--mono); font-size: 10px; color: var(--text2);
  letter-spacing: 0.1em; text-transform: uppercase; margin-top: 5px;
}
.metric-sub {
  font-family: var(--mono); font-size: 10px; color: var(--text3);
  text-align: center; margin-top: 6px;
}

/* ── Answer card — normal response ── */
.ans-card {
  background: linear-gradient(135deg, rgba(99,179,237,0.04), rgba(183,148,244,0.04));
  border: 1px solid rgba(99,179,237,0.2); border-radius: 12px;
  padding: 22px 24px 22px 28px;
  position: relative; overflow: hidden;
  animation: riseIn 0.4s ease both;
}
.ans-card::before {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--blue), var(--purple));
}
/* ── NEW: meta row — label left, version badge right ── */
.ans-meta {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px;
}
.ans-meta-lbl {
  font-family: var(--mono); font-size: 9px; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--text3);
}
.ans-version {
  font-family: var(--mono); font-size: 9px;
  color: rgba(99,179,237,0.5);
  background: rgba(99,179,237,0.07);
  border: 1px solid rgba(99,179,237,0.15);
  border-radius: 100px; padding: 2px 10px;
}
.ans-body {
  font-size: 14.5px; line-height: 1.78; color: #dde4f5;
}

/* ── NEW: Insufficient evidence card — amber ── */
.ins-card {
  background: rgba(251,191,36,0.04);
  border: 1px solid rgba(251,191,36,0.22);
  border-radius: 12px;
  padding: 22px 24px 22px 28px;
  position: relative; overflow: hidden;
  animation: riseIn 0.4s ease both;
}
.ins-card::before {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, #fbbf24, #f59e0b);
}
.ins-header {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 9px; letter-spacing: 0.2em;
  text-transform: uppercase; color: #fbbf24;
  margin-bottom: 12px;
}
.ins-body {
  font-size: 14px; line-height: 1.75; color: #fde68a;
}
.ins-hint {
  margin-top: 14px; font-family: var(--mono); font-size: 11px;
  color: var(--text3); line-height: 1.6;
}

@keyframes riseIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

.source-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.s-pill {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  border-radius: 100px; padding: 5px 13px;
  font-family: var(--mono); font-size: 11px; color: var(--text2);
  transition: all 0.18s;
}
.s-pill:hover { border-color: var(--bdr-hi); color: var(--text); }
.s-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--teal); flex-shrink: 0; }
.s-pill strong { color: #c7d5f0; font-weight: 500; }
.s-pill .sim-val { color: var(--blue); }

/* Pipeline */
.pipeline {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}
.p-row {
  display: flex;
  align-items: stretch;
  border-bottom: 1px solid rgba(255,255,255,0.055);
  transition: background 0.18s;
}
.p-row:last-child { border-bottom: none; }
.p-row:hover { background: rgba(99,179,237,0.025); }

.p-track {
  width: 54px; flex-shrink: 0;
  display: flex; flex-direction: column; align-items: center;
  padding: 16px 0 0;
}
.p-ico {
  width: 34px; height: 34px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  flex-shrink: 0; border: 1px solid; letter-spacing: 0.02em;
}
.p-wire {
  flex: 1; width: 2px; min-height: 12px; margin-top: 4px;
  background: linear-gradient(180deg, rgba(99,179,237,0.28), rgba(183,148,244,0.1));
}
.p-body { flex: 1; padding: 14px 20px 14px 2px; }
.p-num {
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 3px;
}
.p-name { font-size: 14px; font-weight: 600; color: #dde4f5; margin-bottom: 5px; }
.p-desc { font-family: var(--mono); font-size: 11.5px; color: var(--text2); line-height: 1.55; }
.p-chip {
  display: inline-block; font-family: var(--mono); font-size: 9.5px;
  font-weight: 500; letter-spacing: 0.06em;
  padding: 3px 10px; border-radius: 100px; border: 1px solid; margin-top: 8px;
}

.ico-b { background: rgba(99,179,237,0.12);  border-color: rgba(99,179,237,0.35);  color: var(--blue);   }
.ico-v { background: rgba(183,148,244,0.12); border-color: rgba(183,148,244,0.35); color: var(--purple); }
.ico-t { background: rgba(79,209,197,0.12);  border-color: rgba(79,209,197,0.35);  color: var(--teal);   }
.ico-o { background: rgba(251,191,36,0.12);  border-color: rgba(251,191,36,0.35);  color: #fbbf24;       }

.chip-b { color: var(--blue);   border-color: rgba(99,179,237,0.25);  background: rgba(99,179,237,0.09);  }
.chip-v { color: var(--purple); border-color: rgba(183,148,244,0.25); background: rgba(183,148,244,0.09); }
.chip-t { color: var(--teal);   border-color: rgba(79,209,197,0.25);  background: rgba(79,209,197,0.09);  }
.chip-o { color: #fbbf24;       border-color: rgba(251,191,36,0.25);  background: rgba(251,191,36,0.09);  }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <span class="sb-lbl">📥 Feed Your Brain</span>
    <p class="sb-desc">Tag ingested content by topic<br>for targeted recall later.</p>
    """, unsafe_allow_html=True)

    topic = st.text_input("Topic Label", placeholder="RAG · Transformers · LLM Agents")
    ingest_type = st.radio("Source Type", ["📄 PDF", "🌐 URL", "📝 Raw Text"])
    st.write("")

    if ingest_type == "📄 PDF":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if st.button("⬆  Ingest PDF", use_container_width=True):
            if uploaded_file and topic:
                # Sanitize filename (#5) and use tempfile for auto-cleanup (#21)
                safe_name = Path(uploaded_file.name).name
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf", prefix=f"{safe_name}_"
                )
                try:
                    tmp.write(uploaded_file.getbuffer())
                    tmp.close()
                    with st.spinner("Vectorizing…"):
                        ingest_pdf(tmp.name, topic)
                    st.success(f"✓ Stored under **{topic}**")
                except Exception as exc:
                    st.error(f"PDF ingestion failed: {exc}")
                finally:
                    os.unlink(tmp.name)
            else:
                st.warning("Add a topic label and select a file.")

    elif ingest_type == "🌐 URL":
        url = st.text_input("URL", placeholder="https://…")
        if st.button("⬆  Ingest URL", use_container_width=True):
            if url and topic:
                try:
                    with st.spinner("Fetching & vectorizing…"):
                        ingest_url(url, topic)
                    st.success(f"✓ Stored under **{topic}**")
                except Exception as exc:
                    st.error(f"URL ingestion failed: {exc}")
            else:
                st.warning("Add a topic label and a URL.")

    elif ingest_type == "📝 Raw Text":
        raw_text = st.text_area("Notes", placeholder="Paste notes, summaries, excerpts…", height=160)
        if st.button("⬆  Ingest Notes", use_container_width=True):
            if raw_text and topic:
                try:
                    with st.spinner("Storing…"):
                        ingest_text(raw_text, topic)
                    st.success(f"✓ Stored under **{topic}**")
                except Exception as exc:
                    st.error(f"Text ingestion failed: {exc}")
            else:
                st.warning("Add a topic label and some text.")

    st.divider()
    st.markdown('<span class="sb-lbl">📊 Memory Stats</span>', unsafe_allow_html=True)
    if st.button("↻  Refresh Stats", use_container_width=True):
        try:
            collection = get_collection()
            count = collection.count()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-num">{count}</div>
                <div class="metric-lbl">Chunks Stored</div>
            </div>
            <p class="metric-sub">≈ {count * 500:,} characters indexed</p>
            """, unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Could not load stats: {exc}")


# ───────────────────────────────────────────
# HERO
# ───────────────────────────────────────────
st.markdown("""
<div class="nv-hero">
    <div class="nv-tag">Neural Vault · v1.0</div>
    <div class="nv-h1">Your AI Learning Memory</div>
    <p class="nv-sub">A personal second brain — ingest your study material and ask questions grounded entirely in your own notes</p>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────
# QUERY
# ───────────────────────────────────────────
st.markdown('<div class="sec-lbl">💬 Query Your Knowledge</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1], gap="medium")
with col1:
    query = st.text_input("Ask anything", placeholder="What is the difference between RAG and fine-tuning?")
with col2:
    filter_topic = st.text_input("Filter by topic", placeholder="e.g. RAG")

st.write("")
run = st.button("⟶  Search Memory", type="primary")

if run:
    if not query:
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Searching your notes and generating answer…"):
                result = answer(query, topic=filter_topic if filter_topic else None)
            st.session_state["last_result"] = result
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            result = None

# Persist answer across reruns (#24) — e.g. when pipeline toggle triggers rerun
result = st.session_state.get("last_result")

if result:
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    if result.get("insufficient_evidence"):
        # Insufficient evidence — amber card
        insuff_body = _safe(
            result["answer"].replace("INSUFFICIENT_EVIDENCE:", "").strip()
        ).replace("\n", "<br>")
        st.markdown('<div class="sec-lbl">⚠ Insufficient Evidence</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ins-card">
            <div class="ins-header">⚠ &nbsp; Not Enough Evidence in Your Notes</div>
            <div class="ins-body">{insuff_body}</div>
            <div class="ins-hint">
                Try ingesting more content on this topic, then ask again.<br>
                Prompt version used: <strong>v{_safe(result.get("prompt_version", "?"))}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Normal answer card — XSS-safe (#4)
        ans_html = _safe(result["answer"]).replace("\n", "<br>")
        st.markdown('<div class="sec-lbl">🤖 Answer</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ans-card">
            <div class="ans-meta">
                <span class="ans-meta-lbl">Neural Response</span>
                <span class="ans-version">prompt v{_safe(result.get("prompt_version", "?"))}</span>
            </div>
            <div class="ans-body">{ans_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sources — always shown regardless of answer type
    if result.get("sources"):
        st.markdown("<div style='margin-top:22px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📚 Sources Used</div>', unsafe_allow_html=True)
        pills_html = ""
        for i, s in enumerate(result["sources"], start=1):
            try:
                sim = f"{float(s['similarity']):.2f}"
            except Exception:
                sim = "—"
            src = _safe(s.get("source", "—"))
            if len(src) > 40:
                src = src[:40] + "..."
            topic_lbl = _safe(s.get("topic", "—"))
            pills_html += (
                "<span class='s-pill'>"
                "<span class='dot'></span>"
                "<strong>[" + str(i) + "] " + topic_lbl + "</strong>"
                " &nbsp;·&nbsp; " + src + " &nbsp;·&nbsp; "
                "<span class='sim-val'>" + sim + "</span>"
                "</span>"
            )
        st.markdown("<div class='source-pills'>" + pills_html + "</div>", unsafe_allow_html=True)
# ───────────────────────────────────────────
# PIPELINE CHAIN
# ───────────────────────────────────────────
st.markdown("<div style='margin-top:48px'></div>", unsafe_allow_html=True)

PIPELINE = [
    ("01", "ico-b", "01", "Ingest",
     "Feed NeuralVault a PDF, URL, or raw text and assign a topic tag for scoped recall.",
     "chip-b", "PDF &middot; URL &middot; Notes"),
    ("02", "ico-v", "02", "Chunk",
     "Content is split into ~500-character overlapping segments for fine-grained retrieval.",
     "chip-v", "~500 char windows"),
    ("03", "ico-t", "03", "Embed",
     "Each chunk is encoded into a 384-dimensional semantic vector by a local transformer.",
     "chip-t", "all-MiniLM-L6-v2"),
    ("04", "ico-b", "04", "Store",
     "Vectors and metadata are persisted locally in ChromaDB &mdash; private, no cloud required.",
     "chip-b", "ChromaDB &middot; local"),
    ("05", "ico-o", "05", "Retrieve",
     "Your question triggers hybrid search (vector + BM25), then a cross-encoder reranks candidates and selects the top 5.",
     "chip-o", "Top-5 cosine sim"),
    ("06", "ico-v", "06", "Generate",
     "Retrieved chunks + your question are sent to Groq with a strict grounding prompt from prompts.yaml.",
     "chip-v", "Groq &middot; Llama 3.3 70B"),
    ("07", "ico-t", "07", "Grounded Answer",
     "Groq answers using ONLY your notes. If evidence is insufficient, it explicitly refuses rather than hallucinating.",
     "chip-t", "Citation enforced"),
]

def build_pipeline_html(steps):
    last_i = len(steps) - 1
    rows = []
    for i, (ico, ico_cls, num, name, desc, chip_cls, chip_lbl) in enumerate(steps):
        wire = '<div class="p-wire"></div>' if i < last_i else ""
        row = (
            '<div class="p-row">'
            '<div class="p-track">'
            '<div class="p-ico ' + ico_cls + '">' + ico + '</div>'
            + wire +
            '</div>'
            '<div class="p-body">'
            '<div class="p-num">Step ' + num + '</div>'
            '<div class="p-name">' + name + '</div>'
            '<div class="p-desc">' + desc + '</div>'
            '<span class="p-chip ' + chip_cls + '">' + chip_lbl + '</span>'
            '</div>'
            '</div>'
        )
        rows.append(row)
    return '<div class="pipeline">' + "".join(rows) + '</div>'

if "pipeline_open" not in st.session_state:
    st.session_state.pipeline_open = False

arrow = "▼" if st.session_state.pipeline_open else "▶"
toggle_label = f"{arrow}  ⚙️  How NeuralVault Works"

if st.button(toggle_label, key="pipeline_toggle"):
    st.session_state.pipeline_open = not st.session_state.pipeline_open
    st.rerun()

if st.session_state.pipeline_open:
    st.markdown(build_pipeline_html(PIPELINE), unsafe_allow_html=True)

st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)