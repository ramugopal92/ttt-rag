import os
import re
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"

# Confidence gating (same idea as Colab)
MIN_SCORE = 0.33

ABOUT_PAGE_URL = "https://thetechthinker.com/about-the-tech-thinker/"
CONTACT_PAGE_URL = "https://thetechthinker.com/contact-the-tech-thinker/"


def _get_secret(name: str) -> str:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.environ.get(name, "")


@st.cache_resource
def _clients():
    openai_key = _get_secret("OPENAI_API_KEY").strip()
    pinecone_key = _get_secret("PINECONE_API_KEY").strip()
    index_name = _get_secret("PINECONE_INDEX").strip()
    host = _get_secret("PINECONE_HOST").strip()
    namespace = _get_secret("PINECONE_NAMESPACE").strip()

    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets.")
    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY in Streamlit Secrets.")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX in Streamlit Secrets (example: techthinker-rag).")
    if not host:
        raise RuntimeError("Missing PINECONE_HOST in Streamlit Secrets (copy Host from Pinecone dashboard).")
    if not namespace:
        raise RuntimeError("Missing PINECONE_NAMESPACE in Streamlit Secrets (example: ttt_v2).")

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(index_name, host=host)

    return oai, idx, namespace


# ---------------- FAQ / IDENTITY DIRECT ANSWERS (NO RAG) ----------------

def is_greeting(q: str) -> bool:
    ql = q.lower().strip()
    return ql in ["hi", "hello", "hey", "hai", "good morning", "good afternoon", "good evening"]


def is_faq_identity_question(q: str) -> bool:
    ql = q.lower().strip()
    return any(x in ql for x in [
        "who are you",
        "what are you",
        "who created you",
        "who made you",
        "who built you",
        "who owns you",
        "owner",
        "what is your purpose",
        "purpose",
        "where do your answers come from",
        "where do your answers",
        "do you browse",
        "can you browse",
        "do you browse the internet",
        "are you microsoft copilot",
        "are you copilot",
        "are you google gemini",
        "are you gemini",
        "why do you show a source link",
        "how do you avoid wrong answers",
        "how often is your knowledge updated",
        "can you summarize",
        "can you write code",
        "can you generate images",
        "what can you do",
        "what cannot you do",
        "limitations"
    ])


def direct_answer_faq(q: str) -> str:
    ql = q.lower().strip()

    if is_greeting(ql):
        return "Hi 👋 I’m The Tech Thinker AI. Welcome to https://thetechthinker.com/. Ask me anything from The Tech Thinker content."

    if "who are you" in ql or "what are you" in ql:
        return "I am The Tech Thinker AI, an assistant that answers using The Tech Thinker website content."

    if "who created you" in ql or "who made you" in ql or "who built you" in ql:
        return "I was created by The Tech Thinker."

    if "who owns you" in ql or ("owner" in ql and "who" in ql):
        return "I am owned and operated by The Tech Thinker (thetechthinker.com)."

    if "microsoft copilot" in ql or ("copilot" in ql and "microsoft" in ql):
        return "No. I’m The Tech Thinker AI, built specifically for The Tech Thinker content."

    if "google gemini" in ql or ("gemini" in ql and "google" in ql):
        return "No. I’m The Tech Thinker AI, powered by a Retrieval-Augmented Generation system using The Tech Thinker articles."

    if "purpose" in ql:
        return "To help people learn technology and engineering topics using verified content from The Tech Thinker."

    if "where do your answers come from" in ql or "where do your answers" in ql:
        return "From The Tech Thinker website articles stored in my RAG knowledge base."

    if "browse" in ql or "internet" in ql:
        return "No. I only use The Tech Thinker content in my RAG index."

    if "why do you show a source link" in ql or "why source link" in ql:
        return "For transparency—so you can verify the answer in the original The Tech Thinker article."

    if "summarize" in ql:
        return "Yes. Share the article topic or link and I can summarize based on the stored content."

    if "write code" in ql:
        return "I can provide code examples and explanations, but I do not execute code on your device."

    if "generate images" in ql:
        return "Not by default. If image generation tools are added, I can support that."

    if "avoid wrong answers" in ql or "how do you avoid" in ql:
        return "I only answer from retrieved content and refuse to guess when evidence is missing."

    if "how often" in ql and "updated" in ql:
        return "Whenever new The Tech Thinker articles are indexed into the database."

    if "what cannot you do" in ql or "limitations" in ql:
        return "I cannot browse the live internet, access private accounts, run code on your device, or provide medical/legal/financial advice."

    if "what can you do" in ql:
        return "I retrieve and answer from The Tech Thinker articles, explain concepts, summarize, and provide step-by-step guides."

    return "I am The Tech Thinker AI, an assistant that answers using The Tech Thinker website content."


# ---------------- RAG ANSWER ----------------

def _extract_best_url(md: dict) -> str | None:
    return (
        md.get("source_url")
        or md.get("url")
        or md.get("page_url")
        or md.get("source")
        or md.get("link")
        or None
    )


def answer_question(user_query: str) -> dict:
    user_query = (user_query or "").strip()
    if not user_query:
        return {"answer": "Please type a question.", "source_url": None, "confidence": 0.0}

    # ✅ DIRECT FAQ / GREETING GATE (NO PINECONE)
    if is_faq_identity_question(user_query) or is_greeting(user_query):
        ans = direct_answer_faq(user_query)
        return {"answer": ans, "source_url": ABOUT_PAGE_URL, "confidence": 1.0}

    oai, index, namespace = _clients()

    # 1) embed
    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=user_query).data[0].embedding

    # 2) retrieve from your namespace
    res = index.query(
        vector=q_emb,
        top_k=8,
        include_metadata=True,
        namespace=namespace
    )

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    if not matches:
        return {"answer": "I couldn't find relevant content in my sources.", "source_url": None, "confidence": 0.0}

    # 3) best match
    best = matches[0]
    best_score = best.get("score", 0.0) if isinstance(best, dict) else getattr(best, "score", 0.0)
    best_md = best.get("metadata", {}) if isinstance(best, dict) else getattr(best, "metadata", {}) or {}
    best_url = _extract_best_url(best_md)

    # score gate (like Colab MIN_SCORE)
    if float(best_score) < MIN_SCORE:
        return {
            "answer": "❌ This information is not available in The Tech Thinker content.",
            "source_url": best_url,
            "confidence": float(best_score)
        }

    # 4) context
    chunks = []
    for m in matches[:4]:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        txt = md.get("text") or md.get("chunk") or ""
        if txt:
            chunks.append(txt.strip())
    context = "\n\n---\n\n".join(chunks)

    # 5) grounded generation
    prompt = f"""
Answer ONLY from the CONTEXT.
If not supported, reply exactly:
"This information is not available in The Tech Thinker content."

CONTEXT:
{context}

QUESTION:
{user_query}
""".strip()

    out = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    answer = out.choices[0].message.content.strip()
    return {"answer": answer, "source_url": best_url, "confidence": float(best_score)}
