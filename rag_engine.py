import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
NAMESPACE = "ttt_v2"



def _get_secret(name: str) -> str:
    # Streamlit Cloud secrets
    if name in st.secrets:
        return st.secrets[name]
    # fallback (local)
    return os.environ.get(name, "")


@st.cache_resource
def _clients():
    openai_key = _get_secret("OPENAI_API_KEY")
    pinecone_key = _get_secret("PINECONE_API_KEY")
    index_name = _get_secret("PINECONE_INDEX")
    host = _get_secret("PINECONE_HOST")
    namespace = _get_secret("PINECONE_NAMESPACE")  # <-- NEW (ttt_v2)

    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets.")
    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY in Streamlit Secrets.")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX in Streamlit Secrets.")
    if not namespace:
        raise RuntimeError("Missing PINECONE_NAMESPACE in Streamlit Secrets (example: ttt_v2).")

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    # Serverless: host is recommended
    idx = pc.Index(index_name, host=host) if host else pc.Index(index_name)

    return oai, idx, namespace


def answer_question(user_query: str) -> dict:
    """
    Returns:
      {
        "answer": "...",
        "source_url": "https://...",
        "confidence": 0.0-1.0 (Pinecone score)
      }
    """
    oai, index, namespace = _clients()

    # 1) embed query
    q_emb = oai.embeddings.create(
        model=EMBED_MODEL,
        input=user_query
    ).data[0].embedding

    # 2) retrieve (IMPORTANT: namespace)
    res = index.query(
        vector=q_emb,
        top_k=8,
        include_metadata=True,
        namespace=namespace
    )

    matches = res.get("matches", []) if isinstance(res, dict) else res.matches

    if not matches:
        return {
            "answer": "I couldn't find relevant content in my sources.",
            "source_url": None,
            "confidence": 0.0
        }

    # 3) build context + pick best URL
    best = matches[0]
    best_score = best.get("score", 0.0) if isinstance(best, dict) else best.score
    best_md = best.get("metadata", {}) if isinstance(best, dict) else best.metadata
    best_url = best_md.get("url") or best_md.get("source") or None

    chunks = []
    for m in matches[:4]:
        md = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        txt = md.get("text") or md.get("chunk") or ""
        if txt:
            chunks.append(txt.strip())

    context = "\n\n---\n\n".join(chunks)

    # 4) generate (grounded)
    prompt = f"""
You are The Tech Thinker AI.
Answer ONLY from the CONTEXT.
If the context is insufficient, say you don't know and suggest what to check.

CONTEXT:
{context}

QUESTION:
{user_query}
""".strip()

    out = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "Be accurate. No hallucinations. Use only the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = out.choices[0].message.content.strip()
    return {"answer": answer, "source_url": best_url, "confidence": float(best_score)}
