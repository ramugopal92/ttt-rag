def answer_question(user_query: str) -> dict:
    user_query = (user_query or "").strip()
    if not user_query:
        return {"answer": "Please type a question.", "source_url": None}

    # ---------------------------
    # DIRECT FAQ / GREETING GATE
    # ---------------------------
    if is_faq_identity_question(user_query) or is_greeting(user_query):
        ans = direct_answer_faq(user_query)
        return {"answer": ans, "source_url": ABOUT_PAGE_URL}

    # Load clients
    oai, index, namespace = _clients()

    # 1) embed
    q_emb = oai.embeddings.create(
        model=EMBED_MODEL,
        input=user_query
    ).data[0].embedding

    # 2) retrieve
    res = index.query(
        vector=q_emb,
        top_k=8,
        include_metadata=True,
        namespace=namespace
    )

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    if not matches:
        return {"answer": "I couldn't find relevant content in my sources.", "source_url": None}

    # 3) get best match
    best = matches[0]
    best_md = best.get("metadata", {}) if isinstance(best, dict) else getattr(best, "metadata", {}) or {}
    best_url = (
        best_md.get("source_url")
        or best_md.get("url")
        or best_md.get("page_url")
        or best_md.get("source")
        or None
    )

    score = best.get("score", 0.0) if isinstance(best, dict) else getattr(best, "score", 0.0)

    # Score gating
    if float(score) < MIN_SCORE:
        return {
            "answer": "❌ This information is not available in The Tech Thinker content.",
            "source_url": best_url
        }

    # 4) build context
    chunks = []
    for m in matches[:4]:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        txt = md.get("text") or md.get("chunk") or ""
        if txt:
            chunks.append(txt.strip())

    context = "\n\n---\n\n".join(chunks)

    # 5) grounded answer
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

    return {
        "answer": answer,
        "source_url": best_url
    }
