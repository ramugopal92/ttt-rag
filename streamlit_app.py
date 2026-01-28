import streamlit as st
from rag_engine import answer_question

st.set_page_config(page_title="The Tech Thinker AI", page_icon="🤖", layout="centered")

st.title("🤖 The Tech Thinker AI (RAG Demo)")
st.caption("Validated ✅ | Website rollout coming soon")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi machi 👋 Ask me anything from The Tech Thinker content!"}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask a question...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = answer_question(q)
                ans = r.get("answer", "No answer returned.")
                src = r.get("source_url")
                conf = r.get("confidence", None)

                st.markdown(ans)

                meta = []
                if conf is not None:
                    meta.append(f"**Confidence:** {conf:.2f}")
                if src:
                    meta.append(f"**Source:** {src}")
                if meta:
                    st.caption(" • ".join(meta))

            except Exception as e:
                st.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": ans if 'ans' in locals() else "Error"})
