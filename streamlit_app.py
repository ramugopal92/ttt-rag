import streamlit as st

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

    # TEMP reply (we will connect your RAG logic in next step)
    temp_answer = "Demo UI is ready ✅ Next step: connect Pinecone + OpenAI RAG engine."
    with st.chat_message("assistant"):
        st.markdown(temp_answer)

    st.session_state.messages.append({"role": "assistant", "content": temp_answer})
