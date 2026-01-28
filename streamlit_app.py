import streamlit as st
from rag_engine import answer_question

st.set_page_config(
    page_title="The Tech Thinker AI",
    page_icon="logo.png",
    layout="centered"
)

# ---------------------- CENTERED HEADER ----------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:-20px; margin-bottom:10px;">
        <img src="logo.png" width="70" style="margin-bottom:-10px;" />
        <h1 style="margin-bottom:0px; font-size:42px;">The Tech Thinker AI</h1>
        <p style="margin-top:-8px; color:gray; font-size:16px;">
            Powered by <b>The Tech Thinker</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- BOT AVATAR ONLY ----------------
BOT_AVATAR = "logo.png"
USER_AVATAR = None  # default Streamlit user icon

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi machi 👋 Ask me anything from The Tech Thinker content!"}
    ]

# ---------------- SHOW CHAT HISTORY ----------------
for m in st.session_state.messages:
    avatar = BOT_AVATAR if m["role"] == "assistant" else USER_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# ---------------- USER INPUT ----------------
q = st.chat_input("Ask a question...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(q)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                r = answer_question(q)
                ans = r.get("answer", "No answer returned.")
                src = r.get("source_url")

                st.markdown(ans)

                if src:
                    st.caption(f"🔗 **Source:** {src}")

            except Exception as e:
                st.error(f"Error: {e}")
                ans = "Error"

    st.session_state.messages.append({"role": "assistant", "content": ans})
