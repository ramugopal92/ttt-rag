import streamlit as st
from rag_engine import answer_question

# Page config (page_icon can be an emoji or image path; logo.png works if present in repo)
st.set_page_config(page_title="The Tech Thinker AI", page_icon="logo.png", layout="centered")

# ---------- INLINE HEADER ----------
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo.png", width=55)
with col2:
    st.markdown(
        """
        <h1 style='margin-bottom:0px;'>The Tech Thinker AI</h1>
        <p style='margin-top:-10px; color:gray;'>Powered by The Tech Thinker</p>
        """,
        unsafe_allow_html=True
    )

# ---------- CUSTOM AVATARS ----------
BOT_AVATAR = "logo.png"
USER_AVATAR = "logo.png"  # change to None if you want default user icon

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi machi 👋 Ask me anything from The Tech Thinker content!"}
    ]

# ---------- RENDER CHAT HISTORY (WITH AVATARS) ----------
for m in st.session_state.messages:
    avatar = BOT_AVATAR if m["role"] == "assistant" else USER_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# ---------- INPUT ----------
q = st.chat_input("Ask a question...")
if q:
    # user message
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(q)

    # assistant response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                r = answer_question(q)
                ans = r.get("answer", "No answer returned.")
                src = r.get("source_url")

                st.markdown(ans)

                # ✅ Only show Source (NO Confidence)
                if src:
                    st.caption(f"**Source:** {src}")

            except Exception as e:
                ans = "Sorry machi — I hit an error while answering."
                st.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": ans})
