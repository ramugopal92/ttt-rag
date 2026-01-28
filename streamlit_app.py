import streamlit as st
import base64
from rag_engine import answer_question


# ----------- FUNCTION TO LOAD LOGO CORRECTLY ----------
def load_logo_base64(path="logo.png"):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


logo_base64 = load_logo_base64()


# ----------- PAGE CONFIG -----------
st.set_page_config(
    page_title="The Tech Thinker AI",
    page_icon="logo.png",
    layout="centered"
)


# ----------- HEADER (CENTERED, LOGO VISIBLE) -----------
if logo_base64:
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:-20px; margin-bottom:5px;">
            <img src="data:image/png;base64,{logo_base64}" width="70" />
            <h1 style="margin-bottom:0px; font-size:42px;">The Tech Thinker AI Assistant</h1>
            <p style="margin-top:-8px; color:gray; font-size:16px;">
                Powered by <b>TheTechThinker.com</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("⚠️ Logo file not found. Upload `logo.png` to your project folder.")


# ----------- CHAT UI -----------
BOT_AVATAR = "logo.png"
USER_AVATAR = None  # keep default user icon


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there 👋 Ask me anything from The Tech Thinker platform!"}
    ]

for m in st.session_state.messages:
    avatar = BOT_AVATAR if m["role"] == "assistant" else USER_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])


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
