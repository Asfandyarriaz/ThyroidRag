import streamlit as st

def render_user_message(content: str):
    st.markdown(
        f"""
        <div class="user-message">
            <div class="user-content">{_escape_html(content)}</div>
            <span class="user-icon">👤</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_bot_message(content: str):
    # Allow basic markdown while still keeping bubble structure:
    # We'll render markdown inside a div by injecting it as HTML-safe text is hard.
    # This assumes your model output is trusted (your own RAG).
    st.markdown(
        f"""
        <div class="bot-message">
            <span class="bot-icon">🤖</span>
            <div class="bot-content">{_escape_html(content)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_thinking():
    st.markdown(
        """
        <div class="bot-message">
            <span class="bot-icon">🤖</span>
            <div class="bot-content"><span class="typing-indicator">Thinking…</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _escape_html(text: str) -> str:
    # prevent HTML breaking your layout
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
