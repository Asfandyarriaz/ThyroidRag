# ui/components.py
import streamlit as st
import time

def render_user_message(content: str):
    st.markdown(
        f"""
        <div class="user-message">
            <div class="bubble user">{content}</div>
            <div class="avatar user">👤</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_bot_message(content: str):
    st.markdown(
        f"""
        <div class="bot-message">
            <div class="avatar bot">🤖</div>
            <div class="bubble bot">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_thinking():
    st.markdown(
        """
        <div class="bot-message">
            <div class="avatar bot">🤖</div>
            <div class="bubble bot"><span class="typing">Thinking…</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def type_out_bot_message(full_text: str, speed: float = 0.006):
    placeholder = st.empty()
    buf = ""
    for ch in full_text:
        buf += ch
        placeholder.markdown(
            f"""
            <div class="bot-message">
                <div class="avatar bot">🤖</div>
                <div class="bubble bot">{buf}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(speed)
