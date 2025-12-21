import streamlit as st
import time
import html


def _escape(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def render_user_message(content: str):
    content = _escape(content)
    st.markdown(f"""
    <div class="user-message">
        <div class="user-content">{content}</div>
        <span class="user-icon">🧑</span>
    </div>
    """, unsafe_allow_html=True)


def render_bot_message(content: str):
    content = _escape(content)
    st.markdown(f"""
    <div class="bot-message">
        <span class="bot-icon">🤖</span>
        <div class="bot-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_typing_effect(full_text: str, placeholder):
    displayed = ""
    for ch in full_text or "":
        displayed += ch
        placeholder.markdown(f"""
            <div class="bot-message">
                <span class="bot-icon">🤖</span>
                <div class="bot-content">{_escape(displayed)}</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.01)


def show_loader():
    loader = st.empty()
    loader.markdown("""
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            flex-direction:column;
            margin-top:80px;
            gap:12px;
        ">
            <div style="
                width:44px;height:44px;border-radius:50%;
                border:3px solid rgba(255,255,255,0.15);
                border-top:3px solid rgba(255,255,255,0.65);
                animation: spin 1s linear infinite;
            "></div>
            <div style="color:#9aa4b2;font-size:0.95rem;">Loading pipeline…</div>
        </div>

        <style>
            @keyframes spin { from {transform: rotate(0deg);} to {transform: rotate(360deg);} }
        </style>
    """, unsafe_allow_html=True)
    return loader


def show_thinking():
    st.markdown("""
        <div class="bot-message">
            <span class="bot-icon">🤖</span>
            <div class="bot-content"><span class="typing-indicator">Thinking…</span></div>
        </div>
    """, unsafe_allow_html=True)
