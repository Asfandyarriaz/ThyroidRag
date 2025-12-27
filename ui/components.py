import streamlit as st
import time

def render_user_message(content):
    st.markdown(f"""
    <div class="user-message">
        <div class="user-content">{content}</div>
        <span class="user-icon">👤</span>
    </div>
    """, unsafe_allow_html=True)

def render_bot_message(content):
    st.markdown(f"""
    <div class="bot-message">
        <span class="bot-icon">🤖</span>
        <div class="bot-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def render_typing_effect(full_text, placeholder):
    displayed = ""
    for ch in full_text:
        displayed += ch
        placeholder.markdown(f"""
            <div class="bot-message">
                <span class="bot-icon">🤖</span>
                <div class="bot-content">{displayed}</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.015)

def show_thinking():
    ph = st.empty()
    ph.markdown("""
        <div class="bot-message">
            <span class="bot-icon">🤖</span>
            <div class="bot-content"><span class="typing-indicator">Thinking…</span></div>
        </div>
    """, unsafe_allow_html=True)
    return ph
