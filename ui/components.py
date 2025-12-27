# ui/components.py
import streamlit as st


def render_user_message(content: str):
    with st.chat_message("user"):
        st.markdown(content)


def render_bot_message(content: str):
    with st.chat_message("assistant"):
        st.markdown(content)


def show_thinking():
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message("assistant"):
            st.markdown("_Thinking…_")
    return placeholder
