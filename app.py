import streamlit as st
from ui.layout import setup_page, inject_custom_css, page_title
from ui.components import (
    render_user_message, render_bot_message,
    render_typing_effect, show_loader, show_thinking
)

from config import Config
from core.pipeline_loader import init_pipeline

# ----------------------------
# 1) Setup UI
# ----------------------------
setup_page()
inject_custom_css()

# If your ui/layout.py has a fixed title, consider changing it there too.
# This keeps your existing call but changes the intro message below.
page_title()

# ----------------------------
# 2) Init pipeline once
# ----------------------------
if "qa_pipeline" not in st.session_state:
    loader = show_loader()
    st.session_state["qa_pipeline"] = init_pipeline(Config)
    loader.empty()

# ----------------------------
# 3) Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "bot",
        "content": (
            "Hello! 👋\n\n"
            "Welcome to **Thyroid Cancer RAG Assistant**.\n\n"
            "Ask questions about thyroid cancer, and I will answer using **only** the retrieved literature excerpts "
            "from the indexed dataset.\n\n"
            "⚠️ **Disclaimer:** This is a research tool for educational purposes only and **not** medical advice."
        )
    }]

if "bot_typing" not in st.session_state:
    st.session_state["bot_typing"] = False

if "last_bot_index" not in st.session_state:
    st.session_state["last_bot_index"] = -1

# Optional: limit history sent to the backend for speed/cost.
# The UI can keep full chat, but the model only needs a few recent turns.
HISTORY_TURNS = 6  # last 6 messages (including both roles)

def get_recent_history(messages, max_turns=HISTORY_TURNS):
    """
    Returns recent history excluding the latest user message.
    Keeps the most recent N messages to reduce token usage.
    """
    if len(messages) <= 1:
        return []
    history = messages[:-1]  # exclude last user message (current question)
    if max_turns is None or max_turns <= 0:
        return history
    return history[-max_turns:]

# ----------------------------
# 4) Render messages
# ----------------------------
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        if i <= st.session_state["last_bot_index"]:
            render_bot_message(msg["content"])
        else:
            placeholder = st.empty()
            render_typing_effect(msg["content"], placeholder)
            st.session_state["last_bot_index"] = i

if st.session_state["bot_typing"]:
    show_thinking()

# ----------------------------
# 5) Chat input
# ----------------------------
user_input = st.chat_input(
    "Ask a thyroid cancer question…",
    disabled=st.session_state["bot_typing"]
)

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["bot_typing"] = True
    st.rerun()

# ----------------------------
# 6) Answer
# ----------------------------
if st.session_state.get("bot_typing"):
    qa = st.session_state["qa_pipeline"]

    question = st.session_state["messages"][-1]["content"]
    history = get_recent_history(st.session_state["messages"], HISTORY_TURNS)

    # Your pipeline decides how to use history.
    # If you want strictly "no chat memory", pass [] instead of history.
    result = qa.answer(question, history)

    reply = result if isinstance(result, str) else result.get("answer", str(result))

    st.session_state["messages"].append({"role": "bot", "content": reply})
    st.session_state["bot_typing"] = False
    st.rerun()
