# app.py
import streamlit as st

from core.pipeline_loader import init_pipeline
from ui.layout import setup_page, inject_custom_css, page_title
from ui.components import render_user_message, render_bot_message, show_thinking

setup_page()
inject_custom_css()
page_title()

# --- init pipeline once ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = init_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar controls ---
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    mode = st.radio(
        "Answer style",
        ["Short", "Standard", "Evidence"],
        index=0,
        help="Short = quick answer, Standard = fuller, Evidence = includes quotes.",
    )

    st.markdown("---")
    st.markdown("### ✅ Credibility check")
    credibility_on = st.toggle(
        "Enable credibility check mode",
        value=False,
        help="Paste a claim and the assistant will verify it against the indexed thyroid cancer papers.",
    )

    claim_text = ""
    if credibility_on:
        claim_text = st.text_area(
            "Paste the claim to verify",
            placeholder="Example: 'Radioiodine ablation decreases local recurrence risk in papillary thyroid cancer.'",
            height=120,
        )
        st.caption("Tip: Ask in chat too — this box just makes it easier.")

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- render history ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        render_bot_message(msg["content"])

# --- Compose prompt with user preferences ---
def apply_mode_hint(user_text: str) -> str:
    # Keep this lightweight — QA pipeline already routes modes too,
    # but adding a hint improves reliability.
    if mode == "Short":
        return f"{user_text}\n\n(Answer in short mode.)"
    if mode == "Evidence":
        return f"{user_text}\n\n(Include verbatim evidence quotes.)"
    return user_text

# --- input ---
user_input = st.chat_input("Ask about thyroid cancer…")

if user_input:
    # If credibility is enabled and claim box has text, we prefer that claim
    if credibility_on and claim_text.strip():
        combined = f"Check credibility: {claim_text.strip()}"
    else:
        combined = user_input.strip()

    combined = apply_mode_hint(combined)

    st.session_state.messages.append({"role": "user", "content": user_input})
    render_user_message(user_input)

    show_thinking()

    answer = st.session_state.pipeline.answer(
        combined,
        chat_history=st.session_state.messages,
    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    render_bot_message(answer)
