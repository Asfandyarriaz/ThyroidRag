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

# --- Disclaimer on first load ---
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "⚠️ **Disclaimer (Research/Education Only):**\n\n"
            "This assistant retrieves excerpts from a curated thyroid cancer literature dataset and generates "
            "evidence-grounded summaries. It does **not** provide medical advice, diagnosis, or treatment decisions. "
            "For clinical decisions, consult qualified clinicians and official guidelines."
        )
    })

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

    # Left label + hover tooltip (exactly what you asked)
    credibility_on = st.toggle(
        "✅ Credibility check",
        value=False,
        help="Check claims from other sources (e.g., Google/Gemini/other sites) to see if your indexed thyroid cancer papers support them."
    )

    claim_to_check = ""
    verify_clicked = False

    if credibility_on:
        with st.form("cred_form", clear_on_submit=False):
            claim_to_check = st.text_area(
                "Paste the claim to verify",
                placeholder="Example: 'Radioiodine ablation decreases local recurrence risk in papillary thyroid cancer.'",
                height=120,
            )
            verify_clicked = st.form_submit_button("Verify claim")

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

def apply_mode_hint(user_text: str) -> str:
    if mode == "Short":
        return f"{user_text}\n\n(Answer in short mode.)"
    if mode == "Evidence":
        return f"{user_text}\n\n(Include verbatim evidence quotes.)"
    return user_text

# --- If credibility form submitted, run credibility check and append to chat ---
if credibility_on and verify_clicked:
    claim = (claim_to_check or "").strip()
    if claim:
        # Show what user is verifying
        st.session_state.messages.append({"role": "user", "content": f"[Credibility Check] {claim}"})
        render_user_message(f"[Credibility Check] {claim}")

        thinking_ph = show_thinking()

        combined = apply_mode_hint(f"Check credibility: {claim}")
        answer = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)

        thinking_ph.empty()
        st.session_state.messages.append({"role": "assistant", "content": answer})
        render_bot_message(answer)
    else:
        st.sidebar.warning("Paste a claim first.")

# --- Normal chat input ---
user_input = st.chat_input("Ask about thyroid cancer…")

if user_input:
    user_text = user_input.strip()
    combined = apply_mode_hint(user_text)

    st.session_state.messages.append({"role": "user", "content": user_text})
    render_user_message(user_text)

    thinking_ph = show_thinking()

    answer = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)

    thinking_ph.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer})
    render_bot_message(answer)
