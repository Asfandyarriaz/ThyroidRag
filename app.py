# app.py
import re
import streamlit as st

from core.pipeline_loader import init_pipeline
from ui.layout import setup_page, inject_custom_css, page_title
from ui.components import render_user_message, render_bot_message, show_thinking

setup_page()
inject_custom_css()
page_title()


def normalize_output(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return t.strip()


# init pipeline once
if "pipeline" not in st.session_state:
    st.session_state.pipeline = init_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    mode = st.radio(
        "Answer style",
        ["Short", "Standard", "Evidence"],
        index=0,
        help="Short = quick. Standard = fuller. Evidence = includes verbatim quotes.",
    )

    st.markdown("---")

    st.markdown(
        '<span title="Check claims from other sources by verifying them against the indexed thyroid cancer papers.">'
        "### ✅ Credibility Check"
        "</span>",
        unsafe_allow_html=True,
    )
    credibility_on = st.toggle(
        "Enable",
        value=False,
        help="When enabled, paste a claim and press the button to run the check.",
    )

    claim_text = ""
    run_cred = False
    if credibility_on:
        claim_text = st.text_area(
            "Paste the claim to verify",
            placeholder="Example: Radioiodine ablation decreases local recurrence risk in papillary thyroid cancer.",
            height=140,
        )
        run_cred = st.button("Run credibility check", use_container_width=True)

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# show disclaimer once at top (NOT as a chat message)
if "disclaimer_shown" not in st.session_state:
    st.session_state.disclaimer_shown = True
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <b>Disclaimer:</b> This tool is for research/education only and does not provide medical advice. "
        "For clinical decisions, consult guidelines and qualified healthcare professionals."
        "</div>",
        unsafe_allow_html=True,
    )

# render chat history
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


user_input = st.chat_input("Ask about thyroid cancer…")

# sidebar credibility run (explicit)
if run_cred and claim_text.strip():
    combined = f"CREDIBILITY_CHECK: {claim_text.strip()}"
    combined = apply_mode_hint(combined)

    st.session_state.messages.append({"role": "user", "content": f"[Credibility Check] {claim_text.strip()}"})
    render_user_message(f"[Credibility Check] {claim_text.strip()}")

    thinking_ph = show_thinking()
    answer = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)
    thinking_ph.empty()

    answer = normalize_output(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    render_bot_message(answer)

elif user_input:
    ui_text = user_input.strip()
    combined = ui_text

    if ui_text.lower().startswith("[credibility check]"):
        combined = "CREDIBILITY_CHECK: " + ui_text[len("[credibility check]"):].strip()

    combined = apply_mode_hint(combined)

    st.session_state.messages.append({"role": "user", "content": ui_text})
    render_user_message(ui_text)

    thinking_ph = show_thinking()
    answer = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)
    thinking_ph.empty()

    answer = normalize_output(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    render_bot_message(answer)
