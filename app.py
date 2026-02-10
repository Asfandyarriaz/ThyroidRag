# app.py
import re
import json
import streamlit as st

from core.pipeline_loader import init_pipeline
from ui.layout import setup_page, inject_custom_css, page_title
from ui.components import render_user_message, render_bot_message, show_thinking
from ui.json_renderer import JSONRenderer

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
    st.markdown("⚙️ **Controls**")
    mode = st.radio(
        "Answer style",
        ["Short", "Standard", "Evidence"],
        index=1,
        help="Short = quick. Standard = fuller. Evidence = includes verbatim quotes.",
    )

    st.markdown("---")

    # === DIAGNOSTICS SECTION ===
    st.markdown("🔍 **Diagnostics**")
    
    with st.expander("Query Retrieval Analysis", expanded=False):
        st.caption("See what documents are retrieved for any query")
        
        diagnostic_query = st.text_input(
            "Test Query",
            placeholder="Enter a question to diagnose...",
            key="diagnostic_query_input",
            help="Analyze which chunks are retrieved for this query"
        )
        
        chunks_k = st.slider(
            "Chunks per sub-query",
            min_value=5,
            max_value=20,
            value=10,
            key="diagnostic_k",
            help="Number of chunks to retrieve for each generated sub-query"
        )
        
        run_diagnostic = st.button(
            "🔍 Run Diagnostic",
            use_container_width=True,
            type="secondary",
            key="run_diagnostic_btn"
        )
        
        if run_diagnostic and diagnostic_query:
            with st.spinner("Running diagnostic analysis..."):
                try:
                    diagnosis = st.session_state.pipeline.diagnose_retrieval(
                        diagnostic_query,
                        k=chunks_k
                    )
                    
                    st.session_state.diagnostic_results = diagnosis
                    st.success("✅ Diagnostic complete! See results below.")
                    
                except Exception as e:
                    st.error(f"❌ Diagnostic failed: {str(e)}")
                    st.exception(e)
        
        elif run_diagnostic and not diagnostic_query:
            st.warning("⚠️ Please enter a query to diagnose")

    st.markdown("---")

    # === CREDIBILITY CHECK SECTION ===
    st.markdown(
        '<span title="Check claims from other sources by verifying them against the indexed thyroid cancer papers.">'
        "✅ **Credibility Check**"
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
        if "diagnostic_results" in st.session_state:
            del st.session_state.diagnostic_results
        st.rerun()


# show disclaimer once at top
if "disclaimer_shown" not in st.session_state:
    st.session_state.disclaimer_shown = True
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <b>Disclaimer:</b> This tool is for research/education only and does not provide medical advice. "
        "For clinical decisions, consult guidelines and qualified healthcare professionals."
        "</div>",
        unsafe_allow_html=True,
    )

# === DISPLAY DIAGNOSTIC RESULTS ===
if "diagnostic_results" in st.session_state:
    diagnosis = st.session_state.diagnostic_results
    
    with st.container():
        st.markdown("### 🔍 Diagnostic Results")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Question:** {diagnosis['original_question']}")
        with col2:
            if st.button("❌ Close", key="close_diagnostic"):
                del st.session_state.diagnostic_results
                st.rerun()
        
        st.markdown("**Generated Sub-Queries:**")
        for i, sq in enumerate(diagnosis["sub_queries_generated"], 1):
            st.markdown(f"{i}. `{sq}`")
        
        st.markdown("---")
        
        for idx, result in enumerate(diagnosis["retrieval_results"], 1):
            with st.expander(
                f"📊 Sub-Query {idx}: {result['query'][:80]}... ({result['chunks_found']} chunks found)",
                expanded=(idx == 1)
            ):
                if result['sample_chunks']:
                    for chunk in result['sample_chunks']:
                        st.markdown(
                            f"**Rank {chunk['rank']}** | "
                            f"<span class='diagnostic-score'>Score: {chunk['score']:.4f}</span>",
                            unsafe_allow_html=True
                        )
                        
                        st.caption(
                            f"📄 {chunk['title']} ({chunk['year']}) | "
                            f"PMID: {chunk['pmid']} | "
                            f"Evidence Level: {chunk['evidence_level']}"
                        )
                        
                        st.text_area(
                            f"Text Preview (Rank {chunk['rank']})",
                            value=chunk['text_preview'],
                            height=120,
                            disabled=True,
                            key=f"diag_chunk_{idx}_{chunk['rank']}"
                        )
                        
                        if chunk != result['sample_chunks'][-1]:
                            st.markdown("---")
                else:
                    st.warning("⚠️ No chunks retrieved for this sub-query")
        
        st.download_button(
            label="📥 Download Full Diagnostic (JSON)",
            data=json.dumps(diagnosis, indent=2),
            file_name=f"diagnostic_analysis.json",
            mime="application/json",
            help="Download complete diagnostic data for further analysis",
            use_container_width=True
        )
        
        st.markdown("---")

# render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        # Render with HTML support for citations
        with st.chat_message("assistant"):
            st.markdown(msg["content"], unsafe_allow_html=True)


def apply_mode_hint(user_text: str) -> str:
    if mode == "Short":
        return f"{user_text}\n\n(Answer in short mode.)"
    if mode == "Evidence":
        return f"{user_text}\n\n(Include verbatim evidence quotes.)"
    return user_text


def format_answer_from_json(result: dict) -> str:
    """
    Convert JSON response to markdown with citations.
    """
    if "error" in result:
        return f"⚠️ {result['error']}"
    
    json_response = result.get("json_response")
    sources = result.get("sources", {})
    confidence = result.get("confidence", {})
    question_type = result.get("question_type", "definition")
    
    if not json_response:
        return "Unable to generate a structured response."
    
    # Use renderer to convert JSON to markdown
    renderer = JSONRenderer(json_response, sources, confidence)
    
    # Render main answer
    answer_md = renderer.render(question_type)
    
    # Render confidence
    confidence_md = renderer.render_confidence()
    
    # Render sources in expandable section
    sources_md = renderer.render_sources()
    
    # Combine
    full_response = f"""{answer_md}

{confidence_md}

{sources_md}
"""
    
    return full_response


user_input = st.chat_input("Ask about thyroid cancer…")

# sidebar credibility run
if run_cred and claim_text.strip():
    combined = f"CREDIBILITY_CHECK: {claim_text.strip()}"
    combined = apply_mode_hint(combined)

    st.session_state.messages.append({"role": "user", "content": f"[Credibility Check] {claim_text.strip()}"})
    render_user_message(f"[Credibility Check] {claim_text.strip()}")

    thinking_ph = show_thinking()
    result = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)
    thinking_ph.empty()

    answer = format_answer_from_json(result)
    answer = normalize_output(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)

elif user_input:
    ui_text = user_input.strip()
    combined = ui_text

    if ui_text.lower().startswith("[credibility check]"):
        combined = "CREDIBILITY_CHECK: " + ui_text[len("[credibility check]"):].strip()

    combined = apply_mode_hint(combined)

    st.session_state.messages.append({"role": "user", "content": ui_text})
    render_user_message(ui_text)

    thinking_ph = show_thinking()
    result = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)
    thinking_ph.empty()

    answer = format_answer_from_json(result)
    answer = normalize_output(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
