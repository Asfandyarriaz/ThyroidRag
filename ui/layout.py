# ui/layout.py
import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Thyroid Cancer RAG Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def inject_custom_css():
    st.markdown(
        """
<style>
/* ... your existing CSS ... */

/* === CITATION STYLING === */
sup {
  font-size: 0.75em;
  color: #3b82f6;
  cursor: help;
  margin-left: 1px;
}

sup span {
  color: #3b82f6;
  font-weight: 600;
}

sup span:hover {
  color: #1e40af;
  text-decoration: underline;
}

/* Source list styling */
.stMarkdown ul li strong {
  color: #1e40af;
}

/* Tables for comparisons */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
}

table th {
  background-color: #f3f4f6;
  font-weight: 600;
  padding: 0.75rem;
  text-align: left;
  border: 1px solid #e5e7eb;
}

table td {
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
}

table tr:nth-child(even) {
  background-color: #f9fafb;
}
</style>
""",
        unsafe_allow_html=True,
    )

def page_title():
    st.markdown(
        '<h2 class="page-title">🩺 Thyroid Cancer RAG Assistant</h2>',
        unsafe_allow_html=True,
    )
