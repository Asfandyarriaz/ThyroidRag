# ui/layout.py
import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Thyroid Cancer RAG Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# Add this to your ui/layout.py file
# Update your inject_custom_css() function with this:

# Add this to your ui/layout.py file
# Update your inject_custom_css() function with this:

# Add this to your ui/layout.py file
# Update your inject_custom_css() function with this:

def inject_custom_css():
    """Inject custom CSS for citations and collapsible sources."""
    st.markdown("""
    <style>
    /* ===== CITATION LINKS ===== */
    .citation-link {
        color: #1f77b4;
        text-decoration: none;
        font-weight: 600;
        padding: 0 2px;
        transition: all 0.2s ease;
        border-bottom: 1px dotted #1f77b4;
    }

    .citation-link:hover {
        color: #0d5ea8;
        background-color: #e3f2fd;
        border-radius: 3px;
        border-bottom: 1px solid #0d5ea8;
    }

    /* ===== COLLAPSIBLE SOURCES SECTION ===== */
    .sources-collapsible {
        margin-top: 24px;
        margin-bottom: 16px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0;
        background-color: #fafafa;
    }

    .sources-summary {
        cursor: pointer;
        padding: 14px 18px;
        font-size: 16px;
        font-weight: 600;
        color: #424242;
        user-select: none;
        transition: background-color 0.2s ease;
        border-radius: 8px;
        list-style: none;
    }

    .sources-summary::-webkit-details-marker {
        display: none;
    }

    .sources-summary::before {
        content: "▶";
        display: inline-block;
        margin-right: 8px;
        transition: transform 0.3s ease;
        font-size: 12px;
    }

    .sources-collapsible[open] .sources-summary::before {
        transform: rotate(90deg);
    }

    .sources-summary:hover {
        background-color: #f0f0f0;
    }

    /* ===== SOURCES CONTENT ===== */
    .sources-content {
        padding: 18px;
        background-color: #ffffff;
        border-radius: 0 0 8px 8px;
        animation: slideDown 0.3s ease-out;
    }

    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ===== EVIDENCE QUALITY BOX ===== */
    .evidence-quality {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 12px 16px;
        margin-bottom: 16px;
        border-radius: 4px;
        font-size: 14px;
        line-height: 1.6;
    }

    .evidence-quality strong {
        color: #2e7d32;
    }

    .evidence-quality em {
        color: #558b2f;
        font-size: 13px;
    }

    /* ===== SOURCES DIVIDER ===== */
    .sources-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e0e0e0, transparent);
        margin: 16px 0;
    }

    /* ===== INDIVIDUAL SOURCE ITEMS ===== */
    .sources-list {
        font-size: 14px;
        line-height: 1.8;
    }

    .source-item {
        padding: 10px 12px;
        margin-bottom: 8px;
        border-left: 3px solid #1f77b4;
        background-color: #f8f9fa;
        border-radius: 4px;
        transition: all 0.2s ease;
        scroll-margin-top: 100px;
    }

    .source-item:target {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        animation: highlightSource 1.5s ease-in-out;
    }

    @keyframes highlightSource {
        0% {
            background-color: #fff3cd;
            transform: scale(1.02);
        }
        100% {
            background-color: #f8f9fa;
            transform: scale(1);
        }
    }

    .source-item:hover {
        background-color: #e3f2fd;
        border-left-color: #0d5ea8;
    }

    .source-item strong {
        color: #1f77b4;
        margin-right: 6px;
    }

    .source-item a {
        color: #1f77b4;
        text-decoration: none;
        border-bottom: 1px dotted #1f77b4;
    }

    .source-item a:hover {
        color: #0d5ea8;
        border-bottom: 1px solid #0d5ea8;
    }

    /* ===== SMOOTH SCROLLING ===== */
    html {
        scroll-behavior: smooth;
    }

    /* ===== STREAMLIT SPECIFIC FIXES ===== */
    .stMarkdown a.citation-link {
        color: #1f77b4 !important;
    }

    details.sources-collapsible {
        display: block;
    }

    /* ===== FAITHFULNESS BADGE ===== */
    .faithfulness-badge {
        display: inline-block;
        margin-top: 12px;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    .faithfulness-badge strong {
        font-weight: 600;
    }

    .faithfulness-badge.high {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }

    .faithfulness-badge.medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
    }

    .faithfulness-badge.low {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }

    .faithfulness-badge.unavailable {
        background-color: #e2e3e5;
        border-left: 4px solid #6c757d;
        color: #383d41;
    }

    .faithfulness-badge:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .sources-summary {
            font-size: 14px;
            padding: 12px 14px;
        }
        
        .sources-content {
            padding: 14px;
        }
        
        .source-item {
            font-size: 13px;
            padding: 8px 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown(
        '<h2 class="page-title">🩺 Thyroid Cancer RAG Assistant</h2>',
        unsafe_allow_html=True,
    )
