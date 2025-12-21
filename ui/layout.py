import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Thyroid Cancer RAG Assistant",
        page_icon="🩺",
        layout="wide",
    )

def inject_custom_css():
    st.markdown("""
    <style>
        /* Modern ChatGPT-inspired dark theme */
        :root {
            --bg-main: #212121;
            --bg-chat: #2f2f2f;
            --bg-user: #3b3b3b;
            --bg-assistant: #2a2a2a;
            --text-primary: #ececec;
            --text-secondary: #b4b4b4;
            --border: rgba(255,255,255,0.1);
            --accent: #10a37f;
            --input-bg: #40414f;
        }

        /* Main app background */
        .stApp {
            background-color: var(--bg-main) !important;
        }

        /* Hide Streamlit branding */
        #MainMenu, header, footer {
            visibility: hidden;
        }

        /* Container spacing */
        .block-container {
            max-width: 900px;
            padding: 2rem 1rem 8rem 1rem;
        }

        /* Title styling */
        .main-title {
            text-align: center;
            color: var(--text-primary);
            font-size: 2.2rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
            letter-spacing: -0.5px;
        }

        .subtitle {
            text-align: center;
            color: var(--text-secondary);
            font-size: 1rem;
            margin: 0 0 2rem 0;
            font-weight: 400;
        }

        /* Chat message containers */
        .user-message, .bot-message {
            display: flex;
            gap: 12px;
            margin: 1.5rem 0;
            align-items: flex-start;
        }

        .user-message {
            justify-content: flex-end;
        }

        .bot-message {
            justify-content: flex-start;
        }

        /* Icons */
        .user-icon, .bot-icon {
            font-size: 1.3rem;
            min-width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            flex-shrink: 0;
        }

        .user-icon {
            background: var(--accent);
        }

        .bot-icon {
            background: var(--bg-assistant);
            border: 1px solid var(--border);
        }

        /* Message bubbles */
        .user-content, .bot-content {
            padding: 14px 16px;
            border-radius: 16px;
            max-width: 70%;
            line-height: 1.6;
            word-wrap: break-word;
        }

        .user-content {
            background: var(--bg-user);
            color: var(--text-primary);
            border-bottom-right-radius: 4px;
        }

        .bot-content {
            background: var(--bg-assistant);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-bottom-left-radius: 4px;
        }

        /* Typing indicator */
        .typing-indicator {
            color: var(--text-secondary);
            font-style: italic;
            font-size: 0.95rem;
            padding: 0.5rem;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        /* Fix Streamlit markdown colors */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: var(--text-primary) !important;
        }

        /* Links */
        a {
            color: #58a6ff !important;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        /* ===== CHAT INPUT FIXES ===== */
        
        /* Remove any Streamlit default backgrounds */
        [data-testid="stChatInput"] {
            background: transparent !important;
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            padding: 1rem !important;
            background: var(--bg-main) !important;
            border-top: 1px solid var(--border) !important;
            z-index: 999 !important;
        }

        /* Input wrapper */
        [data-testid="stChatInput"] > div {
            max-width: 900px !important;
            margin: 0 auto !important;
            background: transparent !important;
        }

        /* BaseWeb textarea container - THIS IS KEY */
        [data-testid="stChatInput"] [data-baseweb="base-input"] {
            background-color: var(--input-bg) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 12px !important;
        }

        /* The actual textarea */
        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: var(--text-primary) !important;
            font-size: 15px !important;
            padding: 12px 16px !important;
            border: none !important;
            caret-color: var(--text-primary) !important;
        }

        /* Placeholder text */
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-secondary) !important;
            opacity: 0.7 !important;
        }

        /* Focus state */
        [data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
            border-color: rgba(255,255,255,0.3) !important;
            box-shadow: 0 0 0 2px rgba(255,255,255,0.05) !important;
        }

        /* Override any Streamlit autocomplete styling */
        [data-testid="stChatInput"] input:-webkit-autofill,
        [data-testid="stChatInput"] textarea:-webkit-autofill {
            -webkit-text-fill-color: var(--text-primary) !important;
            -webkit-box-shadow: 0 0 0px 1000px var(--input-bg) inset !important;
        }

        /* Send button styling */
        [data-testid="stChatInput"] button {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border: none !important;
            padding: 8px !important;
        }

        [data-testid="stChatInput"] button:hover {
            color: var(--text-primary) !important;
            background-color: rgba(255,255,255,0.1) !important;
        }

        /* Code blocks in messages */
        code {
            background-color: rgba(0,0,0,0.3) !important;
            color: #ff6b6b !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Courier New', monospace !important;
        }

        pre {
            background-color: rgba(0,0,0,0.3) !important;
            padding: 12px !important;
            border-radius: 8px !important;
            overflow-x: auto !important;
        }

        pre code {
            background-color: transparent !important;
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown('<h1 class="main-title">🩺 Thyroid Cancer RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions and get evidence-grounded answers from retrieved literature excerpts.</div>',
        unsafe_allow_html=True
    )
