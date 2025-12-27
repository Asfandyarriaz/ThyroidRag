# ui/layout.py
import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Haroon GPT",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* --- overall app background (ChatGPT-ish dark) --- */
        .stApp {
            background: #0B0F17 !important;
            color: #E7EAF0 !important;
        }

        /* hide streamlit chrome */
        #MainMenu, header, footer {visibility: hidden;}

        /* constrain content width a bit */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 6rem;
            max-width: 980px;
        }

        /* --- Title --- */
        .main-title {
            text-align: center;
            color: #E7EAF0;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0.25rem 0 1.25rem 0;
        }
        .subtitle {
            text-align: center;
            color: #A9B0BE;
            font-size: 0.98rem;
            margin-top: -0.9rem;
            margin-bottom: 1.25rem;
        }

        /* --- Chat bubbles --- */
        .user-message, .bot-message {
            display: flex;
            gap: 10px;
            margin: 12px 0;
            align-items: flex-start;
        }
        .user-message { justify-content: flex-end; }
        .bot-message { justify-content: flex-start; }

        .avatar {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            font-size: 16px;
            flex: 0 0 auto;
        }
        .avatar.user {
            background: #1F2937;
            color: #E7EAF0;
            border: 1px solid rgba(255,255,255,0.10);
        }
        .avatar.bot {
            background: #10A37F;
            color: #0B0F17;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .bubble {
            padding: 12px 14px;
            border-radius: 14px;
            line-height: 1.45;
            max-width: 75%;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .bubble.user {
            background: #111827;
            color: #E7EAF0;
            border-top-right-radius: 6px;
        }
        .bubble.bot {
            background: #0F172A;
            color: #E7EAF0;
            border-top-left-radius: 6px;
        }

        /* --- Sidebar --- */
        section[data-testid="stSidebar"] {
            background: #0F172A !important;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* --- inputs (ChatGPT-ish) --- */
        textarea, input, .stTextInput input {
            color: #E7EAF0 !important;
            caret-color: #E7EAF0 !important;
        }

        /* Streamlit chat input container */
        div[data-testid="stChatInput"] {
            background: transparent !important;
        }

        /* Chat input textarea */
        div[data-testid="stChatInput"] textarea {
            background: #111827 !important;
            color: #E7EAF0 !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 14px !important;
            padding: 12px 14px !important;
            min-height: 52px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
        }

        /* Placeholder text */
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9CA3AF !important;
            opacity: 1 !important;
        }

        /* Fix the white rectangle behind inputs in some Streamlit builds */
        .st-emotion-cache-1c7y2kd, .st-emotion-cache-1y4p8pa, .st-emotion-cache-128upt6 {
            background: transparent !important;
        }

        /* Buttons */
        .stButton button, button[kind="primary"] {
            background: #10A37F !important;
            color: #071A14 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.9rem !important;
            font-weight: 700 !important;
        }
        .stButton button:hover {
            filter: brightness(1.05);
        }

        /* Small helper text */
        .hint {
            color: #9CA3AF;
            font-size: 0.92rem;
        }

        /* Loader */
        .typing {
            display: inline-block;
            color: #A9B0BE;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def page_title():
    st.markdown('<h1 class="main-title">🤖 Haroon GPT</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Thyroid Cancer RAG Assistant</div>', unsafe_allow_html=True)
