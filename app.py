import os
import time
import base64
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_FOLDER  = "chroma_db"
DOCS_FOLDER    = "docs"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SAMPLE_QUESTIONS = [
    "What is photosynthesis?",
    "How does DNA replication work?",
    "What is the speed of light?",
    "What are acids and bases?",
    "How do chemical reactions work?",
    "What is the periodic table?",
    "How does evolution work?",
    "What is Newton's second law?",
]

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciSnapster AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme state ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_tips" not in st.session_state:
    st.session_state.show_tips = False

dark = st.session_state.dark_mode

# ── Full theme variables ────────────────────────────────────────────────────
if dark:
    BG          = "#0d0d12"
    BG2         = "#13131a"
    BG3         = "#1a1a24"
    BG4         = "#22222e"
    TEXT        = "#eeeef2"
    TEXT2       = "#8888a0"
    TEXT3       = "#555568"
    BORDER      = "#2a2a3a"
    BORDER2     = "#363648"
    ACCENT      = "#4f8ef7"
    ACCENT2     = "#6fa8ff"
    ACCENT_BG   = "#1a2540"
    USER_BG     = "#1e2d50"
    USER_TEXT   = "#a8c8ff"
    AI_BG       = "#13131a"
    AI_TEXT     = "#eeeef2"
    SRC_BG      = "#162030"
    SRC_TEXT    = "#5a9adf"
    INPUT_BG    = "#13131a"
    INPUT_TEXT  = "#eeeef2"
    INPUT_BORDER= "#2a2a3a"
    INPUT_FOCUS = "#4f8ef7"
    BTN_BG      = "#1a1a24"
    BTN_TEXT    = "#8888a0"
    BTN_BORDER  = "#2a2a3a"
    BTN_HOVER   = "#22222e"
    ICON_BG_USER= "#2a4070"
    ICON_BG_AI  = "#1e3050"
    SCROLLBAR   = "#2a2a3a"
    SIDEBAR_BG  = "#0f0f16"
    THINKING_BG = "#13131a"
else:
    BG          = "#f7f7fa"
    BG2         = "#ffffff"
    BG3         = "#f0f0f5"
    BG4         = "#e8e8f0"
    TEXT        = "#111118"
    TEXT2       = "#666680"
    TEXT3       = "#aaaabc"
    BORDER      = "#e0e0ec"
    BORDER2     = "#d0d0e0"
    ACCENT      = "#2563eb"
    ACCENT2     = "#3b82f6"
    ACCENT_BG   = "#eff6ff"
    USER_BG     = "#dbeafe"
    USER_TEXT   = "#1e3a8a"
    AI_BG       = "#ffffff"
    AI_TEXT     = "#111118"
    SRC_BG      = "#eff6ff"
    SRC_TEXT    = "#1d4ed8"
    INPUT_BG    = "#ffffff"
    INPUT_TEXT  = "#111118"
    INPUT_BORDER= "#d0d0e0"
    INPUT_FOCUS = "#2563eb"
    BTN_BG      = "#f0f0f5"
    BTN_TEXT    = "#444460"
    BTN_BORDER  = "#d8d8e8"
    BTN_HOVER   = "#e4e4f0"
    ICON_BG_USER= "#bfdbfe"
    ICON_BG_AI  = "#dbeafe"
    SCROLLBAR   = "#d0d0e0"
    SIDEBAR_BG  = "#f0f0f6"
    THINKING_BG = "#f8f8fc"

# ── Logo loader ─────────────────────────────────────────────────────────────
def get_logo_html():
    logo_path = "logo.png"
    for ext in ["logo.png","logo.jpg","logo.jpeg","logo.webp","logo.svg"]:
        if Path(ext).exists():
            logo_path = ext
            break
    if Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = logo_path.split(".")[-1]
        mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
        return f'<img src="data:{mime};base64,{b64}" style="height:44px;width:auto;object-fit:contain;" alt="SciSnapster Logo">'
    # ── LOGO PLACEHOLDER — replace logo.png with your own image file ──
    return '''<div style="
        height:44px; width:44px; border-radius:10px;
        background: linear-gradient(135deg, #2196F3, #4CAF50);
        display:flex; align-items:center; justify-content:center;
        font-weight:900; font-size:18px; color:white; font-family:monospace;
        border: 2px dashed rgba(255,255,255,0.4);
    ">S</div>'''

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"], .stApp {{
    font-family: 'Outfit', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

/* Hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton {{ visibility: hidden !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
[data-testid="stAppViewContainer"] > .main {{ padding: 0 !important; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER} !important;
    width: 300px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 0 !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1rem !important;
}}

/* ── Sidebar buttons ── */
.stButton > button {{
    background: {BTN_BG} !important;
    color: {BTN_TEXT} !important;
    border: 1px solid {BTN_BORDER} !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 0.75rem !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background: {BTN_HOVER} !important;
    border-color: {ACCENT} !important;
    color: {TEXT} !important;
    transform: translateX(3px) !important;
}}

/* ── Chat input — full theme fix ── */
[data-testid="stChatInput"] {{
    background: {BG2} !important;
    border-top: 1px solid {BORDER} !important;
    padding: 1rem 1.5rem !important;
}}
[data-testid="stChatInput"] > div {{
    background: {INPUT_BG} !important;
    border: 1.5px solid {INPUT_BORDER} !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}}
[data-testid="stChatInput"] > div:focus-within {{
    border-color: {INPUT_FOCUS} !important;
    box-shadow: 0 0 0 3px {ACCENT}22 !important;
}}
[data-testid="stChatInput"] textarea {{
    color: {INPUT_TEXT} !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    background: {INPUT_BG} !important;
    caret-color: {ACCENT} !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {TEXT3} !important;
}}
[data-testid="stChatInput"] button {{
    background: {ACCENT} !important;
    border-radius: 10px !important;
    border: none !important;
}}
[data-testid="stChatInput"] button:hover {{
    background: {ACCENT2} !important;
}}
[data-testid="stChatInput"] button svg {{
    fill: white !important;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {SCROLLBAR}; border-radius: 10px; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {ACCENT} !important; }}

/* ── Selectbox, toggle ── */
[data-testid="stSelectbox"] div, [data-testid="stToggle"] div {{
    color: {TEXT} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Load DB ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if not Path(CHROMA_FOLDER).exists():
        pdfs = list(Path(DOCS_FOLDER).glob("*.pdf"))
        if not pdfs:
            return None
        docs = []
        for pdf in pdfs:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_FOLDER)
    else:
        db = Chroma(persist_directory=CHROMA_FOLDER, embedding_function=embeddings)
    return db

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo area
    logo_html = get_logo_html()
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0 1.5rem;">
        {logo_html}
        <div>
            <div style="font-size:1.1rem;font-weight:700;color:{TEXT};letter-spacing:-0.02em;">SciSnapster AI</div>
            <div style="font-size:0.72rem;color:{TEXT2};margin-top:1px;">Science from real sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT3};margin-bottom:4px;">Theme</div>', unsafe_allow_html=True)
    mode_label = "🌙 Dark mode" if dark else "☀️ Light mode"
    if st.button(mode_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:1rem 0;"></div>', unsafe_allow_html=True)

    # Sample questions
    st.markdown(f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT3};margin-bottom:8px;">Try asking</div>', unsafe_allow_html=True)
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q}"):
            st.session_state["prefill"] = q

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:1rem 0;"></div>', unsafe_allow_html=True)

    # Stats
    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    db_exists = Path(CHROMA_FOLDER).exists()
    pdf_count = len(list(Path(DOCS_FOLDER).glob("*.pdf"))) if Path(DOCS_FOLDER).exists() else 0

    st.markdown(f"""
    <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT3};margin-bottom:8px;">Stats</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem;">
        <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:10px;">
            <div style="font-size:1.4rem;font-weight:700;color:{ACCENT};">{msg_count}</div>
            <div style="font-size:0.7rem;color:{TEXT2};">Questions asked</div>
        </div>
        <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:10px;">
            <div style="font-size:1.4rem;font-weight:700;color:{ACCENT};">{pdf_count}</div>
            <div style="font-size:0.7rem;color:{TEXT2};">Documents loaded</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear chat history", key="clear"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:1rem 0;"></div>', unsafe_allow_html=True)

    # About
    st.markdown(f"""
    <div style="font-size:0.72rem;color:{TEXT2};line-height:1.7;">
        Answers grounded in real science documents.<br>
        No hallucinations. Sources always cited.<br><br>
        <span style="color:{TEXT3};">Built by SciSnapster 🎬</span>
    </div>
    """, unsafe_allow_html=True)

# ── Main content area ────────────────────────────────────────────────────────
main = st.container()

with main:
    # Header bar
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:1rem 2rem 0.75rem;
        border-bottom:1px solid {BORDER};
        background:{BG2};
        position:sticky; top:0; z-index:100;
    ">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="
                width:36px;height:36px;border-radius:10px;
                background:{ACCENT_BG};border:1px solid {ACCENT}44;
                display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;
            ">🔭</div>
            <div>
                <div style="font-size:1rem;font-weight:600;color:{TEXT};">Science Assistant</div>
                <div style="font-size:0.72rem;color:{TEXT2};">Powered by real documents</div>
            </div>
        </div>
        <div style="
            display:flex;align-items:center;gap:6px;
            background:{BG3};border:1px solid {BORDER};
            border-radius:20px;padding:4px 12px;
            font-size:0.72rem;color:#22c55e;font-weight:500;
        ">
            <div style="width:6px;height:6px;border-radius:50%;background:#22c55e;"></div>
            Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Empty state
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            padding:4rem 2rem 2rem;text-align:center;
        ">
            <div style="
                width:72px;height:72px;border-radius:20px;
                background:{ACCENT_BG};border:1px solid {ACCENT}44;
                display:flex;align-items:center;justify-content:center;
                font-size:2rem;margin-bottom:1.25rem;
            ">🧬</div>
            <div style="font-size:1.5rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem;letter-spacing:-0.02em;">
                Ask me anything about science
            </div>
            <div style="font-size:0.9rem;color:{TEXT2};max-width:420px;line-height:1.6;margin-bottom:2rem;">
                I search through real textbooks and research documents to give you accurate, cited answers — not guesses.
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:560px;">
                {"".join([f'<div style="background:{BG3};border:1px solid {BORDER};border-radius:20px;padding:6px 14px;font-size:0.8rem;color:{TEXT2};">{q}</div>' for q in SAMPLE_QUESTIONS[:4]])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:1rem 2rem;justify-content:flex-end;">
                <div style="
                    max-width:70%;
                    background:{USER_BG};
                    color:{USER_TEXT};
                    padding:0.85rem 1.1rem;
                    border-radius:18px 18px 4px 18px;
                    font-size:0.95rem;line-height:1.6;
                    font-family:'Outfit',sans-serif;
                ">{msg["content"]}</div>
                <div style="
                    width:36px;height:36px;border-radius:10px;flex-shrink:0;
                    background:{ICON_BG_USER};border:1px solid {ACCENT}44;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;
                ">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                sources_html = f"""
                <div style="
                    display:flex;align-items:center;gap:6px;margin-top:10px;
                    padding-top:10px;border-top:1px solid {BORDER};
                ">
                    <span style="font-size:0.7rem;color:{TEXT3};">SOURCE</span>
                    <span style="
                        font-family:'JetBrains Mono',monospace;
                        font-size:0.7rem;color:{SRC_TEXT};
                        background:{SRC_BG};padding:2px 8px;
                        border-radius:20px;
                    ">📄 {msg["sources"]}</span>
                </div>
                """
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:1rem 2rem;">
                <div style="
                    width:36px;height:36px;border-radius:10px;flex-shrink:0;
                    background:{ICON_BG_AI};border:1px solid {ACCENT}44;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;
                ">🤖</div>
                <div style="
                    max-width:75%;
                    background:{AI_BG};
                    color:{AI_TEXT};
                    padding:0.85rem 1.1rem;
                    border-radius:18px 18px 18px 4px;
                    font-size:0.95rem;line-height:1.7;
                    border:1px solid {BORDER};
                    font-family:'Outfit',sans-serif;
                ">
                    {msg["content"]}
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Indexing notice
    if not Path(CHROMA_FOLDER).exists() and Path(DOCS_FOLDER).exists():
        st.markdown(f"""
        <div style="
            margin:1rem 2rem;padding:1rem 1.25rem;
            background:{ACCENT_BG};border:1px solid {ACCENT}44;
            border-radius:12px;font-size:0.85rem;color:{ACCENT2};
        ">
            ⚡ First run — documents will be indexed when you ask your first question.
        </div>
        """, unsafe_allow_html=True)

# ── Input handling ───────────────────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", "")
question = st.chat_input("Ask a science question...", key="chat_input")
if prefill and not question:
    question = prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner(""):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:0.5rem 2rem;color:{TEXT2};font-size:0.85rem;">
            <div style="
                width:36px;height:36px;border-radius:10px;flex-shrink:0;
                background:{ICON_BG_AI};border:1px solid {ACCENT}44;
                display:flex;align-items:center;justify-content:center;font-size:1rem;
            ">🤖</div>
            <span>Searching documents...</span>
        </div>
        """, unsafe_allow_html=True)

        db = load_db()
        answer = "⚠️ No documents found. Please add PDFs to your docs/ folder and run python rag.py --index"
        sources_str = ""

        if db:
            results  = db.similarity_search(question, k=4)
            context  = "\n\n".join([r.page_content for r in results])
            raw_srcs = list(set([r.metadata.get("source", r.metadata.get("file_path", "")) for r in results]))
            sources  = [Path(s).name if s else "Science Document" for s in raw_srcs]
            sources_str = ", ".join(sources)

            prompt = f"""You are SciSnapster AI — an enthusiastic science assistant.
Answer based ONLY on the provided documents.
If not found, say: "This topic isn't in my current documents yet — but great question!"
Be engaging, clear, and make science exciting.
Always mention the source document or section.

Documents:
{context}

Question: {question}
Answer:"""

            client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
            answer = "⚠️ AI is busy. Please try again in a moment!"
            for attempt in range(4):
                try:
                    response = client.chat.completions.create(
                        model="openrouter/auto",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content
                    break
                except Exception:
                    if attempt < 3:
                        time.sleep(5)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_str
    })
    st.rerun()
