import base64
import html
import os
import time
from pathlib import Path

import streamlit as st


# -- Config -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "SciSnapster AI"
CHROMA_FOLDER = "chroma_db"
DOCS_FOLDER = "docs"
CHROMA_DIR = BASE_DIR / CHROMA_FOLDER
DOCS_DIR = BASE_DIR / DOCS_FOLDER
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"
MODEL_CHOICES = [
    "openrouter/auto",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]
LOGO_CANDIDATES = [
    BASE_DIR / "logo-cropped.png",
    BASE_DIR / "logo.png",
    BASE_DIR / "assets" / "logo.png",
    BASE_DIR / "logo.jpg",
    BASE_DIR / "logo.jpeg",
    BASE_DIR / "logo.webp",
    BASE_DIR / "logo.svg",
]

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

SCIENCE_FACTS = [
    "Lightning strikes Earth about 100 times every second.",
    "Your DNA, if uncoiled, would stretch from Earth to Pluto and back 17 times.",
    "The ocean produces over 50% of Earth's oxygen.",
    "There are more bacteria in your mouth than people on Earth.",
    "A teaspoon of neutron star material weighs about 10 million tons.",
    "Hot water can freeze faster than cold water in some conditions.",
    "Your heart beats about 100,000 times a day.",
    "Earth is the only planet not named after a god.",
    "A single bolt of lightning can carry enough energy to toast thousands of slices of bread.",
    "Octopuses have three hearts and blue blood.",
]


st.set_page_config(
    page_title=APP_NAME,
    page_icon="S",
    layout="wide",
    initial_sidebar_state="auto",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "fact_index" not in st.session_state:
    st.session_state.fact_index = 0
if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# -- Helpers ----------------------------------------------------------------
def asset_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_types = {
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_types.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def get_logo_data_uri() -> str | None:
    for path in LOGO_CANDIDATES:
        if path.exists():
            return asset_to_data_uri(path)
    return None


def logo_html(width: int = 180, height: int = 58, variant: str = "wordmark") -> str:
    src = get_logo_data_uri()
    if src:
        return (
            f'<img class="brand-logo brand-logo--{variant}" src="{src}" '
            f'alt="{APP_NAME} logo" style="width:{width}px;height:{height}px;">'
        )
    return (
        f'<div class="brand-logo brand-logo-fallback brand-logo--{variant}" '
        f'style="width:{width}px;height:{height}px;">{APP_NAME}</div>'
    )


def clean_html(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def source_label(sources: str) -> str:
    if not sources:
        return "Source pending"
    return clean_html(sources)


def set_question(question: str) -> None:
    st.session_state.prefill = question


def get_openrouter_key() -> str:
    env_key = os.environ.get(OPENROUTER_ENV_VAR, "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get(OPENROUTER_ENV_VAR, "")).strip()
    except Exception:
        return ""


def connection_status() -> tuple[str, str]:
    if get_openrouter_key():
        return "AI connected", "ready"
    return "Add API key", "setup"


def document_count() -> int:
    return len(list(DOCS_DIR.glob("*.pdf"))) if DOCS_DIR.exists() else 0


# -- Data -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_db():
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pypdf import PdfReader
    except ImportError as exc:
        return None, str(exc)

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        if not CHROMA_DIR.exists():
            pdfs = list(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []
            if not pdfs:
                return None, ""

            docs = []
            for pdf in pdfs:
                reader = PdfReader(str(pdf))
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": str(pdf), "page": page_number},
                            )
                        )

            splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=90)
            chunks = splitter.split_documents(docs)
            return Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory=str(CHROMA_DIR),
            ), ""

        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        ), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def answer_question(question: str) -> tuple[str, str]:
    db, dependency_error = load_db()
    if not db:
        if dependency_error:
            return (
                f"The app design is ready, but a RAG package is missing: {dependency_error}. Run pip install -r requirements.txt, then restart Streamlit.",
                "",
            )
        return (
            "I do not have any PDFs indexed yet. Add science PDFs to the docs folder, then ask again.",
            "",
        )

    try:
        results = db.similarity_search(question, k=4)
    except Exception as exc:
        return f"I could not search the science database yet: {exc}", ""

    if not results:
        return (
            "I searched the science database, but I did not find a matching source yet.",
            "",
        )

    context = "\n\n".join(result.page_content for result in results)
    context = context[:12000]
    raw_sources = {
        result.metadata.get("source", result.metadata.get("file_path", ""))
        for result in results
    }
    sources = [Path(source).name if source else "Science Document" for source in raw_sources]
    sources_str = ", ".join(sorted(sources))

    prompt = f"""You are SciSnapster AI, an enthusiastic science assistant for the SciSnapster YouTube channel.
Answer based ONLY on the provided documents.
If the answer is not found, say: "This topic isn't in my current documents yet, but great question!"
Keep the answer clear, exciting, and accurate. Mention which document the answer comes from.

Documents:
{context}

Question: {question}
Answer:"""

    openrouter_key = get_openrouter_key()
    if not openrouter_key:
        return (
            f"I found relevant source material, but OpenRouter is not connected yet. Set {OPENROUTER_ENV_VAR}, then I can answer from your documents.",
            sources_str,
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        return f"The OpenAI client package is missing: {exc}", sources_str

    client = OpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=45.0,
    )
    answer = "The AI service is busy right now. Please try again in a moment."

    for attempt, model in enumerate(MODEL_CHOICES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content or answer
            break
        except Exception as exc:
            answer = f"I could not reach the AI service yet: {exc}"
            if attempt < len(MODEL_CHOICES) - 1:
                time.sleep(3)

    return answer, sources_str


# -- Styles -----------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #10121a;
    --muted: #676b7d;
    --soft: #8d92a6;
    --line: #dfe4ee;
    --line-strong: #ccd5e4;
    --paper: #ffffff;
    --wash: #f4f7fb;
    --wash-2: #eaf3ff;
    --blue: #1296db;
    --blue-strong: #0879bc;
    --green: #52ad2e;
    --yellow: #ffd12f;
    --night: #0b1020;
    --shadow-sm: 0 12px 28px rgba(16, 18, 26, 0.08);
    --shadow-md: 0 18px 50px rgba(16, 18, 26, 0.12);
}

*, *::before, *::after {
    box-sizing: border-box;
}

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    color: var(--ink) !important;
}

.stApp {
    background:
        linear-gradient(90deg, rgba(18,150,219,0.06) 1px, transparent 1px),
        linear-gradient(180deg, rgba(82,173,46,0.05) 1px, transparent 1px),
        #f7f9fd !important;
    background-size: 44px 44px, 44px 44px, auto !important;
}

#MainMenu, footer, header, .stDeployButton {
    visibility: hidden !important;
}

.block-container {
    max-width: 100% !important;
    padding: 0 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
    padding: 0 !important;
}

button, textarea, input {
    font-family: 'Outfit', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.92) !important;
    border-right: 1px solid var(--line) !important;
    box-shadow: 10px 0 32px rgba(16,18,26,0.05);
}

section[data-testid="stSidebar"] .block-container {
    padding: 20px 16px 24px !important;
}

.sidebar-brand {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
    padding: 2px 2px 18px;
}

.brand-logo {
    object-fit: contain;
    flex-shrink: 0;
    filter: drop-shadow(0 10px 18px rgba(16,18,26,0.12));
}

.brand-logo--wordmark,
.brand-logo--hero {
    object-position: left center;
    max-width: 100%;
}

.brand-logo--mark {
    object-fit: cover;
    object-position: left center;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fff;
    padding: 2px;
    box-shadow: var(--shadow-sm);
}

.brand-logo--hero {
    filter: drop-shadow(0 16px 28px rgba(16,18,26,0.16));
}

.brand-logo-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 800;
    background: linear-gradient(135deg, var(--blue), var(--green));
    border-radius: 8px;
    padding: 0 10px;
}

.brand-title {
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: 0;
    color: var(--ink);
}

.brand-subtitle {
    color: var(--muted);
    font-size: 0.74rem;
    line-height: 1.25;
    margin-top: 2px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px;
    margin: 2px 0 18px;
}

.metric-card {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #fff;
    padding: 11px 10px;
    box-shadow: var(--shadow-sm);
}

.metric-card strong {
    display: block;
    font-size: 1.45rem;
    line-height: 1;
    letter-spacing: 0;
}

.metric-card span {
    display: block;
    color: var(--muted);
    font-size: 0.68rem;
    margin-top: 6px;
    line-height: 1.2;
}

.metric-card:first-child strong {
    color: var(--blue-strong);
}

.metric-card:last-child strong {
    color: var(--green);
}

.eyebrow {
    margin: 6px 0 9px;
    color: var(--soft);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
}

div[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
}

.stButton > button {
    min-height: 40px !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    background: #fff !important;
    color: #303547 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.55rem 0.72rem !important;
    text-align: left !important;
    box-shadow: none !important;
    transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease, color 0.16s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    color: var(--ink) !important;
    border-color: rgba(18,150,219,0.58) !important;
    background: #f0f8ff !important;
}

.sidebar-divider {
    height: 1px;
    background: var(--line);
    margin: 16px 0;
}

.fact-panel {
    border: 1px solid #f1d36c;
    border-radius: 8px;
    background: #fff9df;
    padding: 12px;
    margin-bottom: 10px;
}

.fact-panel__label {
    color: #8b6504;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.fact-panel__text {
    color: #654c08;
    font-size: 0.82rem;
    line-height: 1.55;
}

/* Header */
.topbar {
    position: sticky;
    top: 0;
    z-index: 40;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 30px;
    background: rgba(255,255,255,0.88);
    border-bottom: 1px solid var(--line);
    backdrop-filter: blur(16px);
}

.topbar__left {
    display: flex;
    align-items: center;
    gap: 13px;
    min-width: 0;
}

.topbar__title {
    font-size: 1.04rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: 0;
    white-space: nowrap;
}

.topbar__subtitle {
    color: var(--muted);
    font-size: 0.76rem;
    margin-top: 2px;
}

.topbar__right {
    display: flex;
    align-items: center;
    gap: 9px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    border-radius: 999px;
    border: 1px solid #bfe6c2;
    background: #effcf1;
    color: #278122;
    padding: 6px 12px;
    font-size: 0.76rem;
    font-weight: 700;
    white-space: nowrap;
}

.status-pill::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 4px rgba(82,173,46,0.16);
}

.status-pill--setup {
    border-color: #f1d36c;
    background: #fff9df;
    color: #8b6504;
}

.status-pill--setup::before {
    background: var(--yellow);
    box-shadow: 0 0 0 4px rgba(255,209,47,0.2);
}

.doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    min-height: 32px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: #fff;
    color: #4b5062;
    padding: 6px 12px;
    font-size: 0.76rem;
    font-weight: 700;
    white-space: nowrap;
}

/* Main canvas */
.chat-stage {
    width: min(1040px, calc(100vw - 48px));
    margin: 0 auto;
    padding: 26px 0 0;
}

.hero {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr);
    gap: 18px;
    align-items: stretch;
    margin-bottom: 24px;
}

.hero-main,
.lab-panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255,255,255,0.92);
    box-shadow: var(--shadow-md);
}

.hero-main {
    padding: 30px;
    overflow: hidden;
    position: relative;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(232,247,255,0.88));
}

.hero-main::before {
    content: "";
    position: absolute;
    inset: 0;
    border-top: 5px solid var(--blue);
    pointer-events: none;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #0c6d9f;
    background: #e7f6ff;
    border: 1px solid #bde8ff;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 800;
    padding: 7px 11px;
    margin-bottom: 18px;
}

.hero-brand {
    margin: 2px 0 18px;
}

.hero-title {
    font-size: 3.8rem;
    line-height: 0.95;
    font-weight: 800;
    letter-spacing: 0;
    max-width: 760px;
    color: var(--ink);
}

.hero-title span {
    color: var(--blue-strong);
}

.hero-copy {
    color: #4c5265;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 620px;
    margin-top: 18px;
}

.hero-stripe {
    display: flex;
    gap: 8px;
    margin-top: 24px;
}

.hero-stripe span {
    display: block;
    height: 9px;
    border-radius: 99px;
}

.hero-stripe span:nth-child(1) {
    width: 78px;
    background: var(--blue);
}

.hero-stripe span:nth-child(2) {
    width: 78px;
    background: var(--green);
}

.hero-stripe span:nth-child(3) {
    width: 78px;
    background: var(--yellow);
}

.lab-panel {
    padding: 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 260px;
    background:
        linear-gradient(180deg, #101827 0%, #111f34 100%);
    border-color: #1c3554;
    color: #f8fbff;
}

.lab-panel__top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
}

.lab-panel__title {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0;
    color: #aab7ca;
    text-transform: uppercase;
}

.molecule {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    align-items: center;
    margin: auto 0;
}

.molecule span {
    aspect-ratio: 1;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
    border: 2px solid rgba(255,255,255,0.22);
}

.molecule span:nth-child(2),
.molecule span:nth-child(7),
.molecule span:nth-child(12),
.molecule span:nth-child(18) {
    background: var(--blue);
    border-color: var(--blue);
}

.molecule span:nth-child(5),
.molecule span:nth-child(9),
.molecule span:nth-child(15) {
    background: var(--green);
    border-color: var(--green);
}

.molecule span:nth-child(11),
.molecule span:nth-child(20) {
    background: var(--yellow);
    border-color: #e8b90b;
}

.lab-panel__footer {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 18px;
}

.lab-chip {
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 8px;
    background: rgba(255,255,255,0.08);
    min-height: 46px;
    padding: 9px 10px;
}

.lab-chip strong {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #f8fbff;
}

.lab-chip span {
    color: #aab7ca;
    font-size: 0.64rem;
}

.messages {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding-bottom: 12px;
}

.message-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.message-row--user {
    justify-content: flex-end;
}

.avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    flex: 0 0 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--line);
    background: #fff;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.avatar img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    padding: 3px;
}

.avatar--user {
    background: #eef7ff;
    color: var(--blue-strong);
    font-size: 1.1rem;
    font-weight: 800;
}

.bubble {
    max-width: min(760px, 78%);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 15px 16px;
    font-size: 0.98rem;
    line-height: 1.7;
    box-shadow: var(--shadow-sm);
}

.bubble--assistant {
    background: #fff;
    color: var(--ink);
    border-left: 5px solid var(--blue);
}

.bubble--user {
    background: #102139;
    color: #f7fbff;
    border-color: #102139;
    border-right: 5px solid var(--yellow);
}

.source-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 13px;
    padding-top: 12px;
    border-top: 1px solid var(--line);
}

.source-strip span:first-child {
    color: var(--soft);
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
}

.source-chip {
    font-family: 'JetBrains Mono', monospace;
    color: #075985;
    background: #e8f7ff;
    border: 1px solid #bde8ff;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 4px 9px;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}

/* Input */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    background: rgba(255,255,255,0.92) !important;
    border-top: 1px solid var(--line) !important;
    backdrop-filter: blur(16px);
}

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.92) !important;
    border-top: 0 !important;
    padding: 14px 32px !important;
    backdrop-filter: blur(16px);
}

[data-testid="stChatInput"] > div {
    width: min(1040px, 100%) !important;
    margin: 0 auto !important;
    background: #fff !important;
    border: 1.5px solid var(--line-strong) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.16s ease, box-shadow 0.16s ease !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 4px rgba(18,150,219,0.15), var(--shadow-sm) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    background: #fff !important;
    caret-color: var(--blue) !important;
    font-size: 0.98rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--soft) !important;
}

[data-testid="stChatInput"] button {
    background: var(--blue) !important;
    border: 0 !important;
    border-radius: 8px !important;
    transition: background 0.16s ease, transform 0.16s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background: var(--blue-strong) !important;
    transform: translateY(-1px);
}

[data-testid="stChatInput"] button svg {
    fill: white !important;
}

.stSpinner > div {
    border-top-color: var(--blue) !important;
}

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #c8d2e1;
    border-radius: 999px;
}

@media (max-width: 1100px) {
    .topbar {
        align-items: flex-start;
        padding: 13px 16px;
    }

    .topbar .brand-logo--wordmark {
        width: 136px !important;
        height: 44px !important;
    }

    .topbar__subtitle,
    .doc-pill {
        display: none;
    }

    .chat-stage {
        width: min(100%, calc(100vw - 24px));
        padding-top: 14px;
    }

    .hero {
        grid-template-columns: 1fr;
        gap: 12px;
    }

    .lab-panel {
        display: none;
    }

    .hero-main {
        padding: 22px;
    }

    .brand-logo--hero {
        width: 250px !important;
        height: 118px !important;
    }

    .hero-copy {
        font-size: 0.94rem;
    }

    .bubble {
        max-width: calc(100vw - 92px);
        font-size: 0.93rem;
    }
}

@media (max-width: 520px) {
    .topbar__title {
        white-space: normal;
        line-height: 1.1;
    }

    .status-pill {
        padding-inline: 10px;
    }

    .topbar {
        gap: 10px;
    }

    .topbar__left {
        gap: 9px;
    }

    .topbar .brand-logo--wordmark {
        width: 112px !important;
        height: 38px !important;
    }

    .hero-title {
        font-size: 2.2rem;
    }

    .brand-logo--hero {
        width: 210px !important;
        height: 96px !important;
    }

    .hero-copy {
        margin-top: 12px;
    }

    .hero-stripe {
        margin-top: 18px;
    }

    .lab-panel__footer {
        grid-template-columns: 1fr;
    }

    .avatar {
        width: 36px;
        height: 36px;
        flex-basis: 36px;
    }

    .message-row {
        gap: 8px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# -- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            {logo_html(190, 76, "wordmark")}
            <div class="brand-subtitle">Science answers from your sources</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    asked_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
    pdf_count = document_count()
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <strong>{asked_count}</strong>
                <span>Questions</span>
            </div>
            <div class="metric-card">
                <strong>{pdf_count}</strong>
                <span>PDFs</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eyebrow">Try Asking</div>', unsafe_allow_html=True)
    for question_text in SAMPLE_QUESTIONS:
        st.button(
            question_text,
            key=f"sample_{question_text}",
            on_click=set_question,
            args=(question_text,),
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    fact = SCIENCE_FACTS[st.session_state.fact_index % len(SCIENCE_FACTS)]
    st.markdown(
        f"""
        <div class="fact-panel">
            <div class="fact-panel__label">Science Fact</div>
            <div class="fact-panel__text">{clean_html(fact)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("New fact", key="new_fact"):
        st.session_state.fact_index += 1
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if st.button("Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.prefill = ""
        st.rerun()


# -- Main UI ----------------------------------------------------------------
status_text, status_state = connection_status()
st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar__left">
            {logo_html(164, 52, "wordmark")}
            <div>
                <div class="topbar__title">Science Assistant</div>
                <div class="topbar__subtitle">Grounded science chat for curious minds</div>
            </div>
        </div>
        <div class="topbar__right">
            <div class="doc-pill">{document_count()} docs loaded</div>
            <div class="status-pill status-pill--{status_state}">{status_text}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<section class="chat-stage">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-main">
                <div class="hero-kicker">Source-backed science assistant</div>
                <div class="hero-brand">{logo_html(330, 160, "hero")}</div>
                <div class="hero-title">Ask science. Get the <span>Snap</span>.</div>
                <div class="hero-copy">
                    Explore biology, chemistry, physics, and space through answers grounded in your own science PDFs.
                </div>
                <div class="hero-stripe" aria-hidden="true">
                    <span></span><span></span><span></span>
                </div>
            </div>
            <aside class="lab-panel">
                <div class="lab-panel__top">
                    <div class="lab-panel__title">Research Mode</div>
                    {logo_html(42, 42, "mark")}
                </div>
                <div class="molecule" aria-hidden="true">
                    <span></span><span></span><span></span><span></span><span></span>
                    <span></span><span></span><span></span><span></span><span></span>
                    <span></span><span></span><span></span><span></span><span></span>
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
                <div class="lab-panel__footer">
                    <div class="lab-chip"><strong>RAG</strong><span>retrieval</span></div>
                    <div class="lab-chip"><strong>PDF</strong><span>sources</span></div>
                    <div class="lab-chip"><strong>AI</strong><span>answers</span></div>
                </div>
            </aside>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<section class="messages">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg.get("role", "")
    content = clean_html(msg.get("content", ""))

    if role == "user":
        st.markdown(
            (
                '<div class="message-row message-row--user">'
                f'<div class="bubble bubble--user">{content}</div>'
                '<div class="avatar avatar--user">You</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        continue

    sources = msg.get("sources", "")
    sources_html = ""
    if sources:
        sources_html = (
            '<div class="source-strip">'
            "<span>Source</span>"
            f'<span class="source-chip">{source_label(sources)}</span>'
            "</div>"
        )

    st.markdown(
        (
            '<div class="message-row">'
            f'<div class="avatar">{logo_html(42, 42, "mark")}</div>'
            f'<div class="bubble bubble--assistant">{content}{sources_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
st.markdown("</section>", unsafe_allow_html=True)
st.markdown("</section>", unsafe_allow_html=True)


# -- Input ------------------------------------------------------------------
question = st.chat_input("Ask a science question...")
if st.session_state.prefill and not question:
    question = st.session_state.prefill
    st.session_state.prefill = ""

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Searching your science sources..."):
        answer, sources_str = answer_question(question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_str,
        }
    )
    st.rerun()

