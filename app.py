import os
import time
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
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
]

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciSnapster AI",
    page_icon="🔭",
    layout="centered"
)

st.markdown("""
<style>
.big-title { font-size: 2.4rem; font-weight: 700; text-align: center; margin-bottom: 0; }
.subtitle  { font-size: 1rem; text-align: center; color: gray; margin-bottom: 1.5rem; }
.source-box { font-size: 0.75rem; color: gray; margin-top: 0.5rem; padding: 0.4rem 0.75rem;
              background: #f5f5f5; border-radius: 8px; border-left: 3px solid #4A90D9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🔭 SciSnapster AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about science — powered by real documents</div>', unsafe_allow_html=True)

# ── Auto-index if no DB exists ───────────────────────────────────────────────
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not Path(CHROMA_FOLDER).exists():
        pdfs = list(Path(DOCS_FOLDER).glob("*.pdf"))
        if not pdfs:
            st.error("❌ No PDFs found in docs/ folder!")
            st.stop()

        with st.spinner(f"📄 Indexing {len(pdfs)} document(s) for the first time... this takes a few minutes!"):
            docs = []
            for pdf in pdfs:
                loader = PyPDFLoader(str(pdf))
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_FOLDER)
        st.success("✅ Documents indexed! Ask away.")
    else:
        db = Chroma(persist_directory=CHROMA_FOLDER, embedding_function=embeddings)

    return db

db = load_db()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("SciSnapster AI answers science questions using real textbooks and research documents — no hallucinations.")
    st.markdown("Built by **SciSnapster** 🎬")

# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.markdown(f'<div class="source-box">📄 Sources: {msg["sources"]}</div>', unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", "")
question = st.chat_input("Ask a science question...")

if prefill and not question:
    question = prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            results  = db.similarity_search(question, k=4)
            context  = "\n\n".join([r.page_content for r in results])
            sources  = list(set([Path(r.metadata.get("source", "unknown")).name for r in results]))

            prompt = f"""You are SciSnapster AI — an enthusiastic, energetic science assistant for the SciSnapster YouTube channel.
Answer the question based ONLY on the provided science documents.
If the answer is not in the documents, say "This topic isn't covered in my current documents, but I'm always learning!"
Be exciting and engaging — make science feel awesome!
Always mention which source or section the answer comes from.
Keep answers clear and easy to understand.

Science documents context:
{context}

Question: {question}

Answer:"""

            client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
            answer = "⚠️ AI is busy right now. Please try again in a moment!"

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

        st.markdown(answer)
        sources_str = ", ".join(sources)
        st.markdown(f'<div class="source-box">📄 Sources: {sources_str}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_str
    })
