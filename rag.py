import os
import sys
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_FOLDER   = "docs"
CHROMA_FOLDER = "chroma_db"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

FREE_MODELS = [
    "openrouter/auto",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-1b-it:free",
    "qwen/qwen-2.5-7b-instruct:free",
]

# ── Step 1: Load and index documents ───────────────────────────────────────────
def index_documents():
    print("📄 Loading PDFs from docs/ folder...")
    docs = []
    for pdf_path in Path(DOCS_FOLDER).glob("*.pdf"):
        print(f"   → Reading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    if not docs:
        print("❌ No PDFs found in docs/ folder. Add your PDFs and try again.")
        return

    print(f"✅ Loaded {len(docs)} pages total.")

    print("✂️  Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    print("🔢 Creating embeddings (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("💾 Saving to local database...")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_FOLDER)
    print("✅ Done! Database saved to chroma_db/ folder.")

# ── Step 2: Answer a question ───────────────────────────────────────────────────
def answer_question(question):
    if not OPENROUTER_KEY:
        print("❌ OPENROUTER_API_KEY is not set. Run: $env:OPENROUTER_API_KEY='your_key'")
        return

    print(f"\n🔍 Searching for: {question}\n")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_FOLDER, embedding_function=embeddings)

    results = db.similarity_search(question, k=4)

    if not results:
        print("❌ No relevant content found in the documents.")
        return

    context = "\n\n".join([r.page_content for r in results])
    sources = list(set([r.metadata.get("source", "unknown") for r in results]))

    prompt = f"""You are a legal assistant for Uzbekistan law. 
Answer the question based ONLY on the provided legal documents.
If the answer is not in the documents, say "This information is not found in the provided documents."
Always mention which article or section the answer comes from if possible.
Answer in the same language as the question.

Legal documents context:
{context}

Question: {question}

Answer:"""

    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    
    for model in FREE_MODELS:
        print(f"⏳ Trying model: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            print(f"✅ answered by: {model}\n")
            print("💬 Answer:")
            print(answer)
            print(f"\n📚 Sources: {', '.join(sources)}")
            return
        except Exception as e:
            print(f"⚠️  Model {model} failed: {str(e)[:80]}. Trying next...")
            time.sleep(3)

    print("❌ All models are currently rate limited. Please wait 1 minute and try again.")

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag.py --index              (index your documents)")
        print("  python rag.py --query 'question'   (ask a question)")
    elif sys.argv[1] == "--index":
        index_documents()
    elif sys.argv[1] == "--query" and len(sys.argv) >= 3:
        answer_question(sys.argv[2])
    else:
        print("❌ Invalid command. Use --index or --query 'your question'")
