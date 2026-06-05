import argparse
import os
import shutil
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DOCS_FOLDER = "docs"
CHROMA_FOLDER = "chroma_db"
DOCS_DIR = BASE_DIR / DOCS_FOLDER
CHROMA_DIR = BASE_DIR / CHROMA_FOLDER
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"

MODEL_CHOICES = [
    "openrouter/auto",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


def get_openrouter_key() -> str:
    return os.environ.get(OPENROUTER_ENV_VAR, "").strip()


def load_dependencies(include_openai: bool = False):
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pypdf import PdfReader
    except ImportError as exc:
        print(f"Missing package: {exc}")
        print("Install everything with: python -m pip install -r requirements.txt")
        raise SystemExit(1)

    openai_client = None
    if include_openai:
        try:
            from openai import OpenAI
        except ImportError as exc:
            print(f"Missing OpenAI client package: {exc}")
            print("Install everything with: python -m pip install -r requirements.txt")
            raise SystemExit(1)
        openai_client = OpenAI

    return PdfReader, Document, HuggingFaceEmbeddings, Chroma, RecursiveCharacterTextSplitter, openai_client


def pdf_files() -> list[Path]:
    return sorted(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []


def source_names(results) -> str:
    raw_sources = {
        result.metadata.get("source", result.metadata.get("file_path", ""))
        for result in results
    }
    names = [Path(source).name if source else "Science Document" for source in raw_sources]
    return ", ".join(sorted(names))


def safe_remove_chroma_dir() -> None:
    target = CHROMA_DIR.resolve()
    root = BASE_DIR.resolve()
    if root not in target.parents and target != root:
        raise RuntimeError(f"Refusing to delete outside the project: {target}")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def print_status() -> None:
    pdfs = pdf_files()
    print("SciSnapster AI status")
    print(f"  Project: {BASE_DIR}")
    print(f"  PDFs: {len(pdfs)} in {DOCS_DIR}")
    for pdf in pdfs:
        print(f"    - {pdf.name}")
    print(f"  Vector DB: {'ready' if CHROMA_DIR.exists() else 'not built'} at {CHROMA_DIR}")
    print(f"  OpenRouter: {'connected' if get_openrouter_key() else 'missing OPENROUTER_API_KEY'}")


def index_documents(rebuild: bool = False) -> None:
    PdfReader, Document, HuggingFaceEmbeddings, Chroma, RecursiveCharacterTextSplitter, _ = load_dependencies()

    pdfs = pdf_files()
    if not pdfs:
        print(f"No PDFs found. Add files to {DOCS_DIR} first.")
        return

    if rebuild:
        print("Rebuilding vector database...")
        safe_remove_chroma_dir()
    elif CHROMA_DIR.exists():
        print(f"Vector database already exists at {CHROMA_DIR}")
        print("Use --rebuild if you want to recreate it from the PDFs.")
        return

    print(f"Loading {len(pdfs)} PDF file(s)...")
    docs = []
    for pdf_path in pdfs:
        print(f"  Reading: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(pdf_path), "page": page_number},
                    )
                )

    if not docs:
        print("No pages were loaded from the PDFs.")
        return

    print(f"Loaded {len(docs)} pages.")
    print("Splitting pages into searchable chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=90)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings. First run can take a few minutes.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DIR))
    print(f"Done. Database saved to {CHROMA_DIR}")


def answer_question(question: str) -> None:
    if not CHROMA_DIR.exists():
        print("The vector database is not built yet.")
        print("Run: python rag.py --index")
        return

    include_openai = bool(get_openrouter_key())
    _, _, HuggingFaceEmbeddings, Chroma, _, OpenAI = load_dependencies(include_openai=include_openai)

    print(f"Searching: {question}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    results = db.similarity_search(question, k=4)

    if not results:
        print("No matching sources found.")
        return

    context = "\n\n".join(result.page_content for result in results)[:12000]
    sources = source_names(results)

    if not get_openrouter_key():
        print("Found relevant sources, but OpenRouter is not connected.")
        print(f"Set your key: $env:{OPENROUTER_ENV_VAR}='your_key'")
        print(f"Sources: {sources}")
        return

    prompt = f"""You are SciSnapster AI, an enthusiastic science assistant for the SciSnapster YouTube channel.
Answer based ONLY on the provided documents.
If the answer is not found, say: "This topic isn't in my current documents yet, but great question!"
Keep the answer clear, exciting, and accurate. Mention which document the answer comes from.

Documents:
{context}

Question: {question}
Answer:"""

    client = OpenAI(
        api_key=get_openrouter_key(),
        base_url="https://openrouter.ai/api/v1",
        timeout=45.0,
    )

    for attempt, model in enumerate(MODEL_CHOICES):
        try:
            print(f"Trying model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            print("\nAnswer:")
            print(response.choices[0].message.content)
            print(f"\nSources: {sources}")
            return
        except Exception as exc:
            print(f"Model failed: {exc}")
            if attempt < len(MODEL_CHOICES) - 1:
                time.sleep(3)

    print("All configured models failed. Try again in a moment.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SciSnapster AI terminal helper")
    parser.add_argument("--status", action="store_true", help="Show project, PDF, DB, and API-key status")
    parser.add_argument("--index", action="store_true", help="Create the vector database from docs/*.pdf")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the vector database")
    parser.add_argument("--query", metavar="QUESTION", help="Ask a question from the indexed PDFs")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.status:
        print_status()
        return 0

    if args.index or args.rebuild:
        index_documents(rebuild=args.rebuild)
        return 0

    if args.query:
        answer_question(args.query)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
