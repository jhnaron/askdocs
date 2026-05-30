from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

DOCS_DIR = Path("docs")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "askdocs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embedding_fn():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def split_into_chunks(text: str) -> list[str]:
    # Overlap keeps us from cutting a sentence right at a chunk boundary
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index():
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}.")

    embedding_fn = get_embedding_fn()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop the existing collection so re-runs always produce a fresh index
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    ids = []
    chunks = []
    metadata = []

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        text = read_pdf(pdf_path)
        file_chunks = split_into_chunks(text)

        for i, chunk in enumerate(file_chunks):
            ids.append(f"{pdf_path.stem}_{i}")
            chunks.append(chunk)
            metadata.append({"source": pdf_path.name, "chunk_index": i})

    collection.add(documents=chunks, ids=ids, metadatas=metadata)
    print(f"Done. Indexed {len(chunks)} chunks from {len(pdf_files)} file(s).")
    return collection


def load_index():
    embedding_fn = get_embedding_fn()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


def get_or_build_index():
    # Use the existing index if it's there, otherwise build it from the docs folder
    try:
        return load_index()
    except Exception:
        print("No index found, building from docs...")
        return build_index()
