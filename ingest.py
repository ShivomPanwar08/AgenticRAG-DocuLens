import os
import re
import uuid
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
load_dotenv()

DATA_PATH = Path("data/Attendance Policy 2025.pdf")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "doculens_docs"

def clean_extracted_text(text: str) -> str:
    """Normalizes whitespace and broken PDF words."""
    cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()
    return cleaned

def get_chroma_client():
    """Initializes persistent native Chroma client."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)

def ingest_pdf_file(file_path: str, display_name: str = None):
    """Processes any PDF safely on Windows without throwing WinError 32."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_label = display_name or path_obj.name
    reader = PdfReader(str(path_obj))
    raw_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = clean_extracted_text(text)
        if cleaned:
            raw_pages.append(
                Document(
                    page_content=cleaned,
                    metadata={"source": doc_label, "page": page_num}
                )
            )

    if not raw_pages:
        raise ValueError("Could not extract any readable text from this PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(raw_pages)

    # Windows-safe: clear collection via client rather than deleting folder on disk
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client
    )
    return len(raw_pages), len(chunks), vector_store

def build_or_get_vector_store():
    """Returns vector store pointing to persistent Chroma directory."""
    client = get_chroma_client()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client
    )

def reset_to_default_doc():
    """Restores baseline Attendance Policy 2025.pdf."""
    return ingest_pdf_file(str(DATA_PATH), display_name="Attendance Policy 2025.pdf")

if __name__ == "__main__":
    pages, chunks, _ = reset_to_default_doc()
    print(f"Attendance policy indexed successfully: {pages} pages, {chunks} chunks.")