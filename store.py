import shutil
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import get_embedding_model

PERSIST_DIR = Path("data") / "chroma"

COLLECTION_NAME = "worldcup_chunks"

_ALLOWED_META_TYPES = (str, int, float, bool)


def _sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, _ALLOWED_META_TYPES):
            clean[key] = value
        elif isinstance(value, list):
            clean[key] = ", ".join(map(str, value))
        else:
            clean[key] = str(value)
    return clean


def _to_documents(documents: Iterable[Document]) -> tuple[list[Document], list[str]]:
    clean_docs: list[Document] = []
    ids: list[str] = []
    for doc in documents:
        match_id = doc.metadata["match_id"]
        clean_meta = _sanitize_metadata(doc.metadata)
        clean_docs.append(
            Document(page_content=doc.page_content, metadata=clean_meta))
        ids.append(str(match_id))
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dupes: set[str] = set()
        for i in ids:
            (dupes if i in seen else seen).add(i)
        raise ValueError(
            f"Duplicate match_id(s): {sorted(dupes)}. IDs must be unique.")

    return clean_docs, ids


def build_store(documents: Iterable[Document], *, force: bool = False) -> Chroma:
    if force and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    docs, ids = _to_documents(documents)

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embedding_model(),
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_store() -> Chroma:
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"No store at {PERSIST_DIR}. Run build_store(documents, force=True) first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(PERSIST_DIR),
    )


if __name__ == "__main__":
    from chunk import records_to_documents
    from clean import clean_matches
    from fetch import fetch_worldcup
    records = clean_matches(fetch_worldcup(2026))

    documents = records_to_documents(records)
    print(f"prepared {len(documents)} documents")

    store = build_store(documents, force=True)
    print(f"built and persisted to {PERSIST_DIR}")

    store = load_store()
    for i, doc in enumerate(store.similarity_search("Mexico vs South Africa", k=3), 1):
        print(f"\n--- result {i} ---")
        print(doc.page_content[:200])
        print(doc.metadata)
