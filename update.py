from fetch import fetch_worldcup
from clean import clean_matches
from chunk import records_to_documents
from store import build_store

YEAR = 2026


def update_store():
    raw = fetch_worldcup(YEAR, force=True)
    records = clean_matches(raw)
    documents = records_to_documents(records)
    build_store(documents, force=True)
    return len(documents)


if __name__ == "__main__":
    print(f"Refreshing World Cup {YEAR} data and rebuilding the store…")
    count = update_store()
    print(f"Done. Rebuilt the store with {count} match documents.")
