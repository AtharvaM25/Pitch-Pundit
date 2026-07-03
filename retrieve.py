from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from store import load_store

VALID_SEARCH_TYPES = {"similarity", "mmr", "similarity_score_threshold"}

DEFAULT_K = 5
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5


def _normalize_filter(metadata_filter: dict | None) -> dict | None:
    if not metadata_filter:
        return None
    if len(metadata_filter) == 1:
        return metadata_filter
    return {"$and": [{key: value} for key, value in metadata_filter.items()]}


def get_retriever(*, search_type: str = "similarity", k: int = DEFAULT_K, metadata_filter: dict | None = None, score_threshold: float | None = None, fetch_k: int = DEFAULT_FETCH_K, lambda_mult: float = DEFAULT_LAMBDA_MULT, store: Chroma | None = None,) -> VectorStoreRetriever:

    if search_type not in VALID_SEARCH_TYPES:
        raise ValueError(
            f"Unknown search_type {search_type!r}. "
            f"Choose one of {sorted(VALID_SEARCH_TYPES)}."
        )

    if store is None:
        store = load_store()

    search_kwargs: dict = {"k": k}

    normalized = _normalize_filter(metadata_filter)
    if normalized is not None:
        search_kwargs["filter"] = normalized

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult
    elif search_type == "similarity_score_threshold":
        if score_threshold is None:
            raise ValueError(
                "search_type='similarity_score_threshold' requires "
                "score_threshold. This is the OFF-TOPIC perimeter guard: "
                "it drops genuinely unrelated queries, not on-topic-but-absent "
                "ones (those are handled by the Component 4 grounding prompt)."
            )
        search_kwargs["score_threshold"] = score_threshold

    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def _print_results(label: str, docs: list) -> None:
    print(f"\n=== {label} ===")
    print(f"returned {len(docs)} document(s)")
    for i, doc in enumerate(docs, 1):
        snippet = doc.page_content[:120].strip().replace("\n", " ")
        print(f"  [{i}] {snippet}")
        print(f"      meta: {doc.metadata}")


if __name__ == "__main__":
    store = load_store()
    print("=== sample metadata (your filterable vocabulary) ===")
    for doc in store.similarity_search("match", k=3):
        print(f"  {doc.metadata}")

    r = get_retriever(search_type="similarity", k=4, store=store)
    _print_results("similarity: 'Who won the final?'",
                   r.invoke("Who won the final?"))

    r = get_retriever(search_type="mmr", k=4, fetch_k=20,
                      lambda_mult=0.7, store=store)
    _print_results("mmr: 'Tell me about Argentina at the tournament'",
                   r.invoke("Tell me about Argentina at the tournament"))

    r = get_retriever(search_type="similarity", k=6,
                      metadata_filter={"phase": "group"}, store=store)
    _print_results("filtered to phase='group'",
                   r.invoke("How did the group stage unfold?"))

    r = get_retriever(search_type="similarity", k=6,
                      metadata_filter={"phase": "knockout"}, store=store)
    _print_results("filtered to phase='knockout'",
                   r.invoke("How did the knockout rounds unfold?"))

    print("\n=== relevance scores for 'Who won the final?' ===")
    for doc, score in store.similarity_search_with_relevance_scores(
        "Who won the final?", k=5
    ):
        print(f"  {score:.3f}  {doc.page_content[:80].strip()}")

    r = get_retriever(search_type="similarity_score_threshold",
                      k=4, score_threshold=0.5, store=store)
    _print_results("threshold (answerable) -> expect docs",
                   r.invoke("Who played in the opening match?"))
    _print_results("threshold (on-topic but ABSENT) -> still returns docs",
                   r.invoke("What did the referee say in his post-match interview?"))
    _print_results("threshold (OFF-topic) -> expect ZERO docs",
                   r.invoke("What is the capital of France?"))
