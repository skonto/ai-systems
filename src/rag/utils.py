from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

def build_langchain_bm25_retriever(docs: list, k: int = 5) -> Optional[BM25Retriever]:
    """
    Constructs a LangChain-compatible BM25 retriever from Chroma documents.

    Args:
        docs: List of dicts with 'id', 'document', and optionally 'metadata'.
        k: Number of documents to retrieve by default.

    Returns:
        A BM25Retriever instance.
    """

    if len(docs) is 0:
        return None

    lc_docs = [
        Document(
            page_content=doc["document"],
            metadata={**(doc.get("metadata") or {}), "id": doc["id"]},
        )
        for doc in docs
        if doc["document"]  # skip if empty
    ]

    retriever = BM25Retriever.from_documents(lc_docs)
    retriever.k = k
    return retriever


def fuse_results(bm25_results: list[Tuple[str, float]], embedding_results: List[Tuple[Document, float]], alpha: float=0.5) -> List[Tuple[Document, float]]:
    """
    Combine BM25 and embedding results using linear weighted score fusion.

    Returns:
        List of (doc_id, fused_score), sorted descending.
    """

    def normalize(results: list[Tuple[str, float]]) ->  Dict:
        if not results:
            return {}
        doc_ids, scores = zip(*results)
        scores_r = np.array(scores).reshape(-1, 1)
        norm = MinMaxScaler().fit_transform(scores_r).flatten()
        return dict(zip(doc_ids, norm))

    norm_bm25 = normalize(bm25_results)
    norm_embed = normalize(
        [(get_doc_id(doc), 1.0 - dist) for doc, dist in embedding_results]
    )

    all_ids = set(norm_bm25) | set(norm_embed)
    fused = defaultdict(float)

    for doc_id in all_ids:
        fused[doc_id] = alpha * norm_bm25.get(doc_id, 0.0) + (
            1 - alpha
        ) * norm_embed.get(doc_id, 0.0)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def fuse_with_bm25(
    embedding_results: List[Tuple[Document, float]],
    bm25_retriever: BM25Retriever,
    query: str,
    alpha: float = 0.2,
    intersection_only: bool = False,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Fuses embedding results with BM25-based relevance scores.

    Args:
        embedding_results: List of (Document, embedding_score)
        bm25_retriever: LangChain BM25Retriever instance
        query: The search query
        alpha: Weight for BM25 scores in fusion
        intersection_only: If True, only docs in embedding_results are considered.
                           If False, union of embedding and BM25 docs is used.

    Returns:
        List of (Document, fused_score), sorted by score descending.
    """
    # Step 1: Map embedding results by doc ID
    embed_docs = {
        get_doc_id(doc): (doc, emb_score) for doc, emb_score in embedding_results
    }
    logger.debug("Printing embed docs")
    for doc, emb_score in embedding_results:
        print(doc)
        print(emb_score)
    # Step 2: Get BM25 docs
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    logger.debug("Printing bm25 docs")
    for doc in bm25_docs:
        print(doc)
    bm25_map = {get_doc_id(doc): doc for doc in bm25_docs}
    bm25_scores = {doc_id: 1.0 for doc_id in bm25_map}

    # Optional: normalize BM25 scores
    if bm25_scores:
        max_score = max(bm25_scores.values())
        bm25_scores = {
            doc_id: score / max_score for doc_id, score in bm25_scores.items()
        }

    # Step 3: Build final doc_id set
    if intersection_only:
        doc_ids = set(embed_docs.keys()) & set(bm25_scores.keys())
    else:
        doc_ids = set(embed_docs.keys()) | set(bm25_scores.keys())

    fused = []
    logger.debug("Printing docs to be fused")
    for doc_id in doc_ids:
        emb_score = 0.0
        doc_to_fuse: Optional[Document] = None
        if doc_id in embed_docs:
            doc_to_fuse, emb_score = embed_docs[doc_id]
        else:
            doc_to_fuse = bm25_map.get(doc_id)
            emb_score = 0.0
            doc_to_fuse= (doc_to_fuse )

        # Sanity check: skip if we failed to get a doc
        if doc_to_fuse is None:
            continue

        bm25_score = bm25_scores.get(doc_id, 0.0)
        fused_score = 0.0
        if bm25_score is None:
            fused_score = emb_score * 1.05
        else:
            fused_score = alpha * bm25_score + emb_score
        logger.debug(doc_to_fuse)
        logger.debug(fused_score)
        fused.append((doc_to_fuse, fused_score))

    # Sort descending by score
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def merge_fused_results_into_context(
    fused_results: List[Tuple[str, float]], all_documents: List[Document], max_tokens: int = 0) -> str:
    """
    Merges fused results into a single context string.

    Args:
        fused_results: List of (doc_id, score)
        all_documents: List[Document]
        max_tokens: Optional maximum tokens/words to limit final context

    Returns:
        A single concatenated context string.
    """
    # Create a mapping from doc_id to content
    doc_map = {doc.metadata.get("id"): doc.page_content for doc in all_documents}

    context = ""
    for doc_id, _ in fused_results:
        content = doc_map.get(doc_id)
        if content:
            context += content.strip() + "\n"

            if max_tokens and len(context.split()) > max_tokens:
                break

    return context.strip()


def get_doc_id(doc: Document) -> str:
    return doc.metadata.get("id") or f"hash:{hash(doc.page_content[:100])}"
