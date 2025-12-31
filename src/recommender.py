"""
recommender.py

Core recommendation logic for book suggestions based on:
- Language filtering
- Fuzzy title matching (full-text search)
- Vector similarity (semantic search)
- Graph-based similarity (shared genres/authors)

Designed for Streamlit, APIs, and batch usage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Iterable, Optional, Any

from neo4j import GraphDatabase, Transaction
from sentence_transformers import SentenceTransformer

# from src.configs.config import Config
from configs.config import Config


# ------------------------------------------------------------------
# Configuration & shared resources
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()


@lru_cache(maxsize=1)
def get_driver():
    cfg = get_config()
    return GraphDatabase.driver(cfg.neo4j_uri, auth=cfg.neo4j_auth)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    cfg = get_config()
    return SentenceTransformer(cfg.embedding_model)


# ------------------------------------------------------------------
# Language mapping (UI-facing convenience)
# ------------------------------------------------------------------

LANGUAGE_MAP: Dict[str, List[str]] = {
    "English": ["eng", "en-us", "en-gb", "en-ca", "enm"],
    "Spanish": ["spa", "glg"],
    "French": ["fre"],
    "German": ["ger"],
    "Japanese": ["jpn"],
    "Chinese": ["zho"],
    "Greek": ["grc"],
    "Italian": ["ita"],
    "Portuguese": ["por"],
    "Russian": ["rus"],
    "Swedish": ["swe"],
    "Arabic": ["ara"],
    "Dutch": ["nl"],
    "Malay": ["msa"],
    "Norwegian": ["nor"],
    "Latin": ["lat"],
    "Welsh": ["wel"],
}


# ------------------------------------------------------------------
# Neo4j query helpers
# ------------------------------------------------------------------

def get_books_by_language(tx: Transaction, language_codes: List[str]) -> List[Any]:
    q = """
    MATCH (b:Book)
    WHERE b.language_code IN $codes
    RETURN b.bookID AS id
    """
    return [r["id"] for r in tx.run(q, codes=language_codes)]


def get_book_by_title(tx: Transaction, title: str, allowed_ids: Iterable) -> Optional[Dict[str, Any]]:
    q = """
    MATCH (b:Book {title: $title})
    WHERE b.bookID IN $allowed_ids
    RETURN b.title AS title, b.description AS description
    LIMIT 1
    """
    return tx.run(q, title=title, allowed_ids=allowed_ids).single()


def find_closest_title_fulltext(
    tx: Transaction,
    query: str,
    allowed_ids: Iterable,
) -> Optional[str]:
    q = """
    CALL db.index.fulltext.queryNodes('book_title_fulltext', $q)
    YIELD node, score
    WHERE node.bookID IN $allowed_ids
    RETURN node.title AS title
    ORDER BY score DESC
    LIMIT 1
    """
    result = tx.run(q, q=f"{query}~3", allowed_ids=allowed_ids).single()
    return result["title"] if result else None


def vector_search(
    tx: Transaction,
    embedding: List[float],
    allowed_ids: Iterable,
    k: int,
    search_k: int = 1000,
) -> List[str]:
    q = """
    CALL db.index.vector.queryNodes(
        'book_embedding_index',
        $search_k,
        $embedding
    )
    YIELD node, score
    WHERE node.bookID IN $allowed_ids
    RETURN node.title AS title, score
    ORDER BY score DESC
    LIMIT $k
    """
    return [dict(r) for r in tx.run(
        q,
        embedding=embedding,
        allowed_ids=allowed_ids,
        k=k,
        search_k=search_k,
    )]


def graph_recommend(
    tx: Transaction,
    title: str,
    allowed_ids: Iterable,
    k: int,
) -> List[str]:
    q = """
    MATCH (b:Book {title: $title})
    WHERE b.bookID IN $allowed_ids

    OPTIONAL MATCH (b)-[:HAS_GENRE]->(g)<-[:HAS_GENRE]-(o:Book)
    WHERE o.bookID IN $allowed_ids

    OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a)<-[:WRITTEN_BY]-(o2:Book)
    WHERE o2.bookID IN $allowed_ids

    WITH collect(DISTINCT o) + collect(DISTINCT o2) AS recs
    UNWIND recs AS r
    WITH r.title AS title, count(*) AS score
    RETURN title, score
    ORDER BY score DESC
    LIMIT $k
    """
    return [dict(r) for r in tx.run(q, title=title, allowed_ids=allowed_ids, k=k)]

def normalize(scores):
    if not scores:
        return scores
    max_s = max(s["score"] for s in scores)
    return [
        {**s, "score": s["score"] / max_s if max_s else 0.0}
        for s in scores
    ]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def recommend_books(
    book_query: str,
    language_codes: List[str],
    top_k: Optional[int] = None,
) -> List[str]:
    """
    Main recommendation entry point.

    Args:
        book_query: User-provided title (exact or fuzzy).
        language_codes: List of language_code values to filter by.
        top_k: Optional override for number of recommendations.

    Returns:
        Ordered list of recommended book titles.
    """
    cfg = get_config()
    k = top_k or cfg.top_k

    driver = get_driver()
    model = get_embedding_model()

    with driver.session() as session:
        allowed_ids = session.execute_read(get_books_by_language, language_codes)
        if not allowed_ids:
            return []

        # Exact match
        book = session.execute_read(get_book_by_title, book_query, allowed_ids)
        if book:
            anchor_title = book_query
        else:
            # Fuzzy title match
            closest_title = session.execute_read(
                find_closest_title_fulltext,
                book_query,
                allowed_ids,
            )
            anchor_title = closest_title
            if not closest_title:
                return []

        book = session.execute_read(get_book_by_title, anchor_title, allowed_ids)
        if not book:
            return []

        full_text = f"{book['title']} {book['description'] or ''}".strip()
        embedding = model.encode(
            full_text,
            normalize_embeddings=cfg.embedding_normalize,
        ).tolist()

        semantic = session.execute_read(
            vector_search,
            embedding,
            allowed_ids,
            k + 1,
        )
        semantic = [
            {**r, "source": "semantic"}
            for r in semantic
            if r["title"] != anchor_title
        ]

        graph = session.execute_read(
            graph_recommend,
            anchor_title,
            allowed_ids,
            k,
        )
        graph = [{**r, "source": "graph"} for r in graph]

        semantic = normalize(semantic)
        graph = normalize(graph)
        
        # Combine scores
        WEIGHTS = {"semantic": 0.6, "graph": 0.4}

        merged = {}

        for r in semantic + graph:
            title = r["title"]
            weighted_score = r["score"] * WEIGHTS[r["source"]]

            if title not in merged:
                merged[title] = {
                    "title": title,
                    "score": weighted_score,
                    "sources": {r["source"]},
                }
            else:
                merged[title]["score"] += weighted_score
                merged[title]["sources"].add(r["source"])

            final = sorted(
                merged.values(),
                key=lambda x: x["score"],
                reverse=True,
            )[:k]
        
        return final