# store_embeddings.py
from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from configs.config import Config
from typing import Any, Dict, List
from neo4j import GraphDatabase, Transaction

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    cfg = Config()
    return SentenceTransformer(cfg.embedding_model)

def get_embedding_dimension() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()

# -----------------------------
# Neo4j I/O
# -----------------------------
def fetch_books_batch(tx: Transaction, skip: int, limit: int) -> List[Dict[str, Any]]:
    """
    Fetch a deterministic batch of books.
    We fetch title/description regardless; we decide what to embed in Python.
    """
    q = """
    MATCH (b:Book)
    RETURN b.bookID AS id,
           b.title AS title,
           b.description AS description
    ORDER BY b.bookID
    SKIP $skip
    LIMIT $limit
    """
    return [dict(r) for r in tx.run(q, skip=skip, limit=limit)]


def write_embeddings_batch(tx: Transaction, rows: List[Dict[str, Any]]) -> None:
    """
    rows: list of dicts:
      {
        "id": ...,
        "embedding": [...]/None,
        "title_embedding": [...]/None,
        "embedding_model": str,
        "embedding_dim": int
      }
    """
    q = """
    UNWIND $rows AS row
    MATCH (b:Book {bookID: row.id})
    SET
      b.embedding_model = row.embedding_model,
      b.embedding_dim = row.embedding_dim,
      b.embedding_updated_at = datetime()
    FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END |
      SET b.embedding = row.embedding
    )
    FOREACH (_ IN CASE WHEN row.title_embedding IS NULL THEN [] ELSE [1] END |
      SET b.title_embedding = row.title_embedding
    )
    """
    tx.run(q, rows=rows)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = Config()

    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=cfg.neo4j_auth)
    model = get_embedding_model()
    dim = get_embedding_dimension()

    write_text = cfg.write_text_embedding
    write_title = cfg.write_title_embedding

    if not write_text and not write_title:
        raise ValueError(
            "Config storage.write_text_embedding and storage.write_title_embedding "
            "are both false. Nothing to do."
        )

    print("📚 Starting embedding job")
    print(f" - Neo4j URI: {cfg.neo4j_uri}")
    print(f" - Model: {cfg.embedding_model} (dim={dim})")
    print(f" - normalize: {cfg.embedding_normalize}")
    print(f" - batch_size: {cfg.embedding_batch_size}")
    print(f" - write_text_embedding: {write_text}")
    print(f" - write_title_embedding: {write_title}")

    total = 0
    skip = 0
    page_size = 500  # DB fetch size; separate from model encode batch_size

    try:
        with driver.session() as session:
            while True:
                books = session.execute_read(fetch_books_batch, skip, page_size)
                if not books:
                    break

                titles = [(b.get("title") or "") for b in books]
                texts = [((b.get("title") or "") + " " + (b.get("description") or "")).strip() for b in books]

                # Encode in batches (SentenceTransformer is faster this way)
                text_embs = None
                title_embs = None

                if write_text:
                    text_embs = model.encode(
                        texts,
                        batch_size=cfg.embedding_batch_size,
                        normalize_embeddings=cfg.embedding_normalize,
                        show_progress_bar=False,
                    )

                if write_title:
                    title_embs = model.encode(
                        titles,
                        batch_size=cfg.embedding_batch_size,
                        normalize_embeddings=cfg.embedding_normalize,
                        show_progress_bar=False,
                    )

                rows: List[Dict[str, Any]] = []
                for i, b in enumerate(books):
                    row = {
                        "id": b["id"],
                        "embedding_model": cfg.embedding_model,
                        "embedding_dim": dim,
                        "embedding": text_embs[i].tolist() if text_embs is not None else None,
                        "title_embedding": title_embs[i].tolist() if title_embs is not None else None,
                    }
                    rows.append(row)

                session.execute_write(write_embeddings_batch, rows)

                total += len(rows)
                print(f"✅ Stored embeddings for {total} books...")

                skip += page_size

        print("🎉 Embeddings successfully added to Neo4j!")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
