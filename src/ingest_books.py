from neo4j import GraphDatabase
import pandas as pd
import ast

# Load the dataset
df = pd.read_csv("data/books_clean.csv")

# Neo4j connection (NO AUTHENTICATION)
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=None
)

def ingest_row(tx, row):
    # --- Create Book node ---
    tx.run("""
        MERGE (b:Book {bookID: $bookID})
        SET b.title = $title,
            b.description = $description,
            b.page_num = $num_pages,
            b.language_code = $language_code,
            b.edition_avgRating = $edition_avgRating,
            b.rating_count = $ratings_count
    """, row)

    # --- Authors ---
    authors = row["authors"].split("/")
    for a in authors:
        tx.run("MERGE (auth:Author {name: $a})", a=a)
        tx.run("""
            MATCH (b:Book {bookID: $bookID})
            MATCH (auth:Author {name: $a})
            MERGE (b)-[:WRITTEN_BY]->(auth)
        """, bookID=row["bookID"], a=a)

    # --- Publisher ---
    tx.run("MERGE (pub:Publisher {name: $p})", p=row["publisher"])
    tx.run("""
        MATCH (b:Book {bookID: $bookID})
        MATCH (pub:Publisher {name: $p})
        MERGE (b)-[:PUBLISHED_BY]->(pub)
    """, bookID=row["bookID"], p=row["publisher"])

    # --- Genres ---
    genres = row["standardized_genres"].split(",")
    for g in genres:
        g = g.strip()
        tx.run("MERGE (genre:Genre {name: $g})", g=g)
        tx.run("""
            MATCH (b:Book {bookID: $bookID})
            MATCH (genre:Genre {name: $g})
            MERGE (b)-[:HAS_GENRE]->(genre)
        """, bookID=row["bookID"], g=g)

    # --- Category ---
    tx.run("MERGE (c:Category {name: $category})", category=row["category"])
    tx.run("""
        MATCH (b:Book {bookID: $bookID})
        MATCH (c:Category {name: $category})
        MERGE (b)-[:HAS_CATEGORY]->(c)
    """, bookID=row["bookID"], category=row["category"])

# --- Execute ingestion ---
with driver.session() as session:
    for _, row in df.iterrows():
        session.execute_write(ingest_row, row.to_dict())

driver.close()
print("Data ingestion to Neo4j completed.")