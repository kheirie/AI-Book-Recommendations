# AI Book Recommendations — v0

AI Book Recommendations is a **learning-focused project** that explores how to build a **hybrid book recommendation system** by combining:

- Knowledge graphs (Neo4j)
- Semantic search using embeddings
- Graph-based similarity (authors, genres)
- A conversational interface (Streamlit + LLM)

> **Project Status**  
> This project was built **for learning and experimentation purposes**.  

## Motivation & Learning Goals

This project was created to:

- Learn how to model books data in a **knowledge graph**
- Understand **vector embeddings** and semantic similarity search
- Combine **graph-based reasoning + semantic retrieval**
- Experiment with **hybrid scoring and explainability**
- Practice structuring a real-world ML/AI project
- Learn how to safely manage configuration and secrets

## Features

- **Book recommendations** based on a given title
- **Semantic similarity** using sentence-transformer embeddings
- **Graph-based similarity** using shared authors and genres
- **Hybrid scoring** (semantic + graph)
- **Explainable results** (shows which signal contributed)
- **Language-based filtering**
- **Conversational interface** using Streamlit
- Secure configuration using environment variables

---

## System Architecture (High Level)

1. **Neo4j Knowledge Graph**
    - Nodes: `Book`, `Author`, `Genre`, `Category`, `Publisher`
    - Relationships: `WRITTEN_BY`, `HAS_GENRE`, `HAS_CATEGORY`, `PUBLISHED_BY`
    - metadata: `title`, `description`, `page_num`, `language_code`, `edition_avgRating`, `rating_count`
    - Stores embeddings of metadata fields (in the form of vectors): `title`, `title`+`description`

2. **Semantic Layer**
   - SentenceTransformers (`all-MiniLM-L6-v2`)
   - Vector index in Neo4j
   - Cosine similarity search

3. **Graph Layer**
   - Shared authors
   - Shared genres
   - Relationship-based scoring

4. **Hybrid Recommender**
   - Normalizes scores
   - Merges results
   - Tracks recommendation sources

5. **Chat Interface**
   - Streamlit UI
   - LLM-based intent parsing
   - Natural language interaction

## Executing the Project 
This section explains how to **set up, run, and modify** the project locally.  

### Environment setup 

```bash
git clone https://github.com/kheirie/AI-Book-Recommendations.git
cd AI-Book-Recommendations
```
### Create and activate virtual environment 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment variables

#### Copy the example file
```bash
cp .env.example .env
```

#### Edit .env and add your Neo4j credentials:
```bash
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

#### Load the variables 
```bash
export $(cat .env | xargs)
```

## Project workFlow

The project follows the below workflow:

1. Data ingestion
2. Embedding generation
3. Recommendation logic
4. Chat interface

### Load Books data into Neo4j graph
```bash
python src/ingest_books.py
```
### Compute and store vector embeddings in Neo4j
```bash
python src/store_embeddings.py
```
### Test the recommender

```bash
python

from src.recommender import recommend_books, LANGUAGE_MAP

recommend_books(
    "The Hobbit",
    LANGUAGE_MAP["English"],
    top_k=5
)
```

### Run Streamlit app 
streamlit run src/chatbot.py

> **Note** 
> Ensure Neo4j is running
> Make sure environment variables are loaded before running the application
    > ```bash
    > export $(cat .env | xargs)
    > ```
> Do not commit your `.env` file to version control.

## Data

### Data Origin

The original dataset used in this project was provided for a different academic project during my **Master’s program at DSTI (Data ScienceTech Institute), France**.

The data was later cleaned and augmented for **learning and experimentation purposes**.

### Data Files

In the `data/` folder, you will find:

- **`books.csv`**  
  The original dataset, which was already cleaned and augmented in a previous academic project.  
  Further details about the original scraping and cleaning process are available in the following repository:  
  https://github.com/kheirie/book-rating-prediction-model.git

- **`books_clean.csv`**  
  The dataset further processed in this project using `src/clean_books.py`.  
  This is the version actively used for data ingestion, embedding generation, and recommendations.

### `books.csv`

The `books.csv` dataset was cleaned and augmented in the following repository:  
https://github.com/kheirie/book-rating-prediction-model.git

It contains **10,264 books** with rich bibliographic, popularity, and geographic metadata.

Below are the columns available at the start of this project:

| Column | Description |
|------|------------|
| `bookID` | Unique book identifier |
| `title` | Book title |
| `authors` | Author(s) |
| `average_rating` | Average user rating |
| `isbn` | ISBN (10) |
| `isbn13` | ISBN (13) |
| `language_code` | Language code |
| `num_pages` | Number of pages |
| `ratings_count` | Number of ratings |
| `text_reviews_count` | Number of text reviews |
| `publication_date` | Publication date |
| `publisher` | Publisher name |
| `publisher_country` | Publisher country |
| `country` | Country associated with the book |
| `latitude` | Geographic latitude |
| `longitude` | Geographic longitude |
| `coordinates` | Combined geographic coordinates |
| `first_published` | First publication year |
| `book_format` | Format (hardcover, paperback, etc.) |
| `edition_avgRating` | Average rating at edition level |
| `added_toShelves` | Number of times added to shelves |
| `first_author` | Primary author |
| `num_contributors` | Number of contributors |
| `size_of_publisher` | Publisher catalog size |
| `size_of_author` | Author catalog size |
| `book_count` | Number of books in series |
| `is_serie` | Whether the book belongs to a series |
| `has_edition` | Whether multiple editions exist |
| `book_age` | Age of the book |
| `category` | High-level category |
| `genre_category` | Genre label |
| `description` | Book description |

### `books_clean.csv`

Additional data processing was performed in this project to make the dataset suitable for **semantic search and recommendation**.

#### Description Cleaning
- Removed HTML tags from the `description` field
- Normalized whitespace
- Filtered out books with **very short or empty descriptions**

This step improves embedding quality and semantic similarity performance.

---

#### Genre Standardization

The original `genre_category` field contained noisy and inconsistent labels.

To address this:
- Genre labels were standardized using a **local Ollama language model**
- Similar or overlapping genres were normalized into a consistent taxonomy

This step improves:
- Graph-based recommendations
- Cross-book genre comparisons
- Overall interpretability of results

### Disclaimer

This dataset and its transformations are used **solely for educational purposes** as part of a learning project.  
The project does not claim ownership of the original data, which is available in  
https://github.com/kheirie/book-rating-prediction-model.git.




