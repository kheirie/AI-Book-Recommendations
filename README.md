# AI Book Recommendations (v0) **Work In Progress**

BookGPT is a hybrid book recommendation system built on:
- Neo4j knowledge graphs
- Sentence-transformer embeddings
- Semantic + graph-based retrieval
- Streamlit conversational UI

## Features
- Semantic similarity using embeddings
- Graph-based recommendations (author, genre)
- Hybrid scoring with explainability
- Language-based filtering
- Streamlit chatbot interface

## Tech Stack
- Python 3.10+
- Neo4j
- SentenceTransformers
- Streamlit
- Ollama (LLM for intent parsing)

## Project Structure
Books-KG/
├── src/
│ ├── chatbot.py
│ ├── recommender.py
│ ├── store_embeddings.py
│ └── configs/
│ ├── config.py
│ └── config.yaml
├── tests/
└── requirements.txt


