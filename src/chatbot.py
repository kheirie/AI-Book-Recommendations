"""
chatbot.py

Streamlit-based conversational UI for BookGPT.
Handles:
- Intent classification
- Book title extraction
- Calling the recommendation engine
"""

from __future__ import annotations

import requests
import streamlit as st
from typing import Literal

from recommender import recommend_books, LANGUAGE_MAP
from configs.config import Config


# Configuration
cfg = Config()

OLLAMA_URL = cfg.ollama_url
OLLAMA_MODEL = cfg.ollama_model


def call_ollama(prompt: str) -> str:
    """
    Sends a prompt to the Ollama API and returns the generated response.

    Args:
        prompt (str): Input text prompt for the model.

    Returns:
        str: Model-generated response text.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("response", "").strip()

def parse_intent(user_message: str) -> Literal["recommend", "chat"]:
    """
    Classifies the user's message intent as either a recommendation request or general chat.

    Args:
        user_message (str): Raw user input message.

    Returns:
        Literal["recommend", "chat"]: Detected intent category.
    """
    
    prompt = f"""
You are a strict classification assistant.

Possible intents:
- recommend : user asks for book suggestions or similar books
- chat      : greetings, small talk, unrelated questions

Return ONLY one word: recommend or chat.

User message:
{user_message}
Answer:
"""
    try:
        reply = call_ollama(prompt).lower()
    except Exception:
        return "chat"

    return "recommend" if "recommend" in reply else "chat"


def extract_book_title(user_message: str) -> str:
    """
    Extracts a book title from the user's message.

    Args:
        user_message (str): Raw user input message.

    Returns:
        str: Extracted book title, or the original message if extraction fails.
    """
    prompt = f"""
Extract ONLY the book title from the sentence below.

Examples:
"I want books like Harry Potter" → Harry Potter
"Recommend me something like The Hobbit" → The Hobbit
"similar to dune please" → dune

Sentence:
{user_message}

Book title:
"""
    try:
        title = call_ollama(prompt)
        return title.strip()
    except Exception:
        return user_message.strip()


# Streamlit UI

st.set_page_config(
    page_title="BookGPT",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Book Recommendations")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: language selection
language_names = list(LANGUAGE_MAP.keys())
default_index = language_names.index("English") if "English" in language_names else 0

selected_language = st.sidebar.selectbox(
    "Preferred language for books",
    language_names,
    index=default_index,
)
language_codes = LANGUAGE_MAP[selected_language]

st.sidebar.markdown(f"**Current language:** {selected_language}")

# User input
user_message = st.chat_input("Ask for book recommendations…")

if user_message:
    st.session_state.history.append(("user", user_message))

    intent = parse_intent(user_message)

    if intent == "recommend":
        book_title = extract_book_title(user_message)

        with st.spinner(f"Finding books similar to '{book_title}'..."):
            try:
                recommendations = recommend_books(
                    book_query=book_title,
                    language_codes=language_codes,
                    top_k=cfg.top_k,
                )
            except Exception as e:
                recommendations = []
                st.session_state.history.append((
                    "assistant",
                    "⚠️ Something went wrong while searching for recommendations."
                ))

        if recommendations:
            response = (
                f"Here are some books similar to **{book_title}**:\n\n"
            )
            
            for r in recommendations:
                response += (
                    f"- **{r['title']}** "
                    f"(score: {r['score']:.2f}, "
                    f"via {', '.join(r['sources'])})\n"
                )
        else:
            response = (
                f"I couldn't find recommendations for **{book_title}** "
                f"in **{selected_language}**."
            )

        st.session_state.history.append(("assistant", response))

    else:
        st.session_state.history.append((
            "assistant",
            "I specialize in book recommendations.\n\n"
            "Try asking:\n"
            "- Recommend books like *The Hobbit*\n"
            "- Similar to *Dune*\n"
            "- Books like *Harry Potter*"
        ))

# Render chat history
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)