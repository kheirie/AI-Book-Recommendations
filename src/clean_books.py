import pandas as pd
import re
import os
from huggingface_hub import InferenceClient
import time
from ollama import Client
import ast

def clean_description(text):
    """
    Clean the book description by removing HTML tags and extra whitespace.

    Args:
        text: The raw description value (can be string-like or NaN).

    Returns:
        A cleaned description string with HTML tags removed and
        normalized whitespace. Returns an empty string for null values.
    """
    
    if pd.isna(text):
        return ""
    
    # Convert to string and remove simple HTML tags
    text = re.sub(r"<.*?>", " ", str(text)) 
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)         
    return text.strip()


def extract_genres_ollama(genre, client, ollama_model = "mistral"):
    """
    Standardizes a book genre into predefined categories using an Ollama model.

    Args:
        genre (str): Raw genre string.
        client: Ollama-compatible chat client.
        ollama_model (str): Model name to use.

    Returns:
        list[str]: Standardized genre labels.
    """
    
    prompt = f"""
    You are a data cleaning assistant. 

    Task: Standardize a single book genre into one or more categories from this list:
    ["Fiction","Nonfiction","Biography","Science","History","Religion",
    "Education","Arts & Design","Children","Comics","Law","Politics","Social Science"]

    Rules:
    1. If none of these genres fit and you cannot find any suitable genre, return ["Unknown"].
    2. If the input contains ',', '/', '&', or 'and', split the genres accordingly.
    3. Always return only a single Python list. Do not include any explanation, Input/Output text, or code formatting. 

    Genre to standardize: "{genre}"

    Return the standardized genre as a Python list only:
    """
    response = client.chat(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response["message"]["content"]
    
    return ast.literal_eval(content)


def clean_books(input_path: str, output_path: str, sleep_seconds=20, max_retries=3):
    """
    Cleans a books dataset and standardizes genres using an Ollama model.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to write the cleaned CSV file.
        sleep_seconds (int): Delay between LLM calls for rate limiting.
        max_retries (int): Maximum retry attempts per genre.

    Returns:
        None
    """
    
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")

    # Clean descriptions and filter out trivial ones
    df["description"] = df["description"].astype(str).apply(clean_description)
    df = df[df["description"].str.len() > 5]

    print("Standardizing genres using Ollama (Mistral)...")

    ollama_client = Client()
    cache = {}
    standardized_genres = []

    # Ensure file exists if appending (write header only once)
    chunk_saved = False

    for i, g in enumerate(df["genre_category"], start=1):
        # Use cached result if we've already seen this genre
        if g in cache:
            standardized_genres.append(cache[g])
            continue

        # Retry logic
        for attempt in range(max_retries):
            try:
                result = extract_genres_ollama(g, ollama_client)
                cache[g] = ", ".join(result)
                standardized_genres.append(cache[g])
                break
            except Exception as e:
                print(f"[{i}] Error processing '{g}' (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    cache[g] = "Unknown"
                    standardized_genres.append("Unknown")
                    
        # Rate limiting
        time.sleep(sleep_seconds)

        # Save every 200 rows to avoid data loss
        if i % 200 == 0:
            df_tmp = df.iloc[:i].copy()
            df_tmp["standardized_genres"] = standardized_genres[:i]
            
            if not chunk_saved:
                df_tmp.to_csv(output_path, index=False)
                chunk_saved = True
            else:
                df_tmp.to_csv(output_path, mode='a', header=False, index=False)
            
            print(f"Progress: Saved {i} rows to {output_path}.")

        if i % 20 == 0 or i == len(df):
            print(f"Processed {i}/{len(df)} genres...")

    # Save final chunk after the loop
    df["standardized_genres"] = standardized_genres
    print("Genre standardization completed.")
    df.to_csv(output_path, index=False) 

    print(f"Cleaned shape: {df.shape}")
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    input_path = "data/books.csv"
    output_path = "data/books_clean.csv"
    clean_books(input_path, output_path)
