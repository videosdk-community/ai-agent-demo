"""
This script builds a Pinecone vector store from the Travel Destinations CSV.
The vector store is used to find relevant travel destination information that can be used
by the AI voice agent when answering travel-related questions.

The script:
1. Loads and processes the Travel Destinations CSV
2. Creates meaningful text chunks from destination data
3. Generates embeddings using OpenAI
4. Stores vectors in Pinecone for later retrieval

Usage:
    python build_pinecone_store.py

Environment variables required:
    - OPENAI_API_KEY
    - PINECONE_API_KEY
    - PINECONE_ENVIRONMENT
    - PINECONE_INDEX_NAME
"""

import os
import time
import hashlib
import csv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configuration
CSV_PATH = "travel_destinations.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Ensure all necessary environment variables are set
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    print("CRITICAL Error: One or more environment variables (OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME) are not set.")
    print("Please ensure your .env file is correctly configured.")
    exit(1)

# Type assertions after validation
assert OPENAI_API_KEY is not None
assert PINECONE_API_KEY is not None
assert PINECONE_INDEX_NAME is not None

def load_travel_destinations_from_csv(csv_path: str) -> List[Document]:
    """Loads travel destinations from CSV and returns a list of Document objects."""
    print(f"Loading travel destinations from CSV: {csv_path}")
    documents = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row_num, row in enumerate(csv_reader, 1):
                # Create a comprehensive text description for each destination
                city = row['City']
                country = row['Country']
                category = row['Category']
                best_time = row['Best_Time_to_Travel']
                
                # Create a rich text description that combines all information
                content = f"""Travel Destination: {city}, {country}

Categories and Attractions: {category}

Best Time to Visit: {best_time}

This destination offers a unique travel experience with various attractions and activities. The best time to visit is {best_time} when the weather and conditions are optimal for exploring {category}."""
                
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": csv_path,
                            "row": row_num,
                            "type": "travel_destination",
                            "city": city,
                            "country": country,
                            "category": category,
                            "best_time": best_time
                        }
                    )
                )
        
        print(f"Successfully loaded {len(documents)} travel destinations from CSV")
        return documents
    
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return []

def generate_deterministic_id(source: str, text_chunk: str, row: int) -> str:
    """Generates a deterministic ID for a text chunk based on its source, content, and row number."""
    hasher = hashlib.md5()
    hasher.update(source.encode('utf-8'))
    hasher.update(text_chunk.encode('utf-8'))
    hasher.update(str(row).encode('utf-8'))
    return hasher.hexdigest()

def build_and_upsert_vector_store(clear_index_first: bool = False):
    """
    Processes the Travel Destinations CSV, generates embeddings, and upserts them to Pinecone.
    
    Args:
        clear_index_first (bool): If True, deletes all vectors from the index before upserting.
    """
    # Load travel destinations from CSV
    documents = load_travel_destinations_from_csv(CSV_PATH)
    
    if not documents:
        print("No travel destinations were loaded from the CSV. Aborting vector store build.")
        return

    # Split documents into chunks (though CSV rows are already well-sized)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Good size for destination descriptions
        chunk_overlap=200,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Total travel destinations split into {len(split_docs)} chunks.")

    if not split_docs:
        print("No document chunks were created. Aborting.")
        return

    # Initialize embeddings model and Pinecone client
    print("Initializing embeddings model and Pinecone client...")
    try:
        # Use environment variable directly to avoid type issues
        embeddings_model = OpenAIEmbeddings()  # Will use OPENAI_API_KEY from environment
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)  # type: ignore
        print(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
    except Exception as e:
        print(f"Failed to initialize OpenAIEmbeddings or Pinecone client: {e}")
        return

    # Clear index if requested
    if clear_index_first:
        print(f"WARNING: Deleting all vectors from index '{PINECONE_INDEX_NAME}' as per request...")
        try:
            index.delete(delete_all=True)
            print("All vectors deleted from the index.")
        except Exception as e_del:
            print(f"Error deleting vectors from index: {e_del}. Proceeding without clearing.")

    # Process and upsert documents in batches
    EMBEDDING_BATCH_SIZE = 64
    total_chunks = len(split_docs)
    print(f"Starting to process and upsert {total_chunks} travel destination chunks in batches of {EMBEDDING_BATCH_SIZE}...")

    for i in tqdm(range(0, total_chunks, EMBEDDING_BATCH_SIZE), desc="Processing Batches"):
        batch_documents = split_docs[i : i + EMBEDDING_BATCH_SIZE]
        current_batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
        total_batches = (total_chunks + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        
        print(f"  Processing batch {current_batch_num}/{total_batches} ({len(batch_documents)} destinations)...")

        batch_texts = [doc.page_content for doc in batch_documents]

        for attempt in range(MAX_RETRIES):
            try:
                print(f"    Attempt {attempt + 1}/{MAX_RETRIES}: Generating embeddings for batch {current_batch_num}...")
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                print(f"    Embeddings generated for batch {current_batch_num}.")

                vectors_to_upsert = []
                for doc, embedding in zip(batch_documents, batch_embeddings):
                    deterministic_id = generate_deterministic_id(
                        doc.metadata['source'],
                        doc.page_content,
                        doc.metadata['row']
                    )
                    metadata_for_pinecone = doc.metadata.copy()
                    metadata_for_pinecone['text'] = doc.page_content
                    
                    vectors_to_upsert.append({
                        "id": deterministic_id,
                        "values": embedding,
                        "metadata": metadata_for_pinecone
                    })
                
                print(f"    Attempt {attempt + 1}/{MAX_RETRIES}: Upserting {len(vectors_to_upsert)} vectors for batch {current_batch_num} to Pinecone...")
                index.upsert(vectors=vectors_to_upsert)
                print(f"    Batch {current_batch_num} successfully upserted.")
                break

            except Exception as e:
                print(f"An error occurred during processing or upserting batch {current_batch_num} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Waiting for {RETRY_DELAY_SECONDS} seconds before retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"Max retries reached for batch {current_batch_num}. Skipping this batch.")

    print("All Travel Destination batches processed and stored in Pinecone.")

if __name__ == "__main__":
    print("Starting the process to build Travel Destinations vector store...")
    build_and_upsert_vector_store(clear_index_first=True)
    print("Process finished.")
