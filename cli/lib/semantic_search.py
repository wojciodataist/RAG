from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import CACHE_DIR, DEFAULT_CHUNK_SIZE, load_movies
import os
import re


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text: str) -> list:
        if len(text) == 0:
            raise ValueError("Text shall not be empty")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        
        movie_strings = []
        for doc in self.documents:
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(f"{CACHE_DIR}/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if os.path.exists(f"{CACHE_DIR}/movie_embeddings.npy"):
            self.embeddings = np.load(f"{CACHE_DIR}/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_vec = self.generate_embedding(query)

        results = []
        for i, movie_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_vec, movie_embedding)
            results.append((score, self.documents[i]))

        results.sort(key=lambda x: x[0], reverse=True)
        
        final_results = []
        for score, doc in results[:limit]:
            movie_dict = {
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            }
            final_results.append(movie_dict)
        return final_results


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    lst = []
    lst.append(text)
    embedding = semantic_search.generate_embedding(lst)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def fixed_size_chunking(text: str, overlap: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words  = words[i: i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    
    return chunks

def chunk_text(text: str, overlap: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    chunks = fixed_size_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def semantic_chunking(text: str, max_chunk_size: int, overlap:int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    step = max_chunk_size - overlap

    chunks = []
    for start in range(0, len(sentences), step):
        window = sentences[start : start + max_chunk_size]
        chunk = " ".join(window)
        chunks.append(chunk)
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int, overlap:int) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
