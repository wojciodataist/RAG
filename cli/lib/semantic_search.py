from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DOCUMENT_PREVIEW_LENGTH,
    load_movies,
    format_search_result)
import os
import re
import json


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
        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
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

def semantic_chunk(text: str, max_chunk_size: int, overlap:int) -> list[str]:
    txt = text.strip()

    if len(txt) == 0:
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", txt)

    if len(sentences) == 1 and not sentences[0].endswith(("?", "!", ".")):
        return [txt]
    
    sentences = [s.strip() for s in sentences if s.strip()]

    step = max_chunk_size - overlap
    chunks = []

    n_sentences = len(sentences)
    i = 0

    while i < n_sentences:
        chunk_sentences = sentences[i: i + max_chunk_size]

        if chunks and len(chunk_sentences) <= overlap:
            break

        chunk = " ".join(chunk_sentences)

        if len(chunk) > 0:
            chunks.append(chunk)
        
        i += step
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int, overlap:int) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        all_chunks = []
        chunk_metadata = []

        for idx, document in enumerate(self.documents):
            self.document_map[document["id"]] = document

            description = document.get("description", "")
            if not description.strip():
                continue

            chunks = semantic_chunk(description, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                chunk_metadata.append({
                    "movie_idx": idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })
            
        self.chunk_embeddings = self.model.encode(all_chunks)

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(all_chunks)
            }, f, indent=2)
        self.chunk_metadata = chunk_metadata

        return self.chunk_embeddings

    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
            
            self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings
    
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results
