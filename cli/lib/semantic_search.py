from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import CACHE_DIR, load_movies
import os


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text: str) -> list:
        if len(text) == 0:
            raise ValueError("Text shall not be empty")
        embedding = self.model.encode(text)
        return list(embedding)[0]
    
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
