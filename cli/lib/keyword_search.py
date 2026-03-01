from .search_utils import DEFAULT_SEARCH_LIMIT, CACHE_DIR, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer
import os
import math
import pickle
from collections import Counter, defaultdict


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
    
    def get_tf(self, doc_id: int, term:str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("This search only supports a single term!")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("This search only supports a single term!")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term:str):
        return self.get_tf(doc_id, term) * self.get_idf(term)
        

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = []
    seen = set()

    query_tokens = tokenize_text(query)
    for token in query_tokens:
        for doc_id in idx.get_documents(token):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(idx.docmap[doc_id])
            if len(results) >= limit:
                return results  
    
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    
    tokens = [t for t in tokens if t]
    stop_words = load_stopwords()
    tokens = [t for t in tokens if t not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def remove_stop_words(text: str) -> list[str]:
    stemmer = PorterStemmer()
    text = tokenize_text(text)
    clean_words = []
    stemmed_list = []
    stop_words = load_stopwords()
    for t in text:
        if t not in stop_words:
            clean_words.append(t)
    for w in clean_words:
        stemmed_list.append(stemmer.stem(w))
    return stemmed_list

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)    
