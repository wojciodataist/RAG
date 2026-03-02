from .search_utils import DEFAULT_SEARCH_LIMIT, CACHE_DIR, BM25_K1, BM25_B, load_movies, load_stopwords, format_search_result
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
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

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
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
    
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
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
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("This search only supports a single term!")
        token = tokens[0]
        df = len(self.index[token])
        total_docs_num = len(self.docmap)
        return math.log((total_docs_num - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        raw_tf = self.get_tf(doc_id, term)

        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        length = 0
        if len(self.doc_lengths) == 0:
            return 0.0
        for doc in self.doc_lengths.values():
            length += doc
        return length / len(self.doc_lengths)
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
        
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = {}
        for movie in self.docmap:
            scores[movie] = 0
            for token in tokens:
                movie_score = self.bm25(movie, token)
                scores[movie] += movie_score
        
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for doc_id, score in sorted_items[:limit]:
            doc = self.docmap[doc_id]
            formatted_movie = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score
            )
            results.append(formatted_movie)
        return results


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

def tfidf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
