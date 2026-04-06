import json
import os
from typing import Any


DEFAULT_ALPHA = 0.5
RRF_K = 60
SEARCH_MULTIPLIER = 5

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DOCUMENT_PREVIEW_LENGTH = 100

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read()
        data = data.splitlines()
    return list(data)


def format_search_result(doc_id: str, title: str, document: str, score:float, **metadata: Any) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, 2),
        "metadata": metadata if metadata else {}
    }


def load_golden_dataset() -> dict:
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        return json.load(f)
