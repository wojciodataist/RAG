import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    chunk_text,
    semantic_chunk_text,
    SemanticSearch,
    ChunkedSemanticSearch
    )
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Get model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Get embedded text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embed_query = subparsers.add_parser("embedquery", help="Generate an embedding for a search query")
    embed_query.add_argument("query", type=str, help="Query to embed")

    search = subparsers.add_parser("search", help="Search movies by meaning")
    search.add_argument("query", type=str, help="Query to search")
    search.add_argument("--limit", type=int, default=5, help="Limit the results")

    chunk = subparsers.add_parser("chunk", help="Split text into fixed-size chunks with optional overlap")
    chunk.add_argument("text", type=str, help="Text to chunk")
    chunk.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk in words")
    chunk.add_argument("--overlap", type=int, help="Number of words to overlap between chunks")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Split text on sentence boundaries to preserve meaning")
    semantic_chunk.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="nieMaximum size of each chunk in sentenceswiem")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="Number of sentences to overlap between chunks")

    subparsers.add_parser("embed_chunks", help="Generate chunk embeddings")

    search_chunked = subparsers.add_parser("search_chunked", help="")
    search_chunked.add_argument("query", type=str, help="")
    search_chunked.add_argument("--limit", type=int, default=5, help="")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            ss = SemanticSearch()
            movies = load_movies()
            ss.load_or_create_embeddings(movies)
            results = ss.search(args.query, args.limit)

            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}")
                print()
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            cs = ChunkedSemanticSearch()
            movies = load_movies()
            
            embeddings = cs.load_or_create_chunk_embeddings(movies)

            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            cs = ChunkedSemanticSearch()
            movies = load_movies()

            cs.load_or_create_chunk_embeddings(movies)
            results = cs.search_chunks(args.query, args.limit)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"   {result["description"]}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
