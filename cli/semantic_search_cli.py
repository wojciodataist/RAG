import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    chunk_text,
    semantic_chunk_text,
    SemanticSearch
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

    chunk = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk.add_argument("text", type=str, help="Text to chunk")
    chunk.add_argument("--chunk-size", type=int, default=200, help="Limit the chunk size")
    chunk.add_argument("--overlap", type=int, help="Number of overlapping characters")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Use semantic chunking")
    semantic_chunk.add_argument("text", type=str, help="Placeholder")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="niewiem")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="")

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
