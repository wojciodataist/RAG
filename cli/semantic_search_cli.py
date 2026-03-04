import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Get model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Get embedded text")
    embed_text_parser.add_argument("text", type=str, help="Text to get embedded text from")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embedded text")
    # verify_embeddings_parser.add_argument("text", type=str, help="Text to verify embedded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
