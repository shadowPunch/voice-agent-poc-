#!/usr/bin/env python3
"""
ingest.py — Populate the ChromaDB knowledge base for voice_agent_v2.

Usage examples
──────────────

# Ingest a plain text file (one document per paragraph)
python3 ingest.py --txt notes.txt

# Ingest a JSONL file where each line is {"text": "...", "id": "optional"}
python3 ingest.py --jsonl docs.jsonl

# Ingest a folder of .txt files
python3 ingest.py --dir ./my_docs

# Ingest raw strings inline
python3 ingest.py --raw "Paris is the capital of France." "The Eiffel Tower is 330m tall."

# Show collection stats
python3 ingest.py --stats

# Clear the collection (destructive!)
python3 ingest.py --clear
"""

# Imported from rtc_server instead of app:
from rtc_server import (
    _chroma_col,
    _embed_model,
    RAG_AVAILABLE,
    CHROMA_PATH,
    CHROMA_COLLECTION,
)

import argparse
import json
import pathlib
import sys

def _ingest(texts: list[str], ids: list[str] | None = None) -> None:
    if not RAG_AVAILABLE:
        sys.exit("chromadb / sentence-transformers not installed.")
    if not texts:
        print("Nothing to ingest.")
        return

    existing = _chroma_col.count()
    if ids is None:
        ids = [str(existing + i) for i in range(len(texts))]

    print(f"Embedding {len(texts)} document(s)…")
    embs = _embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    _chroma_col.add(embeddings=embs.tolist(), documents=texts, ids=ids)
    print(f"Done. Collection '{CHROMA_COLLECTION}' now has {_chroma_col.count()} document(s).")


def main():
    p = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base. Support for .md files")
    p.add_argument("--txt",    metavar="FILE",  help="Plain text file (paragraph-split)")
    p.add_argument("--md",    metavar="FILE",  help="Markdown file (paragraph-split)")
    p.add_argument("--jsonl",  metavar="FILE",  help="JSONL file with {\"text\":\"...\"} entries")
    p.add_argument("--dir",    metavar="DIR",   help="Directory of .txt files")
    p.add_argument("--dir_md",    metavar="DIR",   help="Directory of .md files")
    p.add_argument("--raw",    metavar="STR", nargs="+", help="Raw strings to ingest directly")
    p.add_argument("--stats",  action="store_true", help="Print collection statistics")
    p.add_argument("--clear",  action="store_true", help="Delete all documents in the collection")
    args = p.parse_args()

    if args.stats:
        print(f"Collection : {CHROMA_COLLECTION}")
        print(f"Path       : {CHROMA_PATH}")
        print(f"Documents  : {_chroma_col.count()}")
        return

    if args.clear:
        ids = _chroma_col.get()["ids"]
        if ids:
            _chroma_col.delete(ids=ids)
        print(f"Cleared {len(ids)} document(s).")
        return

    texts, ids = [], []

    if args.txt:
        src = pathlib.Path(args.txt).read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in src.split("\n\n") if p.strip()]
        texts.extend(paragraphs)

    if args.md:
        src = pathlib.Path(args.md).read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in src.split("\n\n") if p.strip()]
        texts.extend(paragraphs)

    if args.jsonl:
        for i, line in enumerate(pathlib.Path(args.jsonl).read_text().splitlines()):
            obj = json.loads(line)
            texts.append(obj["text"])
            if "id" in obj:
                ids.append(obj["id"])

    if args.dir:
        for ext in ("*.txt", "*.md"):
            for f in sorted(pathlib.Path(args.dir).glob(ext)):
                paragraphs = [p.strip() for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
                texts.extend(paragraphs)
    if args.dir_md:
        for f in sorted(pathlib.Path(args.dir_md).glob("*.md")):
            paragraphs = [p.strip() for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
            texts.extend(paragraphs)
    if args.raw:
        texts.extend(args.raw)

    _ingest(texts, ids if ids else None)


if __name__ == "__main__":
    main()