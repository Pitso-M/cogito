import os
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/01_cleaned_docs.json"
OUTPUT_FILE  = "data/02_chunks.json"

CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 100   # characters

# ── Setup ─────────────────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
    length_function=len
)

# ── Main ──────────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split each scene into chunks. Every chunk inherits its parent's metadata
    and gets a unique chunk_id.
    """
    all_chunks = []
    chunk_counter = 0

    for doc in tqdm(docs, desc="Chunking scenes"):
        text   = doc["text"]
        splits = splitter.split_text(text)

        for i, chunk_text in enumerate(splits):
            all_chunks.append({
                "chunk_id":    chunk_counter,
                "film":        doc["film"],
                "scene_id":    doc["scene_id"],
                "heading":     doc["heading"],
                "chunk_index": i,           # position within parent scene
                "total_chunks": len(splits), # how many chunks this scene produced
                "text":        chunk_text
            })
            chunk_counter += 1

    return all_chunks


def print_stats(chunks: list[dict]) -> None:
    """Print a summary of chunk distribution."""
    lengths = [len(c["text"]) for c in chunks]
    by_film = {}
    for c in chunks:
        by_film.setdefault(c["film"], []).append(len(c["text"]))

    print(f"\n📊 Chunk Statistics:")
    print(f"   Total chunks:   {len(chunks)}")
    print(f"   Avg length:     {sum(lengths) // len(lengths)} chars")
    print(f"   Min length:     {min(lengths)} chars")
    print(f"   Max length:     {max(lengths)} chars")

    print(f"\n   By film:")
    for film, lens in sorted(by_film.items()):
        print(f"   • {film}: {len(lens)} chunks, avg {sum(lens)//len(lens)} chars")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("=== Chapter 3: Chunking ===\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"📂 Loaded {len(docs)} scenes from {INPUT_FILE}")

    chunks = chunk_documents(docs)
    print_stats(chunks)

    # Save to disk for Chapter 4
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to {OUTPUT_FILE}")

    # Preview: show ALL chunks for the first multi-chunk scene
    multi_chunk_scenes = [c for c in chunks if c["total_chunks"] > 1]
    if multi_chunk_scenes:
        first = multi_chunk_scenes[0]
        scene_chunks = [
            c for c in chunks
            if c["film"] == first["film"] and c["scene_id"] == first["scene_id"]
        ]

        print(f"\n--- Preview: Multi-chunk Scene ---")
        print(f"Film:    {first['film']}")
        print(f"Heading: {first['heading']}")
        print(f"Chunks:  {first['total_chunks']} total\n")

        for c in scene_chunks:
            print(f"{'─' * 60}")
            print(f"Chunk {c['chunk_index'] + 1} of {c['total_chunks']} "
                  f"({len(c['text'])} chars):")
            print(f"{'─' * 60}")
            print(c["text"])
            print()