import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
#chromadb.config.Settings(anonymized_telemetry=False)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE       = "data/02_chunks.json"
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "starwars"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE       = 64    # chunks to embed and index at once
MIN_CHUNK_LENGTH = 10    # skip chunks shorter than this

# ── Setup ─────────────────────────────────────────────────────────────────────

def load_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    before = len(chunks)
    short_chunks = [c for c in chunks if len(c["text"]) < MIN_CHUNK_LENGTH]
    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_LENGTH]
    
    if short_chunks:
        print(f"   ⚠️  Filtered out {len(short_chunks)} chunks shorter than {MIN_CHUNK_LENGTH} chars:")
        for c in short_chunks:
            print(f"      • chunk_id {c['chunk_id']} | {c['film']} | {c['heading']}")
            print(f"        Text: '{c['text']}'")
    return chunks


def build_chroma_collection(chunks, model):
    """
    Create (or recreate) the ChromaDB collection and index all chunks in batches.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists so we start clean
    existing = client.list_collections()
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"   🗑️  Deleted existing '{COLLECTION_NAME}' collection")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text
    )

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in tqdm(range(total_batches), desc="Embedding & indexing"):
        batch = chunks[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]

        texts     = [c["text"] for c in batch]
        ids       = [str(c["chunk_id"]) for c in batch]
        metadatas = [
            {
                "film":         c["film"],
                "scene_id":     c["scene_id"],
                "heading":      c["heading"],
                "chunk_index":  c["chunk_index"],
                "total_chunks": c["total_chunks"]
            }
            for c in batch
        ]

        # Embed the whole batch at once — much faster than one at a time
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return collection


def verify_collection(collection) -> None:
    """Run a quick sanity query to confirm everything indexed correctly."""
    count = collection.count()
    print(f"\n✅ Collection '{COLLECTION_NAME}' contains {count} indexed chunks")

    # Test query
    test_query = "I am your father"
    results = collection.query(
        query_embeddings=None,
        query_texts=[test_query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Test query: '{test_query}'")
    print(f"   Top 3 results:\n")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"   Result {i+1} — {meta['film']} | {meta['heading']}")
        print(f"   Similarity: {1 - dist:.3f}")
        print(f"   Text: {doc[:150]}...")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print("=== Chapter 4: Embedding & Indexing ===\n")

    print(f"📂 Loading chunks from {INPUT_FILE}...")
    chunks = load_chunks(INPUT_FILE)
    print(f"   {len(chunks)} chunks ready for embedding")

    print(f"\n🤖 Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   Model loaded — embedding dim: {model.get_sentence_embedding_dimension()}")

    print(f"\n📦 Building ChromaDB collection in '{CHROMA_DIR}/'...")
    collection = build_chroma_collection(chunks, model)

    verify_collection(collection)

    print("💾 Knowledge base persisted to disk — ready for querying!")