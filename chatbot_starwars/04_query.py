import os
import json
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
import ollama

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "starwars"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL    = "llama3.2"
TOP_K           = 5
MIN_SIMILARITY  = 0.45
LOG_FILE        = "data/query_log.jsonl"

SYSTEM_PROMPT = """You are a research assistant with access ONLY to excerpts 
from the Star Wars original trilogy screenplays — A New Hope, The Empire 
Strikes Back, and Return of the Jedi.

STRICT RULES:
- Answer using ONLY information present in the provided context chunks.
- If the context does not contain enough information, respond with exactly:
  "The original trilogy screenplays don't contain enough information to answer this."
- NEVER use outside knowledge, expanded universe material, or anything 
  not explicitly stated in the provided context.
- NEVER speculate or infer beyond what the text directly supports.
- When quoting dialogue, attribute it to the correct character.
- Keep answers concise and grounded in the provided text."""

# ── Logging ───────────────────────────────────────────────────────────────────

def log_query(query: str, chunks: list[dict], outcome: str, answer: str = None) -> None:
    """
    Append a structured log entry to the JSONL log file.
    outcome: 'answered' | 'rejected_no_chunks' | 'rejected_low_similarity'
    """
    entry = {
        "timestamp":      datetime.now().isoformat(),
        "query":          query,
        "outcome":        outcome,
        "top_similarity": chunks[0]["similarity"] if chunks else None,
        "chunks_retrieved": [
            {
                "film":       c["film"],
                "heading":    c["heading"],
                "similarity": round(c["similarity"], 4)
            }
            for c in chunks
        ],
        "answer": answer
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def print_log_summary() -> None:
    """Print a summary of the query log at startup."""
    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        return

    answered = sum(1 for e in entries if e["outcome"] == "answered")
    rejected = sum(1 for e in entries if e["outcome"].startswith("rejected"))

    print(f"📋 Query log: {len(entries)} total queries "
          f"({answered} answered, {rejected} rejected)\n")


# ── Setup ─────────────────────────────────────────────────────────────────────

def load_resources():
    print("🤖 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("📦 Connecting to ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"   Connected — {collection.count()} chunks indexed\n")

    return model, collection


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, model, collection) -> list[dict]:
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1 - dist
        if similarity >= MIN_SIMILARITY:
            chunks.append({
                "text":       doc,
                "film":       meta["film"],
                "heading":    meta["heading"],
                "similarity": similarity
            })

    return chunks


def is_context_sufficient(chunks: list[dict]) -> tuple[bool, str]:
    """
    Gate the LLM call. Returns (is_sufficient, rejection_reason).
    """
    if not chunks:
        return False, "rejected_no_chunks"
    if chunks[0]["similarity"] < MIN_SIMILARITY:
        return False, "rejected_low_similarity"
    return True, "answered"


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {c['film']} — {c['heading']} "
            f"(relevance: {c['similarity']:.2f})]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Generation ────────────────────────────────────────────────────────────────

def ask(query: str, model, collection) -> None:
    chunks = retrieve(query, model, collection)
    sufficient, outcome = is_context_sufficient(chunks)

    if not sufficient:
        fallback = ("The original trilogy screenplays don't contain "
                    "enough information to answer this.")
        print(f"\n⚠️  {fallback}\n")
        log_query(query, chunks, outcome)
        return

    # Show retrieval sources
    print(f"\n📚 Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"   • {c['film']} | {c['heading']} "
              f"(similarity: {c['similarity']:.3f})")

    context  = format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context from the screenplays:\n\n{context}\n\nQuestion: {query}"
        }
    ]

    print(f"\n💬 Answer:\n")

    # Stream response and capture full answer for logging
    full_answer = []
    stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_answer.append(token)

    print("\n")

    log_query(query, chunks, outcome, answer="".join(full_answer))


# ── Interactive Loop ──────────────────────────────────────────────────────────

def main():
    print("=== Star Wars RAG Chatbot ===")
    print(f"Model: {OLLAMA_MODEL} via Ollama")
    print("Type 'quit' or 'exit' to stop\n")

    print_log_summary()
    model, collection = load_resources()

    while True:
        try:
            query = input("🎙️  Ask anything: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nMay the Force be with you. 👋")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("\nMay the Force be with you. 👋")
            break

        ask(query, model, collection)


if __name__ == "__main__":
    main()