import sys

print(f"Python version: {sys.version}")
print()

# --- LangChain ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✅ LangChain text splitters: OK")
except ImportError as e:
    print(f"❌ LangChain: {e}")

# --- Sentence Transformers ---
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    test_embedding = model.encode("The Force is strong with this one.")
    print(f"✅ sentence-transformers: OK (embedding dim: {len(test_embedding)})")
except Exception as e:
    print(f"❌ sentence-transformers: {e}")

# --- ChromaDB ---
try:
    import chromadb
    client = chromadb.Client()
    col = client.create_collection("test")
    col.add(documents=["test doc"], ids=["1"])
    print("✅ ChromaDB: OK")
except Exception as e:
    print(f"❌ ChromaDB: {e}")

# --- Ollama ---
try:
    import ollama
    models = ollama.list()
    model_names = [m.model for m in models.models]
    if model_names:
        print(f"✅ Ollama: OK — models available: {', '.join(model_names)}")
    else:
        print("⚠️  Ollama: connected but no models pulled yet. Run: ollama pull llama3.2")
except Exception as e:
    print(f"❌ Ollama: {e} — is Ollama running? Try: ollama serve")

# --- Script files ---
import os
script_dir = "scripts"
if os.path.exists(script_dir):
    files = [f for f in os.listdir(script_dir) if f.endswith(".txt")]
    if files:
        print(f"✅ Script files found: {', '.join(files)}")
    else:
        print("⚠️  scripts/ folder exists but no .txt files found yet")
else:
    print("⚠️  scripts/ folder not found — create it and drop your .txt files in")