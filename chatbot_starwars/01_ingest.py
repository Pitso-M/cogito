import os
import re
import json
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPTS_DIR = "scripts"
OUTPUT_FILE  = "data/01_cleaned_docs.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def derive_film_title(filename: str) -> str:
    """Turn a filename like 'a_new_hope.txt' into 'A New Hope'."""
    name = os.path.splitext(filename)[0]
    return name.replace("-", " ").title()


def clean_script(raw_text: str) -> str:
    """
    Remove common screenplay formatting artifacts from a raw .txt script.
    Returns cleaned plain text.
    """
    # Remove page numbers (e.g. standalone numbers or 'Page 12')
    text = re.sub(r'(?m)^\s*\d+\.\s*$', '', raw_text)
    text = re.sub(r'(?m)^\s*Page\s+\d+\s*$', '', text, flags=re.IGNORECASE)

    # Remove (CONT'D), (V.O.), (O.S.) and similar annotations
    text = re.sub(r'\(CONT\'D\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(V\.O\.?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(O\.S\.?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(O\.C\.?\)', '', text, flags=re.IGNORECASE)

    # Remove parenthetical stage directions on their own line e.g. (quietly)
    text = re.sub(r'(?m)^\s*\(.*?\)\s*$', '', text)

    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    # Strip leading whitespace per line
    text = '\n'.join(line.lstrip() for line in text.splitlines())

    return text.strip()


def extract_scenes(cleaned_text: str, film_title: str) -> list[dict]:
    """
    Split a cleaned script into scenes using INT./EXT. headers as boundaries.
    Each scene becomes a document with metadata.
    Falls back to returning the whole script as one document if no scenes found.
    """
    # Match scene headings like INT. DEATH STAR - DAY or EXT. TATOOINE - NIGHT
    scene_pattern = re.compile(
        r'(?m)^(?:\d+\s+)?((?:INT|EXT|INT\/EXT|I\/E)[\.\s].+?)$'
    )

    # Find all scene header positions
    headers = [(m.start(), m.group(1).strip()) for m in scene_pattern.finditer(cleaned_text)]

    if not headers:
        # No scene headers found — return whole script as single document
        print(f"  ⚠️  No scene headers found in '{film_title}', treating as single document.")
        return [{
            "film":     film_title,
            "scene_id": 0,
            "heading":  "FULL SCRIPT",
            "text":     cleaned_text
        }]

    scenes = []
    for i, (start, heading) in enumerate(headers):
        # Text runs from after this header to the start of the next header (or EOF)
        end = headers[i + 1][0] if i + 1 < len(headers) else len(cleaned_text)
        scene_text = cleaned_text[start:end].strip()

        if len(scene_text) < 30:
            # Skip near-empty scenes
            continue

        scenes.append({
            "film":     film_title,
            "scene_id": i + 1,
            "heading":  heading,
            "text":     scene_text
        })

    return scenes


def load_and_process_scripts(scripts_dir: str) -> list[dict]:
    """Load all .txt files, clean them, and extract scenes."""
    all_docs = []
    txt_files = sorted([f for f in os.listdir(scripts_dir) if f.endswith(".txt")])

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{scripts_dir}/'")

    for filename in tqdm(txt_files, desc="Processing scripts"):
        filepath = os.path.join(scripts_dir, filename)
        film_title = derive_film_title(filename)

        print(f"\n📄 Loading: {filename} → '{film_title}'")

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        print(f"   Raw size: {len(raw_text):,} characters")

        cleaned = clean_script(raw_text)
        print(f"   Cleaned size: {len(cleaned):,} characters")

        scenes = extract_scenes(cleaned, film_title)
        print(f"   Scenes extracted: {len(scenes)}")

        all_docs.extend(scenes)

    return all_docs


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("=== Chapter 2: Ingestion & Cleaning ===\n")
    docs = load_and_process_scripts(SCRIPTS_DIR)

    print(f"\n✅ Total scenes extracted: {len(docs)}")
    print(f"   Films: {list({d['film'] for d in docs})}")

    # Save to disk for Chapter 3
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to {OUTPUT_FILE}")

    # Preview first scene
    print("\n--- Preview: First Scene ---")
    first = docs[0]
    print(f"Film:    {first['film']}")
    print(f"Scene:   {first['scene_id']} — {first['heading']}")
    print(f"Length:  {len(first['text'])} characters")
    print(f"Text preview:\n{first['text'][:300]}...")