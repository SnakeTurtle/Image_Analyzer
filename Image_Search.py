"""
Image_Search.py

Console-based search over the ChromaDB image collection built by
Image_Vectorization.py (see docs/ARCHITECTURE.md).

    text prompt -> CLIP text encoder -> 512-d text vector
    text vector -> ChromaDB query    -> n nearest neighbor images

The text vector comes from the SAME CLIP model that produced the stored
image vectors, so text and image live in one embedding space.

Querying only — vectorization/ingest lives in Image_Vectorization.py.
"""

from pathlib import Path

import chromadb
import torch
from transformers import CLIPModel, CLIPProcessor

# --- Configuration (kept in sync with Image_Vectorization.py) ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CHROMA_PATH = "E:/Image_Analyzer/chroma_data"
COLLECTION_NAME = "image_embeddings"

DEFAULT_RESULTS = 5
EXIT_COMMANDS = ("quit", "exit", "q")


# ================================================================
# CLIP: text -> vector (same model/space as the stored image vectors)
# ================================================================

def setup_model() -> tuple[CLIPModel, CLIPProcessor]:
    """Loads and returns the CLIP model and processor."""
    print(f"Loading CLIP model: {CLIP_MODEL_ID}...")
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to("cpu")
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        print("✅ CLIP models loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"🚨 Fatal error loading models: {e}")
        exit()


def _unwrap_embedding(raw_output) -> torch.Tensor:
    """
    transformers 5.x: get_text_features/get_image_features return a
    BaseModelOutputWithPooling with the PROJECTED embedding in .pooler_output.
    Older versions returned the tensor directly. Handles both.
    """
    if isinstance(raw_output, torch.Tensor):
        return raw_output
    return raw_output.pooler_output


def get_text_embedding(prompt: str, model: CLIPModel, processor: CLIPProcessor) -> list[float] | None:
    """Converts a text prompt into a CLIP text embedding as a plain float list."""
    try:
        inputs = processor(text=prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            text_tensor = _unwrap_embedding(model.get_text_features(**inputs))

        return text_tensor.cpu().numpy()[0].tolist()

    except Exception as e:
        print(f"❌ Failed to embed the prompt: {e}")
        return None


# ================================================================
# ChromaDB: vector -> nearest neighbor images
# ================================================================

def initialize_chroma() -> chromadb.Collection:
    """Connects to the persistent on-disk collection created by the ingest pipeline."""
    print("Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ Connected to collection: {COLLECTION_NAME} ({collection.count()} records)")
        return collection
    except Exception as e:
        print(f"🛑 Critical Error initializing ChromaDB: {e}")
        exit()


def search_images(collection: chromadb.Collection, query_vector: list[float], n_results: int) -> None:
    """Queries ChromaDB and prints the top n matching images, best first."""
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        print(f"❌ A fatal error occurred during database querying: {e}")
        return

    ids = results.get("ids", [[]])[0]
    if not ids:
        print("No results found in the database.")
        return

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print(f"\n✨ TOP {len(ids)} MATCHES ✨")
    for rank, (metadata, distance) in enumerate(zip(metadatas, distances), start=1):
        filepath = metadata.get("filepath", "unknown")
        description = metadata.get("description", "")
        similarity = 1.0 - distance  # cosine distance -> similarity score
        print(f"\n#{rank}  [similarity: {similarity:.4f}]")
        print(f"    Path:        {filepath}")
        print(f"    Description: {description}")


# ================================================================
# MAIN EXECUTION BLOCK (console loop)
# ================================================================

if __name__ == "__main__":
    model, processor = setup_model()
    collection = initialize_chroma()

    total = collection.count()
    if total == 0:
        print("🚨 The collection is empty. Run Image_Vectorization.py first to ingest images.")
        exit()

    print("\nDescribe the image you're looking for (or type 'quit' to exit).")
    print("-" * 60)

    while True:
        try:
            prompt = input("\nSearch: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break

        query_vector = get_text_embedding(prompt, model, processor)
        if query_vector is None:
            continue

        search_images(collection, query_vector, min(DEFAULT_RESULTS, total))