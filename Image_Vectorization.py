"""
Image_Vectorization.py

Full ingest pipeline (see docs/ARCHITECTURE.md):

    image -> CLIP image encoder  -> 512-d vector
    image -> Gemma (LM Studio)   -> short text description (metadata only, NOT embedded)
    vector + metadata -> ChromaDB (persistent, on-disk, upsert-safe)

Querying is intentionally NOT implemented here.
"""

import base64
from pathlib import Path

import chromadb
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# --- Configuration ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_DIR = "E:/Image_Analyzer/Images"
CHROMA_PATH = "E:/Image_Analyzer/chroma_data"
COLLECTION_NAME = "image_embeddings"

LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"  # OpenAI-compatible endpoint
LMSTUDIO_MODEL = "google/gemma-4-e4b"  # model loaded in LM Studio
DESCRIBE_WITH_LLM = True  # set False to skip Gemma and use filename-based descriptions

# Max size for the image sent to the LLM (keeps base64 payloads small)
MAX_DESCRIBE_DIM = 768


# ================================================================
# CLIP: image -> vector
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
    transformers 5.x: get_image_features/get_text_features return a
    BaseModelOutputWithPooling with the PROJECTED embedding in .pooler_output.
    Older versions returned the tensor directly. Handles both.
    """
    if isinstance(raw_output, torch.Tensor):
        return raw_output
    return raw_output.pooler_output


def get_image_embedding(image_path: Path, model: CLIPModel, processor: CLIPProcessor) -> list[float] | None:
    """Loads an image and returns its CLIP embedding as a plain float list."""
    print(f"   -> Vectorizing {image_path.name}...")
    try:
        image = Image.open(str(image_path)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_tensor = _unwrap_embedding(model.get_image_features(**inputs))

        return image_tensor.cpu().numpy()[0].tolist()

    except Exception as e:
        print(f"   [ERROR] Failed to vectorize {image_path.name}: {e}")
        return None


# ================================================================
# Gemma (LM Studio): image -> short text description
# ================================================================

import requests
import base64
from pathlib import Path
from PIL import Image # Assuming these imports are available globally

# --- (Rest of the setup variables: DESCRIBE_WITH_LLM, LMSTUDIO_MODEL, etc.)


def describe_image(image_path: Path) -> str:
    """
    Sends the image to the local LM Studio server and asks for a short
    description. Falls back robustly if parsing fails or description is empty.
    
    NOTE: This function now prints the full LLM response payload 
          for debugging purposes.
    """
    if not DESCRIBE_WITH_LLM:
        return f"Image file: {image_path.stem}"

    # --- Image Preparation ---
    try:
        image = Image.open(str(image_path)).convert("RGB")
        image.thumbnail((MAX_DESCRIBE_DIM, MAX_DESCRIBE_DIM))
        temp_path = image_path.with_suffix(".describe_tmp.jpg")
        image.save(temp_path, format="JPEG", quality=85)
        b64_image = base64.b64encode(temp_path.read_bytes()).decode("utf-8")
        temp_path.unlink() 

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return f"Image file: {image_path.stem}"


    # --- Payload Construction (Improved Prompt) ---
    user_prompt = (
        "You are an expert metadata captioning service. Analyze the provided image "
        "and return a single, concise descriptive sentence (maximum 20 words). "
        "Your entire response MUST CONTAIN ONLY the description itself. "
        "Do not include greetings, explanations, punctuation marks, or any surrounding text."
    )

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }
        ],
        "max_tokens": 60,
        "temperature": 0.2,
    }


    # --- API Call and Request Error Handling ---
    data = None # Initialize data for scope safety
    try:
        response = requests.post(LMSTUDIO_API_URL, json=payload, timeout=120)
        print(f"Received HTTP Status Code: {response.status_code}")

        # If we get a non-200 code, raise an error and fall through the try/except block
        response.raise_for_status() 
        data = response.json()
        
        # ⭐ DEBUGGING STEP: Log the full payload for inspection ⭐
        print("\n--- LLM API Payload Received (Debug) ---")
        import json
        print(json.dumps(data, indent=2))
        print("-------------------------------------------\n")


    except requests.exceptions.HTTPError as e:
        # Catches 4xx or 5xx errors
        print(f" [ERROR] HTTP Error communicating with LM Studio API (Status {e.response.status_code}): {e}")
        return f"Image file: {image_path.stem}"

    except requests.exceptions.RequestException as e:
        # Handles timeouts, connection errors, etc.
        print(f" [ERROR] Network Request failed to LM Studio API: {e}")
        return f"Image file: {image_path.stem}"


    # --- Response Parsing and Content Extraction ---
    description = ""
    try:
        if data and "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content")
            if content is not None:
                description = content.strip()

    except Exception as e:
        # Catches structural JSON errors
        print(f" [ERROR] Failed to parse LLM response structure: {e}")


    # --- Final Fallback Logic ---
    if description:
        return description
    else:
        # This block runs if the model returned a 200 OK, but 'content' was empty or None.
        print(f"   [WARN] Empty or unreadable description from LLM for {image_path.name}, using fallback.")
        return f"Image file: {image_path.stem}"


# ================================================================
# ChromaDB: vector + metadata -> persistent storage
# ================================================================

def initialize_chroma() -> chromadb.Collection:
    """Connects to (or creates) the persistent on-disk collection."""
    print("Initializing ChromaDB Client...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ Connected/created collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"🛑 Critical Error initializing ChromaDB: {e}")
        exit()


def store_image(collection: chromadb.Collection, image_path: Path, vector: list[float], description: str) -> None:
    """
    Upserts one image into ChromaDB. The file path is used as a stable ID,
    so re-running the pipeline updates records instead of duplicating them.
    """
    collection.upsert(
        ids=[str(image_path)],
        embeddings=[vector],
        metadatas=[{
            "filepath": str(image_path),
            "description": description,
            "model": CLIP_MODEL_ID,
        }],
        documents=[description],
    )


# ================================================================
# MAIN EXECUTION BLOCK
# ================================================================

if __name__ == "__main__":
    model, processor = setup_model()
    collection = initialize_chroma()

    image_paths = (
        list(Path(IMAGE_DIR).glob("**/*.jpg"))
        + list(Path(IMAGE_DIR).glob("**/*.jpeg"))
        + list(Path(IMAGE_DIR).glob("**/*.png"))
    )

    if not image_paths:
        print(f"🚨 No supported image files (.jpg, .jpeg, .png) found in {IMAGE_DIR}")
        exit()

    print(f"\nFound {len(image_paths)} image(s). Starting ingest pipeline...\n")

    succeeded, failed = 0, 0
    for img_path in image_paths:
        vector = get_image_embedding(img_path, model, processor)
        if vector is None:
            failed += 1
            continue

        description = describe_image(img_path)
        store_image(collection, img_path, vector, description)

        print(f"   ✅ Stored {img_path.name}  |  description: \"{description}\"")
        succeeded += 1

    print("\n===================================================")
    print(f"✨ INGEST COMPLETE: {succeeded} stored, {failed} failed.")
    print(f"   Collection '{COLLECTION_NAME}' now has {collection.count()} record(s).")
    print("===================================================")