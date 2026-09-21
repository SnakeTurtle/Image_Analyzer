import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path 
import os

# --- Configuration ---
MODEL_ID = "openai/clip-vit-base-patch32" # The specific model ID
IMAGE_DIR = "E:/Image_Analyzer/Images"           # !!! CHANGE THIS TO YOUR DIRECTORY PATH !!!


def setup_model() -> tuple[CLIPModel, CLIPProcessor]:
    """Loads and returns the necessary CLIP model and processor."""
    print(f"Loading CLIP model: {MODEL_ID}...")
    try:
        # Initialize models. Use .to("cuda") if you have a GPU.
        model = CLIPModel.from_pretrained(MODEL_ID).to("cpu") 
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        print("✅ Models loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"🚨 Fatal error loading models: {e}")
        exit()


def get_image_embedding(image_path: Path, model, processor) -> list[float] | None:
    """
    Takes a local file path, loads the image, and generates its embedding vector.
    *** This function now contains multiple fallback attempts to extract the tensor ***
    """
    print(f"   -> Processing {image_path.name}...")
    try:
        # 1. Open Image using PIL from the local path
        image = Image.open(str(image_path)).convert("RGB")

        # 2. Preprocess and Generate Vector (The core embedding logic)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            raw_output = model.get_image_features(**inputs)
            
            image_tensor = None
            # --- START OF ROBUST EXTRACTION LOGIC (The Fix) ---
            try:
                # Attempt 1: The previously assumed attribute
                image_tensor = raw_output.image_embeds
            except AttributeError:
                 # Attempt 2: Accessing the general output state
                try:
                    # Many feature extraction models store the result here
                    image_tensor = raw_output.last_hidden_state
                except AttributeError:
                     pass # Continue to next attempt

            # Fallback Check: If we still don't have a tensor, raise error gracefully
            if image_tensor is None or not isinstance(image_tensor, torch.Tensor):
                 raise AttributeError("Could not locate the primary embedding tensor in the model output object.")


        # 3. Convert the PyTorch tensor to a standard Python list (the final vector)
        embedding_vector = image_tensor.cpu().numpy()[0].tolist()
        
        return embedding_vector

    except Exception as e:
        print(f"   [ERROR] Failed to process {image_path.name}: {e}")
        return None


def embed_directory(directory_path: str, model, processor) -> list[tuple[str, list[float]]]:
    """ (This function remains unchanged as it was correct.) """
    print("\n===================================================")
    print("   STARTING BATCH EMBEDDING PROCESS")
    print(f"   Target Directory: {directory_path}")
    print("===================================================\n")

    image_paths = list(Path(directory_path).glob('**/*.jpg')) + \
                  list(Path(directory_path).glob('**/*.jpeg')) + \
                  list(Path(directory_path).glob('**/*.png'))

    if not image_paths:
        print("🚨 WARNING: No supported image files (.jpg, .jpeg, .png) found in the directory.")
        return []

    all_embeddings = []
    for img_path in image_paths:
        vector = get_image_embedding(img_path, model, processor)
        if vector is not None:
            # Store the file path and its corresponding vector
            all_embeddings.append((str(img_path), vector))

    return all_embeddings


# -------------------- MAIN EXECUTION BLOCK --------------------

if __name__ == "__main__":
    model, processor = setup_model()
    
    # Step B: Process all images in the local directory
    all_image_embeddings = embed_directory(IMAGE_DIR, model, processor)

    if all_image_embeddings:
        print("\n===================================================")
        print("✨ EMBEDDING PROCESS COMPLETE SUCCESSFULLY! ✨")
        print(f"Total files successfully embedded: {len(all_image_embeddings)}")
    else:
        print("\n🔴 Failed to generate any embeddings. Please check the directory path and image files.")

    # Assuming all_image_embeddings = [ (file_path_1, vector_1), (file_path_2, vector_2), ... ]

    if all_image_embeddings:
        print("\n===================================================")
        print("✨ EMBEDDING PROCESS COMPLETE SUCCESSFULLY! ✨")
        print(f"Total files successfully embedded: {len(all_image_embeddings)}")
        # Example of how to access the results:
        first_path, first_vector = all_image_embeddings[0]
        
        # 1. Display the Path of the first image
        print("\n--- Sample Vector Inspection ---")
        print(f"File path analyzed: {Path(first_path).name}")

        # 2. Show only a sample of the vector (The FIX IS HERE)
        sample_vector = first_vector
        formatted_samples = ""
        for x in sample_vector[:5]:
            formatted_samples = formatted_samples + str(x[:4])
        
        # Instead of printing the list, we join the formatted strings with spaces and print that single string.
        print("Vector Sample (First 5 dimensions):", " ".join(formatted_samples))

        # 3. Display the total dimensionality
        dimension = len(first_vector)
        print(f"\nTotal Vector Dimensionality: {dimension}")


