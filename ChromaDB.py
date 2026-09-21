import chromadb
import numpy as np
from pathlib import Path 


# --- CONFIGURATION ---
COLLECTION_NAME = "image_embeddings"
CHROMA_PATH = "E:/Image_Analyzer/chroma_data" # NOTE: Your custom path here

def initialize_chroma():
    """Initializes or connects to a local ChromaDB client using the safer get_or_create method."""
    print("Initializing ChromaDB Client...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME, 
            metadata={"hnsw:space": "cosine"} # Define similarity metric
        )
        print(f"✅ Successfully connected/created collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"🛑 Critical Error initializing ChromaDB: {e}")
        exit()


def generate_mock_embeddings(file_paths):
    """
    Generates mock embeddings. This remains the data preparation step.
    """
    print("\n[SIMULATION MODE]: Generating mock embeddings...")
    embeddings = []
    for path in file_paths:
        # Generate a vector of random floats with fixed dimension (e.g., 512)
        random_vector = np.random.uniform(-1, 1, size=(512,)).tolist()
        embeddings.append(random_vector)
    return embeddings


def add_data_to_db(collection: chromadb.Collection, file_paths: list[str], embeddings: list[list]):
    """
    Adds the metadata (file paths) and their corresponding vectors to ChromaDB.
    """
    print("\n================================================")
    print("   STARTING DATA UPLOAD TO CHROMADB")
    print(f"   Attempting to add {len(embeddings)} items...")

    # 1. Prepare the data for bulk insertion
    ids = [str(i) for i in range(len(file_paths))] 
    metadatas = file_paths # The list of simple strings (the fixed fix!)

    # 2. Insert the data into ChromaDB
    try:
        collection.add(
            embeddings=embeddings,
            documents=metadatas,  # Passing a list of strings for metadata
            ids=ids               # Unique identifiers for each item
        )
        print("✅ SUCCESS! All data has been successfully indexed into ChromaDB.")

    except Exception as e:
        print(f"❌ ERROR during database insertion: {e}")


def find_by_vector(collection: chromadb.Collection, query_vector: list[float]):
    """
    Finds the most similar file path in the database given a vector input, 
    with robust error handling for inconsistent result lengths.
    """
    print("\n================================================")
    print("   PERFORMING VECTOR SEARCH QUERY")

    # --- Step 1: Query Execution (The list must be passed as a single item in a list) ---
    try:
        results = collection.query(
            query_embeddings=[query_vector], # Pass the query vector in a LIST!
            n_results=3,                      
            include=['documents', 'metadatas', 'distances']
        )

        # Check if any results were returned at all
        if not results or not results['ids']:
             print("No results found in the database.")
             return

        # --- Step 2: Determine Safe Iteration Limit (THE FIX!) ---
        # We calculate the minimum length of the key result lists.
        # This prevents 'list index out of range' errors if one list is shorter than others.
        max_results = min(
            len(results['ids'][0]), 
            len(results['metadatas'][0]), 
            len(results['distances'][0])
        )

        print("\n================================================")
        print("✨ SEARCH RESULTS FOUND ✨")
        
        # --- Step 3: Iteration using the safe limit ---
        for i in range(max_results):
            file_path = results['metadatas'][i][0]
            distance = results['distances'][i][0]

            # Check if file_path was successfully retrieved (The NoneType check)
            if file_path is None:
                print("-" * 40)
                print("⚠️ WARNING: Could not retrieve the file path for this result.")
                # We still report the score, but skip printing a path
                similarity_score = 1 - distance 
                print(f"✅ Similarity Score (Distance): {similarity_score:.4f} (Found by vector math, but metadata missing)")
                continue # Skip to the next item

            # If file_path is valid, we proceed:
            print("-" * 40)
            print(f"🔍 Found File: {Path(file_path).name}")
            similarity_score = 1 - distance
            print(f"✅ Similarity Score: {similarity_score:.4f} (Closer to 1 is best)")

    except Exception as e:
        print(f"❌ A fatal error occurred during database querying: {e}")



# ================================================
# MAIN EXECUTION BLOCK
# ================================================

if __name__ == "__main__":
    
    # --- PHASE 0: Setup (This needs to run first!) ---
    collection = initialize_chroma()


    # 1. Define the file list and generate embeddings (The setup phase)
    mock_file_paths = [
        "E:/Image_Analyzer/Images/The_Yard_Burger.jpg",
        "E:/Image_Analyzer/Images/The_Yard_Burger_Finished.jpg"
    ]

    # 2. Generate the Vectors (Mock Data) and store them in ChromaDB
    embeddings = generate_mock_embeddings(mock_file_paths)
    add_data_to_db(collection, mock_file_paths, embeddings)


    # --- PHASE 1: Query Test (This is your final goal) ---

    # Simulate a specific search vector. We use the first generated vector as our query!
    if embeddings:
        query_vector = embeddings[0] 
        print("\n--- TESTING THE QUERY FUNCTIONALITY ---")
        print(f"Querying with Vector from file: {Path(mock_file_paths[0]).name}")

        # Run the search function using the mock vector!
        find_by_vector(collection, query_vector)
