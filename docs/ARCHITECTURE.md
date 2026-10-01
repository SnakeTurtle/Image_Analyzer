# Key Rule
The query vector and the stored vectors MUST come from the same model/space.
CLIP is trained to place matching images and text close together, so we use
CLIP for BOTH the image vectors and the query text vectors. Descriptions from
Gemma are human-readable metadata, NOT the retrieval key.

# Pipeline of Vectorizing an Image
Pass the image to CLIP:
  CLIP image encoder creates a 512-d vector (single embedding space for image + text)

Pass the image to Gemma (via LM Studio):
  Gemma gathers a short text description to store as metadata (not embedded)

Pass the CLIP vector + Metadata to ChromaDB:
  ChromaDB upserts the vector into a persistent on-disk collection
  alongside metadata: { filepath, description, model name }

# Pipeline for Querying the Database
Input a text prompt:
  Prompt goes directly to CLIP's text encoder to get a text vector
  (no LLM hop in the query path — same model, same space as stored image vectors)

Pass the vector to ChromaDB:
  Use the vector to look through the database on disk to find images that match that description
  Upon finding an image return n nearest neighbors to that vector image
  Return image file path from metadata / Return image
