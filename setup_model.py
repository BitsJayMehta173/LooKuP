from sentence_transformers import SentenceTransformer

print("Downloading the base model...")
# This will download the model automatically into the Hugging Face cache
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model is ready!")
