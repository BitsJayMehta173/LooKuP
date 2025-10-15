import pickle
import numpy as np
from sentence_transformers import util

EMBEDDINGS_FILE = "model/embeddings.pkl"

def search(query, top_k=5):
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)

    notes = data["notes"]
    embeddings = data["embeddings"]

    from fine_tune_notes import setup_model
    model = setup_model()
    query_embedding = model.encode(query)

    similarities = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(notes[i], similarities[i]) for i in top_indices]
    return results

if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = search(query)
    for note, score in results:
        print(f"Score: {score:.4f}\n{note}\n{'-'*40}")
