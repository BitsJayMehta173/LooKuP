import os
import pickle
from fine_tune_notes import setup_model

NOTES_DIR = "notes"
EMBEDDINGS_FILE = "model/embeddings.pkl"

def build_embeddings():
    os.makedirs("model", exist_ok=True)
    model = setup_model()

    notes = []
    for file_name in os.listdir(NOTES_DIR):
        if file_name.endswith(".txt"):
            with open(os.path.join(NOTES_DIR, file_name), "r", encoding="utf-8") as f:
                notes.append(f.read())

    embeddings = model.encode(notes, show_progress_bar=True)

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"notes": notes, "embeddings": embeddings}, f)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    build_embeddings()
