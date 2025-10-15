import os
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt')

EMBEDDINGS_DIR = "embeddings"
MODEL_DIR = "model"

def load_notes():
    notes = []
    file_names = []
    notes_dir = "notes"
    if not os.path.exists(notes_dir):
        os.makedirs(notes_dir)
    for fname in os.listdir(notes_dir):
        if fname.endswith(".txt"):
            path = os.path.join(notes_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                sentences = nltk.tokenize.sent_tokenize(text)
                notes.extend(sentences)
                notes.append(text)  # also store full text
                file_names.extend([fname]* (len(sentences)+1))
    return notes, file_names

def build_embeddings():
    notes, file_names = load_notes()
    if not notes:
        print("No notes found to embed.")
        return

    model = SentenceTransformer(MODEL_DIR)
    embeddings = model.encode(notes, show_progress_bar=True)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    np.save(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, "files.npy"), file_names)
    print("Embeddings saved.")

if __name__ == "__main__":
    build_embeddings()
