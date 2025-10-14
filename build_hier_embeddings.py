from sentence_transformers import SentenceTransformer
import os, pickle
from tqdm import tqdm
import numpy as np

NOTES_DIR = "notes"
MODEL_DIR = "trained_notepad_model"
OUT_FILE = "hier_embeddings.pkl"

print("🔄 Loading fine-tuned model...")
model = SentenceTransformer(MODEL_DIR)

data = []

for file in tqdm(os.listdir(NOTES_DIR), desc="Encoding notes"):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(NOTES_DIR, file)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        continue

    # Split by lines or paragraphs
    sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 5]

    # Sentence-level embeddings
    sent_embs = model.encode(sentences, show_progress_bar=False)

    # File-level embedding (average of all sentence embeddings)
    file_emb = np.mean(sent_embs, axis=0)

    data.append({
        "path": path,
        "sentences": sentences,
        "sent_embs": sent_embs,
        "file_emb": file_emb
    })

with open(OUT_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Hierarchical embeddings saved to {OUT_FILE}")
