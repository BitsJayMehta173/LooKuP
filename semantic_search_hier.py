from sentence_transformers import SentenceTransformer, util
import numpy as np, pickle, os
from tabulate import tabulate

MODEL_DIR = "trained_notepad_model"
EMB_FILE = "hier_embeddings.pkl"

# Load model and embeddings
print("🔄 Loading model and embeddings...")
model = SentenceTransformer(MODEL_DIR)
with open(EMB_FILE, "rb") as f:
    data = pickle.load(f)
print("✅ Model and embeddings loaded.\n")

def rank_results(query):
    query_emb = model.encode(query)

    ranked = []
    for item in data:
        # Document similarity
        doc_sim = util.cos_sim(query_emb, item["file_emb"]).item()

        # Sentence-level similarity
        sent_sims = util.cos_sim(query_emb, item["sent_embs"])[0].cpu().numpy()
        top_idx = np.argmax(sent_sims)
        best_sentence = item["sentences"][top_idx]
        best_sent_score = sent_sims[top_idx]

        # Combine both scores (weighted)
        final_score = 0.7 * best_sent_score + 0.3 * doc_sim

        ranked.append({
            "file": os.path.basename(item["path"]),
            "best_sentence": best_sentence,
            "sentence_score": best_sent_score,
            "doc_score": doc_sim,
            "final": final_score
        })

    ranked.sort(key=lambda x: x["final"], reverse=True)
    return ranked[:5]

while True:
    query = input("Enter your search query (or 'exit'): ").strip()
    if query.lower() == 'exit':
        break

    results = rank_results(query)

    rows = []
    for i, r in enumerate(results, 1):
        rows.append([
            i,
            round(r["final"], 3),
            r["file"],
            r["best_sentence"][:120]
        ])

    print("\n" + tabulate(rows, headers=["Rank", "Score", "File", "Matched Sentence"], tablefmt="rounded_outline"))
    print()
