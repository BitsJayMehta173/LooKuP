import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sentence_transformers import util, SentenceTransformer
from fine_tune_notes import train_model
from build_embeddings import build_embeddings

MODEL_DIR = "model"
EMBEDDINGS_DIR = "embeddings"

class NotesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Notes Search App")
        self.root.geometry("600x400")

        self.search_var = tk.StringVar()
        tk.Label(root, text="Search Query:").pack(pady=5)
        tk.Entry(root, textvariable=self.search_var, width=50).pack(pady=5)
        tk.Button(root, text="Search", command=self.search_notes).pack(pady=5)
        tk.Button(root, text="Retrain Model", command=self.retrain_model).pack(pady=5)

        self.results_box = tk.Listbox(root, width=80, height=15)
        self.results_box.pack(pady=10)

        # Load model and embeddings
        self.model = None
        self.embeddings = None
        self.files = None
        self.load_model_embeddings()

    def load_model_embeddings(self):
        if os.path.exists(MODEL_DIR):
            self.model = SentenceTransformer(MODEL_DIR)
        else:
            messagebox.showwarning("Warning", "Model not found. Train the model first.")
            return

        try:
            emb_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
            files_path = os.path.join(EMBEDDINGS_DIR, "files.npy")
            self.embeddings = np.load(emb_path)
            self.files = np.load(files_path)
        except:
            messagebox.showwarning("Warning", "Embeddings not found. Build embeddings first.")

    def search_notes(self):
        query = self.search_var.get()
        if not query:
            return
        if self.model is None or self.embeddings is None:
            messagebox.showwarning("Warning", "Model or embeddings missing.")
            return

        q_emb = self.model.encode([query])
        cos_scores = util.cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
        top_idx = np.argsort(-cos_scores)[:5]

        self.results_box.delete(0, tk.END)
        for idx in top_idx:
            self.results_box.insert(tk.END, f"{self.files[idx]} (score: {cos_scores[idx]:.3f})")

    def retrain_model(self):
        success = train_model()
        if success:
            build_embeddings()
            self.load_model_embeddings()
            messagebox.showinfo("Info", "Retraining complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NotesApp(root)
    root.mainloop()
