# 🧠 LooKuP A Smart Notepad Search (USING TRANSFORMERS)

## 💡 Problem

Writing lots of notes in Notepad can get messy. Finding the right file or sentence manually is slow and frustrating. This project solves it with a **local AI-powered search system** that understands your notes by meaning, not just keywords.

---

## 🚀 Features

- Searches **by sentence, paragraph, and whole file**  
- Ranks results by **semantic similarity**  
- Works **offline**, no internet required  
- Learns your writing style for more relevant results  

---

## 📂 Project Structure

```bash
lookup/
├── notes/                   # Folder containing all your .txt notes
├── fine_tune_notes.py        # Train a lightweight model on your writing style
├── build_hier_embeddings.py  # Build embeddings for hierarchical organization
├── semantic_search_hier.py   # Run semantic search across notes
├── requirements.txt          # Project dependencies
└── README.md                 # This file

```

## ⚙️ Setup & Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train the model**

```bash
python fine_tune_notes.py
```

3. **Build embeddings**

```bash
python build_hier_embeddings.py
```

4. **Run the search**

```bash
python semantic_search_hier.py
```

Type your query to see the most relevant sentences and files. Type `exit` to quit.

---

## 🔄 Updating Notes

If you add or edit `.txt` files in `notes/`, run:

```bash
python build_hier_embeddings.py
```

Re-train the model only if your writing style or topics change significantly.

---

## 👨‍💻 Author

**Jay Mehta** — built to make searching through random notes fast, intelligent, and personal.

---

✅ This is **one single block of Markdown**. Copy and save it directly as `README.md`. It won’t split or break anywhere.

---

If you want, I can also make a **slightly more visual version with emojis and numbered steps** that looks really friendly for beginners.

---

This will render nicely in GitHub or any Markdown viewer.  

If you want, I can now combine this with your previous sections into **one full, ready-to-use `README.md` file** in a single block so nothing ever breaks. Do you want me to do that?
