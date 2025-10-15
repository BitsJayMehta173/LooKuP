# Lookup Notes: Semantic Search for Your Personal Notes

## Project Aim

**Lookup Notes** is a Python-based tool that allows you to quickly search and retrieve relevant information from a large collection of personal notes. Unlike traditional file search methods that rely on exact keyword matches, this project uses **semantic embeddings** to understand the meaning of your queries and notes, enabling more accurate and context-aware search results.

The system is designed to help users who frequently write notes but struggle to find specific information quickly. By converting your notes into vector embeddings and comparing them with the embedding of your query, it identifies the most relevant notes even if the exact words do not match.

---

## Features

- **Automatic model setup**: Downloads the necessary pretrained SentenceTransformer model on the first run.
- **Embeddings generation**: Converts all notes into embeddings for efficient semantic search.
- **Semantic search**: Finds relevant notes based on the meaning of your query, not just keywords.
- **Lightweight and flexible**: Works with plain `.txt` files and can be extended for hierarchical search.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/lookup.git
cd lookup
```

2. Set up a virtual environment (recommended):

python -m venv venv
# Activate the environment
# Windows:
venv\Scripts\activate
# Linux / MacOS:
source venv/bin/activate

3. Install dependencies:

```bash
pip install -r requirements.txt
```

# Lookup Notes: Semantic Search for Your Personal Notes

## Usage

1. **Add your notes** in the `notes/` folder as plain `.txt` files.

2. **Download & set up the model:**

```bash
python fine_tune_notes.py
```

3. **Build embeddings from your notes:**

```bash
python build_embeddings.py
```

4. **Search your notes semantically:**

```bash
python semantic_search_hier.py
```

Enter a query, and the tool will return the most relevant notes based on semantic similarity.

Enter a query, and the tool will return the most relevant notes based on semantic similarity.

Project Structure
lookup/
├── notes/                  # Your .txt notes
├── model/                  # Will contain pretrained model and embeddings
├── fine_tune_notes.py      # Setup & fine-tune embeddings
├── build_embeddings.py     # Build embeddings from notes
├── semantic_search_hier.py # Search notes using embeddings
├── requirements.txt
├── README.md
└── .gitignore

How It Solves the Problem

Many users take extensive notes but struggle to locate specific information later. Traditional search tools only match exact keywords, which often misses context or related ideas. Lookup Notes addresses this by:

Converting all notes and queries into vector embeddings using SentenceTransformer.

Comparing the semantic similarity between query and notes.

Returning the most relevant notes, even if the query does not contain exact keywords.

This makes it fast and intuitive to locate information across hundreds or thousands of notes.

# Notes

Users do not need any HuggingFace API token. The required model is downloaded automatically.

Only .txt files in the notes/ folder are processed.

Embeddings are stored in model/embeddings.pkl for faster subsequent searches.