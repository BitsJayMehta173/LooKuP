import os
import shutil
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import nltk

nltk.download('punkt')

MODEL_DIR = "model"
BACKUP_DIR = os.path.join(MODEL_DIR, "backup")
OUTPUT_MODEL_DIR = MODEL_DIR

def backup_model():
    if not os.path.exists(MODEL_DIR):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    shutil.copytree(MODEL_DIR, backup_path, ignore=shutil.ignore_patterns("backup"))
    print(f"Backup created at {backup_path}")
    return backup_path

def load_notes():
    notes = []
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
                notes.append(text)  # full file as context
    return notes

def train_model():
    notes = load_notes()
    print(f"Training on {len(notes)} sentences.")
    if not notes:
        print("No notes found to train on.")
        return False

    backup_path = backup_model()

    try:
        if os.path.exists(MODEL_DIR):
            model = SentenceTransformer(MODEL_DIR)
            print("Loaded existing model for retraining...")
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Creating new model...")

        examples = [InputExample(texts=[n, n]) for n in notes]
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            show_progress_bar=True
        )

        # Save to temp folder first to avoid Windows file-lock issue
        temp_model_dir = MODEL_DIR + "_temp"
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        model.save(temp_model_dir)

        # Replace old model folder
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        os.rename(temp_model_dir, MODEL_DIR)

        print("Model training complete and saved.")
        return True

    except Exception as e:
        print(f"Error during training: {e}")
        if backup_path and os.path.exists(backup_path):
            if os.path.exists(MODEL_DIR):
                shutil.rmtree(MODEL_DIR)
            shutil.copytree(backup_path, MODEL_DIR)
            print(f"Restored backup from {backup_path}")
        return False

if __name__ == "__main__":
    train_model()
