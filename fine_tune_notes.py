import os
import nltk
from sentence_transformers import SentenceTransformer, models

nltk.download('punkt')

MODEL_DIR = "model"
PRETRAINED_MODEL = "all-MiniLM-L6-v2"

def setup_model():
    """Download or load the SentenceTransformer model."""
    model_path = os.path.join(MODEL_DIR, PRETRAINED_MODEL)
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        if os.path.exists(model_path):
            print("Loading existing model...")
            model = SentenceTransformer(model_path)
        else:
            print("Downloading pretrained model...")
            word_embedding_model = models.Transformer(PRETRAINED_MODEL)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            model.save(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using fallback pretrained model from HuggingFace...")
        model = SentenceTransformer(PRETRAINED_MODEL)
        model.save(model_path)
        return model

def train_model():
    model = setup_model()
    print("Model ready for fine-tuning or embedding creation.")

if __name__ == "__main__":
    train_model()
