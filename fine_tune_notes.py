from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

NOTES_DIR = "notes"
OUTPUT_MODEL = "trained_notepad_model"

# 1. Collect all lines from all .txt notes
texts = []
for file in os.listdir(NOTES_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(NOTES_DIR, file), "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) > 10]
            texts.extend(lines)

if len(texts) < 2:
    raise ValueError("Not enough text found in your notes folder to train.")

# 2. Create basic positive pairs (same-file or nearby lines = similar)
train_examples = []
for i in range(0, len(texts) - 1, 2):
    train_examples.append(InputExample(texts=[texts[i], texts[i + 1]], label=1.0))

# 3. Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Prepare training setup
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 5. Fine-tune model
print("🚀 Training model on your personal notes...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    output_path=OUTPUT_MODEL
)

print(f"✅ Fine-tuned model saved at: {OUTPUT_MODEL}")
