import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. EMOTION SPACE (FEATURE BASIS)
# =========================================================

emotions = [
    "happy", "sad", "anger", "fear", "frustration", "surprise",
    "guilt", "pride", "gratitude", "awe", "calmness",
    "motivation", "relief", "nostalgia", "loneliness",
    "boredom", "admiration", "confusion"
]

# Emotion polarity mapping (blended sentiment logic)
positive_emotions = {"happy", "pride", "gratitude", "awe", "calmness", "motivation", "relief", "admiration"}
negative_emotions = {"sad", "anger", "fear", "frustration", "guilt", "loneliness", "boredom", "confusion"}

# =========================================================
# 2. DATASET GENERATION (1000 SAMPLES)
# =========================================================

def generate_sentence(emotion_combo):
    return "I feel " + " and ".join(emotion_combo)

def label_emotions(combo):
    pos = sum(e in positive_emotions for e in combo)
    neg = sum(e in negative_emotions for e in combo)

    return 1 if pos >= neg else 0   # positive / negative

data = []

for _ in range(1000):
    k = random.randint(2, 4)  # mix of emotions
    combo = random.sample(emotions, k)
    sentence = generate_sentence(combo)
    label = label_emotions(combo)
    data.append((sentence, combo, label))

# =========================================================
# 3. FEATURE ENGINEERING (EMOTION → VECTOR)
# =========================================================

emotion_to_idx = {e: i for i, e in enumerate(emotions)}

def encode_emotions(combo):
    vec = np.zeros(len(emotions))
    for e in combo:
        vec[emotion_to_idx[e]] += 1
    return vec

# =========================================================
# 4. DATASET CLASS (WITH NORMALIZATION)
# =========================================================

class SentimentDataset(Dataset):
    def __init__(self, data):
        X = []
        y = []

        for _, combo, label in data:
            X.append(encode_emotions(combo))
            y.append(label)

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

        # NORMALIZATION (VERY IMPORTANT)
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


dataset = SentimentDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================================================
# 5. MODEL (LINEAR + NON-LINEARITY FOR FLEXIBILITY)
# =========================================================

class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


model = SentimentModel(len(emotions))

# =========================================================
# 6. COMPILATION
# =========================================================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================================================
# 7. TRAINING
# =========================================================

epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for X_batch, y_batch in dataloader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================================================
# 8. EVALUATION
# =========================================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in dataloader:
        outputs = model(X_batch)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total

print("\nAccuracy:", round(accuracy, 4))

# =========================================================
# 9. TEST CASES (MIXED EMOTIONS)
# =========================================================

test_sentences = [
    ["happy", "gratitude", "calmness"],
    ["anger", "frustration", "confusion"],
    ["nostalgia", "loneliness", "awe"],
    ["fear", "relief", "gratitude"],
    ["boredom", "sad", "admiration"]
]

print("\n===== TEST RESULTS =====")

for combo in test_sentences:
    vec = encode_emotions(combo)

    # normalize using training stats
    vec = (vec - dataset.mean) / dataset.std

    x = torch.tensor(vec, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0

    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"

    print("\nSentence:", "I feel " + " and ".join(combo))
    print("Prediction:", sentiment)
    print("Probability:", round(prob, 4))

    # =====================================================
    # PROVENANCE REPORT (ONLY REASONS)
    # =====================================================

    print("Provenance Report (Reasons):")

    pos_count = sum(e in positive_emotions for e in combo)
    neg_count = sum(e in negative_emotions for e in combo)

    print("- Positive emotion count:", pos_count)
    print("- Negative emotion count:", neg_count)
    print("- Dominant polarity determined by higher count")
    print("- Non-linear layer captured interaction between mixed emotions")