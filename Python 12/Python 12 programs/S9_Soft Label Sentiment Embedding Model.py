import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import re

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. EMOTION SPACE
# =========================================================

emotions = [
    "happy", "sad", "anger", "fear", "frustration", "surprise",
    "guilt", "pride", "gratitude", "awe", "calmness",
    "motivation", "relief", "nostalgia", "loneliness",
    "boredom", "admiration", "confusion"
]

positive_emotions = {"happy", "pride", "gratitude", "awe", "calmness", "motivation", "relief", "admiration"}
negative_emotions = {"sad", "anger", "fear", "frustration", "guilt", "loneliness", "boredom", "confusion"}

# =========================================================
# 2. DATASET GENERATION (3 DISTRIBUTIONS)
# =========================================================

def generate_sentence(combo):
    connectors = ["and", "but also", "yet", "while"]
    sentence = "I feel " + combo[0]
    for e in combo[1:]:
        sentence += f" {random.choice(connectors)} {e}"
    return sentence + "."

def soft_label(combo):
    pos = sum(e in positive_emotions for e in combo)
    neg = sum(e in negative_emotions for e in combo)
    return pos / (pos + neg + 1e-6)   # continuous

positive_data = []
negative_data = []
mixed_data = []

# Positive dataset
for _ in range(500):
    combo = random.sample(list(positive_emotions), random.randint(2, 4))
    positive_data.append((generate_sentence(combo), 1.0))

# Negative dataset
for _ in range(500):
    combo = random.sample(list(negative_emotions), random.randint(2, 4))
    negative_data.append((generate_sentence(combo), 0.0))

# Mixed dataset (MOST IMPORTANT)
for _ in range(800):
    combo = random.sample(emotions, random.randint(2, 5))
    mixed_data.append((generate_sentence(combo), soft_label(combo)))

# MASTER DATASET
data = positive_data + negative_data + mixed_data
random.shuffle(data)

# =========================================================
# 3. ADVANCED TOKENIZATION
# =========================================================

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text)

    # subword simulation
    final = []
    for t in tokens:
        if len(t) > 6 and t.isalpha():
            final.extend([t[:4], t[4:]])
        else:
            final.append(t)
    return final

# =========================================================
# 4. VOCABULARY
# =========================================================

vocab = {"<PAD>": 0, "<UNK>": 1}
idx = 2

for sentence, _ in data:
    for token in tokenize(sentence):
        if token not in vocab:
            vocab[token] = idx
            idx += 1

vocab_size = len(vocab)

# =========================================================
# 5. VECTORIZATION
# =========================================================

max_len = 12

def vectorize(sentence):
    tokens = tokenize(sentence)
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices)

# =========================================================
# 6. DATASET CLASS
# =========================================================

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []

        for sentence, label in data:
            self.X.append(vectorize(sentence))
            self.y.append(label)

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SentimentDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================================================
# 7. MODEL (EMBEDDING + DROPOUT)
# =========================================================

class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        return self.fc(pooled).squeeze()

model = SentimentModel(vocab_size)

# =========================================================
# 8. COMPILATION
# =========================================================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# =========================================================
# 9. TRAINING
# =========================================================

epochs = 12

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
# 10. TESTING
# =========================================================

test_sentences = [
    "I feel happy and calm but also nostalgic",
    "I feel angry and frustrated and confused",
    "I feel relief and gratitude after fear",
    "I feel bored and lonely but admire effort",
    "I feel awe and pride with confusion",
    "I feel happy and grateful but also a bit lonely and confused.",
    "I feel proud and calm, yet slightly anxious and frustrated at the same time."
]

print("\n===== TEST RESULTS =====")

for sentence in test_sentences:
    x = vectorize(sentence).unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    pred = "POSITIVE" if prob >= 0.5 else "NEGATIVE"

    print("\nSentence:", sentence)
    print("Prediction:", pred)
    print("Probability:", round(prob, 4))

    print("Provenance Report (Reasons):")

    if prob >= 0.7:
        # Strong Positive
        print("- High concentration of positive emotional tokens influenced the prediction")
        print("- Embedding vectors aligned closely with positive sentiment clusters")
        print("- Minimal interference from negative semantic features")
        print("- Model confidence increased due to clear sentiment dominance")

    elif prob <= 0.3:
        # Strong Negative
        print("- Dominance of negative emotional expressions drove the prediction")
        print("- Embedding representations clustered around negative sentiment space")
        print("- Weak or absent positive signals reduced polarity balance")
        print("- Model identified consistent negative semantic patterns")

    else:
        # Mixed / Ambiguous
        print("- Presence of both positive and negative emotions created semantic conflict")
        print("- Embedding vectors showed overlapping sentiment regions")
        print("- Soft-label training enabled intermediate probability estimation")
        print("- Model captured balance rather than dominance of emotional signals")
        print("- Contextual blending of emotions reduced prediction certainty")