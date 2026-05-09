import torch
import torch.nn as nn
import torch.optim as optim
import random

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. DATASET (100 AI-RELATED WORDS & PHRASES)
# =========================================================

vocab_list = [
    "nlp", "agentic ai", "machine learning", "deep learning", "tensorflow",
    "pytorch", "nvidia", "openai", "claude", "gemini",
    "csp", "planning", "neural networks", "logistic regression",
    "linear regression", "gradient descent", "weight update",
    "api calls", "knowledge representation", "reasoning",
    "chain of thought", "lstm", "llm", "hallucination",
    "chatbot personality", "autonomous agents", "codex",
    "claude code", "transformers", "attention mechanism",
    "self attention", "tokenization", "word embeddings",
    "feature engineering", "data preprocessing", "normalization",
    "batch training", "backpropagation", "loss function",
    "optimizer", "relu", "sigmoid", "softmax",
    "overfitting", "underfitting", "regularization",
    "dropout", "fine tuning", "transfer learning",
    "reinforcement learning", "q learning", "policy gradient",
    "reward function", "exploration", "exploitation",
    "state space", "action space", "environment",
    "multi agent systems", "planning algorithms",
    "constraint satisfaction", "search algorithms",
    "heuristics", "a star", "breadth first search",
    "depth first search", "probabilistic models",
    "bayesian networks", "markov chains",
    "hidden markov model", "speech recognition",
    "computer vision", "image classification",
    "object detection", "segmentation",
    "generative ai", "diffusion models",
    "gan", "autoencoders",
    "vector databases", "semantic search",
    "retrieval augmented generation",
    "prompt engineering", "zero shot learning",
    "few shot learning", "embedding space",
    "cosine similarity", "euclidean distance",
    "model deployment", "api integration",
    "scalability", "latency",
    "edge ai", "cloud ai",
    "distributed training", "gpu acceleration",
    "cuda", "parallel computing"
]

vocab_size = len(vocab_list)
word_to_idx = {w: i for i, w in enumerate(vocab_list)}

# =========================================================
# 2. GENERATE CO-OCCURRENCE PAIRS (SIMULATED CONTEXT)
# =========================================================

pairs = []

for _ in range(2000):
    w1, w2 = random.sample(vocab_list, 2)
    pairs.append((word_to_idx[w1], word_to_idx[w2]))

# =========================================================
# 3. EMBEDDING MODEL
# =========================================================

embedding_dim = 16

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)


model = EmbeddingModel(vocab_size, embedding_dim)

# =========================================================
# 4. TRAINING (MAKE SIMILAR WORDS CLOSER)
# =========================================================

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CosineEmbeddingLoss()

epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for w1, w2 in pairs:
        v1 = model(torch.tensor(w1))
        v2 = model(torch.tensor(w2))

        target = torch.tensor(1.0)  # similar pair

        loss = loss_fn(v1, v2, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================================================
# 5. SIMILARITY FUNCTION
# =========================================================

def get_similar_words(word, top_k=3):
    if word not in word_to_idx:
        return ["<UNK>"]

    idx = word_to_idx[word]
    target_vec = model.embeddings.weight[idx]

    similarities = []

    for i, other_word in enumerate(vocab_list):
        if other_word == word:
            continue

        other_vec = model.embeddings.weight[i]
        sim = torch.cosine_similarity(target_vec, other_vec, dim=0).item()
        similarities.append((other_word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in similarities[:top_k]]

# =========================================================
# 6. TEST: 5 RANDOM STRINGS (5 WORDS EACH)
# =========================================================

print("\n===== SIMILARITY TESTS =====")

for i in range(5):
    random_words = random.sample(vocab_list, 5)
    print(f"\nTest {i+1}: {random_words}")

    for word in random_words:
        similar = get_similar_words(word)
        print(f"Word: {word} → Similar: {similar}")

# =========================================================
# 7. EXPLANATION
# =========================================================

print("\n===== EXPLANATION =====")

print("""
How Embeddings Work:

- Each word is mapped to a dense vector (embedding)
- During training, similar words are pushed closer in vector space
- Cosine similarity measures closeness between vectors

Why Similar Words Get Similar Vectors:

- Words appearing in similar contexts (e.g., 'pytorch' and 'tensorflow')
  get similar updates during training
- The model learns relationships through co-occurrence patterns

Example:
- 'deep learning', 'neural networks', 'backpropagation'
  tend to cluster together

Key Insight:
Embeddings convert discrete words into continuous space,
where semantic meaning is captured geometrically.

This is the foundation of:
- NLP models
- Transformers (like GPT, BERT)
- Semantic search systems
""")