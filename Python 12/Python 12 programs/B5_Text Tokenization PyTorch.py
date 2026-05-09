import torch

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. PREDEFINED SENTENCES (FOCUSED ON MODEL IMPROVEMENT)
# =========================================================

sentences = [
    "Adding non linearity using relu layers improves linear models",
    "Gradual capture of multi scale features enhances prediction accuracy",
    "Normalizing features stabilizes training and improves convergence",
    "Class balancing helps in handling imbalanced heatwave datasets",
    "Feature engineering with multi level atmospheric data boosts performance"
]

# =========================================================
# 2. BASIC TOKENIZATION FUNCTION
# =========================================================

def tokenize(sentence):
    sentence = sentence.lower()
    tokens = sentence.split()   # basic whitespace tokenization
    return tokens

# =========================================================
# 3. BUILD VOCABULARY
# =========================================================

vocab = {}
index = 0

for sentence in sentences:
    tokens = tokenize(sentence)
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1

print("\n===== VOCABULARY =====")
print(vocab)

# =========================================================
# 4. CONVERT TOKENS → TENSORS (PYTORCH)
# =========================================================

def tokens_to_tensor(tokens, vocab):
    indices = [vocab.get(token, -1) for token in tokens]  # unknown = -1
    return torch.tensor(indices)

# =========================================================
# 5. TOKENIZE PREDEFINED SENTENCES
# =========================================================

print("\n===== TOKENIZATION (PREDEFINED SENTENCES) =====")

for i, sentence in enumerate(sentences):
    tokens = tokenize(sentence)
    tensor = tokens_to_tensor(tokens, vocab)

    print(f"\nSentence {i+1}: {sentence}")
    print("Tokens:", tokens)
    print("Tensor:", tensor)

# =========================================================
# 6. USER INPUT TOKENIZATION
# =========================================================

user_input = input("\nEnter a sentence for tokenization: ")

user_tokens = tokenize(user_input)
user_tensor = tokens_to_tensor(user_tokens, vocab)

print("\n===== USER INPUT TOKENIZATION =====")
print("Tokens:", user_tokens)
print("Tensor:", user_tensor)

# =========================================================
# 7. EXPLANATION
# =========================================================

print("\n===== EXPLANATION =====")

print("""
Tokenization Process:
- Converted text → lowercase
- Split into words (basic tokenization)
- Built vocabulary from predefined sentences
- Converted tokens → numerical indices
- Represented tokens as PyTorch tensors

Why PyTorch?
- Enables seamless integration into deep learning pipelines
- Tokens can directly be fed into embedding layers
- Provides flexibility for custom NLP pipelines

Limitations:
- Basic tokenization (no punctuation handling)
- No subword tokenization (like BPE, WordPiece)
- Unknown words mapped to -1

This is a foundational step before building NLP models in PyTorch.
""")