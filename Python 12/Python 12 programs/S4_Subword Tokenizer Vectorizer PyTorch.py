import torch
import re
from collections import defaultdict

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. DATASET (15 SENTENCES - ADVANCED TOKENIZATION FOCUS)
# =========================================================

sentences = [
    "Advanced tokenization handles punctuation correctly.",
    "Subword tokenization like BPE improves unknown word handling.",
    "WordPiece splits rare words into meaningful units.",
    "Punctuation such as commas, periods, and exclamation marks must be handled.",
    "Unknown words require contextual mapping to existing vocabulary.",
    "Tokenization must preserve semantic meaning across sentences.",
    "Subword units help in representing rare and unseen words.",
    "Normalization and tokenization improve NLP model performance.",
    "Efficient tokenization reduces vocabulary size significantly.",
    "Handling edge cases like hyphenated-words is important.",
    "Tokenization should separate punctuation from words.",
    "Unknown tokens can be mapped using similarity or embeddings.",
    "Context aware tokenization improves downstream tasks.",
    "Subword tokenization balances vocabulary and expressiveness.",
    "Robust tokenization ensures better generalization."
]

# =========================================================
# 2. ADVANCED TOKENIZER
# =========================================================

def tokenize(sentence):
    sentence = sentence.lower()

    # Handle punctuation (separate it)
    tokens = re.findall(r"\w+|[^\w\s]", sentence)

    return tokens


# =========================================================
# 3. SIMPLE SUBWORD TOKENIZATION (SIMULATED BPE-LIKE)
# =========================================================

def subword_tokenize(token):
    # If word is long, split into subwords
    if len(token) > 6 and token.isalpha():
        return [token[:4], token[4:]]
    return [token]


# =========================================================
# 4. BUILD VOCABULARY
# =========================================================

vocab = {}
index = 0

for sentence in sentences:
    tokens = tokenize(sentence)

    for token in tokens:
        sub_tokens = subword_tokenize(token)

        for st in sub_tokens:
            if st not in vocab:
                vocab[st] = index
                index += 1

# Add UNK token
vocab["<UNK>"] = index

print("\n===== VOCAB SIZE =====")
print(len(vocab))


# =========================================================
# 5. UNKNOWN WORD HANDLING (SIMILARITY FALLBACK)
# =========================================================

def get_closest_token(token):
    # simple similarity: match prefix
    for v in vocab:
        if token[:3] == v[:3]:
            return v
    return "<UNK>"


# =========================================================
# 6. VECTORIZE SENTENCE → PYTORCH TENSOR
# =========================================================

def vectorize(sentence):
    tokens = tokenize(sentence)

    indices = []

    for token in tokens:
        sub_tokens = subword_tokenize(token)

        for st in sub_tokens:
            if st in vocab:
                indices.append(vocab[st])
            else:
                closest = get_closest_token(st)
                indices.append(vocab[closest])

    return torch.tensor(indices)


# =========================================================
# 7. DISPLAY 5 SAMPLE SENTENCES
# =========================================================

print("\n===== SAMPLE VECTORIZATION OUTPUT =====")

for i in range(5):
    sentence = sentences[i]
    tensor = vectorize(sentence)

    print(f"\nSentence {i+1}: {sentence}")
    print("Tokens:", tokenize(sentence))
    print("Vector:", tensor)


# =========================================================
# 8. EXPLANATION
# =========================================================

print("\n===== EXPLANATION =====")

print("""
Vectorization Process:

1. Tokenization:
   - Lowercasing
   - Regex-based punctuation separation

2. Subword Tokenization:
   - Simulated BPE-like splitting
   - Long words split into smaller units

3. Vocabulary Creation:
   - Built from all tokens + subwords
   - Includes <UNK> for unseen tokens

4. Unknown Word Handling:
   - Prefix-based similarity matching
   - Falls back to <UNK> if no match found

5. PyTorch Integration:
   - Tokens converted into tensors
   - Ready for embedding layers

Why This Matters:

- Reduces vocabulary size
- Handles rare/unseen words
- Preserves contextual meaning
- Enables robust NLP pipelines

This is a simplified version of real systems like:
- BPE (Byte Pair Encoding)
- WordPiece (used in BERT)
""")