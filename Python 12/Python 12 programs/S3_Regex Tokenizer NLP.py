import re

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. DATASET (5 COMPLEX SENTENCES)
# =========================================================

sentences = [
    "Advanced sentiment-analysis pipelines utilize subword-extraction and vectorization methodologies for high-dimensional representations.",
    "Soft-label approximation combined with embedding regularization improves generalization in sentiment-classification systems.",
    "Vocabulary-expansion strategies and tokenization-mechanisms handle out-of-distribution generalization effectively.",
    "Dropout-regularization within neural-network architectures enhances robustness against overfitting in sentiment-prediction models.",
    "Rectified Linear Unit activation facilitates non-linear transformation during embedding-vector optimization processes."
]

# =========================================================
# 2. TOKENIZATION FUNCTION
# =========================================================

def tokenize(sentence):
    # Lowercase for normalization
    sentence = sentence.lower()

    # Handle phrases (keep hyphenated words together)
    tokens = re.findall(r"[a-zA-Z\-]+|[.,]", sentence)

    return tokens

# =========================================================
# 3. PROCESS & DISPLAY TOKENS
# =========================================================

print("\n===== TOKENIZATION OUTPUT =====")

for i, sentence in enumerate(sentences):
    tokens = tokenize(sentence)

    print(f"\nSentence {i+1}:")
    print(sentence)
    print("Tokens:", tokens)

# =========================================================
# 4. EXPLANATION
# =========================================================

print("\n===== EXPLANATION =====")

print("""
Tokenization Strategy Used:

1. Lowercasing:
   - Ensures uniform representation

2. Regex-based extraction:
   - Captures words and punctuation
   - Preserves hyphenated complex terms (e.g., 'sentiment-analysis')

3. Handles long words and phrases:
   - Keeps domain-specific terminology intact

Why Important for ML:

- Converts raw text into structured units
- Enables downstream processes like:
    • vectorization
    • embedding
    • sentiment classification

This step is essential before applying any machine learning model.
""")