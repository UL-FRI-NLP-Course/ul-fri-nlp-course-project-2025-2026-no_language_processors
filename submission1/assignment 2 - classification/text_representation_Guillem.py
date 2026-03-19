"""
Text Representation Techniques Assignment (Student Version)
===========================================================

Implement core text-representation methods for spam classification.

Covers:
- text preprocessing (tokenisation, normalisation, stop-word removal)
- vocabulary construction
- Bag-of-Words (Count Vectorizer) from scratch
- TF-IDF from scratch (with data-leakage-safe IDF)
- hand-crafted feature engineering

Complete all TODO sections.
"""
import math
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("spam.csv")

STOP_WORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "i me my we our you your he she it his her its they them their "
    "this that these those and but or nor so yet for of in on at to "
    "by with from up out about into through during before after above "
    "below between each few more most other some such no than then "
    "too very just here there what which who whom when where why how "
    "all both each few more most other some such only own same so "
    "than too s t ll re ve d m".split()
)


# ---------------------------------------------------------------------------
# Step 1 - Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load the SMS spam CSV and return (texts, labels) where label 1 = spam."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    texts = df["text"].astype(str).tolist()
    labels = (df["label"].str.lower() == "spam").astype(int).tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# Step 2 - Preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Normalise and tokenise a single SMS message.

    1. Lowercase.
    2. Remove punctuation and digits (keep alphabetic only).
    3. Split on whitespace.
    4. Optionally filter stop words.
    5. Filter single-character tokens.
    """
    # TODO: lowercase text
    text = text.lower()

    # TODO: remove punctuation and digits
    text = re.sub(r"[\d.]", "", text)

    # TODO: split into tokens
    tokens: list[str] = text.split()

    # TODO: if remove_stopwords is True, remove tokens from STOP_WORDS
    if remove_stopwords: 
        tokens = [token for token in tokens if token not in STOP_WORDS]

    # TODO: remove single-character tokens
    for token in tokens:
        if len(token) == 1:
            tokens.remove(token)

    return tokens


# ---------------------------------------------------------------------------
# Step 3 - Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(corpus: list[str], max_vocab: int = 3000) -> dict[str, int]:
    """
    Build a token -> index mapping from the training corpus.

    1. Preprocess every document.
    2. Count token frequencies.
    3. Keep the top max_vocab tokens.
    4. Return {token: index} ordered by descending frequency.
    """

    tokens = [token for sentence in corpus for token in preprocess(sentence)]    
    frequencies = Counter(tokens)
    
    top_tokens = frequencies.most_common(max_vocab)
    
    vocab_mapping = {token: i for i, (token, count) in enumerate(top_tokens)}

    return vocab_mapping

# ---------------------------------------------------------------------------
# Step 4 - Bag-of-Words (Count Vectorizer)
# ---------------------------------------------------------------------------

def count_vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    """
    Convert texts to a count (Bag-of-Words) matrix.

    Returns numpy array of shape (n_docs, vocab_size), dtype int32.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)

    # TODO: for each document:
    # - preprocess document
    # - increment matrix[i, vocab[token]] for each token found in vocab

    for documents in texts: 
        tokens = preprocess(documents)
        for token in tokens: 
            if token in vocab: 
                matrix[texts.index(documents), vocab[token]] += 1

    return matrix


# ---------------------------------------------------------------------------
# Step 5 - TF-IDF
# ---------------------------------------------------------------------------

def compute_idf(corpus_tokens: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    """
    Compute the IDF vector for all vocabulary tokens.

    Formula (smoothed):
        IDF(t) = log((1 + N) / (1 + df(t))) + 1
    """
    n_docs = len(corpus_tokens)
    vocab_size = len(vocab)
    df = np.zeros(vocab_size, dtype=np.float64)

    # TODO: compute document frequency per token
    # Hint: use set(tokens) so each token counts once per document
    for doc in corpus_tokens:
        unique_tokens = set(doc)
        for token in unique_tokens:
            if token in vocab:
                idx = vocab[token] 
                df[idx] += 1
    
    # TODO: compute and return smoothed IDF vector
    return np.log((1 + n_docs) / (1 + df)) + 1

def tfidf_vectorize(
    texts: list[str],
    vocab: dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """
    Convert texts to a TF-IDF matrix.

        TF(t, d)     = count(t, d) / len(d)
        TF-IDF(t, d) = TF(t, d) * IDF(t)

    IDF is passed in (pre-computed on training data) to prevent leakage.

    Returns numpy array of shape (n_docs, vocab_size), dtype float64.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

    for text in texts: 
        tokens = preprocess(text)
        doc_len = len(tokens)

        counts = Counter(tokens)

        for token, count in counts.items():
            if token in vocab: 
                matrix[texts.index(text), vocab[token]] += count / doc_len * idf[vocab[token]]
    
    return matrix 


# ---------------------------------------------------------------------------
# Step 6 - Custom hand-crafted features
# ---------------------------------------------------------------------------

_SPAM_WORDS = frozenset(
    "free win winner won cash prize claim urgent call txt text reply "
    "guaranteed offer discount limited mobile ringtone download bonus "
    "selected congratulations awarded voucher reward collect".split()
)


def extract_custom_features(text: str) -> list[float]:
    """
    Hand-crafted feature vector for one SMS message (9 features).

    Suggested features:
    1. Message length (characters)
    2. Number of tokens
    3. Number of digits
    4. Number of uppercase characters
    5. Uppercase character ratio
    6. Number of punctuation marks (!, ?, .)
    7. Number of currency symbols ($, GBP, EUR)
    8. Count of spam-indicator words (lowercased)
    9. Type-token ratio (unique tokens / total tokens)
    """
    # TODO: compute all 9 features and return as list[float]
    # Tip: protect against division by zero for empty text/tokens.
    message_length = len(text)
    tokens = preprocess(text)
    num_tokens = len(tokens)
    num_digits = sum(c.isdigit() for c in text)
    num_uppercase = sum(1 for c in text if c.isupper())
    num_lowercase = sum(1 for c in text if c.islower()) 
    uppercase_ratio = num_uppercase / num_lowercase
    num_punctuation = sum(1 for c in text if c in string.punctuation)
    num_currency = sum(1 for c in text if c in "$£€")
    num_spam_words = sum(1 for token in tokens if token.lower() in _SPAM_WORDS)
    type_token_ratio = len(set(tokens)) / num_tokens

    indicators = [message_length, num_tokens, num_digits, num_uppercase, uppercase_ratio, num_punctuation, num_currency, num_spam_words, type_token_ratio]

    return indicators



# ---------------------------------------------------------------------------
# Step 7 - Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    """Train logistic regression and print a classification report."""

    # TODO: train LogisticRegression(max_iter=1000, random_state=42)
    # TODO: predict on X_test
    # TODO: compute macro F1
    # TODO: print section header and classification_report
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print('\n \n')
    print(f'Evaluating: {label}')
    print(70*'-')

    print(50*'=')
    print('Logistic regression for spam text classification')
    print(f'F1 score is {macro_f1}')
    print(50*'=')

    print('')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

def evaluate_KNN(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    """Train logistic regression and print a classification report."""

    # TODO: train LogisticRegression(max_iter=1000, random_state=42)
    # TODO: predict on X_test
    # TODO: compute macro F1
    # TODO: print section header and classification_report
    
    model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print('\n \n')
    print(f'Evaluating: {label}')
    print(70*'-')

    print(50*'=')
    print('KNN for spam text classification')
    print(f'F1 score is {macro_f1}')
    print(50*'=')

    print('')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading dataset...")
    texts, labels = load_dataset(DATA_PATH)

    print(f"  Total samples : {len(texts)}")
    print(f"  Spam          : {sum(labels)}  ({sum(labels)/len(labels):.1%})")
    print(f"  Ham           : {len(labels)-sum(labels)}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )

    # TODO: build vocabulary from X_train_raw only (avoid leakage)
    vocab: dict[str, int] = build_vocabulary(X_train_raw, max_vocab = 1000)

    # --- Bag-of-Words ---
    # TODO: vectorize train/test with cou
    # nt_vectorize and evaluate
    count_vectorizer_train = count_vectorize(X_train_raw, vocab)
    count_vectorizer_test = count_vectorize(X_test_raw, vocab)

    # --- TF-IDF ---
    # TODO: preprocess X_train_raw -> tokens
    tokens_corpus = [] 
    for text in X_train_raw:
        tokens = preprocess(text)
        tokens_corpus.append(tokens)

    # TODO: compute idf on training tokens only
    idf = compute_idf(tokens_corpus, vocab)

    # TODO: vectorize train/test with tfidf_vectorize and evaluate
    tfidf_vectorize_train = tfidf_vectorize(X_train_raw, vocab, idf)
    tfidf_vectorize_test = tfidf_vectorize(X_test_raw, vocab, idf)


    # --- Custom features ---
    # TODO: build numpy feature matrices from extract_custom_features
    # TODO: evaluate custom features
    
    evaluate(count_vectorizer_train, count_vectorizer_test, y_train, y_test, "Bag-of-Words")
    evaluate(tfidf_vectorize_train, tfidf_vectorize_test, y_train, y_test, "TF-IDF")
    
    
    # --- Custom approach (optional) ---
    # TODO: add your own representation idea and evaluate it 
    # (e.g., use scikit-learn tf-idf implementation and try to find the best hyperparameter values)
    # (run script using machine learning models)
    evaluate_KNN(count_vectorizer_train, count_vectorizer_test, y_train, y_test, "Bag-of-Words")
    evaluate_KNN(tfidf_vectorize_train, tfidf_vectorize_test, y_train, y_test, "TF-IDF")
    

if __name__ == "__main__":
    main()

