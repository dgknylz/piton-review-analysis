"""Text preprocessing helpers used by the notebook, training, and app."""

from __future__ import annotations

import argparse
import re
from functools import lru_cache


FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def _as_text(text: str | object) -> str:
    if text is None:
        return ""
    return str(text)


def ensure_nltk_resources(download: bool = False) -> None:
    """Optionally download NLTK resources outside CI/lightweight test runs."""
    if not download:
        return

    import nltk

    for package in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _get_stop_words() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return FALLBACK_STOP_WORDS


@lru_cache(maxsize=1)
def _get_wordnet_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")
        return lemmatizer
    except Exception:
        return None


def lowercase_text(text: str) -> str:
    """Lowercase a text value."""
    return _as_text(text).lower()


def clean_text(text: str) -> str:
    """Remove HTML fragments, punctuation-like characters, and extra whitespace."""
    text = _as_text(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove common English stop words."""
    stop_words = _get_stop_words()
    tokens = [token for token in _as_text(text).split() if token.lower() not in stop_words]
    return " ".join(tokens)


def _fallback_lemmatize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def lemmatize_text(text: str) -> str:
    """Lemmatize tokens with NLTK WordNet when available, otherwise use a light fallback."""
    lemmatizer = _get_wordnet_lemmatizer()
    tokens = _as_text(text).split()
    if lemmatizer is None:
        return " ".join(_fallback_lemmatize_token(token) for token in tokens)
    return " ".join(lemmatizer.lemmatize(token) for token in tokens)


def preprocess_text(text: str) -> str:
    """Apply the full preprocessing pipeline."""
    lowered = lowercase_text(text)
    cleaned = clean_text(lowered)
    no_stopwords = remove_stopwords(cleaned)
    return lemmatize_text(no_stopwords)


def demonstrate_preprocessing_steps(text: str) -> dict[str, str]:
    """Return each intermediate preprocessing step for notebook display."""
    lowercased = lowercase_text(text)
    cleaned = clean_text(lowercased)
    without_stopwords = remove_stopwords(cleaned)
    lemmatized = lemmatize_text(without_stopwords)
    return {
        "raw": _as_text(text),
        "lowercased": lowercased,
        "cleaned": cleaned,
        "stopword_removed": without_stopwords,
        "lemmatized": lemmatized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage preprocessing resources.")
    parser.add_argument("--download-nltk", action="store_true", help="Download optional NLTK corpora.")
    args = parser.parse_args()
    ensure_nltk_resources(download=args.download_nltk)
    if args.download_nltk:
        print("NLTK resources downloaded.")


if __name__ == "__main__":
    main()

