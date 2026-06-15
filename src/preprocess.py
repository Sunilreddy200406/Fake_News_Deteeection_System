import re
from collections import Counter
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "chunkers/maxent_ne_chunker": "maxent_ne_chunker",
        "corpora/words": "words",
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> List[str]:
    tokens = word_tokenize(text)
    filtered = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatized = [lemmatizer.lemmatize(token) for token in filtered]
    return lemmatized


def preprocess_text(text: str) -> str:
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned, stop_words, lemmatizer)
    return " ".join(tokens)


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    processed = preprocess_text(text)
    tokens = processed.split()
    if not tokens:
        return []
    frequencies = Counter(tokens)
    return [word for word, _ in frequencies.most_common(top_k)]


def extract_entities(text: str) -> List[str]:
    ensure_nltk_resources()
    text_for_ner = re.sub(r"\s+", " ", text).strip()
    if not text_for_ner:
        return []

    tokens = word_tokenize(text_for_ner)
    tagged = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(tagged, binary=False)

    entities = []
    for chunk in chunks:
        if hasattr(chunk, "label"):
            entity = " ".join(c[0] for c in chunk)
            entities.append(entity)

    # Keep unique entities while preserving order.
    seen = set()
    deduped = []
    for item in entities:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:10]
