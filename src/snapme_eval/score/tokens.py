"""Compare sets of food words using the SNAPMe paper's token F1 calculation.

The stopword list, punctuation handling, WordNet lemmatizer and F1 formula follow the authors' notebook,
JulesLarke-USDA/SNAPMe/scripts/ingredient_prediction_Inverse_Cooking.ipynb. Callers select the reference:
raw ASA24 descriptions for token_f1, or published mixed-dish ingredient sets for token_f1_2023.
"""
from __future__ import annotations

import re
import string
from functools import cache

PUNCT = string.punctuation[0:11] + string.punctuation[13:]  # keeps '-' as the authors did
STOPWORDS = {'ns', 'nfs', '', 'dry', 'water', 'black', 'brown', 'bayo', 'cut', 'cooked', 'cooking', 'as',
             'to', 'of', 'flavor', 'fat', 'eaten', 'made', 'with', 'raw', 'brewed', 'fried', 'toasted',
             'boiled', 'from', 'fresh', 'type', 'coated', 'baked', 'or', 'broiled', 'part', 'and', 'method',
             'skin', 'not', 'stewed', 'canned', 'table', 'reduced', 'added', 'in', 'hot', 'granulated',
             'ground', 'Mexican', 'blend', 'flavored', 'home', 'recipe', 'purchased', 'c', 'at', 'a',
             'bakery', 'replacement', 'free', 'powder', '0', '1', '2', '20', '99', '100'}


@cache
def _lemmatizer():
    import os

    import nltk
    from nltk.stem import WordNetLemmatizer

    if "SNAPME_ROOT" in os.environ:   # where `snapme-eval fetch` puts WordNet; NLTK_DATA is read by nltk itself
        nltk.data.path.append(os.path.join(os.environ["SNAPME_ROOT"], "env", "nltk_data"))
    return WordNetLemmatizer()


def clean(text: str) -> set[str]:
    text = "".join(ch for ch in text.lower() if ch not in PUNCT)
    toks = re.split(r"[-\W+]", text)
    wn = _lemmatizer()
    return {wn.lemmatize(t) for t in toks if t not in STOPWORDS}


def f1(truth: set[str], pred: set[str]) -> float:
    """The authors' string_detection(a=truth, b=pred)."""
    tp = sum(any(m == t for m in truth) for t in pred)
    fn = len(truth) - tp
    fp = len(pred) - sum(any(m == t for m in pred) for t in truth)
    try:
        p, r = tp / (tp + fp), tp / (tp + fn)
        return 2 * p * r / (p + r)
    except ZeroDivisionError:
        return 0.0
