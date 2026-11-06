"""
Keyword search (BM25) written in plain Python.

Why BM25 and not embeddings? Member questions are full of exact terms that must match
exactly: "17-month", "jumbo", "APY", "Galena", "routing number". Keyword search is
precise for that, needs no model download or API call, and answers in milliseconds.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

STOPWORDS = set("""
a an the and or of to in on for at by with from as is are was were be been it its this that these
those what which who whom whose how when where why do does did can could would should will may
you your yours i me my we our us they them their there here please tell give get know about
any some all much many dupaco s has have had like would want need right now today current
currently also just still im let general
""".split())

# Member words -> the words the website uses for the same thing (added to the query).
QUERY_EXPANSIONS = {
    "mortgage": "home loan",
    "branches": "branch office locations",
    "credit card": "visa",
}

# Different words members use for the same thing.
SYNONYMS = {
    "cd": "certificate", "cds": "certificate", "mo": "month", "yr": "year", "yrs": "year",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "eighteen": "18", "twenty": "20", "thirty": "30", "sixty": "60",
    "interest": "rate",
}

WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
RATE_OR_PRICE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?%|\d+\.\d+")


def searchable_text(text: str) -> str:
    """Text used for the search index: rate and price VALUES are removed.

    Members search with words ("17-month special certificate"), not with the values
    we are trying to find. Without the values, a long rate table is not penalised
    for its length, and stale numbers cannot attract a search hit.
    """
    return RATE_OR_PRICE.sub(" ", text)


class Tokenizer:
    """Lower-cases, splits into words, maps synonyms, and stems.

    Stemming = cut every word to its first 6 letters, so "recognition" / "recognized" and
    "eligible" / "eligibility" match. Crude, but simple and predictable.
    """
    STEM_LENGTH = 6

    def tokens(self, text: str) -> List[str]:
        words: List[str] = WORD.findall(text.lower())
        out = []
        for word in words:
            word = SYNONYMS.get(word, word)
            if word not in STOPWORDS:
                out.append(self._stem(word))
        return out

    def _stem(self, word: str) -> str:
        if word[0].isdigit() or len(word) <= 3:
            return word
        if word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        return word[: self.STEM_LENGTH]


class BM25Index:
    """Classic BM25 ranking over an inverted index (term -> list of (doc_id, term count))."""

    def __init__(self, tokenizer: Tokenizer, k1: float = 1.2, b: float = 0.75):
        self.tokenizer = tokenizer
        self.k1, self.b = k1, b
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_length: Dict[int, int] = {}
        self._idf: Dict[str, float] = {}
        self._length_norm: Dict[int, float] = {}
        self._dirty = True

    def add(self, doc_id: int, text: str) -> None:
        counts = Counter(self.tokenizer.tokens(text))
        self.doc_length[doc_id] = sum(counts.values())
        for term, count in counts.items():
            self.postings[term].append((doc_id, count))
        self._dirty = True

    def _refresh_stats(self) -> None:
        """Recompute IDF and average length after new documents were added."""
        n = len(self.doc_length) or 1
        avg_length = sum(self.doc_length.values()) / n or 1.0
        self._idf = {
            term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }
        # Long documents are penalised a little so they don't win just by containing more words.
        self._length_norm = {
            doc_id: self.k1 * (1 - self.b + self.b * length / avg_length)
            for doc_id, length in self.doc_length.items()
        }
        self._dirty = False

    def search(self, query: str, allowed: Callable[[int], bool], top_n: int = 50) -> List[Tuple[int, float]]:
        """Return the best (doc_id, score) pairs among documents where allowed(doc_id) is True."""
        if self._dirty:
            self._refresh_stats()
        for phrase, extra in QUERY_EXPANSIONS.items():
            if phrase in query.lower():
                query += " " + extra
        terms = set(self.tokenizer.tokens(query))
        scores: Dict[int, float] = defaultdict(float)
        matched: Dict[int, int] = defaultdict(int)
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, count in self.postings[term]:
                scores[doc_id] += idf * count * (self.k1 + 1) / (count + self._length_norm[doc_id])
                matched[doc_id] += 1
        # Coordination factor: a chunk containing more of the question's words ranks higher
        # (so "lobby" + "hours" beats a chunk that just says "hours" many times).
        ranked = sorted(((d, s * matched[d] / len(terms)) for d, s in scores.items() if allowed(d)),
                        key=lambda x: -x[1])
        return ranked[:top_n]
