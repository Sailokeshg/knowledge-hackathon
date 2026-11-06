"""
Retriever: finds the evidence for a question, as the website stood on a given snapshot.

Steps
  1. BM25 keyword search over the chunks visible in that snapshot.
  2. Re-rank: multiply relevance by how trustworthy and how fresh the source is.
  3. Collapse near-duplicates. The same rate table is often copied onto several pages
     (rates page, product page, calculator, promo page). If the copies show the SAME numbers
     we keep one and cite all of them. If they show DIFFERENT numbers, one copy is stale:
     we keep the most trustworthy, most recently dated copy and note the disagreement.
  4. Drop calculators / listing pages when better sources exist, and keep the evidence
     diverse (at most a few chunks per page).
"""
import re
from dataclasses import dataclass, field
from typing import List, Set

from .chunker import Chunk
from .pages import SourcePolicy
from .search import BM25Index
from .store import VersionStore

NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
NUMBER_OR_DATE = re.compile(r"\$?\s?\d[\d,./]*%?")
RATE_QUESTION = re.compile(r"\b(rates?|apy|apr|yield|dividends?|fees?|cost|price|percent|interest)\b", re.I)


@dataclass
class Evidence:
    """One piece of evidence: the winning chunk plus what we learned while choosing it."""
    chunk: Chunk
    score: float
    also_on: List[str] = field(default_factory=list)       # other URLs with the same content
    overruled: List[Chunk] = field(default_factory=list)   # stale copies with other numbers


class Retriever:
    CANDIDATES = 100         # how many BM25 hits we look at
    TABLE_BOOST = 1.5        # for rate / fee questions
    DUPLICATE_OVERLAP = 0.6  # similarity above which two chunks are "the same content"
    MAX_PER_PAGE = 3         # at most this many chunks from one URL
    MIN_TRUST = 0.5          # calculators / listing pages are used only if nothing better matches

    def __init__(self, store: VersionStore, index: BM25Index, policy: SourcePolicy):
        self.store, self.index, self.policy = store, index, policy

    def retrieve(self, question: str, snapshot_id: str, k: int = 8) -> List[Evidence]:
        today = self.store.snapshot_date(snapshot_id)
        hits = self.index.search(question, lambda cid: self.store.is_live(cid, snapshot_id), self.CANDIDATES)

        # Step 2: weight relevance by source trust and freshness.
        # A question about a rate / APY / fee is best answered by a rate table, so tables get a boost.
        wants_numbers = bool(RATE_QUESTION.search(question))
        scored = []
        for chunk_id, relevance in hits:
            chunk = self.store.chunks[chunk_id]
            removed = not self.store.is_crawled(chunk.url, snapshot_id)
            score = relevance * self.policy.search_weight(chunk.page, today, removed)
            if wants_numbers and chunk.is_table:
                score *= self.TABLE_BOOST
            scored.append((score, chunk))
        scored.sort(key=lambda pair: -pair[0])

        # Step 3: group near-duplicates; each group is led by its most current copy.
        groups: List[List[tuple]] = []
        signatures = []
        for score, chunk in scored:
            signature = self._signature(chunk)
            for group, group_signature in zip(groups, signatures):
                if self._same_content(chunk, group[0][1], signature, group_signature):
                    group.append((score, chunk))
                    break
            else:
                groups.append([(score, chunk)])
                signatures.append(signature)

        evidence = [self._pick_winner(group, snapshot_id) for group in groups]
        evidence.sort(key=lambda e: -e.score)

        # Step 4: drop weak sources (calculators, listing pages) when better ones exist,
        # and keep the evidence diverse - no single page may fill all the slots.
        trusted = [e for e in evidence if e.chunk.page.trust >= self.MIN_TRUST]
        if len(trusted) >= 3:
            evidence = trusted
        chosen, per_page = [], {}
        for item in evidence:
            if per_page.get(item.chunk.url, 0) < self.MAX_PER_PAGE:
                chosen.append(item)
                per_page[item.chunk.url] = per_page.get(item.chunk.url, 0) + 1
        return chosen[:k]

    # ------------------------------------------------------------------ helpers

    def _pick_winner(self, group, snapshot_id) -> Evidence:
        """Inside a group of near-identical chunks, keep the copy most likely to be current."""
        def currency(item):
            chunk = item[1]
            present = self.store.is_crawled(chunk.url, snapshot_id)
            return self.policy.freshness_key(chunk.page, chunk.effective, present) + (item[0],)

        best_score = max(score for score, _ in group)
        winner = max(group, key=currency)[1]
        result = Evidence(winner, best_score)
        for _, chunk in group:
            if chunk is winner:
                continue
            if self._numbers(chunk) == self._numbers(winner):
                if chunk.url != winner.url and chunk.url not in result.also_on:
                    result.also_on.append(chunk.url)
            else:
                result.overruled.append(chunk)
        return result

    @staticmethod
    def _signature(chunk: Chunk) -> Set[str]:
        """What a chunk says with every number replaced by '#'.

        Two copies of the same content share a signature even when one copy has stale numbers.
          tables: the set of body rows   "| #-# model year | # months | # apr |"
          text:   the set of word triples "has over #", "over # employees", ...
        """
        masked = NUMBER_OR_DATE.sub("#", chunk.text.lower().replace("–", "-"))
        if chunk.is_table:
            rows = [l for l in masked.split("\n") if l.startswith("|") and not re.match(r"^\|[\s\-|:]+$", l)]
            return {re.sub(r"\s+", " ", row) for row in rows[1:]}       # skip the header row
        words = re.findall(r"[a-z#]+", masked)
        return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}

    def _same_content(self, a: Chunk, b: Chunk, sig_a: Set[str], sig_b: Set[str]) -> bool:
        if a.is_table != b.is_table or self._is_business(a) != self._is_business(b):
            return False   # a business rate table is never a stale copy of a personal one
        if not sig_a or not sig_b:
            return False
        return len(sig_a & sig_b) / len(sig_a | sig_b) >= self.DUPLICATE_OVERLAP

    @staticmethod
    def _is_business(chunk: Chunk) -> bool:
        first_line = chunk.text.split("\n")[0]
        return "business" in f"{chunk.url} {chunk.heading} {first_line}".lower()

    @staticmethod
    def _numbers(chunk: Chunk) -> Set[str]:
        return set(NUMBER.findall(chunk.text))
