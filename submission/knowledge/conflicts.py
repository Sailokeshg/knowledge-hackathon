"""
Conflict detection: where does the website contradict itself, and who is right?

1. FactExtractor turns content into small facts: (key, value).
     table row     "| 10 Month Fixed Jumbo Certificate Special | 3.96% | 4.00% |"
                     -> ("personal: 10-month fixed jumbo certificate special: apy", "4.00%")
     label line    "6-month certificate APY: 3.50%"
                     -> ("personal: 6-month certificate apy", "3.50%")
     boilerplate   "It has ... 24 branch office locations"
                     -> ("number of dupaco branch office locations", "24")
2. ConflictDetector groups facts by key across every page and every snapshot.
   A key with two or more different values is a conflict. The winner is chosen with the
   same SourcePolicy rules the search uses, so answers and conflicts always agree.
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .chunker import Chunk
from .pages import SourcePolicy
from .store import VersionStore

VALUE = re.compile(r"^\$?\d[\d,]*(?:\.\d+)?%?$")
LABEL_LINE = re.compile(r"^\s*([A-Za-z0-9][^:|\n]{3,60}):\s*(\$?\d[\d,]*(?:\.\d+)?%?)\s*$", re.M)


def normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"(\d+)[ -](month|year)s?\b", r"\1-\2", text)     # "10 Month" -> "10-month"
    text = re.sub(r"(\d),(\d)", r"\1\2", text)                        # "$100,000" -> "$100000"
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip(" :|")


def same_number(a: str, b: str) -> bool:
    """'7.29%' and '7.29' (or '$2.5' and '$2.50') are the same value, not a conflict."""
    def number(v):
        return float(re.sub(r"[^\d.]", "", v) or 0)
    return number(a) == number(b)


@dataclass
class Fact:
    key: str          # grouping key: same key = same fact (audience + row label + metric)
    topic: str        # readable topic for the report
    value: str        # e.g. "4.00%"
    quote: str        # the source line, for citations
    chunk: Chunk


@dataclass
class ConflictFinding:
    topic: str
    values: List[str]           # winner first
    sources: List[Fact]         # one example fact per value
    resolution: str
    reason: str
    current: bool               # True if the disagreement is visible on today's website
    cross_page: bool            # True if different pages disagree (not just a change over time)


class FactExtractor:
    """Pulls (key, value) facts out of rate tables, 'Label: value' lines and boilerplate sentences."""

    # One value column per table, in this order of preference (APY says more than dividend rate).
    METRICS = [("apy", re.compile(r"apy|yield", re.I)), ("apr", re.compile(r"apr|annual percentage rate", re.I)),
               ("rate", re.compile(r"rate|dividend", re.I)), ("fee", re.compile(r"fee", re.I))]
    IGNORE_COLUMN = re.compile(r"min\.?|minimum|payment|example|points|rate for", re.I)

    # Organisation facts repeated in every press release ("About Dupaco ..."). They drift over time.
    SENTENCE_FACTS = [
        (re.compile(r"(\d+) branch (?:office )?locations", re.I), "number of dupaco branch office locations"),
        (re.compile(r"serves residents in (\d+) counties", re.I), "number of counties dupaco serves"),
    ]

    def facts(self, chunk: Chunk) -> List[Fact]:
        if chunk.is_table:
            return self._table_facts(chunk)
        return self._label_facts(chunk) + self._sentence_facts(chunk)

    # ---------------------------------------------------------------- tables

    def _table_facts(self, chunk: Chunk) -> List[Fact]:
        lines = chunk.text.split("\n")
        caption = lines[0] if not lines[0].startswith("|") else chunk.heading.split(" > ")[-1]
        rows = [self._cells(l) for l in lines if l.startswith("|") and not re.match(r"^\|[\s\-|:]+$", l)]
        if len(rows) < 2:
            return []
        header, body = rows[0], rows[1:]
        metric, value_col = self._value_column(header)
        if value_col is None:
            return []
        label_cols = [i for i, h in enumerate(header) if i != value_col and not self.IGNORE_COLUMN.search(h)
                      and not any(pattern.search(h) for _, pattern in self.METRICS)]
        facts = []
        for row in body:
            if len(row) != len(header):
                continue
            label = normalize(" | ".join(row[i] for i in label_cols if row[i]))
            value = re.sub(r"\s*(APR|APY)$", "", row[value_col]).strip()
            if not label or not VALUE.match(value):
                continue
            topic = f"{normalize(caption)}: {label}"
            # Labels like "$25,000-$99,999.99" or "Standard" don't say which product they belong
            # to, so the caption becomes part of the key. Specific labels match across pages.
            name = label if self._is_specific(label) else topic
            key = f"{self._audience(chunk, caption)}: {name}: {metric}"
            facts.append(Fact(key, f"{topic} ({metric})", value, " | ".join(row), chunk))
        return facts

    def _value_column(self, header: List[str]):
        for metric, pattern in self.METRICS:
            for i, name in enumerate(header):
                if pattern.search(name) and not self.IGNORE_COLUMN.search(name):
                    return metric, i
        return None, None

    # ---------------------------------------------------------------- text

    def _label_facts(self, chunk: Chunk) -> List[Fact]:
        """Short lines such as '6-month certificate APY: 3.50%' or 'Routing Number: 273974581'."""
        facts = []
        for match in LABEL_LINE.finditer(chunk.text):
            label, value = normalize(match.group(1)), match.group(2)
            if "%" in value or "$" in value:          # only rates and prices
                key = f"{self._audience(chunk, label)}: {label}"
                facts.append(Fact(key, label, value, match.group(0).strip(), chunk))
        return facts

    def _sentence_facts(self, chunk: Chunk) -> List[Fact]:
        facts = []
        for pattern, topic in self.SENTENCE_FACTS:
            for match in pattern.finditer(chunk.text):
                facts.append(Fact(topic, topic, match.group(1), match.group(0), chunk))
        return facts

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _is_specific(label: str) -> bool:
        """True for labels like '17-month fixed certificate special' or '2023-26 model year | 60-month':
        they contain a number AND a word that names something (not just a tier or term)."""
        words = [w for w in re.findall(r"[a-z]{3,}", label)
                 if w not in ("month", "year", "less", "and", "over", "more", "than")]
        return bool(re.search(r"\d", label)) and len(words) >= 1

    @staticmethod
    def _audience(chunk: Chunk, caption: str) -> str:
        """Business and personal products share names but are different facts."""
        return "business" if "business" in f"{chunk.url} {caption}".lower() else "personal"

    @staticmethod
    def _cells(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]


class ConflictDetector:
    MAX_REPORTED = 30   # listing every tiny change would bury the real contradictions

    def __init__(self, store: VersionStore, extractor: FactExtractor, policy: SourcePolicy):
        self.store, self.extractor, self.policy = store, extractor, policy

    def detect(self) -> List[ConflictFinding]:
        facts_by_key: Dict[str, List[Fact]] = defaultdict(list)
        for chunk in self.store.chunks:
            if self.store.live[chunk.chunk_id]:
                for fact in self.extractor.facts(chunk):
                    facts_by_key[fact.key].append(fact)

        findings = [self._resolve(facts) for facts in facts_by_key.values() if self._disagree(facts)]
        # Most important first: contradictions on today's website, then different pages
        # disagreeing, then plain changes over time. Trusted pages before less trusted ones.
        findings.sort(key=lambda f: (not f.current, not f.cross_page, -f.sources[0].chunk.page.trust, f.topic))
        unique, seen = [], set()
        for finding in findings:
            signature = tuple(sorted(finding.values))
            if signature not in seen:      # e.g. Roth and Traditional IRA rows with identical numbers
                seen.add(signature)
                unique.append(finding)
        return unique[: self.MAX_REPORTED]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _disagree(facts: List[Fact]) -> bool:
        first = facts[0].value
        return any(not same_number(first, f.value) for f in facts[1:])

    def _last_seen(self, fact: Fact) -> str:
        return max(self.store.snapshot_date(s) for s in self.store.live[fact.chunk.chunk_id])

    def _on_site_now(self, fact: Fact) -> bool:
        latest = self.store.latest_id
        return self.store.is_live(fact.chunk.chunk_id, latest) and self.store.is_crawled(fact.chunk.url, latest)

    def _currency(self, fact: Fact) -> tuple:
        """Bigger = more likely to be true now. Same rules as the search uses."""
        return self.policy.freshness_key(fact.chunk.page, fact.chunk.effective,
                                         self._on_site_now(fact), self._last_seen(fact))

    def _resolve(self, facts: List[Fact]) -> ConflictFinding:
        winner = max(facts, key=self._currency)
        # One representative fact per distinct value (the most current copy of that value).
        best_per_value: List[Fact] = []
        for fact in sorted(facts, key=self._currency, reverse=True):
            if not any(same_number(fact.value, kept.value) for kept in best_per_value):
                best_per_value.append(fact)
        losers = best_per_value[1:]
        live_now = [f for f in facts if self._on_site_now(f)]
        return ConflictFinding(
            topic=winner.topic,
            values=[f.value for f in best_per_value],
            sources=best_per_value,
            resolution=winner.value,
            reason=self._explain(winner, losers[0]),
            current=any(not same_number(live_now[0].value, f.value) for f in live_now[1:]) if live_now else False,
            cross_page=len({f.chunk.url for f in best_per_value}) > 1,
        )

    def _explain(self, winner: Fact, loser: Fact) -> str:
        w, l = self._currency(winner), self._currency(loser)
        wp, lp = winner.chunk.page, loser.chunk.page
        if w[0] != l[0]:
            return (f"The {wp.kind.replace('_', ' ')} page {winner.chunk.url} is more authoritative than "
                    f"the {lp.kind.replace('_', ' ')} page {loser.chunk.url}, which shows {loser.value}.")
        if w[1] != l[1]:
            return (f"{loser.value} was last seen on {loser.chunk.url} in the {self._last_seen(loser)} "
                    f"snapshot; the current website ({winner.chunk.url}) shows {winner.value}.")
        if w[2] != l[2]:
            return (f"{winner.value} is the more recently dated value ({w[2]} vs {l[2] or 'undated'}) "
                    f"on {winner.chunk.url}; {loser.chunk.url} still shows {loser.value}.")
        return f"{winner.value} comes from the newest snapshot ({w[3]}) of {winner.chunk.url}."
