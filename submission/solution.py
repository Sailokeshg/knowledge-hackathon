"""
A knowledge system that knows what is true NOW.

The scorer calls:   build()  ->  ingest(snapshot) x5  ->  answer(question, as_of)  ->  conflicts()

How the pieces fit together (one class per job):

    VersionStore      keeps every version of every page, per snapshot date (history, never overwritten)
    MarkdownChunker   cuts pages into small chunks, keeping rate tables whole
    BM25Index         keyword search over the chunks
    SourcePolicy      the trust rules: rate sheet > product page > PDF > news > blog > calculator
    Retriever         search + trust/freshness re-ranking + drop stale duplicate copies
    DeepSeekAnswerer  writes a short answer from the chosen evidence (ExtractiveAnswerer if no API key)
    ConflictDetector  finds facts with different values across pages/snapshots and picks the winner
"""
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# The contract types live in harness/contract.py (found the same way the starter does it).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())))
sys.path.insert(0, str(HERE))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

from knowledge.answerers import REFUSAL_TEXT, DeepSeekAnswerer, Draft, ExtractiveAnswerer  # noqa: E402
from knowledge.chunker import MarkdownChunker  # noqa: E402
from knowledge.conflicts import ConflictDetector, FactExtractor  # noqa: E402
from knowledge.dates import date_in_question  # noqa: E402
from knowledge.pages import SourcePolicy  # noqa: E402
from knowledge.retriever import Evidence, Retriever  # noqa: E402
from knowledge.search import BM25Index, Tokenizer  # noqa: E402
from knowledge.store import VersionStore  # noqa: E402


class KnowledgeSystem:
    def __init__(self, use_llm: bool = True):
        self.policy = SourcePolicy()
        self.tokenizer = Tokenizer()
        self.index = BM25Index(self.tokenizer)
        self.store = VersionStore(self.policy, MarkdownChunker(), self.index)
        self.retriever = Retriever(self.store, self.index, self.policy)
        self.conflict_detector = ConflictDetector(self.store, FactExtractor(), self.policy)
        self.fallback_writer = ExtractiveAnswerer(self.tokenizer)
        self.llm_writer = self._make_llm_writer() if use_llm else None
        self._answers: Dict[tuple, Answer] = {}   # same question twice -> same answer, instantly

    # ------------------------------------------------------------------ contract

    def ingest(self, snapshot: Snapshot) -> None:
        self.store.add_snapshot(snapshot)
        self._answers.clear()

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        as_of = as_of or self._date_mentioned(question)
        snapshot_id = self.store.snapshot_for(as_of)
        cache_key = (" ".join(question.lower().split()).strip(" ?.!"), snapshot_id)
        if cache_key not in self._answers:
            today = as_of or self.store.snapshot_date(snapshot_id)
            evidence = self.retriever.retrieve(question, snapshot_id)
            draft = self._write(question, evidence, today)
            self._answers[cache_key] = self._to_answer(draft, snapshot_id)
        return self._answers[cache_key]

    def conflicts(self) -> List[Conflict]:
        result = []
        for finding in self.conflict_detector.detect():
            sources = [Source(url=f.chunk.url, snapshot_id=self._last_snapshot(f.chunk.chunk_id), quote=f.quote)
                       for f in finding.sources]
            result.append(Conflict(topic=finding.topic, values=finding.values, sources=sources,
                                   resolution=finding.resolution, reason=finding.reason))
        return result

    # ------------------------------------------------------------------ helpers

    def _write(self, question: str, evidence: List[Evidence], today: str) -> Draft:
        """Use the LLM when available; fall back to quoting the evidence if the call fails."""
        if self.llm_writer:
            try:
                return self.llm_writer.answer(question, evidence, today)
            except Exception as error:  # network trouble, rate limits, ... never crash the scorer
                print(f"[knowledge] LLM call failed, using extractive answer: {error}", file=sys.stderr)
        return self.fallback_writer.answer(question, evidence, today)

    def _to_answer(self, draft: Draft, snapshot_id: str) -> Answer:
        if not draft.answered:
            return Answer(text=REFUSAL_TEXT, sources=[], confident=False)
        sources, notes = [], []
        for item in draft.used:
            sources.append(Source(url=item.chunk.url, snapshot_id=snapshot_id, quote=item.chunk.text[:200]))
            sources += [Source(url=url, snapshot_id=snapshot_id) for url in item.also_on[:3]]
            for stale in item.overruled[:2]:
                notes.append(f"{stale.url} shows different numbers; we trusted {item.chunk.url} "
                             f"({item.chunk.page.kind.replace('_', ' ')} page, more authoritative/current).")
        return Answer(text=draft.text, sources=sources, confident=True,
                      conflict_note=" ".join(notes) or None)

    def _date_mentioned(self, question: str) -> Optional[str]:
        """'as of May 2026' in the question text works like as_of, if it falls inside our snapshots."""
        mentioned = date_in_question(question)
        if mentioned and self.store.snapshots and mentioned >= self.store.snapshots[0][1]:
            return mentioned
        return None

    def _last_snapshot(self, chunk_id: int) -> Optional[str]:
        live = self.store.live[chunk_id]
        return max(live, key=self.store.snapshot_date) if live else None

    @staticmethod
    def _make_llm_writer() -> Optional[DeepSeekAnswerer]:
        _load_dotenv()
        if os.getenv("KS_USE_LLM", "1") == "0":
            return None
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("[knowledge] no DEEPSEEK_API_KEY set, using extractive answers", file=sys.stderr)
            return None
        try:
            return DeepSeekAnswerer()
        except ImportError:
            print("[knowledge] 'openai' package not installed, using extractive answers", file=sys.stderr)
            return None


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a local .env file (if one exists) without overriding real env vars."""
    for folder in [HERE, *HERE.parents][:3]:
        env_file = folder / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                match = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
                if match:
                    os.environ.setdefault(match.group(1), match.group(2).strip("'\""))
            return


def build() -> KnowledgeSystem:
    return KnowledgeSystem()
