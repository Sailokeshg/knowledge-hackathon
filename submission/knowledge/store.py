"""
VersionStore: remembers every version of every page, and what the website looked like
on each snapshot date.

We never overwrite history. For each snapshot we keep a "view": url -> the page version
visible on that date. This is what makes `as_of` questions possible: to answer
"as of May 31", we search only the chunks that were visible in the May 19 view.
"""
from typing import Dict, List, Optional, Set, Tuple

from .chunker import Chunk, MarkdownChunker
from .dates import effective_date
from .pages import PageVersion, SourcePolicy
from .search import BM25Index, searchable_text


class VersionStore:
    def __init__(self, policy: SourcePolicy, chunker: MarkdownChunker, index: BM25Index):
        self.policy = policy
        self.chunker = chunker
        self.index = index

        self.snapshots: List[Tuple[str, str]] = []          # (snapshot_id, taken_on), oldest first
        self.views: Dict[str, Dict[str, PageVersion]] = {}  # snapshot_id -> {url: visible version}
        self.crawled: Dict[str, Set[str]] = {}              # snapshot_id -> urls actually in that crawl

        self.chunks: List[Chunk] = []                       # chunk_id -> Chunk
        self.live: List[Set[str]] = []                      # chunk_id -> snapshot_ids where visible
        self._chunk_ids: Dict[tuple, int] = {}              # (url, heading, text) -> chunk_id
        self._version_chunks: Dict[tuple, List[int]] = {}   # (url, page text) -> its chunk_ids

    # ------------------------------------------------------------------ ingest

    def add_snapshot(self, snapshot) -> None:
        """Record one crawl. Pages missing from it are carried forward from the last view,
        because a crawl can simply miss a page (they get a lower weight later)."""
        view = dict(self.views[self.latest_id]) if self.snapshots else {}
        for page in snapshot.pages:
            old = view.get(page.url)
            if old is None or old.markdown != page.markdown:
                view[page.url] = self.policy.make_version(
                    page.url, page.title, page.markdown, snapshot.snapshot_id, snapshot.taken_on)

        self.snapshots.append((snapshot.snapshot_id, snapshot.taken_on))
        self.snapshots.sort(key=lambda s: s[1])
        self.views[snapshot.snapshot_id] = view
        self.crawled[snapshot.snapshot_id] = {p.url for p in snapshot.pages}

        for version in view.values():
            for chunk_id in self._chunks_of(version):
                self.live[chunk_id].add(snapshot.snapshot_id)

    def _chunks_of(self, version: PageVersion) -> List[int]:
        """Chunk a page version (once) and return its chunk ids. Unchanged chunks are reused."""
        key = (version.url, version.markdown)
        if key not in self._version_chunks:
            ids = []
            for heading, text, is_table in self.chunker.split(version.markdown):
                chunk_key = (version.url, heading, text)
                if chunk_key not in self._chunk_ids:
                    chunk_id = len(self.chunks)
                    self.chunks.append(Chunk(chunk_id, version.url, heading, text, is_table,
                                             effective_date(text), version))
                    self.live.append(set())
                    self.index.add(chunk_id, searchable_text(f"{version.title}\n{heading}\n{text}"))
                    self._chunk_ids[chunk_key] = chunk_id
                ids.append(self._chunk_ids[chunk_key])
            self._version_chunks[key] = ids
        return self._version_chunks[key]

    # ------------------------------------------------------------------ lookups

    @property
    def latest_id(self) -> str:
        return self.snapshots[-1][0]

    def snapshot_for(self, as_of: Optional[str]) -> str:
        """The newest snapshot taken on or before `as_of` (the latest one if as_of is None)."""
        if not as_of:
            return self.latest_id
        eligible = [sid for sid, taken in self.snapshots if taken <= as_of]
        return eligible[-1] if eligible else self.snapshots[0][0]

    def snapshot_date(self, snapshot_id: str) -> str:
        return dict(self.snapshots)[snapshot_id]

    def is_crawled(self, url: str, snapshot_id: str) -> bool:
        """False when the page was only carried forward from an older crawl."""
        return url in self.crawled.get(snapshot_id, ())

    def is_live(self, chunk_id: int, snapshot_id: str) -> bool:
        return snapshot_id in self.live[chunk_id]
