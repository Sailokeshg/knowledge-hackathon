"""
Turns a page of markdown into small searchable chunks.

Two ideas matter here:
  * Tables are kept whole, together with their caption and "rates effective" note.
    Splitting a rate table row-by-row would lose which product / tier a number belongs to.
  * Every chunk remembers its section heading, so "4.10%" is never separated
    from "17 Month Fixed Certificate Special".
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from .pages import PageVersion


@dataclass
class Chunk:
    chunk_id: int
    url: str
    heading: str                 # section path, e.g. "Rates > Special Certificates"
    text: str                    # cleaned text shown to the answer writer
    is_table: bool
    effective: Optional[str]     # newest "rates effective as of" date inside the chunk
    page: PageVersion            # page version where this chunk first appeared


class MarkdownChunker:
    MAX_CHARS = 900           # target size of a text chunk (small chunks = precise search)
    MAX_TABLE_CHARS = 3500    # bigger tables are split into row groups (header repeated)

    def split(self, markdown: str):
        """Yield (heading, text, is_table) pieces for one page."""
        blocks = [b.strip() for b in re.split(r"\n\s*\n", self.clean(markdown)) if b.strip()]
        headings = {}            # heading level -> heading text
        buffer: List[str] = []   # paragraphs waiting to become a chunk

        def heading_path():
            return " > ".join(headings[level] for level in sorted(headings))

        def flush():
            if buffer:
                yield heading_path(), "\n\n".join(buffer), False
                buffer.clear()

        for i, block in enumerate(blocks):
            heading = re.match(r"^(#{1,6})\s+(.*)", block)
            if heading:
                yield from flush()
                level = len(heading.group(1))
                headings = {lvl: txt for lvl, txt in headings.items() if lvl < level}
                headings[level] = heading.group(2).strip()
            elif self._is_table(block):
                # The short line right before a table is usually its caption ("Special Certificates").
                caption = buffer.pop() if buffer and len(buffer[-1]) < 80 else ""
                yield from flush()
                note = self._effective_note(blocks, i)
                for rows in self._split_table(block):
                    yield heading_path(), "\n".join(p for p in (caption, rows, note) if p), True
            else:
                buffer.append(block)
                if sum(len(b) for b in buffer) > self.MAX_CHARS:
                    # Overlap: a short last paragraph ("Branch hours are:") is repeated at the
                    # start of the next chunk, so a list is never cut off from its intro line.
                    carry = buffer[-1] if len(buffer[-1]) < 200 else None
                    yield from flush()
                    if carry:
                        buffer.append(carry)
        yield from flush()

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def clean(markdown: str) -> str:
        """Remove the noise that a web crawl leaves in the markdown."""
        text = markdown.replace("\\r\\n", " ").replace("\\\\", "")
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)                  # images
        text = re.sub(r"\[([^\]]*)\]\((?:tel|mailto):[^)]*\)", r"\1", text)  # phone / email links
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)[^)]*\)", r"\1 (\2)", text)  # keep link URLs
        text = re.sub(r"\[\]\([^)]*\)", "", text)                           # empty links
        text = re.sub(r"(%|APR|APY)\d\b", r"\1", text)                    # footnote marks: "4.20%1"
        text = re.sub(r"\\([*_#\[\]])", r"\1", text)                       # escaped markdown
        text = text.replace("**", "")
        # Drop the "Handpicked for You" teaser section and pop-up leftovers.
        text = re.sub(r"## Handpicked for You.*?(?=\n## |\Z)", "", text, flags=re.S)
        lines = [l for l in text.split("\n") if not re.match(r"^\s*(✕|\* \* \*|Continue .*Cancel)", l)]
        return "\n".join(lines)

    @staticmethod
    def _is_table(block: str) -> bool:
        lines = block.split("\n")
        return len(lines) >= 3 and all(l.lstrip().startswith("|") for l in lines)

    @staticmethod
    def _effective_note(blocks, table_index) -> str:
        """The 'Rates are effective as of ...' sentence printed under a table, if any."""
        for block in blocks[table_index + 1: table_index + 3]:
            if re.match(r"^\S*\s*Rates (are )?effective", block):
                return block.split("\n")[0][:200]
        return ""

    def _split_table(self, table: str):
        """Yield the table, or row groups of it (each with the header) if it is very large."""
        if len(table) <= self.MAX_TABLE_CHARS:
            yield table
            return
        lines = table.split("\n")
        header, rows = lines[:2], lines[2:]
        group: List[str] = []
        for row in rows:
            group.append(row)
            if sum(len(r) for r in group) > self.MAX_TABLE_CHARS:
                yield "\n".join(header + group)
                group = []
        if group:
            yield "\n".join(header + group)
