"""
Answer writers: turn the retrieved evidence into a short member-facing answer.

Both classes share one small interface:  answer(question, evidence, today) -> Draft

  * DeepSeekAnswerer    - the main path. The LLM reads only the evidence we picked (already
                          filtered for freshness and trust) and writes a short answer as JSON.
  * ExtractiveAnswerer  - the no-API fallback. Quotes the best-matching table row or sentence.

All "what is current" decisions are made BEFORE this step, in plain Python. The model only
has to read and phrase, which keeps it fast and predictable.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .retriever import Evidence
from .search import Tokenizer

DATE_WORDS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|"
                        r"november|december|(?:19|20)\d\d)\b", re.I)

# No digits on purpose: a refusal must not state any number.
REFUSAL_TEXT = ("I don't know. That isn't listed on the website, "
                "so please contact Dupaco directly and a team member can help.")


@dataclass
class Draft:
    text: str
    answered: bool
    used: List[Evidence] = field(default_factory=list)


class ExtractiveAnswerer:
    """Fallback without an LLM: return the table row or sentence that best matches the question."""

    MIN_COVERAGE = 0.75   # the best evidence must contain 3/4 of the question's words

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def answer(self, question: str, evidence: List[Evidence], today: str) -> Draft:
        # Dates in the question ("in January 2026") pick the snapshot; they are not search words.
        wanted = set(self.tokenizer.tokens(DATE_WORDS.sub(" ", question)))
        if not evidence or not wanted:
            return Draft(REFUSAL_TEXT, answered=False)
        best = evidence[0]
        page_words = set(self.tokenizer.tokens(f"{best.chunk.page.title} {best.chunk.heading} {best.chunk.text}"))
        if len(wanted & page_words) / len(wanted) < self.MIN_COVERAGE:
            return Draft(REFUSAL_TEXT, answered=False)      # the site doesn't really talk about this
        units = self._table_rows(best.chunk.text) if best.chunk.is_table else self._sentences(best.chunk.text)
        ranked = sorted(units, key=lambda u: -len(wanted & set(self.tokenizer.tokens(u))))
        text = ranked[0].strip()[:600] if ranked else ""
        return Draft(text or REFUSAL_TEXT, answered=bool(text), used=[best])

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 10]

    @staticmethod
    def _table_rows(text: str) -> List[str]:
        """Rewrite table rows as 'Column: value; Column: value' so they read as sentences."""
        lines = [l for l in text.split("\n") if l.startswith("|") and not re.match(r"^\|[\s\-|:]+$", l)]
        if len(lines) < 2:
            return [text]
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[1:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells) if c) + ".")
        return rows


class DeepSeekAnswerer:
    """Writes the answer with DeepSeek, grounded only in the evidence it is given.

    DeepSeek's API is OpenAI-compatible, so we use the `openai` SDK with DeepSeek's base URL.
    """

    BASE_URL = "https://api.deepseek.com"

    SYSTEM_PROMPT = """You are the member-service assistant for Dupaco Community Credit Union.
Answer the member's question using ONLY the website excerpts provided.

How to answer:
- Be direct and brief: one to three plain sentences, no markdown. Quote numbers exactly as the excerpts write them (for example "4.10% APY").
- Answer exactly what was asked and respect every qualifier: term (6-month vs 12-month), balance or loan-amount tier, rate vs APY vs APR, regular vs special or jumbo, personal vs business, model year. Do not quote numbers for other products or tiers unless the member asks for a list, a comparison or a range.
- Assume the member means personal (consumer) products unless they mention a business.
- If a product's rate depends on a tier (loan amount or balance) and the member did not name one, give the range from the lowest to the highest tier and say which tier each end applies to (for example "from 5.10% for loans up to $100,000 to 5.40% for larger loans").
- If the question is broad (for example "what are the mortgage rates" or "what are your lobby hours"), lead with the most common product or schedule (for example the 30-year fixed purchase loan) and say briefly that details vary.
- If the member asks what is offered (all cards, all certificates, all options), list every matching item in the excerpts with its key number.
- When an excerpt states a count or milestone that answers the question (for example "the third time" or "five years in a row"), include it.
- State only what is true today. Excerpts are ordered most trustworthy and most current first; official rate and product pages beat calculators, promotions, blog posts and older press releases. Never mention old, previous or superseded values, and do not add history or background the member did not ask for.
- Today's date is given. If an excerpt describes an event that has already happened (a branch closing, an offer ending), answer accordingly.
- Never answer with a question, and never ask for a ZIP code, member number or other details instead of answering.
- Do not mention "excerpts", "sources" or "facts" in the answer; speak as the credit union.
- For "how do I..." account-service questions that the excerpts do not explain step by step, answer by pointing the member to the contact options the excerpts give (Shine Online Banking, a branch, or the phone number).
- If the member asks for a specific fact (a rate, fee, hours, address, date) and the excerpts do not contain it for exactly what was asked, set "answered" to false. Never substitute a similar product's number (a 30-year fixed rate is not a 7/1 ARM rate) and never invent details for a branch, product or fee that the excerpts do not mention.
- In "sources", list the [n] numbers of the excerpts you used.

Reply with a JSON object only, in exactly this shape:
{"answer": "<one to three sentences>", "answered": true, "sources": [1, 3]}"""

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI  # imported here so the rest of the system works without the package

        self.model = model or os.getenv("KS_MODEL", "deepseek-flash")
        # Each answer makes at most 2 calls x 2 attempts x 12 s, so it always stays under the
        # 60-second limit (a normal answer takes 1-2 s).
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=self.BASE_URL,
                             timeout=12.0, max_retries=1)

    def answer(self, question: str, evidence: List[Evidence], today: str) -> Draft:
        if not evidence:
            return Draft(REFUSAL_TEXT, answered=False)
        data = self._ask(self._prompt(question, evidence, today))
        if not data.get("answered") or not str(data.get("answer", "")).strip():
            return Draft(REFUSAL_TEXT, answered=False)
        used = [evidence[i - 1] for i in data.get("sources", []) if isinstance(i, int) and 1 <= i <= len(evidence)]
        return Draft(str(data["answer"]).strip(), answered=True, used=used or evidence[:1])

    def _ask(self, prompt: str) -> dict:
        """One chat call in JSON mode. DeepSeek can occasionally return blank content, so retry once."""
        for _ in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=800,
                temperature=0.0,                                  # same question -> same answer
                extra_body={"thinking": {"type": "disabled"}},   # no long reasoning: fast answers
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return json.loads(content)
        raise RuntimeError("DeepSeek returned an empty response twice")

    @staticmethod
    def _prompt(question: str, evidence: List[Evidence], today: str) -> str:
        parts = [f"Today's date: {today}. The excerpts show the website as it stood on this date.",
                 "", "Website excerpts (most trustworthy and most current first):"]
        for n, item in enumerate(evidence, start=1):
            chunk, page = item.chunk, item.chunk.page
            details = [page.kind.replace("_", " ") + " page"]
            if page.published:
                details.append(f"published {page.published}")
            if chunk.effective:
                details.append(f"rates effective {chunk.effective}")
            parts += ["", f"[{n}] {page.title} - {chunk.url}",
                      f"({', '.join(details)}; section: {chunk.heading or 'top of page'})",
                      chunk.text]
        parts += ["", f"Member question: {question}", "", "Reply with the JSON object."]
        return "\n".join(parts)
