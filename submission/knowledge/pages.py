"""
Page versions and the rules for "which page should I believe?".

Every page we ingest becomes a PageVersion: the page text plus WHERE and WHEN we saw it.
SourcePolicy holds the trust rules. They live in one place so that search ranking and
conflict resolution always agree with each other.
"""
import re
from dataclasses import dataclass
from typing import Optional

from .dates import published_date, years_between


@dataclass(frozen=True)
class PageVersion:
    """One copy of one web page, as captured by a snapshot."""
    url: str
    title: str
    markdown: str
    snapshot_id: str              # first snapshot that contained this exact text
    snapshot_date: str            # ISO date of that snapshot
    kind: str                     # "rate_sheet", "product", "news", "blog", "calculator", ...
    trust: float                  # 0..1, how authoritative this kind of page is
    published: Optional[str] = None   # ISO date printed on a blog post / press release


class SourcePolicy:
    """Decides how much each page can be trusted.

    The ranking a careful human expert would use:
        official rate sheets and product pages  >  PDF documents
        >  promo landing pages and press releases  >  blog posts
        >  calculators (often never updated)  >  tag / author listing pages
    """

    # (regex on the URL, page kind, trust score). The first match wins.
    RULES = [
        (r"/tag/|/author/|/newsroom/?$|/learn/blog/?$", "listing", 0.2),
        (r"calculator", "calculator", 0.4),
        (r"/learn/|/business/resources/|dupacodaily|/20\d\d/\d\d/", "blog", 0.5),
        (r"/newsroom/", "news", 0.6),
        (r"/hubfs/", "document", 0.7),
        (r"-lp$|switch-and-earn|checking-offer|makethemost|money-makeover", "promo", 0.6),
        (r"rates", "rate_sheet", 1.0),
    ]
    DEFAULT_KIND, DEFAULT_TRUST = "product", 1.0

    REMOVED_PENALTY = 0.6      # page was in an older crawl but is missing from this one
    MIN_AGE_FACTOR = 0.8       # an old article still keeps most of its weight

    def classify(self, url: str):
        """Return (kind, trust) for a URL."""
        for pattern, kind, trust in self.RULES:
            if re.search(pattern, url, re.I):
                return kind, trust
        return self.DEFAULT_KIND, self.DEFAULT_TRUST

    def make_version(self, url, title, markdown, snapshot_id, snapshot_date) -> PageVersion:
        kind, trust = self.classify(url)
        published = published_date(markdown) if kind in ("news", "blog") else None
        return PageVersion(url, title, markdown, snapshot_id, snapshot_date, kind, trust, published)

    def search_weight(self, page: PageVersion, today: str, removed: bool) -> float:
        """Multiplier applied to a search score. Less trusted pages, old articles and removed
        pages count less. It is a gentle nudge: some facts (branch hours, awards) only appear
        in press releases, so those must still be findable."""
        weight = 0.4 + 0.6 * page.trust
        if page.published:
            age = years_between(page.published, today)
            weight *= max(self.MIN_AGE_FACTOR, 1.0 - 0.03 * age)
        if removed:
            weight *= self.REMOVED_PENALTY
        return weight

    @staticmethod
    def freshness_key(page: PageVersion, content_date: Optional[str], present: bool,
                      seen_on: Optional[str] = None) -> tuple:
        """Sort key for 'which copy is the current truth?'. Bigger wins.

        1. the more authoritative kind of page (an official rate page beats a promo page
           or a calculator, even when the promo page is still online)
        2. the page is still on the website (not only in an older crawl)
        3. the most recently dated content ("rates effective as of", publish date)
        4. the most recent crawl it was seen in
        """
        return (page.trust, present, content_date or page.published or "", seen_on or page.snapshot_date)
