"""
Small date helpers.

The website writes dates in a few different styles. We turn all of them into
ISO strings ("2026-09-01") because ISO strings sort correctly as plain text.
"""
import calendar
import re
from datetime import date
from typing import Optional

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})

# "June 18, 2024" or "Sep 17, 2026"
LONG_DATE = re.compile(r"\b([A-Z][a-z]{2,8})\.? (\d{1,2}), (\d{4})\b")
# "Rates are effective as of 09/01/2026"
EFFECTIVE_DATE = re.compile(r"effective (?:as of )?(\d{1,2})/(\d{1,2})/(\d{4})", re.I)
# "as of May 2026", "in October 2026"
MONTH_YEAR = re.compile(r"\b([A-Za-z]{3,9}) (\d{4})\b")
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _iso(year: int, month: int, day: int) -> Optional[str]:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def published_date(text: str) -> Optional[str]:
    """Find the publish date printed near the top of a blog post or press release."""
    match = LONG_DATE.search(text[:400])
    if not match or match.group(1).lower() not in MONTHS:
        return None
    return _iso(int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2)))


def effective_date(text: str) -> Optional[str]:
    """Return the newest 'Rates are effective as of MM/DD/YYYY' date in the text, if any."""
    found = [_iso(int(y), int(m), int(d)) for m, d, y in EFFECTIVE_DATE.findall(text)]
    found = [f for f in found if f]
    return max(found) if found else None


def date_in_question(question: str) -> Optional[str]:
    """If a question names a date ("as of May 2026"), return the last day of that period.

    Used only when the caller did not pass `as_of` explicitly.
    """
    iso = ISO_DATE.search(question)
    if iso:
        return _iso(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    for word, year in MONTH_YEAR.findall(question):
        month = MONTHS.get(word.lower())
        if month:
            last_day = calendar.monthrange(int(year), month)[1]
            return _iso(int(year), month, last_day)
    return None


def years_between(older: str, newer: str) -> float:
    """Age in years between two ISO dates (0 if the order is reversed)."""
    days = (date.fromisoformat(newer) - date.fromisoformat(older)).days
    return max(0.0, days / 365.0)
