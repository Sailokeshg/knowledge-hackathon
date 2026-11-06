# NOTES: a knowledge system that knows what is true *now*

## How it works

```
ingest(snapshot)                                   answer(question, as_of)
      │                                                    │
      ▼                                                    ▼
 VersionStore ── keeps every version of every page   pick the snapshot for as_of
      │          per snapshot (history is never              │
      │          overwritten)                                ▼
      ▼                                              Retriever
 MarkdownChunker ── small chunks; rate tables          1. BM25 keyword search (only chunks
      │             kept whole with caption               visible in that snapshot)
      ▼             + "rates effective" date           2. re-rank by source trust + freshness
 BM25Index ──────── keyword search                     3. collapse duplicate copies; stale
                                                          copies with other numbers lose
                                                       4. drop calculators if better exists
                                                             │
 conflicts()                                                 ▼
      │                                              DeepSeekAnswerer (or ExtractiveAnswerer)
      ▼                                              writes a short answer from the chosen
 FactExtractor ── (topic, value) facts from tables   evidence only; "I don't know" when the
 ConflictDetector ── same topic, different values    evidence doesn't answer the question
                  -> pick the winner, explain why
```

Each class has one job (`submission/knowledge/`):

| Class | File | Job |
|---|---|---|
| `KnowledgeSystem` | `solution.py` | The facade the scorer calls: `ingest`, `answer`, `conflicts` |
| `VersionStore` | `store.py` | History of every page; which chunks were visible in each snapshot |
| `MarkdownChunker` | `chunker.py` | Cleans crawl noise and splits pages into chunks (tables stay whole) |
| `Tokenizer`, `BM25Index` | `search.py` | Keyword search in plain Python (no external services) |
| `SourcePolicy` | `pages.py` | **The trust rules**: what kind of page is this and how far to trust it |
| `Retriever` | `retriever.py` | Search, then trust/freshness re-ranking, then duplicate collapsing |
| `DeepSeekAnswerer`, `ExtractiveAnswerer` | `answerers.py` | Turn evidence into a short answer |
| `FactExtractor`, `ConflictDetector` | `conflicts.py` | Find contradictions and pick the right value |

## How we decide what is current

All "which value is right?" decisions are made in plain Python by `SourcePolicy`.
Search ranking and conflict resolution both use it, so the answers and the conflict report
always agree. The model only reads and phrases. The rules, in order:

1. **Answer from the right moment.** Without `as_of` we use the newest snapshot. With `as_of`
   (or a date written in the question, like "as of May 2026") we use the newest snapshot
   taken on or before that date, and the model is told that date is "today".
2. **Official pages beat secondary pages.** Rate sheets and product pages (trust 1.0) beat
   PDFs (0.7), promo landing pages and press releases (0.6), blog posts (0.5), calculators (0.4)
   and tag/author listing pages (0.2). Calculators and promo pages are often never updated
   (for example, the vehicle calculator still shows the old 5.34% auto rate).
3. **Pages still on the site beat pages that only existed in older crawls.** A page missing
   from a crawl is kept (a crawl can simply miss a page) but gets a lower weight.
4. **Newer dated content beats older.** We read "Rates are effective as of MM/DD/YYYY" under each
   table and the publish date of news/blog posts. This is how the stale mortgage table on the
   homebuying-guidance page (effective 04/22/2026) loses to the home-loans table (09/25/2026),
   and how "22 branch locations" (2026 press release) beats "24" (2024 press release).
5. **Duplicates are compared, not just ranked.** The same table often appears on several pages.
   The retriever masks the numbers and compares what is left. Copies with the same numbers are
   merged (and all cited); a copy with different numbers is treated as stale and is never shown
   to the model, so the old number cannot leak into the answer.
6. **Qualifiers are part of the fact.** "17-month special" vs "17-month jumbo special",
   APY vs dividend rate, personal vs business, each balance tier: these are different facts,
   never conflicts. Rate tables are kept whole so a number never loses its row and column.
7. **Be honest.** If the evidence does not answer exactly what was asked, the answer is an
   "I don't know, please contact Dupaco" message with `confident=False` and no numbers in it.

## Conflicts

`FactExtractor` turns every rate-table row into a fact keyed by *audience + row label + metric*
(for example `personal: 10-month fixed jumbo certificate special: apy -> 4.00%`). It also reads
short `Label: value` lines and the press-release boilerplate ("22 branch office locations").
`ConflictDetector` groups facts across all pages and all snapshots. Any key with two or more
different values is a conflict. The winner is chosen with the `SourcePolicy` rules above, and
the reason is written out ("the rate sheet page is more authoritative than the promo page...").
Contradictions visible on today's website come first, then disagreements between pages, then
changes over time. The list is capped at 30 to keep precision high.

## Why these choices

- **BM25 instead of embeddings:** member questions hinge on exact tokens ("17-month", "jumbo",
  "APY", "Galena"). Keyword search is precise for that, needs no model download, and runs in
  about 30 ms. Two simple tweaks made it work well here: stemming by keeping only the first 6
  letters of each word (so "recognition" and "recognized" match), and a coordination factor
  (chunks that contain more of the question's words rank higher).
- **The LLM only writes:** retrieval, freshness and conflict decisions are deterministic and
  testable. DeepSeek gets about 8 short excerpts that have already been filtered, and returns
  JSON (`answer`, `answered`, `sources`) in JSON mode. Thinking mode is switched off and
  temperature is 0, which keeps answers fast, short, grounded and consistent.
- **Consistency:** identical questions are answered from a cache. Paraphrases go through the same
  deterministic retrieval, so they usually see the same evidence.
- **Never crash the scorer:** if the API call fails (network, rate limit, missing key) the
  system falls back to the extractive answer writer.

## API keys

- `DEEPSEEK_API_KEY` (**paid DeepSeek API**). It is used by `DeepSeekAnswerer` (model
  `deepseek-flash`, non-thinking mode, one call per question). DeepSeek's API is
  OpenAI-compatible, so the code uses the `openai` package pointed at `https://api.deepseek.com`.
  Set the key in the environment, or copy `.env.example` to `.env` at the repo root.
- Optional: `KS_MODEL` to change the model (for example `deepseek-v4-pro`), and `KS_USE_LLM=0`
  to force the no-API mode.
- Without a key the system still runs end to end (extractive answers).

## Running

```bash
pip install -r requirements.txt
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

Ingest of all five snapshots takes about 1 second; retrieval takes about 30 ms per question.

## Results on the practice set (40 questions)

| Answer writer | Final score | Stale values repeated | Consistency | p95 latency |
|---|---|---|---|---|
| Starter solution (for reference) | 0.09 | - | - | 0.1 s |
| No API key (extractive) | 0.32 | 0 | 1.0 | 0.03 s |
| DeepSeek V4.1 Flash (`deepseek-flash`) | 0.82 - 0.85 (two runs) | 0 | 1.0 | 1.6 - 1.8 s |

The harness self-test (`harness/selftest`, which includes a conflicts gold file) scores 1.0:
the one conflict is found and resolved to the right value.

Questions still missed, and why:
- "Get me the rate for the certificate of accounts": the item has no expected values, so it
  cannot score (our answer lists the certificate rates).
- Volunteer Time Off "last year": the eval expects 294 (2023), but the site has a newer press
  release saying 256 employees used it in 2024. We answer with the newer number by design.
- Membership eligibility: the join page says "live or work within our charter area" but does not
  name the states there.
- "Do any savings tiers pay 0.35% APY?": the business Investor's Choice money market does pay
  0.35%; the eval expects "no" based on personal Share Savings.
- "Do all branches keep the same lobby hours?": the answer needs two schedules from two different
  press releases; we usually return one.
- Forbes recognition: the model sometimes leaves out "the fifth time" (the only answer that
  changed between the two runs).

## Known limitations

- Fact matching for conflicts needs the same row label on both pages. When a tier is renamed
  (for example "2023-26 Model Year" becoming "2024-27 Model Year"), the old and new rows are not
  linked.
- Some facts live only in press releases (branch lobby hours), so the answer quotes the most
  typical schedule and says hours vary by branch.
- BM25 has no notion of synonyms beyond a small hand-made list (mortgage -> home loan,
  credit card -> visa, branches -> branch office locations).
