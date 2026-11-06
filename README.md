# Knowledge system that knows what's true *now*

My solution to the interface.ai knowledge hackathon: a system that reads five snapshots of a
credit union's website, answers member questions with the **current** value and a source,
lists where the website contradicts itself, and says "I don't know" when the site doesn't say.

- **Solution:** [`submission/`](submission/) (`solution.py` + the `knowledge/` package)
- **Approach and decisions:** [`submission/NOTES.md`](submission/NOTES.md)
- **Scorer and practice questions:** `harness/` and `eval/` (unchanged from the kit)

The website snapshots are **not** in this repository (they were shared for the assignment only).
Put the kit's `snapshots_zip/` folder at the repo root; the scorer unpacks it on first run.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env      # then add your DEEPSEEK_API_KEY (optional, see NOTES.md)
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

Without an API key the system still runs, using extractive (quote-the-evidence) answers.
