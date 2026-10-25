# 📈 Autonomous Daily Financial News Reporter

A zero-cost agent that pulls the day's financial headlines from RSS feeds, has
**Google Gemini Pro** synthesize them into a macro/market briefing, and delivers
a responsive HTML newsletter to your inbox via **Gmail SMTP**, all scheduled by
**GitHub Actions**.

```
RSS feeds ──► clean / dedupe / 24h filter ──► Gemini (structured JSON) ──► HTML + text ──► Gmail SMTP
                                                                                  └──► output/brief-YYYY-MM-DD.html
```

## Features

- **Parallel RSS ingestion** (CNBC, MarketWatch, Bloomberg, Investing.com, Federal Reserve by default). One broken feed never fails the run.
- **Cleaning pipeline**: strips HTML, de-duplicates by link and title, keeps only the last `LOOKBACK_HOURS`.
- **Structured Gemini output**: uses `response_schema` and validates with Pydantic, so the result is always well-formed: headline, executive summary, sentiment, macro themes, outlook, sector watch, risks and top stories.
- **Hallucination guard**: the model cites articles by numeric id, and links are only taken from real fetched articles. Unknown ids are dropped.
- **Safe, responsive email**: table-based layout with inline CSS and a mobile media query; every piece of text is HTML-escaped. Includes a plain-text alternative.
- **Resilient**: exponential-backoff retries for Gemini (429/5xx/network/invalid JSON) and SMTP. Credential errors are never retried.
- **Private delivery**: recipients go in `Bcc`.
- **Fully offline test suite**: no keys or network needed.

## Project structure

```
.
├── .github/workflows/daily_brief.yml   # cron + manual trigger, tests gate the run
├── .env.example                        # configuration template
├── agent.py                            # the agent
├── test_agent.py                       # pytest suite
├── requirements.txt                    # runtime deps
└── requirements-dev.txt                # + pytest
```

## Prerequisites

1. **Gemini API key**: create one at <https://aistudio.google.com/apikey>.
2. **Gmail App Password**: turn on 2-Step Verification for your Google account, then create a 16-character app password at <https://myaccount.google.com/apppasswords>. Your normal password will **not** work.

## Local usage

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env        # then fill in your values

pytest -q                   # run the tests
python agent.py --dry-run   # build output/brief-<date>.html without emailing
python agent.py             # build and send
```

CLI flags: `--dry-run`, `--output-dir PATH`, `-v/--verbose`.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | | Google AI Studio API key |
| `GMAIL_ADDRESS` | ✅ (unless dry-run) | | Sender Gmail address |
| `GMAIL_APP_PASSWORD` | ✅ (unless dry-run) | | 16-char app password (spaces are OK) |
| `RECIPIENT_EMAILS` | | `GMAIL_ADDRESS` | Comma-separated recipients (Bcc) |
| `GEMINI_MODEL` | | `gemini-pro-latest` | Any Gemini model id, e.g. `gemini-2.5-pro` |
| `RSS_FEEDS` | | built-in list | Comma-separated feed URLs |
| `REPORT_TIMEZONE` | | `UTC` | IANA timezone for the report date |
| `LOOKBACK_HOURS` | | `24` | Recency window for articles |
| `MAX_ARTICLES_PER_FEED` | | `15` | Per-feed cap |
| `MAX_TOTAL_ARTICLES` | | `60` | Cap on articles sent to Gemini |
| `SENDER_NAME` | | `Daily Market Brief` | Display name in the From header |
| `DRY_RUN` | | `false` | Skip email delivery |

## GitHub Actions deployment

1. Push this repo to GitHub.
2. Go to **Settings → Secrets and variables → Actions** and add these **Secrets**:
   - `GEMINI_API_KEY`
   - `GMAIL_ADDRESS`
   - `GMAIL_APP_PASSWORD`
   - `RECIPIENT_EMAILS` (optional)
3. Optionally add **Variables**: `GEMINI_MODEL`, `RSS_FEEDS`, `REPORT_TIMEZONE` (the workflow defaults to `Asia/Bangkok`), `LOOKBACK_HOURS`.
4. Go to **Actions → Daily Financial Brief → Run workflow** and tick *dry run* for a first test. The generated HTML is attached to the run as an artifact.

The workflow runs at **23:00 UTC daily (06:00 ICT)**. To change it, edit the `cron` line in [`daily_brief.yml`](.github/workflows/daily_brief.yml). Every run runs the test suite first, and the brief is only sent if the tests pass.

### Cost

- **GitHub Actions**: free for public repos. Private repos get 2,000 free minutes/month, and one run takes about 1–2 minutes.
- **Gemini API**: one request per day. Free-tier access to Pro models depends on Google's current quotas. If Pro is not available on your tier, set `GEMINI_MODEL` to a Flash model.
- **Gmail SMTP**: free (personal accounts are limited to about 500 recipients/day).

> **Note:** GitHub pauses scheduled workflows in public repos after 60 days without repository activity. Re-enable the workflow from the Actions tab, or push a commit, to resume it.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Configuration error or unexpected failure (see logs) |
| `2` | No articles could be fetched, so nothing was sent |

## Disclaimer

The analysis is AI-generated, may contain errors, and is not investment advice.
