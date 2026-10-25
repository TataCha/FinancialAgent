#!/usr/bin/env python3
"""Autonomous Daily Financial News Reporter Agent.

Pipeline:
    1. Fetch recent market headlines from a set of RSS feeds (in parallel).
    2. Clean, de-duplicate and filter them to a recency window.
    3. Ask Gemini (Pro) for a structured JSON macro/market analysis.
    4. Render a responsive HTML newsletter (+ plain-text fallback).
    5. Deliver it through Gmail SMTP (or save locally with --dry-run).

All configuration is read from environment variables (see .env.example).
"""

from __future__ import annotations

import argparse
import calendar
import html
import logging
import os
import re
import smtplib
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import feedparser
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError, field_validator

try:  # httpx is a transitive dependency of google-genai.
    import httpx

    _TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (httpx.TransportError,)
except ImportError:  # pragma: no cover
    _TRANSPORT_ERRORS = ()

__version__ = "1.0.0"

logger = logging.getLogger("financial_news_agent")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gemini-3.5-flash-lite"

DEFAULT_FEEDS: tuple[str, ...] = (
    # CNBC – Top News
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    # CNBC – Economy
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    # MarketWatch – Top Stories
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    # Bloomberg – Markets
    "https://feeds.bloomberg.com/markets/news.rss",
    # Investing.com – Economy news
    "https://www.investing.com/rss/news_14.rss",
    # Federal Reserve – Press releases
    "https://www.federalreserve.gov/feeds/press_all.xml",
)

USER_AGENT = f"Mozilla/5.0 (compatible; FinancialNewsAgent/{__version__})"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
FEED_TIMEOUT_SECONDS = 20
SUMMARY_MAX_CHARS = 500
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})

SYSTEM_INSTRUCTION = """\
You are a senior macro strategist writing a concise, institutional-quality \
daily market briefing for busy investors.

Rules:
- Base every statement ONLY on the numbered articles provided. Never invent \
figures, quotes, prices, dates or events. If information is thin, say so.
- Synthesize: connect individual headlines into macro themes (monetary policy, \
inflation, growth, labour, earnings, geopolitics, commodities, FX, rates).
- Explain the "so what" for markets: likely impact on equities, bonds, \
currencies and commodities.
- Reference supporting articles by their numeric id in `article_ids` and \
`top_stories.article_id`. Only use ids that appear in the input.
- Neutral, professional tone. No investment advice, no hype, no emojis.
- market_sentiment must be one of: bullish, bearish, neutral, mixed.
- sector_watch.outlook must be one of: positive, negative, neutral.
"""

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class AgentError(Exception):
    """Base error for the agent."""


class ConfigError(AgentError):
    """Invalid or missing configuration."""


class FeedError(AgentError):
    """A feed could not be fetched or parsed."""


class BriefParseError(AgentError):
    """The model response could not be parsed into a MarketBrief."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _split_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in re.split(r"[,\n;]", value) if item.strip())


def _parse_positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{key} must be > 0, got {value}")
    return value


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    gmail_address: str = ""
    gmail_app_password: str = ""
    recipients: tuple[str, ...] = ()
    gemini_model: str = DEFAULT_MODEL
    feeds: tuple[str, ...] = DEFAULT_FEEDS
    lookback_hours: int = 24
    max_articles_per_feed: int = 15
    max_total_articles: int = 60
    report_timezone: str = "UTC"
    sender_name: str = "Daily Market Brief"

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, require_email: bool = True
    ) -> "Config":
        env = os.environ if env is None else env

        api_key = (env.get("GEMINI_API_KEY") or "").strip()
        missing = [] if api_key else ["GEMINI_API_KEY"]

        gmail_address = (env.get("GMAIL_ADDRESS") or "").strip()
        # Gmail displays app passwords in groups of 4 separated by spaces.
        gmail_password = re.sub(r"\s+", "", env.get("GMAIL_APP_PASSWORD") or "")
        recipients = _split_list(env.get("RECIPIENT_EMAILS")) or (
            (gmail_address,) if gmail_address else ()
        )

        if require_email:
            if not gmail_address:
                missing.append("GMAIL_ADDRESS")
            if not gmail_password:
                missing.append("GMAIL_APP_PASSWORD")
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

        for addr in recipients:
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", addr):
                raise ConfigError(f"Invalid recipient email address: {addr!r}")

        feeds = _split_list(env.get("RSS_FEEDS")) or DEFAULT_FEEDS
        for url in feeds:
            if urlparse(url).scheme not in {"http", "https"}:
                raise ConfigError(f"Invalid feed URL (must be http/https): {url!r}")

        tz_name = (env.get("REPORT_TIMEZONE") or "UTC").strip()
        try:
            ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigError(f"Unknown REPORT_TIMEZONE: {tz_name!r}") from exc

        return cls(
            gemini_api_key=api_key,
            gmail_address=gmail_address,
            gmail_app_password=gmail_password,
            recipients=recipients,
            gemini_model=(env.get("GEMINI_MODEL") or DEFAULT_MODEL).strip(),
            feeds=feeds,
            lookback_hours=_parse_positive_int(env, "LOOKBACK_HOURS", 24),
            max_articles_per_feed=_parse_positive_int(env, "MAX_ARTICLES_PER_FEED", 15),
            max_total_articles=_parse_positive_int(env, "MAX_TOTAL_ARTICLES", 60),
            report_timezone=tz_name,
            sender_name=(env.get("SENDER_NAME") or "Daily Market Brief").strip(),
        )


# --------------------------------------------------------------------------- #
# News ingestion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Article:
    title: str
    link: str
    source: str
    summary: str = ""
    published: datetime | None = None


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)  # entities may have decoded into tags
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


def _entry_datetime(entry: Mapping[str, Any]) -> datetime | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        struct = entry.get(key)
        if struct:
            try:
                return datetime.fromtimestamp(calendar.timegm(struct), tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
    return None


def fetch_url(url: str, timeout: int = FEED_TIMEOUT_SECONDS) -> bytes:
    """Download raw feed bytes with a timeout and a polite User-Agent."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def parse_feed(content: bytes, feed_url: str, max_items: int = 15) -> list[Article]:
    """Parse RSS/Atom bytes into Articles."""
    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        raise FeedError(f"Unparseable feed {feed_url}: {parsed.get('bozo_exception')}")

    source = clean_text(parsed.feed.get("title")) or urlparse(feed_url).netloc
    articles: list[Article] = []
    for entry in parsed.entries:
        if len(articles) >= max_items:
            break
        title = clean_text(entry.get("title"))
        link = (entry.get("link") or "").strip()
        if not title or urlparse(link).scheme not in {"http", "https"}:
            continue
        summary = clean_text(entry.get("summary") or entry.get("description"))
        if summary == title:
            summary = ""
        articles.append(
            Article(
                title=title,
                link=link,
                source=source,
                summary=truncate(summary, SUMMARY_MAX_CHARS),
                published=_entry_datetime(entry),
            )
        )
    return articles


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def dedupe_articles(articles: Sequence[Article]) -> list[Article]:
    """Drop articles sharing a link or a (normalized) title, keeping the first."""
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Article] = []
    for article in articles:
        link_key = article.link.split("?")[0].rstrip("/").lower()
        title_key = _normalize_title(article.title)
        if link_key in seen_links or title_key in seen_titles:
            continue
        seen_links.add(link_key)
        seen_titles.add(title_key)
        unique.append(article)
    return unique


def filter_recent(
    articles: Sequence[Article], now: datetime, lookback_hours: int
) -> list[Article]:
    """Keep articles published within the window. Undated articles are kept."""
    cutoff = now - timedelta(hours=lookback_hours)
    future_tolerance = now + timedelta(hours=1)
    return [
        a
        for a in articles
        if a.published is None or cutoff <= a.published <= future_tolerance
    ]


def fetch_all_articles(
    feeds: Sequence[str],
    *,
    fetcher: Callable[[str], bytes] = fetch_url,
    now: datetime | None = None,
    lookback_hours: int = 24,
    max_per_feed: int = 15,
    max_total: int = 60,
) -> list[Article]:
    """Fetch every feed in parallel; failures of individual feeds are logged, not fatal."""
    now = now or datetime.now(timezone.utc)

    def _load(url: str) -> list[Article]:
        try:
            items = parse_feed(fetcher(url), url, max_per_feed)
            logger.info("Fetched %d items from %s", len(items), url)
            return items
        except Exception as exc:  # noqa: BLE001 - one bad feed must not kill the run
            logger.warning("Skipping feed %s: %s", url, exc)
            return []

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(feeds)))) as pool:
        results = list(pool.map(_load, feeds))  # preserves feed order

    articles = [a for batch in results for a in batch]
    articles = dedupe_articles(filter_recent(articles, now, lookback_hours))
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    articles.sort(key=lambda a: a.published or epoch, reverse=True)
    return articles[:max_total]


# --------------------------------------------------------------------------- #
# Gemini analysis
# --------------------------------------------------------------------------- #
# NOTE: fields intentionally have no defaults – the Gemini response_schema
# converter does not accept default values.


class Theme(BaseModel):
    title: str = Field(description="Short name of the macro theme.")
    analysis: str = Field(description="2-4 sentence synthesis across the articles.")
    market_impact: str = Field(description="Likely impact on equities, rates, FX, commodities.")
    article_ids: list[int] = Field(description="Ids of supporting articles.")


class SectorOutlook(BaseModel):
    sector: str
    outlook: str = Field(description="One of: positive, negative, neutral.")
    rationale: str

    @field_validator("outlook")
    @classmethod
    def _normalize_outlook(cls, value: str) -> str:
        value = value.strip().lower()
        return value if value in {"positive", "negative", "neutral"} else "neutral"


class TopStory(BaseModel):
    article_id: int
    why_it_matters: str


class MarketBrief(BaseModel):
    headline: str = Field(description="One-line headline capturing the day's dominant story.")
    executive_summary: str = Field(description="3-5 sentence overview for a busy reader.")
    market_sentiment: str = Field(description="One of: bullish, bearish, neutral, mixed.")
    sentiment_rationale: str
    key_themes: list[Theme] = Field(description="3-5 most important macro themes.")
    macro_outlook: str = Field(description="Forward-looking paragraph: what to watch next.")
    sector_watch: list[SectorOutlook] = Field(description="2-5 sectors in focus.")
    risks_to_watch: list[str] = Field(description="2-5 key risks.")
    top_stories: list[TopStory] = Field(description="3-6 must-read articles.")

    @field_validator("market_sentiment")
    @classmethod
    def _normalize_sentiment(cls, value: str) -> str:
        value = value.strip().lower()
        return value if value in {"bullish", "bearish", "neutral", "mixed"} else "mixed"


def build_prompt(articles: Sequence[Article], report_date: datetime) -> str:
    lines = [
        f"Report date: {report_date:%A, %B %d, %Y}",
        f"Below are {len(articles)} recent financial news articles. "
        "Produce today's market briefing as JSON matching the schema.",
        "",
    ]
    for idx, article in enumerate(articles, start=1):
        when = article.published.strftime("%Y-%m-%d %H:%M UTC") if article.published else "n/a"
        lines.append(f"[{idx}] {article.title}")
        lines.append(f"    Source: {article.source} | Published: {when}")
        if article.summary:
            lines.append(f"    Summary: {article.summary}")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_brief(text: str | None) -> MarketBrief:
    if not text or not text.strip():
        raise BriefParseError("Empty response from model")
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return MarketBrief.model_validate_json(cleaned)
    except ValidationError as exc:
        raise BriefParseError(f"Model response did not match schema: {exc}") from exc


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, BriefParseError):
        return True
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in RETRYABLE_HTTP_CODES
    return isinstance(exc, (ConnectionError, TimeoutError, *_TRANSPORT_ERRORS))


def generate_brief(
    client: Any,
    model: str,
    articles: Sequence[Article],
    report_date: datetime,
    *,
    max_attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> MarketBrief:
    """Call Gemini with structured output and exponential-backoff retries."""
    prompt = build_prompt(articles, report_date)
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
        response_mime_type="application/json",
        response_schema=MarketBrief,
    )
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Requesting analysis from %s (attempt %d/%d)", model, attempt, max_attempts)
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            return parse_brief(getattr(response, "text", None))
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Gemini call failed (%s); retrying in %.1fs", exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

SENTIMENT_COLORS = {
    "bullish": "#15803d",
    "bearish": "#b91c1c",
    "neutral": "#475569",
    "mixed": "#b45309",
}
OUTLOOK_STYLE = {
    "positive": ("▲", "#15803d"),
    "negative": ("▼", "#b91c1c"),
    "neutral": ("●", "#64748b"),
}

_EMAIL_CSS = """
body{margin:0;padding:0;background:#f1f5f9;-webkit-text-size-adjust:100%;}
a{color:#2563eb;}
@media only screen and (max-width:620px){
  .container{width:100% !important;border-radius:0 !important;}
  .px{padding-left:18px !important;padding-right:18px !important;}
  .h1{font-size:22px !important;line-height:28px !important;}
}
"""

_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _paragraphs(text: str, style: str) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return "".join(f'<p style="{style}">{_e(p)}</p>' for p in parts)


def _section(title: str, body: str) -> str:
    return (
        '<tr><td class="px" style="padding:8px 32px 20px 32px;">'
        f'<h2 style="margin:0 0 12px 0;font-family:{_FONT};font-size:13px;letter-spacing:1.2px;'
        f'text-transform:uppercase;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:8px;">'
        f"{_e(title)}</h2>{body}</td></tr>"
    )


def _lookup(articles: Sequence[Article], article_id: int) -> Article | None:
    return articles[article_id - 1] if 1 <= article_id <= len(articles) else None


def render_html(
    brief: MarketBrief,
    articles: Sequence[Article],
    report_date: datetime,
    model: str = DEFAULT_MODEL,
) -> str:
    """Render the brief as a responsive, email-client-safe HTML document.

    All model/feed text is HTML-escaped; links come only from fetched feeds.
    """
    p_style = f"margin:0 0 12px 0;font-family:{_FONT};font-size:15px;line-height:24px;color:#334155;"
    small = f"font-family:{_FONT};font-size:13px;line-height:20px;color:#64748b;"
    sentiment_color = SENTIMENT_COLORS.get(brief.market_sentiment, "#475569")

    # Executive summary + sentiment
    summary_html = _paragraphs(brief.executive_summary, p_style) + (
        f'<p style="margin:4px 0 0 0;{small}">'
        f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;background:{sentiment_color};'
        f'color:#ffffff;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.8px;">'
        f"{_e(brief.market_sentiment)}</span>&nbsp; {_e(brief.sentiment_rationale)}</p>"
    )

    # Themes
    theme_blocks = []
    for theme in brief.key_themes:
        links, seen = [], set()
        for aid in theme.article_ids:
            article = _lookup(articles, aid)
            if article and article.link not in seen:
                seen.add(article.link)
                links.append(
                    f'<a href="{_e(article.link)}" style="color:#2563eb;text-decoration:none;">'
                    f"{_e(article.source)}</a>"
                )
        sources = f'<p style="margin:6px 0 0 0;{small}">Sources: {" · ".join(links)}</p>' if links else ""
        theme_blocks.append(
            '<div style="margin:0 0 16px 0;padding:14px 16px;background:#f8fafc;border-left:4px solid #2563eb;border-radius:6px;">'
            f'<h3 style="margin:0 0 6px 0;font-family:{_FONT};font-size:16px;color:#0f172a;">{_e(theme.title)}</h3>'
            f'<p style="margin:0 0 8px 0;font-family:{_FONT};font-size:14px;line-height:22px;color:#334155;">{_e(theme.analysis)}</p>'
            f'<p style="margin:0;font-family:{_FONT};font-size:14px;line-height:22px;color:#0f172a;">'
            f"<strong>Market impact:</strong> {_e(theme.market_impact)}</p>{sources}</div>"
        )

    # Sectors
    sector_rows = []
    for s in brief.sector_watch:
        symbol, color = OUTLOOK_STYLE.get(s.outlook, OUTLOOK_STYLE["neutral"])
        sector_rows.append(
            "<tr>"
            f'<td valign="top" style="padding:8px 8px 8px 0;font-family:{_FONT};font-size:14px;color:{color};'
            f'font-weight:700;white-space:nowrap;">{symbol} {_e(s.sector)}</td>'
            f'<td valign="top" style="padding:8px 0;font-family:{_FONT};font-size:14px;line-height:21px;color:#334155;">'
            f"{_e(s.rationale)}</td></tr>"
        )
    sectors_html = (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{"".join(sector_rows)}</table>'
    )

    risks_html = "<ul style=\"margin:0;padding-left:20px;\">" + "".join(
        f'<li style="margin:0 0 6px 0;font-family:{_FONT};font-size:14px;line-height:22px;color:#334155;">{_e(r)}</li>'
        for r in brief.risks_to_watch
    ) + "</ul>"

    # Top stories (only valid ids → real links)
    story_items = []
    for story in brief.top_stories:
        article = _lookup(articles, story.article_id)
        if not article:
            continue
        story_items.append(
            '<div style="margin:0 0 14px 0;">'
            f'<a href="{_e(article.link)}" style="font-family:{_FONT};font-size:15px;font-weight:600;'
            f'color:#1d4ed8;text-decoration:none;line-height:22px;">{_e(article.title)}</a>'
            f'<p style="margin:2px 0 0 0;{small}"><em>{_e(article.source)}</em> — {_e(story.why_it_matters)}</p></div>'
        )

    sections = [
        _section("Executive Summary", summary_html),
        _section("Key Macro Themes", "".join(theme_blocks)) if theme_blocks else "",
        _section("Outlook", _paragraphs(brief.macro_outlook, p_style)),
        _section("Sector Watch", sectors_html) if sector_rows else "",
        _section("Risks to Watch", risks_html) if brief.risks_to_watch else "",
        _section("Top Stories", "".join(story_items)) if story_items else "",
    ]

    date_label = f"{report_date:%A, %B} {report_date.day}, {report_date.year}"
    preheader = truncate(brief.executive_summary, 140)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="x-apple-disable-message-reformatting">
<title>{_e(brief.headline)}</title>
<style>{_EMAIL_CSS}</style>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{_e(preheader)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:24px 8px;">
<table role="presentation" class="container" width="600" cellpadding="0" cellspacing="0"
       style="width:600px;max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;">
  <tr><td class="px" style="padding:28px 32px;background:#0f172a;">
    <p style="margin:0 0 6px 0;font-family:{_FONT};font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:#93c5fd;">
      Daily Market Brief · {_e(date_label)}</p>
    <h1 class="h1" style="margin:0;font-family:{_FONT};font-size:26px;line-height:32px;color:#ffffff;">{_e(brief.headline)}</h1>
  </td></tr>
  <tr><td style="height:16px;line-height:16px;">&nbsp;</td></tr>
  {"".join(sections)}
  <tr><td class="px" style="padding:20px 32px;background:#f8fafc;border-top:1px solid #e2e8f0;">
    <p style="margin:0;{small}">Generated automatically from {len(articles)} articles using {_e(model)}.
    AI-generated analysis may contain errors. This is not investment advice.</p>
  </td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


def render_text(brief: MarketBrief, articles: Sequence[Article], report_date: datetime) -> str:
    """Plain-text alternative for clients that do not render HTML."""
    out = [
        f"DAILY MARKET BRIEF — {report_date:%A, %B} {report_date.day}, {report_date.year}",
        "=" * 60,
        brief.headline,
        "",
        "EXECUTIVE SUMMARY",
        brief.executive_summary,
        f"Sentiment: {brief.market_sentiment.upper()} — {brief.sentiment_rationale}",
        "",
        "KEY MACRO THEMES",
    ]
    for theme in brief.key_themes:
        out += [f"* {theme.title}", f"  {theme.analysis}", f"  Market impact: {theme.market_impact}"]
    out += ["", "OUTLOOK", brief.macro_outlook]
    if brief.sector_watch:
        out += ["", "SECTOR WATCH"]
        out += [f"* {s.sector} ({s.outlook}): {s.rationale}" for s in brief.sector_watch]
    if brief.risks_to_watch:
        out += ["", "RISKS TO WATCH"] + [f"* {r}" for r in brief.risks_to_watch]
    stories = [(s, _lookup(articles, s.article_id)) for s in brief.top_stories]
    stories = [(s, a) for s, a in stories if a]
    if stories:
        out += ["", "TOP STORIES"]
        for story, article in stories:
            out += [f"* {article.title} ({article.source})", f"  {story.why_it_matters}", f"  {article.link}"]
    out += ["", "AI-generated analysis may contain errors. Not investment advice."]
    return "\n".join(out)


def build_subject(brief: MarketBrief, report_date: datetime) -> str:
    prefix = f"Market Brief {report_date:%b} {report_date.day}: "
    return prefix + truncate(clean_text(brief.headline), 110)


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


def build_message(config: Config, subject: str, html_body: str, text_body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((config.sender_name, config.gmail_address))
    msg["To"] = config.gmail_address
    # Recipients go in Bcc so subscribers don't see each other's addresses.
    msg["Bcc"] = ", ".join(config.recipients)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    return msg


def send_email(
    config: Config,
    subject: str,
    html_body: str,
    text_body: str,
    *,
    smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP_SSL,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Send via Gmail SMTP over implicit TLS with retries on transient errors."""
    msg = build_message(config, subject, html_body, text_body)
    for attempt in range(1, max_attempts + 1):
        try:
            with smtp_factory(
                SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=30
            ) as server:
                server.login(config.gmail_address, config.gmail_app_password)
                server.send_message(msg)
            logger.info("Email sent to %d recipient(s)", len(config.recipients))
            return
        except smtplib.SMTPAuthenticationError:
            raise  # never retry bad credentials
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, OSError) as exc:
            if attempt >= max_attempts:
                raise
            delay = 2.0 * attempt
            logger.warning("SMTP send failed (%s); retrying in %.0fs", exc, delay)
            sleep(delay)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(
    config: Config,
    *,
    dry_run: bool = False,
    output_dir: Path = Path("output"),
    fetcher: Callable[[str], bytes] = fetch_url,
    client: Any = None,
    now: datetime | None = None,
    smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP_SSL,
) -> int:
    """Execute the full pipeline. Returns a process exit code."""
    now = now or datetime.now(timezone.utc)
    report_date = now.astimezone(ZoneInfo(config.report_timezone))

    articles = fetch_all_articles(
        config.feeds,
        fetcher=fetcher,
        now=now,
        lookback_hours=config.lookback_hours,
        max_per_feed=config.max_articles_per_feed,
        max_total=config.max_total_articles,
    )
    logger.info("Collected %d unique recent articles", len(articles))
    if not articles:
        logger.error("No articles collected from any feed; aborting without sending.")
        return 2

    client = client or genai.Client(api_key=config.gemini_api_key)
    brief = generate_brief(client, config.gemini_model, articles, report_date)

    html_body = render_html(brief, articles, report_date, config.gemini_model)
    text_body = render_text(brief, articles, report_date)
    subject = build_subject(brief, report_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"brief-{report_date:%Y-%m-%d}.html"
    out_file.write_text(html_body, encoding="utf-8")
    logger.info("Saved newsletter to %s", out_file)

    if dry_run:
        logger.info("Dry run: skipping email delivery. Subject would be: %s", subject)
        return 0

    send_email(config, subject, html_body, text_body, smtp_factory=smtp_factory)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily financial news reporter agent")
    parser.add_argument("--dry-run", action="store_true", help="Generate the brief but do not send email.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Where to save the HTML.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    dry_run = args.dry_run or _parse_bool(os.environ.get("DRY_RUN"))

    try:
        config = Config.from_env(require_email=not dry_run)
        return run(config, dry_run=dry_run, output_dir=args.output_dir)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except smtplib.SMTPAuthenticationError:
        logger.error("Gmail rejected the credentials. Use a 16-char App Password, not your account password.")
        return 1
    except Exception:  # noqa: BLE001
        logger.exception("Agent run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
