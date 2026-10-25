"""Offline test-suite for agent.py (no network, no real API keys)."""

from __future__ import annotations

import json
import smtplib
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent
from agent import (
    Article,
    BriefParseError,
    Config,
    ConfigError,
    MarketBrief,
    build_prompt,
    build_subject,
    clean_text,
    dedupe_articles,
    fetch_all_articles,
    filter_recent,
    generate_brief,
    parse_brief,
    parse_feed,
    render_html,
    render_text,
    run,
    send_email,
    truncate,
)

NOW = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def rss(title: str, items: list[tuple[str, str, str, datetime | None]]) -> bytes:
    body = []
    for t, link, desc, pub in items:
        pub_xml = f"<pubDate>{format_datetime(pub)}</pubDate>" if pub else ""
        body.append(
            f"<item><title>{t}</title><link>{link}</link>"
            f"<description><![CDATA[{desc}]]></description>{pub_xml}</item>"
        )
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        f"<title>{title}</title><link>https://example.com</link>"
        f"{''.join(body)}</channel></rss>"
    ).encode()


FEED_A = rss(
    "Market Wire",
    [
        ("Fed holds rates steady", "https://a.com/fed", "<p>The <b>Fed</b> held &amp; signalled patience.</p>", NOW - timedelta(hours=2)),
        ("Oil jumps 3%", "https://a.com/oil", "Brent rallied on supply worries.", NOW - timedelta(hours=5)),
        ("Old news", "https://a.com/old", "Stale.", NOW - timedelta(days=3)),
    ],
)
FEED_B = rss(
    "Econ Daily",
    [
        ("FED HOLDS RATES STEADY!", "https://b.com/fed-dup", "Duplicate title.", NOW - timedelta(hours=1)),
        ("Treasury yields slip", "https://b.com/yields", "10y yield fell 5bp.", NOW - timedelta(hours=3)),
    ],
)


def make_fetcher(mapping: dict[str, bytes]):
    def fetcher(url: str) -> bytes:
        if url not in mapping:
            raise ConnectionError(f"boom: {url}")
        return mapping[url]

    return fetcher


def brief_dict(**overrides) -> dict:
    data = {
        "headline": "Fed patience steadies markets as oil climbs",
        "executive_summary": "The Fed held rates.\nOil rose on supply fears.",
        "market_sentiment": "Mixed",
        "sentiment_rationale": "Dovish Fed offset by energy-driven inflation risk.",
        "key_themes": [
            {
                "title": "Monetary policy",
                "analysis": "The Fed signalled patience.",
                "market_impact": "Supportive for duration.",
                "article_ids": [1, 99],
            }
        ],
        "macro_outlook": "Watch CPI next week.",
        "sector_watch": [{"sector": "Energy", "outlook": "POSITIVE", "rationale": "Higher crude."}],
        "risks_to_watch": ["Oil supply shock"],
        "top_stories": [
            {"article_id": 2, "why_it_matters": "Inflation input."},
            {"article_id": 42, "why_it_matters": "Hallucinated id."},
        ],
    }
    data.update(overrides)
    return data


@pytest.fixture
def articles() -> list[Article]:
    return [
        Article("Fed holds rates steady", "https://a.com/fed", "Market Wire", "The Fed held.", NOW),
        Article("Oil jumps 3%", "https://a.com/oil", "Market Wire", "Brent rallied.", NOW),
    ]


@pytest.fixture
def brief() -> MarketBrief:
    return MarketBrief.model_validate(brief_dict())


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(text=item)


class FakeClient:
    def __init__(self, *responses):
        self.models = FakeModels(responses)


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, context=None, timeout=None):
        self.host, self.port, self.context, self.timeout = host, port, context, timeout
        self.logged_in = None
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)


BASE_ENV = {
    "GEMINI_API_KEY": "test-key",
    "GMAIL_ADDRESS": "me@gmail.com",
    "GMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_defaults(self):
        cfg = Config.from_env(BASE_ENV)
        assert cfg.gmail_app_password == "abcdefghijklmnop"  # spaces stripped
        assert cfg.recipients == ("me@gmail.com",)  # falls back to sender
        assert cfg.feeds == agent.DEFAULT_FEEDS
        assert cfg.gemini_model == agent.DEFAULT_MODEL
        assert cfg.lookback_hours == 24

    def test_overrides(self):
        env = BASE_ENV | {
            "RECIPIENT_EMAILS": "a@x.com, b@y.org",
            "RSS_FEEDS": "https://f1.com/rss,https://f2.com/rss",
            "GEMINI_MODEL": "gemini-2.5-pro",
            "LOOKBACK_HOURS": "48",
            "REPORT_TIMEZONE": "Asia/Bangkok",
        }
        cfg = Config.from_env(env)
        assert cfg.recipients == ("a@x.com", "b@y.org")
        assert cfg.feeds == ("https://f1.com/rss", "https://f2.com/rss")
        assert cfg.gemini_model == "gemini-2.5-pro"
        assert cfg.lookback_hours == 48
        assert cfg.report_timezone == "Asia/Bangkok"

    def test_missing_required(self):
        with pytest.raises(ConfigError, match="GEMINI_API_KEY.*GMAIL_ADDRESS.*GMAIL_APP_PASSWORD"):
            Config.from_env({})

    def test_email_optional_for_dry_run(self):
        cfg = Config.from_env({"GEMINI_API_KEY": "k"}, require_email=False)
        assert cfg.recipients == ()

    @pytest.mark.parametrize(
        "key,value",
        [
            ("LOOKBACK_HOURS", "abc"),
            ("LOOKBACK_HOURS", "0"),
            ("RECIPIENT_EMAILS", "not-an-email"),
            ("RSS_FEEDS", "ftp://bad.com/feed"),
            ("REPORT_TIMEZONE", "Mars/Olympus"),
        ],
    )
    def test_invalid_values(self, key, value):
        with pytest.raises(ConfigError):
            Config.from_env(BASE_ENV | {key: value})


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class TestIngestion:
    def test_clean_text(self):
        assert clean_text("<p>Hello&nbsp;<b>world</b> &amp; co</p>") == "Hello world & co"
        assert clean_text("&lt;script&gt;x&lt;/script&gt;") == "x"
        assert clean_text(None) == ""

    def test_truncate(self):
        assert truncate("short", 10) == "short"
        out = truncate("the quick brown fox jumps over", 15)
        assert out.endswith("…") and len(out) <= 15

    def test_parse_feed(self):
        items = parse_feed(FEED_A, "https://a.com/rss", max_items=10)
        assert [a.title for a in items] == ["Fed holds rates steady", "Oil jumps 3%", "Old news"]
        assert items[0].source == "Market Wire"
        assert items[0].summary == "The Fed held & signalled patience."
        assert items[0].published == NOW - timedelta(hours=2)

    def test_parse_feed_respects_max_items_and_skips_bad_links(self):
        feed = rss("S", [("No link", "javascript:alert(1)", "", None), ("Ok", "https://ok.com", "", None)])
        assert [a.title for a in parse_feed(feed, "https://s.com", 5)] == ["Ok"]
        assert len(parse_feed(FEED_A, "https://a.com", max_items=1)) == 1

    def test_parse_feed_garbage_raises(self):
        with pytest.raises(agent.FeedError):
            parse_feed(b"this is not xml at all <<<", "https://bad.com")

    def test_dedupe(self):
        arts = [
            Article("Fed holds rates", "https://a.com/1?utm=x", "A"),
            Article("Fed Holds Rates!", "https://b.com/2", "B"),  # same title
            Article("Different", "https://a.com/1", "C"),  # same link sans query
            Article("Unique", "https://c.com/3", "D"),
        ]
        assert [a.source for a in dedupe_articles(arts)] == ["A", "D"]

    def test_filter_recent_keeps_undated(self):
        arts = [
            Article("new", "https://x/1", "S", published=NOW - timedelta(hours=1)),
            Article("old", "https://x/2", "S", published=NOW - timedelta(hours=30)),
            Article("undated", "https://x/3", "S"),
            Article("future", "https://x/4", "S", published=NOW + timedelta(days=1)),
        ]
        assert [a.title for a in filter_recent(arts, NOW, 24)] == ["new", "undated"]

    def test_fetch_all_articles_end_to_end(self):
        fetcher = make_fetcher({"https://a.com/rss": FEED_A, "https://b.com/rss": FEED_B})
        result = fetch_all_articles(
            ["https://a.com/rss", "https://b.com/rss", "https://down.com/rss"],
            fetcher=fetcher,
            now=NOW,
        )
        titles = [a.title for a in result]
        # Sorted newest first, old item filtered, duplicate title dropped, broken feed ignored.
        assert titles == ["Fed holds rates steady", "Treasury yields slip", "Oil jumps 3%"]

    def test_fetch_all_articles_max_total(self):
        fetcher = make_fetcher({"https://a.com/rss": FEED_A})
        assert len(fetch_all_articles(["https://a.com/rss"], fetcher=fetcher, now=NOW, max_total=1)) == 1


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


class TestGemini:
    def test_build_prompt(self, articles):
        prompt = build_prompt(articles, NOW)
        assert "[1] Fed holds rates steady" in prompt
        assert "[2] Oil jumps 3%" in prompt
        assert "Sunday, October 25, 2026" in prompt

    def test_parse_brief_normalizes(self):
        b = parse_brief(json.dumps(brief_dict(market_sentiment="Euphoric")))
        assert b.market_sentiment == "mixed"
        assert b.sector_watch[0].outlook == "positive"

    def test_parse_brief_strips_code_fences(self):
        b = parse_brief("```json\n" + json.dumps(brief_dict()) + "\n```")
        assert b.headline.startswith("Fed patience")

    @pytest.mark.parametrize("bad", [None, "", "not json", json.dumps({"headline": "x"})])
    def test_parse_brief_invalid(self, bad):
        with pytest.raises(BriefParseError):
            parse_brief(bad)

    def test_generate_brief_success_and_request_shape(self, articles):
        client = FakeClient(json.dumps(brief_dict()))
        b = generate_brief(client, "gemini-test", articles, NOW, sleep=lambda _: None)
        assert b.market_sentiment == "mixed"
        call = client.models.calls[0]
        assert call["model"] == "gemini-test"
        assert "[1] Fed holds rates steady" in call["contents"]
        assert call["config"].response_mime_type == "application/json"
        assert call["config"].response_schema is MarketBrief

    def test_generate_brief_retries_transient_errors(self, articles):
        delays: list[float] = []
        client = FakeClient(ConnectionError("reset"), "garbage", json.dumps(brief_dict()))
        b = generate_brief(client, "m", articles, NOW, sleep=delays.append, base_delay=1)
        assert b.headline
        assert delays == [1, 2]
        assert len(client.models.calls) == 3

    def test_generate_brief_gives_up(self, articles):
        client = FakeClient("bad", "bad")
        with pytest.raises(BriefParseError):
            generate_brief(client, "m", articles, NOW, max_attempts=2, sleep=lambda _: None)

    def test_generate_brief_does_not_retry_fatal(self, articles):
        client = FakeClient(ValueError("bad request"), json.dumps(brief_dict()))
        with pytest.raises(ValueError):
            generate_brief(client, "m", articles, NOW, sleep=lambda _: None)
        assert len(client.models.calls) == 1

    def test_is_retryable_api_error_codes(self):
        def api_error(code):
            err = agent.genai_errors.APIError.__new__(agent.genai_errors.APIError)
            err.code = code
            return err

        assert agent._is_retryable(api_error(429))
        assert agent._is_retryable(api_error(503))
        assert not agent._is_retryable(api_error(400))
        assert not agent._is_retryable(api_error(403))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


class TestRendering:
    def test_render_html_structure(self, brief, articles):
        out = render_html(brief, articles, NOW, "gemini-test")
        assert out.startswith("<!DOCTYPE html>")
        assert 'name="viewport"' in out and "@media only screen" in out
        for section in ("Executive Summary", "Key Macro Themes", "Outlook", "Sector Watch", "Risks to Watch", "Top Stories"):
            assert section in out
        assert 'href="https://a.com/fed"' in out  # theme source from id 1
        assert 'href="https://a.com/oil"' in out  # top story id 2
        assert "Hallucinated id." not in out  # invalid id 42 dropped
        assert "gemini-test" in out
        assert "Sunday, October 25, 2026" in out

    def test_render_html_escapes_untrusted_content(self, articles):
        evil = MarketBrief.model_validate(
            brief_dict(headline="<script>alert('x')</script>", risks_to_watch=["<img src=x onerror=1>"])
        )
        evil_articles = [Article('Bad "title" <b>', 'https://a.com/?q="><script>', "Src")] + articles
        out = render_html(evil, evil_articles, NOW)
        assert "<script>" not in out
        assert "<img src=x" not in out
        assert "&lt;script&gt;" in out
        assert '"><script>' not in out

    def test_render_text(self, brief, articles):
        out = render_text(brief, articles, NOW)
        assert "EXECUTIVE SUMMARY" in out
        assert "Sentiment: MIXED" in out
        assert "https://a.com/oil" in out
        assert "Hallucinated id." not in out

    def test_subject(self, brief):
        assert build_subject(brief, NOW) == "Market Brief Oct 25: Fed patience steadies markets as oil climbs"


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class TestEmail:
    def setup_method(self):
        FakeSMTP.instances.clear()

    def test_send_email(self):
        cfg = Config.from_env(BASE_ENV | {"RECIPIENT_EMAILS": "a@x.com,b@y.com"})
        send_email(cfg, "Subj", "<p>hi</p>", "hi", smtp_factory=FakeSMTP)
        smtp = FakeSMTP.instances[0]
        assert (smtp.host, smtp.port) == ("smtp.gmail.com", 465)
        assert smtp.context is not None
        assert smtp.logged_in == ("me@gmail.com", "abcdefghijklmnop")
        msg = smtp.sent[0]
        assert msg["Subject"] == "Subj"
        assert msg["Bcc"] == "a@x.com, b@y.com"
        assert msg.get_body(("html",)).get_content().strip() == "<p>hi</p>"
        assert msg.get_body(("plain",)).get_content().strip() == "hi"

    def test_send_email_retries_then_succeeds(self):
        attempts = {"n": 0}

        def flaky(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise smtplib.SMTPServerDisconnected("dropped")
            return FakeSMTP(*args, **kwargs)

        cfg = Config.from_env(BASE_ENV)
        send_email(cfg, "S", "<p/>", "t", smtp_factory=flaky, sleep=lambda _: None)
        assert attempts["n"] == 2
        assert FakeSMTP.instances[0].sent

    def test_send_email_auth_error_not_retried(self):
        calls = {"n": 0}

        def bad_auth(*args, **kwargs):
            calls["n"] += 1
            raise smtplib.SMTPAuthenticationError(535, b"bad creds")

        cfg = Config.from_env(BASE_ENV)
        with pytest.raises(smtplib.SMTPAuthenticationError):
            send_email(cfg, "S", "<p/>", "t", smtp_factory=bad_auth, sleep=lambda _: None)
        assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


class TestRun:
    def setup_method(self):
        FakeSMTP.instances.clear()

    def _config(self, **extra):
        return Config.from_env(BASE_ENV | {"RSS_FEEDS": "https://a.com/rss,https://b.com/rss"} | extra)

    def test_full_pipeline_sends_email(self, tmp_path: Path):
        fetcher = make_fetcher({"https://a.com/rss": FEED_A, "https://b.com/rss": FEED_B})
        code = run(
            self._config(REPORT_TIMEZONE="Asia/Bangkok"),
            output_dir=tmp_path,
            fetcher=fetcher,
            client=FakeClient(json.dumps(brief_dict())),
            now=NOW,
            smtp_factory=FakeSMTP,
        )
        assert code == 0
        html_file = tmp_path / "brief-2026-10-25.html"
        assert html_file.exists()
        assert "Fed patience steadies markets" in html_file.read_text()
        assert FakeSMTP.instances and FakeSMTP.instances[0].sent

    def test_dry_run_does_not_send(self, tmp_path: Path):
        fetcher = make_fetcher({"https://a.com/rss": FEED_A})
        code = run(
            self._config(),
            dry_run=True,
            output_dir=tmp_path,
            fetcher=fetcher,
            client=FakeClient(json.dumps(brief_dict())),
            now=NOW,
            smtp_factory=FakeSMTP,
        )
        assert code == 0
        assert list(tmp_path.glob("brief-*.html"))
        assert FakeSMTP.instances == []

    def test_no_articles_aborts(self, tmp_path: Path):
        client = FakeClient()
        code = run(self._config(), output_dir=tmp_path, fetcher=make_fetcher({}), client=client, now=NOW)
        assert code == 2
        assert client.models.calls == []

    def test_main_config_error_exit_code(self, monkeypatch):
        monkeypatch.setattr(agent, "load_dotenv", lambda *a, **k: None)
        for key in ("GEMINI_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "DRY_RUN"):
            monkeypatch.delenv(key, raising=False)
        assert agent.main([]) == 1
