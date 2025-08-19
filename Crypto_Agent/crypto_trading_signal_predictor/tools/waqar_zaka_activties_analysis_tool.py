from __future__ import annotations
import os
import re
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Iterable, Tuple

from pydantic import BaseModel, Field

# --- Optional deps: import guarded ---
try:
    import tweepy  # X (Twitter)
except Exception:  # pragma: no cover
    tweepy = None

try:
    from googleapiclient.discovery import build  # YouTube
except Exception:  # pragma: no cover
    build = None

try:
    from telethon import TelegramClient  # Telegram
    from telethon.tl.functions.messages import GetHistoryRequest
    from telethon.tl.types import PeerChannel
except Exception:  # pragma: no cover
    TelegramClient = None

try:
    import feedparser  # RSS fallback
except Exception:  # pragma: no cover
    feedparser = None

try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # VADER
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None

# You indicated this decorator exists in your repo
try:
    from agents import function_tool
except Exception:
    # Fallback no-op decorator to keep file runnable in isolation
    def function_tool(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

# ---------------- Config -----------------
# Fill these with Waqar Zaka's *official* public channels.
OFFICIAL_CHANNELS: Dict[str, List[str]] = {
    "x": [
        # e.g., "@ZakaWaqar"  (placeholder — replace with official)
    ],
    "youtube": [
        # e.g., "UCxxxxxxxxxxxxxxxx" (Channel ID) or channel handle without '@'
    ],
    "telegram": [
        # e.g., "@WaqarZakaOfficial" (public channel username)
    ],
    # Optional future additions:
    # "facebook": ["<page-id-or-handle>"],
    # "instagram": ["<business-account-id-or-handle>"],
    # "rss": ["https://<official-site>/feed"],
}

DEFAULT_LOOKBACK_HOURS = 72
DEFAULT_MAX_POSTS_PER_SOURCE = 30

# -------------- Models -------------------
class WaqarZakaAnalysisInput(BaseModel):
    lookback_hours: int = Field(
        DEFAULT_LOOKBACK_HOURS, ge=1, le=24*14,
        description="How many recent hours of posts to fetch per source",
    )
    max_posts_per_source: int = Field(
        DEFAULT_MAX_POSTS_PER_SOURCE, ge=1, le=200,
        description="Maximum posts to pull per channel per platform",
    )
    platforms: Optional[List[str]] = Field(
        None, description="Subset of platforms to query (default: all configured)"
    )
    channels_override: Optional[Dict[str, List[str]]] = Field(
        None, description="Explicit platform->channels mapping to override OFFICIAL_CHANNELS"
    )

class SocialPost(BaseModel):
    platform: str
    channel: str
    id: str
    url: Optional[str] = None
    text: str
    created_at: datetime
    raw: Optional[Dict[str, Any]] = None

class CoinMention(BaseModel):
    symbol: str
    count: int

class WaqarZakaAnalysisOutput(BaseModel):
    fetched_at: datetime
    window_start: datetime
    platforms_scanned: List[str]
    channels_scanned: Dict[str, List[str]]
    total_posts: int
    posts: List[SocialPost]
    sentiment: Dict[str, float]  # {neg, neu, pos, compound}
    top_mentions: List[CoinMention]

# -------------- Utilities ----------------
TICKER_REGEX = re.compile(r"(?<![A-Z0-9])\$?([A-Z]{2,10})(?![A-Z0-9])")
CRYPTO_WORDS = {
    "btc", "eth", "bnb", "sol", "xrp", "ada", "doge", "matic", "dot", "avax", "trx",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _within_window(dt: datetime, window_start: datetime) -> bool:
    try:
        return dt.replace(tzinfo=timezone.utc) >= window_start
    except Exception:
        return False


def _hash_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _extract_mentions(text: str) -> List[str]:
    mentions: List[str] = []
    for m in TICKER_REGEX.findall(text or ""):
        token = m.upper().strip("$")
        if 2 <= len(token) <= 10:
            mentions.append(token)
    # Simple heuristics for common crypto words
    for w in CRYPTO_WORDS:
        if re.search(fr"(?i)(?<![a-z]){re.escape(w)}(?![a-z])", text or ""):
            mentions.append(w.upper())
    return mentions


def _aggregate_mentions(posts: List[SocialPost]) -> List[CoinMention]:
    counts: Dict[str, int] = {}
    for p in posts:
        for sym in _extract_mentions(p.text):
            counts[sym] = counts.get(sym, 0) + 1
    return [CoinMention(symbol=k, count=v) for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]


def _sentiment_scores(texts: Iterable[str]) -> Dict[str, float]:
    joined = "\n".join([t for t in texts if t])
    if not joined:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    if SentimentIntensityAnalyzer is None:
        # Fallback neutral if NLTK not available
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(joined)

# -------------- Adapters -----------------
class XAdapter:
    def __init__(self, bearer: Optional[str]):
        self.bearer = bearer
        self.client = None
        if bearer and tweepy is not None:
            self.client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    def fetch_user_id(self, handle: str) -> Optional[str]:
        if not self.client or not handle:
            return None
        try:
            uname = handle.lstrip("@")
            resp = self.client.get_user(username=uname, user_fields=["id"])
            return str(resp.data.id) if resp and resp.data else None
        except Exception:
            return None

    def fetch(self, channel: str, window_start: datetime, limit: int) -> List[SocialPost]:
        if not self.client:
            return []
        uid = self.fetch_user_id(channel)
        if not uid:
            return []
        posts: List[SocialPost] = []
        try:
            resp = self.client.get_users_tweets(
                id=uid,
                max_results=min(100, max(5, limit)),
                tweet_fields=["created_at","text","id"]
            )
            if not resp or not resp.data:
                return []
            for t in resp.data:
                created_at = t.created_at.replace(tzinfo=timezone.utc) if t.created_at else _now_utc()
                if not _within_window(created_at, window_start):
                    continue
                url = f"https://x.com/{channel.lstrip('@')}/status/{t.id}"
                posts.append(SocialPost(
                    platform="x", channel=channel, id=str(t.id), url=url, text=t.text or "", created_at=created_at, raw={"tweet": t.data}
                ))
        except Exception:
            return posts
        return posts[:limit]


class YouTubeAdapter:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.svc = None
        if api_key and build is not None:
            self.svc = build("youtube", "v3", developerKey=api_key)

    def _channel_uploads_playlist(self, channel_id_or_handle: str) -> Optional[str]:
        if not self.svc:
            return None
        try:
            if channel_id_or_handle.startswith("UC"):  # likely channel ID
                resp = self.svc.channels().list(part="contentDetails", id=channel_id_or_handle).execute()
            else:
                # Treat as handle without '@'
                handle = channel_id_or_handle.lstrip("@")
                resp = self.svc.channels().list(part="contentDetails", forHandle=handle).execute()
            items = resp.get("items", [])
            if not items:
                return None
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except Exception:
            return None

    def fetch(self, channel: str, window_start: datetime, limit: int) -> List[SocialPost]:
        if not self.svc:
            return []
        uploads = self._channel_uploads_playlist(channel)
        if not uploads:
            return []
        posts: List[SocialPost] = []
        try:
            resp = self.svc.playlistItems().list(part="snippet,contentDetails", playlistId=uploads, maxResults=min(50, limit)).execute()
            for item in resp.get("items", []):
                sn = item.get("snippet", {})
                ts = sn.get("publishedAt")
                created_at = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else _now_utc()
                if not _within_window(created_at, window_start):
                    continue
                vid = sn.get("resourceId", {}).get("videoId", "")
                title = sn.get("title", "")
                desc = sn.get("description", "")
                text = f"{title}\n\n{desc}".strip()
                url = f"https://www.youtube.com/watch?v={vid}" if vid else None
                posts.append(SocialPost(platform="youtube", channel=channel, id=vid or _hash_id(title), url=url, text=text, created_at=created_at, raw=item))
        except Exception:
            return posts
        return posts[:limit]


class TelegramAdapter:
    def __init__(self, api_id: Optional[int], api_hash: Optional[str], session_name: Optional[str] = None):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name or "tg_session"
        self.client = None
        if TelegramClient and api_id and api_hash:
            try:
                # NOTE: On first run, Telethon may prompt for phone login in a console.
                self.client = TelegramClient(self.session_name, api_id, api_hash)
            except Exception:
                self.client = None

    def fetch(self, channel: str, window_start: datetime, limit: int) -> List[SocialPost]:
        if not self.client:
            return []
        posts: List[SocialPost] = []
        try:
            with self.client:
                entity = self.client.get_entity(channel)
                history = self.client(GetHistoryRequest(
                    peer=entity,
                    limit=min(100, max(5, limit)),
                    offset_date=None,
                    offset_id=0,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0
                ))
                for msg in history.messages:
                    if not getattr(msg, "message", None):
                        continue
                    dt = getattr(msg, "date", None) or _now_utc()
                    created_at = dt.replace(tzinfo=timezone.utc)
                    if not _within_window(created_at, window_start):
                        continue
                    mid = str(getattr(msg, "id", _hash_id(channel, msg.message)))
                    url = None  # For public channels, you could construct t.me link if username is known
                    posts.append(SocialPost(platform="telegram", channel=channel, id=mid, url=url, text=msg.message, created_at=created_at, raw={}))
        except Exception:
            return posts
        return posts[:limit]


class RSSAdapter:
    def __init__(self):
        self.enabled = feedparser is not None

    def fetch(self, url: str, window_start: datetime, limit: int) -> List[SocialPost]:
        if not self.enabled:
            return []
        posts: List[SocialPost] = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                title = getattr(entry, "title", "")
                desc = getattr(entry, "summary", "")
                link = getattr(entry, "link", None)
                ts = getattr(entry, "published", None) or getattr(entry, "updated", None)
                created_at = _now_utc()
                if ts:
                    try:
                        created_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pass
                if not _within_window(created_at, window_start):
                    continue
                pid = _hash_id(url, link or title)
                text = (title + "\n\n" + desc).strip()
                posts.append(SocialPost(platform="rss", channel=url, id=pid, url=link, text=text, created_at=created_at, raw={}))
        except Exception:
            return posts
        return posts


# -------------- Orchestrator -------------
class Collector:
    def __init__(self, cfg: Dict[str, List[str]]):
        self.cfg = cfg
        self.adapters = {
            "x": XAdapter(os.getenv("X_BEARER_TOKEN")),
            "youtube": YouTubeAdapter(os.getenv("YT_API_KEY")),
            "telegram": TelegramAdapter(
                api_id=int(os.getenv("TELEGRAM_API_ID")) if os.getenv("TELEGRAM_API_ID") else None,
                api_hash=os.getenv("TELEGRAM_API_HASH"),
                session_name=os.getenv("TELEGRAM_BOT_SESSION") or "tg_session",
            ),
            # Optional: add adapters for facebook/instagram using Graph API
        }
        self.rss_adapter = RSSAdapter()

    def run(self, window_start: datetime, per_source_limit: int, platforms: Optional[List[str]] = None) -> Tuple[List[SocialPost], Dict[str, List[str]]]:
        posts: List[SocialPost] = []
        scanned: Dict[str, List[str]] = {}
        plats = platforms or list(self.cfg.keys())
        for platform in plats:
            sources = self.cfg.get(platform, [])
            scanned[platform] = sources
            if platform == "rss":
                for url in sources:
                    posts.extend(self.rss_adapter.fetch(url, window_start, per_source_limit))
                continue
            adapter = self.adapters.get(platform)
            if not adapter:
                continue
            for ch in sources:
                try:
                    posts.extend(adapter.fetch(ch, window_start, per_source_limit))
                except Exception:
                    continue
        # De-dupe by (platform, id) or content hash
        seen: set = set()
        uniq: List[SocialPost] = []
        for p in sorted(posts, key=lambda x: x.created_at, reverse=True):
            key = (p.platform, p.id)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq, scanned


# -------------- Public Tool --------------
@function_tool
class WaqarZakaTradingAnalysis(BaseModel):
    """Use this to pull Waqar Zaka's trading analysis from his official channels across social platforms."""
    input: WaqarZakaAnalysisInput

    class Config:
        title = "waqar_zaka_trading_analysis"

    def run(self) -> WaqarZakaAnalysisOutput:
        cfg = dict(OFFICIAL_CHANNELS)
        if self.input.channels_override:
            # Only override specified keys, keep others
            for k, v in self.input.channels_override.items():
                cfg[k] = v
        platforms = self.input.platforms or list(cfg.keys())

        window_start = _now_utc() - timedelta(hours=self.input.lookback_hours)
        collector = Collector(cfg)
        posts, scanned = collector.run(window_start, self.input.max_posts_per_source, platforms)

        sentiment = _sentiment_scores(p.text for p in posts)
        mentions = _aggregate_mentions(posts)

        return WaqarZakaAnalysisOutput(
            fetched_at=_now_utc(),
            window_start=window_start,
            platforms_scanned=platforms,
            channels_scanned=scanned,
            total_posts=len(posts),
            posts=posts,
            sentiment=sentiment,
            top_mentions=mentions[:20],
        )


# -------------- CLI (optional) ----------
if __name__ == "__main__":
    # Lightweight smoke test without hitting APIs
    sample_input = WaqarZakaAnalysisInput(
        lookback_hours=24,
        max_posts_per_source=10,
        platforms=[p for p in OFFICIAL_CHANNELS.keys()],
    )
    tool = WaqarZakaTradingAnalysis(input=sample_input)
    out = tool.run()
    print("Fetched:", out.total_posts, "posts from", out.platforms_scanned)
    print("Top mentions:", [(m.symbol, m.count) for m in out.top_mentions])
    print("Sentiment:", out.sentiment)
