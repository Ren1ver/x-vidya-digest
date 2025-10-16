import os
import requests
import datetime as dt
from pathlib import Path
import yaml
from dotenv import load_dotenv

# === Load environment variables from .env ===
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
BEARER = os.getenv("X_BEARER_TOKEN")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")

if not BEARER or not WEBHOOK:
    raise SystemExit("❌ Missing X_BEARER_TOKEN or DISCORD_WEBHOOK_URL in .env")

# === Load config from YAML ===
cfg_path = Path(__file__).with_name("query_config.yml")
cfg = yaml.safe_load(cfg_path.read_text())

LANG = cfg.get("lang", "en")
MAX_RESULTS = min(cfg.get("max_results", 50), 100)
KEYWORDS = cfg.get("keywords", [])
VERBS = cfg.get("verbs", [])
ACCOUNTS = cfg.get("accounts", [])
BLOCK_TERMS = cfg.get("block_terms", [])

# === Build one compact Free-tier-friendly query ===
def build_query():
    parts = []

    if KEYWORDS or VERBS:
        all_terms = KEYWORDS + VERBS
        parts.append("(" + " OR ".join([f'"{t}"' if " " in t else t for t in all_terms]) + ")")

    if ACCOUNTS:
        parts.append("(" + " OR ".join([f"from:{acc}" for acc in ACCOUNTS]) + ")")

    parts.append("-is:retweet")
    parts.append(f"lang:{LANG}")

    return " ".join(parts)

# === Call X API v2 ===
def fetch_tweets(query: str):
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": MAX_RESULTS,
        "tweet.fields": "created_at,public_metrics",
        "expansions": "author_id",
        "user.fields": "username"
    }
    headers = {
        "Authorization": f"Bearer {BEARER}"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# === Scoring / summarizing ===
def score(tweet, user):
    metrics = tweet.get("public_metrics", {})
    text = tweet.get("text", "").lower()
    kw_hits = sum(k.lower() in text for k in KEYWORDS + VERBS)
    block_hits = sum(b.lower() in text for b in BLOCK_TERMS)
    bonus = 3 if user.get("username", "").lower() in [a.lower() for a in ACCOUNTS] else 0
    score = (
        kw_hits * 5
        - block_hits * 5
        + metrics.get("like_count", 0) * 0.5
        + metrics.get("retweet_count", 0) * 0.7
        + metrics.get("reply_count", 0) * 0.3
        + bonus
    )
    return score

def summarize(data, top_n=8):
    tweets = data.get("data", [])
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    summaries = []

    for t in tweets:
        user = users.get(t["author_id"], {})
        t["_user"] = user
        t["_score"] = score(t, user)

    top = sorted(tweets, key=lambda x: x["_score"], reverse=True)[:top_n]

    for t in top:
        u = t["_user"]
        txt = t["text"].strip().replace("\n", " ")
        txt = txt if len(txt) < 220 else txt[:217] + "…"
        url = f"https://x.com/{u.get('username','')}/status/{t['id']}"
        summaries.append(f'• **@{u.get("username","")}** — {txt}\n<{url}>')

    return "\n".join(summaries)

# === Send to Discord ===
def post_to_discord(summary_text, title="🎮 Daily Gaming Digest"):
    embed = {
        "title": title,
        "description": summary_text,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "footer": {"text": "Powered by X + Python"}
    }
    resp = requests.post(WEBHOOK, json={"embeds": [embed]}, timeout=30)
    resp.raise_for_status()

# === Main ===
def main():
    query = build_query()
    data = fetch_tweets(query)
    if not data.get("data"):
        print("No tweets found.")
        post_to_discord("🛑 No relevant tweets found today.")
        return

    summary = summarize(data)
    today = dt.datetime.now().strftime("%b %d, %Y")
    post_to_discord(summary, title=f"🎮 Gaming Digest — {today}")
    print("✅ Posted to Discord.")

if __name__ == "__main__":
    main()
