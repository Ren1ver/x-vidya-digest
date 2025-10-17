import os
import re
import time
import requests
import feedparser  # not used now, but harmless; remove if you want
from datetime import datetime, timedelta
from dotenv import load_dotenv
from readability import Document
from bs4 import BeautifulSoup
from html import unescape
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, urljoin

# -------- ENV --------
load_dotenv()
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
if not DISCORD_WEBHOOK:
    raise SystemExit("❌ DISCORD_WEBHOOK_URL missing in .env")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
OLLAMA_URL = (os.getenv("OLLAMA_URL", "http://localhost:11434") or "").rstrip("/")

# -------- CONFIG --------
SOURCES = {
    "News in Vidya": [
        "https://www.belloflostsouls.net/",
    ],
    "The Skinny": [
        "https://www.dexerto.com/",
    ],
}

# Per-site scraping limits & posting behavior
MAX_ARTICLES_PER_SITE = 3          # try to find up to N real articles per site
MAX_PER_CATEGORY_POST = 9          # hard cap per category (if you want)
HOURS_LOOKBACK = 24                # skip obviously old links if we can detect date
UA = {"User-Agent": "Mozilla/5.0 (DiscordDigestBot/2.0)"}
MAX_DISCORD_LEN = 1900

# -------- Utility & cleaning --------
MD_ESCAPE_RE = re.compile(r'([*_`~])')
GENERIC_SNIPPET_RE = re.compile(
    r"(read more|click here|watch the (?:trailer|video)|subscribe|sign up|follow us|via [A-Za-z]+)",
    re.IGNORECASE,
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")

def md_escape(s: str) -> str:
    return MD_ESCAPE_RE.sub(r'\\\1', (s or "").strip())

def normalize_ws(s: str) -> str:
    s = re.sub(r'\r\n?', '\n', s or "")
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

def strip_html(text: str) -> str:
    soup = BeautifulSoup(text or "", "lxml")
    plain = soup.get_text(" ").strip()
    plain = unescape(plain)
    return re.sub(r"\s+", " ", plain)

def format_bold_first_word_italic(raw: str) -> str:
    txt = normalize_ws(raw or "")
    m = re.match(r"^(\S+)(.*)$", txt, flags=re.DOTALL)
    if not m:
        return f"*{md_escape(txt)}*"
    first, rest = m.group(1), m.group(2).lstrip()
    first_esc = md_escape(first)
    rest_esc  = md_escape(rest)
    return f"***{first_esc}** {rest_esc}*" if rest_esc else f"***{first_esc}***"

def split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(t) if p.strip()]

def clean_snippet_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s*[\[\(](?:photo|image|video|gallery|credit|via)[^\]\)]*[\]\)]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    t = GENERIC_SNIPPET_RE.sub("", t)
    return t.strip()

def best_sentences(text: str, max_sentences: int = 3, min_words: int = 6, max_chars: int = 480) -> str:
    sents = [s for s in split_sentences(text) if len(s.split()) >= min_words]
    out, total = [], 0
    for s in sents:
        if total + len(s) + (1 if out else 0) > max_chars:
            break
        out.append(s)
        total += len(s) + (1 if out else 0)
        if len(out) >= max_sentences:
            break
    if not out and sents:
        s0 = sents[0][:max_chars]
        out = [s0 + ("…" if len(sents[0]) > max_chars else "")]
    return " ".join(out).strip()

# -------- URL normalization & heuristics --------
def normalize_url(u: str) -> str:
    try:
        p = urlparse((u or "").strip())
        scheme = (p.scheme or "https").lower()
        netloc = (p.netloc or "").lower().lstrip("www.").replace("amp.", "")
        path = re.sub(r"/amp/?$", "/", p.path or "", flags=re.IGNORECASE)
        path = re.sub(r"//+", "/", path).rstrip("/")
        qs_pairs = parse_qsl(p.query or "", keep_blank_values=True)
        drop_keys = {
            "fbclid","gclid","yclid","mc_cid","mc_eid","ref","ref_src","ref_url",
            "spm","trk","igshid","mkt_tok","_hsenc","_hsmi","feature","sharetype",
            "s","source","clid","cmpid","ncid"
        }
        drop_prefixes = ("utm_", "utm-", "icid")
        kept = []
        for k, v in qs_pairs:
            kl = k.lower()
            if kl in drop_keys or any(kl.startswith(pref) for pref in drop_prefixes):
                continue
            kept.append((k, v))
        query = urlencode(sorted(kept))
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return (u or "").strip()

def same_domain(u: str, base: str) -> bool:
    try:
        a, b = urlparse(u), urlparse(base)
        return a.netloc.split(":")[0].lower().lstrip("www.") == b.netloc.split(":")[0].lower().lstrip("www.")
    except Exception:
        return False

def looks_like_article_url(u: str) -> bool:
    """Heuristics to prefer article pages over hubs."""
    path = urlparse(u).path.lower()
    # prefer dated or /news/ paths and avoid tag/category/home/etc.
    if re.search(r"/20\d{2}/\d{2}/\d{2}/", path):  # /YYYY/MM/DD/
        return True
    if "/news/" in path or "/story/" in path or "/article" in path:
        return True
    if any(seg in path for seg in ["/tag/", "/category/", "/page/", "/author/", "/about", "/contact", "/privacy"]):
        return False
    return path.count("/") >= 2  # has some depth

# -------- Fetching --------
def fetch_html(url: str, timeout=15) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            return r.text
    except Exception:
        pass
    return ""

def extract_opengraph_article(html: str) -> bool:
    soup = BeautifulSoup(html or "", "lxml")
    og = soup.find("meta", {"property": "og:type"})
    if og and str(og.get("content","")).lower().strip() == "article":
        return True
    # schema.org Article
    if soup.find(attrs={"itemtype": re.compile("schema.org/(News)?Article", re.I)}):
        return True
    return False

def discover_article_links(section_url: str, max_links=5) -> list[str]:
    """Grab candidate article URLs from a site/section page."""
    html = fetch_html(section_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    hrefs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        absu = urljoin(section_url, href)
        if not same_domain(absu, section_url):
            continue
        absu = normalize_url(absu)
        if looks_like_article_url(absu):
            hrefs.add(absu)

    # Score candidates by presence of og:article and path length
    scored = []
    for u in hrefs:
        h = fetch_html(u)
        if not h:
            continue
        is_article = extract_opengraph_article(h)
        score = (2 if is_article else 0) + min(3, len(urlparse(u).path.split("/")))
        scored.append((score, u))
        if len(scored) >= max_links * 3:
            # fetched enough to pick best
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    uniq = []
    seen = set()
    for _, u in scored:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
        if len(uniq) >= max_links:
            break
    return uniq

def fetch_article_text_and_title(url: str, hard_cap=4000) -> tuple[str, str]:
    """Return (title, text) extracted via Readability with minimal cleanup."""
    html = fetch_html(url)
    if not html:
        return "", ""
    try:
        doc = Document(html)
        title = strip_html(doc.short_title() or "")
        content = BeautifulSoup(doc.summary(), "lxml")
        for sel in ["script","style","noscript","form","iframe"]:
            for t in content.select(sel):
                t.decompose()
        noisy = ["share","social","subscribe","newsletter","footer","header","byline","related","tags","cookie"]
        for cls in noisy:
            for t in content.find_all(attrs={"class": lambda c: c and cls in c.lower()}):
                t.decompose()
        text = content.get_text("\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return title.strip(), text.strip()[:hard_cap]
    except Exception:
        return "", ""

# -------- Ollama (generative blurbs) --------
def _ollama_chat(messages, temperature=0.45, ctx=8192, timeout=60):
    data = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.08, "num_ctx": ctx},
    }
    url = f"{OLLAMA_URL}/api/chat"
    r = requests.post(url, json=data, timeout=timeout)
    if r.status_code == 404:
        raise FileNotFoundError("/api/chat not available")
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()

def _ollama_generate(prompt, temperature=0.45, ctx=8192, timeout=60):
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.08, "num_ctx": ctx},
    }
    url = f"{OLLAMA_URL}/api/generate"
    r = requests.post(url, json=data, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def generate_blurb_with_ollama(title: str, url: str, article_text: str, min_words=70, max_words=130) -> str:
    """
    Ask the model to write a vivid but factual 2–3 sentence blurb
    using ONLY the provided article text.
    """
    context_trim = article_text[:3000]
    sys_prompt = (
        "You are a gaming news copy editor. Using ONLY the article text I give you, "
        f"write a vivid, neutral 2–3 sentence blurb ({min_words}-{max_words} words) that captures the key facts "
        "and context (who/what/when/where/why/impact). Do NOT invent facts or speculate. "
        "No quotes, no emojis, no links, no bullet points. Output plain text only."
    )
    user_prompt = f"TITLE: {title}\nURL: {url}\nARTICLE TEXT:\n{context_trim}\n\nReturn only the blurb text."

    try:
        out = _ollama_chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.45,
            ctx=4096,
        )
    except FileNotFoundError:
        out = _ollama_generate(sys_prompt + "\n\n" + user_prompt, temperature=0.45, ctx=4096)
    except requests.RequestException:
        out = ""

    return normalize_ws(out)

CLICKBAIT_RE = re.compile(
    r"\b(here'?s|this|these|things|you won'?t believe|shocking|surprising|amazing|must[- ]see|what we know|everything you need to know|explained)\b",
    re.IGNORECASE,
)

def tidy_headline_from_title(title: str, max_words: int = 10) -> str:
    """Deterministic, no-clickbait headline from the page title."""
    t = (title or "").strip()
    # remove leading site prefix like 'Kotaku - ' or 'Kotaku: '
    t = re.sub(r"^\s*[\w.&'’\- ]+\s*[-:]\s*", "", t)
    # strip quotes and collapse spaces
    t = t.strip(' "\'“”‘’')
    t = re.sub(r"\s+", " ", t)
    # remove clickbait-y fillers
    t = CLICKBAIT_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    # cap words and drop trailing punctuation
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    t = re.sub(r"[.?!,:;–—-]+$", "", t)
    # title case lightly (keep ALLCAPS/acronyms as-is)
    parts = []
    for w in t.split():
        if w.isupper():
            parts.append(w)
        else:
            parts.append(w.capitalize())
    out = " ".join(parts).strip()
    # safety: if we trimmed to nothing, use first few words from original
    if not out:
        out = " ".join((title or "Update").split()[:max_words])
    return out

def generate_headline_with_ollama(title: str, article_text: str, min_words=6, max_words=10) -> str:
    """
    Ask Ollama for a concise, factual mini-headline using ONLY article text.
    Returns plain text (no markdown). Falls back to '' on error.
    """
    if not article_text:
        return ""
    ctx = article_text[:1800]
    sys_prompt = (
        "You are a precise gaming news editor. From ONLY the provided article text, "
        f"write a clear, factual mini-headline ({min_words}-{max_words} words). "
        "No clickbait, no hype, no site names, no emojis, no quotes, no trailing punctuation. "
        "Focus on the main subject + action. Output plain text only."
    )
    user_prompt = f"TITLE: {title}\nARTICLE TEXT:\n{ctx}\n\nReturn only the mini-headline."

    try:
        out = _ollama_chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.25,  # tight for factual tone
            ctx=2048,
        )
    except FileNotFoundError:
        try:
            out = _ollama_generate(sys_prompt + "\n\n" + user_prompt, temperature=0.25, ctx=2048)
        except requests.RequestException:
            return ""
    except requests.RequestException:
        return ""

    # sanitize and validate length
    h = normalize_ws(out)
    h = re.sub(r"[“”\"'‘’]+", "", h)           # remove quotes if any
    h = re.sub(r"[.?!,:;–—-]+$", "", h).strip() # drop trailing punctuation
    wc = len(h.split())
    if wc < min_words or wc > max_words:
        return ""
    return h


# -------- Summarizer (no header; per-article blurbs) --------
def summarize_source_with_ollama(category_name: str, site_urls: list[str]) -> list[str]:
    """
    For each site URL, discover article links, pull full text, produce:
      **Mini Headline**
      *Blurb…* [link](url)
    Returns a list of formatted items for the category.
    """
    lines = []
    seen_urls = set()
    seen_blurbs = set()

    for site in site_urls:
        candidates = discover_article_links(site, max_links=MAX_ARTICLES_PER_SITE)
        for art_url in candidates:
            if len(lines) >= MAX_PER_CATEGORY_POST:
                break
            norm_u = normalize_url(art_url)
            if norm_u in seen_urls:
                continue
            seen_urls.add(norm_u)

            title, fulltext = fetch_article_text_and_title(art_url, hard_cap=4000)
            if not fulltext:
                html = fetch_html(art_url)
                fulltext = strip_html(html)[:2000] if html else ""

            # --- Generate mini headline (model -> fallback) ---
            mini_head = generate_headline_with_ollama(title or "", fulltext)
            if not mini_head:
                mini_head = tidy_headline_from_title(title or "", max_words=10)

            # --- Generate blurb (model -> extractive fallback) ---
            blurb_raw = ""
            if fulltext:
                blurb_raw = generate_blurb_with_ollama(title or "", norm_u, fulltext, min_words=70, max_words=130)
            if not blurb_raw:
                cleaned = clean_snippet_text(fulltext or title or "")
                blurb_raw = best_sentences(cleaned, max_sentences=3, min_words=6, max_chars=480)

            # dedupe by blurb text to avoid near-duplicates across sites
            key = re.sub(r"[^\w\s]", " ", (blurb_raw or "").lower())
            key = re.sub(r"\s+", " ", key).strip()
            if key in seen_blurbs or not blurb_raw:
                continue
            seen_blurbs.add(key)

            # --- Final formatting ---
            mini_head_fmt = f"**{md_escape(mini_head)}**"
            blurb_fmt = format_bold_first_word_italic(blurb_raw)
            item = f"{mini_head_fmt}\n{blurb_fmt} [link]({norm_u})"
            lines.append(item)

        if len(lines) >= MAX_PER_CATEGORY_POST:
            break

    return lines


# -------- Discord --------
def send_discord_message(content: str):
    payload = {"content": content, "flags": 4}  # 4 = SUPPRESS_EMBEDS (no previews)
    r = requests.post(DISCORD_WEBHOOK, json=payload)
    if r.status_code not in (200, 204):
        raise SystemExit(f"Discord post failed: {r.status_code} {r.text}")
    time.sleep(0.9)

def post_to_discord(digest: dict):
    """
    Sends each category as one or more quoted messages.
    The category header appears ONLY on the first chunk for that category.
    """
    for section, lines in digest.items():
        header = f"## __**{section}**__"
        if not lines:
            send_discord_message(f"{header}\n>>> (No recent items found.)")
            continue

        # Build chunks that respect Discord's length limit
        chunks = []
        current = ""
        for line in lines:  # 'lines' is already a list of formatted blurbs
            addition = ("" if current == "" else "\n\n") + line
            # use the correct prefix length depending on whether this is the first chunk
            prefix_len = len(f"{header}\n>>> ") if len(chunks) == 0 else len(">>> ")
            if prefix_len + len(current) + len(addition) > MAX_DISCORD_LEN:
                if current:  # flush
                    chunks.append(current)
                    current = line
                else:
                    # single line is too long; send it alone
                    chunks.append(line)
                    current = ""
            else:
                current += addition
        if current.strip():
            chunks.append(current.strip())

        # Send: header on the first message only
        for i, chunk in enumerate(chunks):
            if i == 0:
                send_discord_message(f"{header}\n>>> {chunk}")
            else:
                send_discord_message(f">>> {chunk}")


# -------- Main --------
if __name__ == "__main__":
    digest = {}
    for category, site_urls in SOURCES.items():
        digest[category] = summarize_source_with_ollama(category, site_urls)
    post_to_discord(digest)
