"""Fetch -> filter -> cluster into events -> pre-score -> Gemini scoring -> docs/events.json"""
import os, re, json, time, calendar, datetime as dt
import feedparser, requests, numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")  # check current free-tier models
WINDOW_H, TOP_N, BATCH, SIM = 48, 60, 15, 0.78
TIER_W = {"primary": 3, "wire": 2, "general": 1}
JUNK = re.compile(r"horoscope|celebrity|box office|bollywood|transfer rumou?r|football|cricket score|viral video|best deals", re.I)
SIGNAL = re.compile(r"\b(law|bill|ordinance|treaty|sanction|ban|court|verdict|ruling|outbreak|epidemic|earthquake|flood|cyclone|ceasefire|war|attack|tariff|export|regulation|notification|agreement|summit|election|budget|shortage|dam|border)\b", re.I)
CATS = ["Geopolitics", "Economy", "Health", "Environment", "Security", "Tech & Science", "Law & Policy", "Local Civic", "Other"]
SCALES = ["Local", "State", "National", "Regional", "Global"]

def fetch():
    cutoff = time.time() - WINDOW_H * 3600
    out, seen = [], set()
    for f in json.load(open(os.path.join(HERE, "feeds.json"))):
        try:
            d = feedparser.parse(f["url"])
        except Exception as e:
            print("feed failed", f["name"], e); continue
        for e in d.entries:
            t = e.get("published_parsed") or e.get("updated_parsed")
            ts = calendar.timegm(t) if t else time.time()
            title = (e.get("title") or "").strip()
            link = e.get("link")
            if ts < cutoff or not title or not link or link in seen or JUNK.search(title):
                continue
            seen.add(link)
            snip = re.sub(r"<[^>]+>", " ", e.get("summary", ""))[:300].strip()
            out.append(dict(title=title, link=link, snip=snip, source=f["name"], tier=f["tier"], ts=ts))
    print(len(out), "articles")
    return out

def cluster(arts):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = model.encode([a["title"] + ". " + a["snip"][:150] for a in arts], normalize_embeddings=True)
    cents, groups = [], []
    for i, v in enumerate(vecs):
        if cents:
            sims = np.array(cents) @ v
            j = int(sims.argmax())
            if sims[j] >= SIM:
                groups[j].append(i)
                c = vecs[groups[j]].mean(axis=0); cents[j] = c / np.linalg.norm(c)
                continue
        cents.append(v); groups.append([i])
    events = []
    for g in groups:
        items = sorted((arts[i] for i in g), key=lambda a: -TIER_W[a["tier"]])
        srcs = sorted({a["source"] for a in items})
        tier = max(items, key=lambda a: TIER_W[a["tier"]])["tier"]
        pre = TIER_W[tier] * 2 + min(len(srcs), 5) + 2 * bool(SIGNAL.search(" ".join(a["title"] for a in items)))
        events.append(dict(pre=pre, tier=tier, sources=srcs, n_sources=len(srcs), ts=max(a["ts"] for a in items),
                           articles=[dict(title=a["title"], link=a["link"], source=a["source"]) for a in items[:5]],
                           snip=items[0]["snip"]))
    return sorted(events, key=lambda e: -e["pre"])

PROMPT = """You are a news editor who ranks events by real-world IMPACT, not popularity or drama.
For each event return JSON: {"id":int,"skip":bool,"category":one of %s,"scale":one of %s,
"scores":{"scale":0-10 people/money affected,"severity":0-10 safety/rights/livelihood,"irreversibility":0-10,
"novelty":0-10 genuinely new vs repeat,"cascade":0-10 second-order effects},
"summary":"max 2 factual sentences using ONLY the given text","why":"one sentence on why it matters"}.
skip=true for gossip, sports, opinion, promotions. Quiet primary documents (laws, rulings, regulations, notices) can be highly important.
Return ONLY a JSON array.
EVENTS: %s"""

def gemini(batch):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    body = {"contents": [{"parts": [{"text": PROMPT % (CATS, SCALES, json.dumps(batch, ensure_ascii=False))}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    for attempt in range(4):
        r = requests.post(url, headers={"x-goog-api-key": KEY}, json=body, timeout=120)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1)); continue
        r.raise_for_status()
        return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    return []

def score(events):
    top = events[:TOP_N]
    for s in range(0, len(top), BATCH):
        chunk = top[s:s + BATCH]
        batch = [dict(id=i, tier=e["tier"], n_sources=e["n_sources"], headlines=[a["title"] for a in e["articles"][:3]], text=e["snip"])
                 for i, e in enumerate(chunk)]
        try:
            res = gemini(batch)
        except Exception as ex:
            print("gemini batch failed:", ex); continue
        for r in res:
            try:
                e = chunk[r["id"]]
                sc = r["scores"]
                e.update(skip=bool(r.get("skip")), category=r.get("category", "Other"), scale=r.get("scale", "National"),
                         summary=r.get("summary", ""), why=r.get("why", ""),
                         impact=round(sum(sc.values()) / len(sc), 1))
            except Exception:
                continue
        time.sleep(6)  # stay under free-tier rate limits
    out = []
    for e in top:
        if "impact" not in e or e["skip"] or e["impact"] < 4:
            continue
        e["under_radar"] = e["impact"] >= 6 and e["n_sources"] <= 2
        e["gap"] = round(e["impact"] / 10 - min(e["n_sources"], 10) / 10, 2)
        out.append(e)
    return sorted(out, key=lambda e: -e["impact"])

if __name__ == "__main__":
    if not KEY:
        raise SystemExit("Set GEMINI_API_KEY")
    arts = fetch()
    evs = score(cluster(arts)) if arts else []
    for e in evs:
        e.pop("snip", None)
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "events": evs},
              open(os.path.join(HERE, "docs", "events.json"), "w"), ensure_ascii=False, indent=1)
    print(len(evs), "events written")
