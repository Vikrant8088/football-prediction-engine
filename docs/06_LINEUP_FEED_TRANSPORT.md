# 06 — Predicted-lineup feed: the datacenter-IP block, and how to run it

## The problem, measured

The predicted-lineup archiver reads Fantasy Football Pundit (FFP), which publishes each
club's predicted XI with a per-player **Start %** days before the deadline. That signal
is the only pre-deadline route to the minutes headroom Phase 6f measured (+4.59/GW
ceiling), so it is worth keeping alive.

FFP **blocks GitHub Actions' datacenter IPs**. Confirmed directly on 2026-07-30: from a
residential IP the page is ~896 KB / 20 teams; from CI every attempt returns a 204-byte
stub (`has_start_pct=False`), across all three retries. It is not intermittent and it is
not our parser — the request never reaches the real page.

We checked the alternatives before settling:

| option | verdict |
|---|---|
| Keyless proxies (allorigins, corsproxy, codetabs, thingproxy) | fail (520 / 403 / DNS) — FFP blocks them too |
| FotMob JSON API | now requires signed tokens — dead |
| fpledits / thefantasytool | no per-player probability exposed; JS-rendered |
| Confirmed-lineup APIs (API-Football, Sportmonks) | publish ~1 h before kickoff — **too late** for the FPL deadline |
| `r.jina.ai` (Jina Reader), **keyless** | works from a residential IP (HTML the parser reads at 20 teams / 403 players) but **403s from GitHub's datacenter IP** — free tiers throttle datacenter traffic |
| **`r.jina.ai` with a FREE key** | **works from CI** — the free Jina key (no card) lifts the datacenter throttle |
| Paid scraping proxy (ScraperAPI, ZenRows) | works too; the alternative if you prefer it |

Days-ahead predicted lineups *with probabilities* only come from scraped fantasy sites,
and those block datacenter IPs. So does every free *keyless* reader — datacenter
throttling is universal, which is the honest catch: **there is no zero-signup cloud
fix.** But one *free* key (Jina, ~2 min, no credit card) is enough.

## How it runs

`fetch_html` tries transports in order and returns the first that actually carries
lineups:

1. **direct** — all a residential/local run needs; the fallbacks are never reached.
2. **the Jina reader** (`r.jina.ai`, `X-Return-Format: html`) — the cloud fallback when
   direct is blocked. Keyless it 403s from CI; with `JINA_API_KEY` it rides a Bearer
   header and serves CI.
3. **a paid proxy** — used *instead of* Jina when `LINEUP_FETCH_PROXY` is set.

### To turn the CI archiver on (pick one, both free)

**Option 1 — free Jina key (simplest):**

1. Get a free key at <https://jina.ai/reader> (no credit card).
2. GitHub repo → **Settings → Secrets and variables → Actions → New secret**, name
   **`JINA_API_KEY`**, value the key.

**Option 2 — scraping-proxy key:** free tier from ScraperAPI / ZenRows (~1 000 req/mo;
this needs ~30). Add secret **`LINEUP_FETCH_PROXY`** = the template with a `{url}`
placeholder, e.g. `https://api.scraperapi.com/?api_key=YOUR_KEY&url={url}`.

Either key lives only in the secret — never in code or git, exactly like the
API-Football rule. With neither set, the CI archiver fails (correctly — the feed is
genuinely unreachable from a datacenter without one); local runs are unaffected.

### Local, from a residential IP (also zero cost)

The archiver already works from any residential machine. Run it on a schedule:

```
python -m research.data.predicted_lineups
git add research/data/lineups/ && git commit -m "chore(lineups): snapshot" && git push
```

On Windows, drive it with Task Scheduler daily; the snapshot the pipeline actually uses
is the latest one before each deadline, so a run that misses a day is not fatal. The cost
is that the machine must be on when it fires — the dependency CI was meant to remove.

## Why this is safe to leave unconfigured

The predicted-lineup feed drives a **declared secondary variant**, not the pre-registered
primary. With no snapshot, `lineup_start_pct` returns `{}`, no lineup variant is locked,
and the primary squad is unaffected. So the season is never blocked on this — configuring
the proxy only switches the secondary A/B back on.
