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
| **`r.jina.ai` (Jina Reader), free + keyless** | **works** — returns HTML the existing parser reads at 20 teams / 403 players, identical to a residential fetch |
| Paid scraping proxy (ScraperAPI, ZenRows) | works too; the reliable upgrade if the free reader is rate-limited |

Days-ahead predicted lineups *with probabilities* only come from scraped fantasy sites,
and those block datacenter IPs. So the fix is transport, not source — and a free reader
that fetches server-side from its own IPs is enough.

## How it runs

`fetch_html` tries transports in order and returns the first that actually carries
lineups:

1. **direct** — all a residential/local run needs; the fallbacks are never reached.
2. **the free Jina reader** (`r.jina.ai`, `X-Return-Format: html`) — the automatic cloud
   fallback when direct is blocked. No key, no signup, no config. **This is why the
   daily CI archiver works out of the box.**
3. **a paid proxy** — used *instead of* Jina when `LINEUP_FETCH_PROXY` is set.

### Optional: a paid proxy for extra reliability

Jina is a free public service, so it can rate-limit or wobble. If the archiver starts
failing on it, set a paid scraping-proxy key as the reliable upgrade:

1. Free tier from ScraperAPI / ZenRows / ScrapingBee (~1 000 req/mo; this needs ~30).
2. GitHub repo → **Settings → Secrets and variables → Actions → New secret**, name
   **`LINEUP_FETCH_PROXY`**, value the provider's template with a `{url}` placeholder:

   ```
   https://api.scraperapi.com/?api_key=YOUR_KEY&url={url}
   ```

The key lives only in the secret — never in code or git, exactly like the API-Football
rule. Unset, Jina is used.

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
