"""Tests for the predicted-lineup archiver (`research/data/predicted_lineups.py`).

The parser is the load-bearing part: a snapshot is unrecoverable, so a silent parse
regression would archive wrong data forever. These pin the real markup shape (two
tables per club — the XI and the alternatives — each with a per-player `Start %`)
against a fixture, so no network is touched.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from research.data import predicted_lineups as pl

# Trimmed from the real page, keeping its actual structure: an <h2> club heading,
# then the predicted-XI table, then the "Potential Starters" table.
FIXTURE = """
<style>.x{width:100%}</style>
<h2>Arsenal Predicted Lineup</h2>
<figure class="wp-block-table"><table class="has-fixed-layout">
<thead><tr><th>Player</th><th>Pos</th><th>Start %</th></tr></thead>
<tbody>
<tr><td>David Raya</td><td class="c">GK</td><td class="c">90%</td></tr>
<tr><td>Cristhian Mosquera</td><td class="c">RB</td><td class="c">60%</td></tr>
</tbody></table></figure>
<figure class="wp-block-table"><table class="has-fixed-layout">
<thead><tr><th>Potential Starters</th><th>Pos</th><th>Start %</th></tr></thead>
<tbody>
<tr><td>Riccardo Calafiori</td><td class="c">LB</td><td class="c">60%</td></tr>
</tbody></table></figure>
<h2>Nottingham Forest Predicted Lineup</h2>
<figure class="wp-block-table"><table class="has-fixed-layout">
<thead><tr><th>Player</th><th>Pos</th><th>Start %</th></tr></thead>
<tbody>
<tr><td>Matz Sels</td><td class="c">GK</td><td class="c">85%</td></tr>
</tbody></table></figure>
<h2>Categories</h2>
<table><tr><td>not a lineup</td></tr></table>
"""


class TestParse(unittest.TestCase):
    def setUp(self):
        self.teams = pl.parse(FIXTURE)
        self.by_team = {t["team"]: t for t in self.teams}

    def test_finds_each_club_once(self):
        self.assertEqual([t["team"] for t in self.teams],
                         ["Arsenal", "Nottingham Forest"])

    def test_ignores_tables_without_a_start_percentage(self):
        # The trailing "Categories" table must not become a club.
        self.assertNotIn("Categories", self.by_team)

    def test_reads_name_position_and_percentage(self):
        raya = self.by_team["Arsenal"]["players"][0]
        self.assertEqual(raya["name"], "David Raya")
        self.assertEqual(raya["position"], "GK")
        self.assertEqual(raya["start_pct"], 90)
        self.assertTrue(raya["predicted_xi"])

    def test_keeps_alternatives_and_flags_them_as_non_xi(self):
        players = self.by_team["Arsenal"]["players"]
        alt = [p for p in players if not p["predicted_xi"]]
        self.assertEqual(len(alt), 1)
        self.assertEqual(alt[0]["name"], "Riccardo Calafiori")
        self.assertEqual(alt[0]["start_pct"], 60)   # the rotation signal we want

    def test_maps_club_name_towards_the_engines_canonical(self):
        self.assertEqual(self.by_team["Nottingham Forest"]["canonical_team"],
                         "Nott'm Forest")
        self.assertEqual(self.by_team["Arsenal"]["canonical_team"], "Arsenal")


class TestSnapshot(unittest.TestCase):
    def test_snapshot_records_provenance_and_counts(self):
        snap = pl.build_snapshot(FIXTURE, "2026-27", gameweek=1,
                                 deadline_time="2026-08-21T17:30:00Z")
        self.assertEqual(snap["season"], "2026-27")
        self.assertEqual(snap["gameweek"], 1)
        self.assertEqual(snap["team_count"], 2)
        self.assertEqual(snap["player_count"], 4)
        self.assertEqual(snap["source"], pl.SOURCE)
        self.assertEqual(snap["parser_version"], pl.PARSER_VERSION)
        self.assertIn("fetched_at", snap)
        json.dumps(snap)                       # must round-trip

    def test_save_writes_a_labelled_timestamped_file(self):
        snap = pl.build_snapshot(FIXTURE, "2026-27", gameweek=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = pl.save_snapshot(snap, archive_dir=Path(tmp))
            self.assertTrue(path.exists())
            self.assertIn("GW07", path.name)
            self.assertEqual(path.parent.name, "2026-27")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["team_count"], 2)

    def test_preseason_snapshot_is_still_archived(self):
        snap = pl.build_snapshot(FIXTURE, "2026-27", gameweek=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = pl.save_snapshot(snap, archive_dir=Path(tmp))
            self.assertIn("preseason", path.name)


class TestFailsClosedOnEmptyParse(unittest.TestCase):
    """A 200 that carries no lineups (bot challenge, consent wall) must never be
    archived. An empty snapshot is not data — it is a failed fetch wearing data's
    clothes, and the archive is unrecoverable, so silence is the worst outcome."""

    CHALLENGE_PAGE = "<html><head><title>Just a moment...</title></head><body></body></html>"

    def test_empty_parse_raises_and_archives_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            original = pl.fetch_html
            pl.fetch_html = lambda *a, **k: self.CHALLENGE_PAGE
            try:
                with self.assertRaises(pl.LineupFetchError):
                    pl.archive_now(season="2026-27", gameweek=1, archive_dir=archive)
            finally:
                pl.fetch_html = original
            self.assertEqual(list(archive.rglob("*.json")), [],
                             "an empty parse must leave the archive untouched")

    def test_diagnose_distinguishes_a_challenge_from_a_real_page(self):
        challenge = pl.diagnose(self.CHALLENGE_PAGE)
        self.assertIn("Just a moment", challenge)
        self.assertIn("has_start_pct=False", challenge)
        self.assertIn("has_start_pct=True", pl.diagnose(FIXTURE))


class TestFetchRetriesTransients(unittest.TestCase):
    """The site answers 200 with a challenge page intermittently (seen live: two CI
    runs three minutes apart returned 0 teams then 20). Retrying keeps that transient
    from punching an unfillable hole in the archive."""

    class _Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def _patch(self, pages):
        """Serve `pages` in order; record how many requests were made."""
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return self._Response(pages[min(len(calls) - 1, len(pages) - 1)])

        return fake_get, calls

    def setUp(self):
        self._real_get = pl.requests.get
        self._real_sleep = pl.time.sleep
        pl.time.sleep = lambda *_: None          # don't serve the crawl delay in tests
        pl._LAST_FETCH[0] = 0.0

    def tearDown(self):
        pl.requests.get = self._real_get
        pl.time.sleep = self._real_sleep

    def test_retries_until_the_real_page_arrives(self):
        fake_get, calls = self._patch(["<html><title>Just a moment...</title></html>",
                                       FIXTURE])
        pl.requests.get = fake_get
        page = pl.fetch_html(attempts=3)
        self.assertIn(pl.LINEUP_MARKER, page)
        self.assertEqual(len(calls), 2, "should have retried exactly once")

    def test_gives_up_after_attempts_and_returns_the_last_page(self):
        fake_get, calls = self._patch(["<html><title>Just a moment...</title></html>"])
        pl.requests.get = fake_get
        page = pl.fetch_html(attempts=3)
        self.assertNotIn(pl.LINEUP_MARKER, page)   # caller then raises LineupFetchError
        self.assertEqual(len(calls), 3)


class TestCurrentSeason(unittest.TestCase):
    def test_july_starts_the_new_season(self):
        self.assertEqual(
            pl.current_season(datetime(2026, 7, 17, tzinfo=timezone.utc)), "2026-27")

    def test_may_still_belongs_to_the_old_season(self):
        self.assertEqual(
            pl.current_season(datetime(2027, 5, 1, tzinfo=timezone.utc)), "2026-27")


if __name__ == "__main__":
    unittest.main()
