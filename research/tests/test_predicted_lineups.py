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


class TestInSeasonHeadings(unittest.TestCase):
    """The archiver has only ever seen PRESEASON pages, where the heading is exactly
    "Arsenal Predicted Lineup". Once fixtures exist the site is likely to append the
    opponent or gameweek, or vary the spelling. The old anchored regex matched none of
    those and would have punched a silent, unrecoverable hole in the archive on the
    first real gameweek. These pin the tolerant behaviour."""

    def _one_team(self, heading):
        page = ("<h2>%s</h2>"
                "<table><thead><tr><th>Player</th><th>Pos</th><th>Start %%</th></tr>"
                "</thead><tbody>"
                "<tr><td>David Raya</td><td>GK</td><td>90%%</td></tr>"
                "</tbody></table>" % heading)
        teams = pl.parse(page)
        return teams[0]["team"] if teams else None

    def test_opponent_suffix_still_resolves_the_team(self):
        self.assertEqual(self._one_team("Arsenal Predicted Lineup vs Chelsea"), "Arsenal")

    def test_gameweek_suffix_still_resolves_the_team(self):
        self.assertEqual(self._one_team("Arsenal Predicted Lineup (GW20)"), "Arsenal")

    def test_spelling_variants_resolve(self):
        self.assertEqual(self._one_team("Arsenal Predicted Line-up"), "Arsenal")
        self.assertEqual(self._one_team("Arsenal Predicted XI"), "Arsenal")

    def test_multiword_team_names_are_captured_whole(self):
        self.assertEqual(self._one_team("Aston Villa Predicted Lineup vs Spurs"),
                         "Aston Villa")

    def test_the_section_header_is_not_mistaken_for_a_team(self):
        # "Predicted Starting Lineups" has a word between Predicted and Line, so it must
        # not be captured — otherwise the page title becomes a phantom club.
        self.assertIsNone(
            self._one_team("Premier League Team News - Predicted Starting Lineups"))


class TestParseHealth(unittest.TestCase):
    """The all-or-nothing checks miss a PARTIAL parse: club headings resolve but a
    table-structure change halves the players per team. `parse_health` is that alarm."""

    def _team(self, name, n):
        return {"team": name, "players": [{"name": "P%d" % i} for i in range(n)]}

    def test_a_full_parse_is_not_degraded(self):
        health = pl.parse_health([self._team("A", 18), self._team("B", 20)])
        self.assertFalse(health["degraded"])
        self.assertEqual(health["thin_teams"], [])
        self.assertEqual(health["median_players_per_team"], 20)

    def test_a_median_below_the_xi_is_degraded(self):
        # Every team lost its alternatives AND part of its XI: structural drift.
        teams = [self._team("A", 5), self._team("B", 6), self._team("C", 4)]
        health = pl.parse_health(teams)
        self.assertTrue(health["degraded"])
        self.assertEqual(sorted(health["thin_teams"]), ["A", "B", "C"])

    def test_one_thin_team_among_full_ones_is_flagged_but_not_degraded(self):
        # A single light-news team is not a structural failure; it is recorded, not fatal.
        teams = [self._team("A", 18), self._team("B", 19), self._team("C", 6)]
        health = pl.parse_health(teams)
        self.assertFalse(health["degraded"], "the median team is still whole")
        self.assertEqual(health["thin_teams"], ["C"])


class TestPendingPredictions(unittest.TestCase):
    """Early in a season (and for weeks after the rollover) FFP lists the predicted XI
    with the Start % shown as 'TBD' before it firms the numbers. That is a VALID page,
    not a failed fetch — treating it as junk collapsed the whole page to zero teams and
    fail-closed the archiver for 8 days after the 2026/27 rollover."""

    TBD_PAGE = """
    <h2>Arsenal Predicted Lineup</h2>
    <table><thead><tr><th>Player</th><th>Pos</th><th>Start %</th></tr></thead>
    <tbody>
    <tr><td>David Raya</td><td>GK</td><td>TBD</td></tr>
    <tr><td>William Saliba</td><td>CB</td><td>TBD</td></tr>
    </tbody></table>
    """

    def test_a_tbd_percentage_keeps_the_player_as_pending(self):
        teams = pl.parse(self.TBD_PAGE)
        self.assertEqual(len(teams), 1, "the team must not vanish just because % is TBD")
        players = teams[0]["players"]
        self.assertEqual(len(players), 2)
        self.assertTrue(all(p["start_pct"] is None for p in players))
        self.assertEqual(players[0]["name"], "David Raya")

    def test_snapshot_flags_predictions_pending(self):
        snap = pl.build_snapshot(self.TBD_PAGE, "2026-27")
        self.assertEqual(snap["team_count"], 1)
        self.assertTrue(snap["predictions_pending"])
        self.assertEqual(snap["numeric_start_pcts"], 0)

    def test_a_numeric_page_is_not_pending(self):
        snap = pl.build_snapshot(FIXTURE, "2025-26")
        self.assertFalse(snap["predictions_pending"])
        self.assertGreater(snap["numeric_start_pcts"], 0)

    def test_archiver_saves_a_pending_page_instead_of_failing_closed(self):
        real = pl.fetch_html
        pl.fetch_html = lambda *a, **k: self.TBD_PAGE + (
            "<table><tr><td>x</td></tr></table>")  # a second, non-lineup table
        try:
            tmp = Path(tempfile.mkdtemp())
            snap = pl.archive_now(season="2026-27", gameweek=1, archive_dir=tmp)
            self.assertEqual(snap["team_count"], 1)
            self.assertTrue(snap["predictions_pending"])
        finally:
            pl.fetch_html = real


class TestPromotedFeedNames(unittest.TestCase):
    """FFP drops the 'City' that the promoted clubs carry in FPL. Without a mapping
    their whole squad's Start % is dropped as an unknown team."""

    def test_coventry_and_hull_meet_fpls_names(self):
        self.assertEqual(pl.TEAM_NAME_FIXES.get("Coventry"), "Coventry City")
        self.assertEqual(pl.TEAM_NAME_FIXES.get("Hull"), "Hull City")

    def test_the_fix_is_applied_in_the_parsed_canonical_team(self):
        page = ("<h2>Coventry Predicted Lineup</h2>"
                "<table><thead><tr><th>Player</th><th>Pos</th><th>Start %</th></tr>"
                "</thead><tbody><tr><td>Ellis Simms</td><td>FW</td><td>80%</td></tr>"
                "</tbody></table>")
        teams = pl.parse(page)
        self.assertEqual(teams[0]["canonical_team"], "Coventry City")


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
