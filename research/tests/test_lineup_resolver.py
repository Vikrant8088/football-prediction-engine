"""Tests for the predicted-lineup name resolver.

Every case here is taken from the live feed. A wrong id is worse than a known gap —
it silently attributes one player's rotation risk to another — so the threshold
behaviour and the ambiguous-first-name cases are pinned as hard as the happy path.
"""

import unittest

import pandas as pd

from research.data import lineup_resolver as lr


def _players():
    """A miniature FPL bootstrap, using real awkward names."""
    rows = [
        # id, web_name,        full_name,                         team
        (1, "Raya", "David Raya Martin", "Arsenal"),
        (2, "Gabriel", "Gabriel dos Santos Magalhães", "Arsenal"),
        (3, "Jesus", "Gabriel Fernando de Jesus", "Arsenal"),
        (4, "Martinelli", "Gabriel Teodoro Martinelli Silva", "Arsenal"),
        (5, "Ødegaard", "Martin Ødegaard", "Arsenal"),
        (6, "Gyökeres", "Viktor Gyökeres", "Arsenal"),
        # FPL carries the extra surname, which is what makes the source's
        # concatenated "KepaArrizabalaga" genuinely hard: not equal even squashed.
        (7, "Kepa", "Kepa Arrizabalaga Revuelta", "Arsenal"),
        (8, "Lewis-Skelly", "Myles Lewis-Skelly", "Arsenal"),
        (9, "McTominay", "Scott McTominay", "Manchester United"),
    ]
    return pd.DataFrame(
        [{"id": i, "web_name": w, "full_name": f, "team": t} for i, w, f, t in rows])


class TestNormalize(unittest.TestCase):
    def test_strips_accents(self):
        self.assertEqual(lr.normalize("Gabriel Magalhães"), "gabriel magalhaes")

    def test_maps_letters_that_do_not_decompose(self):
        # Ø is a distinct letter, not O + diacritic: NFKD alone leaves it untouched.
        self.assertEqual(lr.normalize("Martin Ødegaard"), "martin odegaard")
        self.assertEqual(lr.normalize("Gyökeres"), "gyokeres")

    def test_strips_punctuation(self):
        self.assertEqual(lr.normalize("Lewis-Skelly"), "lewis skelly")

    def test_squash_removes_spaces(self):
        self.assertEqual(lr.squash("Kepa Arrizabalaga"), "kepaarrizabalaga")


class TestResolveName(unittest.TestCase):
    def setUp(self):
        players = _players()
        self.arsenal = [lr._candidate(r) for _, r in
                        players[players["team"] == "Arsenal"].iterrows()]
        self.united = [lr._candidate(r) for _, r in
                       players[players["team"] == "Manchester United"].iterrows()]

    def _id(self, name, candidates=None):
        return lr.resolve_name(name, candidates or self.arsenal)[0]

    def test_short_name_matches_longer_official_name(self):
        self.assertEqual(self._id("David Raya"), 1)          # vs "David Raya Martin"

    def test_concatenated_name_still_matches(self):
        # The source's real glitch: the space is missing, AND FPL holds an extra
        # surname — so this resolves only via the unique squashed-prefix rule.
        player_id, _, method = lr.resolve_name("KepaArrizabalaga", self.arsenal)
        self.assertEqual(player_id, 7)
        self.assertEqual(method, "squash_prefix")

    def test_squash_prefix_never_fires_when_ambiguous(self):
        # "Gabrie" prefixes all three Arsenal Gabriels and is nobody's web_name, so it
        # can only reach the prefix rule — where uniqueness must reject it rather than
        # pick one. This is the guard that makes the prefix rule safe.
        player_id, _, _ = lr.resolve_name("Gabrie", self.arsenal)
        self.assertIsNone(player_id)

    def test_camel_case_surname_is_not_broken(self):
        # The reason we squash rather than split on camel case.
        self.assertEqual(self._id("Scott McTominay", self.united), 9)

    def test_accent_stripped_source_name_matches(self):
        self.assertEqual(self._id("Martin Odegaard"), 5)
        self.assertEqual(self._id("Viktor Gyokeres"), 6)

    def test_ambiguous_first_names_resolve_to_the_right_gabriel(self):
        self.assertEqual(self._id("Gabriel Magalhaes"), 2)
        self.assertEqual(self._id("Gabriel Jesus"), 3)
        self.assertEqual(self._id("Gabriel Martinelli"), 4)

    def test_hyphenated_name_matches(self):
        self.assertEqual(self._id("Myles Lewis-Skelly"), 8)

    def test_bare_first_name_resolves_only_via_fpls_own_short_name(self):
        # FPL's web_name for Magalhães IS "Gabriel" (Jesus is "Jesus", Martinelli is
        # "Martinelli"), so this is the right answer — and it must come from an exact
        # web_name hit, never from a loose token-subset or prefix guess among the three.
        player_id, _, method = lr.resolve_name("Gabriel", self.arsenal)
        self.assertEqual(player_id, 2)
        self.assertEqual(method, "exact_web")

    def test_unknown_player_is_rejected_not_guessed(self):
        self.assertIsNone(self._id("Cristiano Ronaldo"))


class TestResolveSnapshot(unittest.TestCase):
    def _snapshot(self):
        return {"teams": [{
            "team": "Arsenal", "canonical_team": "Arsenal",
            "players": [
                {"name": "David Raya", "position": "GK", "start_pct": 90,
                 "predicted_xi": True},
                {"name": "KepaArrizabalaga", "position": "GK", "start_pct": 10,
                 "predicted_xi": False},
                {"name": "Someone Unknown", "position": "CF", "start_pct": 30,
                 "predicted_xi": False},
            ]}]}

    def test_builds_start_pct_by_player_id(self):
        result = lr.resolve_snapshot(self._snapshot(), _players())
        self.assertEqual(result["start_pct"][1], 90)
        self.assertEqual(result["start_pct"][7], 10)

    def test_pending_tbd_player_resolves_by_name_but_adds_no_numeric_signal(self):
        # A 'TBD' Start % parses as start_pct=None. It must resolve as a NAME (so the
        # match rate is honest) yet never enter the start_pct map — and never crash on
        # int(None), which it used to on every early-season page.
        snap = {"teams": [{"team": "Arsenal", "canonical_team": "Arsenal", "players": [
            {"name": "David Raya", "position": "GK", "start_pct": None,
             "predicted_xi": True}]}]}
        result = lr.resolve_snapshot(snap, _players())
        self.assertEqual(result["stats"]["matched"], 1, "the name still resolved")
        self.assertEqual(result["start_pct"], {}, "but no numeric signal was added")

    def test_unmatched_are_reported_never_dropped_silently(self):
        result = lr.resolve_snapshot(self._snapshot(), _players())
        self.assertEqual(len(result["unmatched"]), 1)
        self.assertEqual(result["unmatched"][0]["name"], "Someone Unknown")
        self.assertEqual(result["stats"]["matched"], 2)
        self.assertEqual(result["stats"]["players"], 3)

    def test_duplicate_player_keeps_the_higher_probability(self):
        snap = self._snapshot()
        snap["teams"][0]["players"].append(
            {"name": "David Raya", "position": "GK", "start_pct": 95,
             "predicted_xi": False})
        result = lr.resolve_snapshot(snap, _players())
        self.assertEqual(result["start_pct"][1], 95)

    def test_unknown_team_is_skipped_and_recorded(self):
        snap = {"teams": [{"team": "Barcelona", "canonical_team": "Barcelona",
                           "players": [{"name": "X", "start_pct": 50,
                                        "position": "CF", "predicted_xi": True}]}]}
        result = lr.resolve_snapshot(snap, _players())
        self.assertEqual(result["start_pct"], {})
        self.assertIn("Barcelona", result["stats"]["unknown_teams"])


if __name__ == "__main__":
    unittest.main()
