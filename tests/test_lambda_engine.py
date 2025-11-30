"""
QEPC Tests: lambda_engine
Basic sanity checks for the compute_lambda function.

These tests are intentionally simple and beginner friendly:
- they make sure the function runs,
- and that obvious relationships (rest advantage, offensive strength)
  move the lambdas in the expected direction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qepc.core import lambda_engine


def test_rest_advantage_increases_home_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If the home team has more rest than the away team, and everything else is equal,
    the home lambda should get a bump.
    """

    # Remove random noise so the test is deterministic
    monkeypatch.setattr(lambda_engine.np.random, "normal", lambda loc, scale: 1.0)

    # Two games with the same teams and stats, different rest only
    schedule = pd.DataFrame(
        {
            "Home Team": ["Alpha", "Alpha"],
            "Away Team": ["Beta", "Beta"],
            # Game 0: equal rest, Game 1: big home rest edge
            "home_rest_days": [3.0, 4.0],
            "away_rest_days": [3.0, 1.0],
            "home_b2b": [False, False],
            "away_b2b": [False, False],
        }
    )

    team_stats = pd.DataFrame(
        {
            "Team": ["Alpha", "Beta"],
            "ORtg": [110.0, 110.0],
            "DRtg": [110.0, 110.0],
            "Pace": [100.0, 100.0],
            "Volatility": [12.0, 12.0],
        }
    )

    result = lambda_engine.compute_lambda(schedule, team_stats, include_situational=True)

    lambda_home_equal_rest = result.loc[0, "lambda_home"]
    lambda_home_more_rest = result.loc[1, "lambda_home"]

    assert lambda_home_more_rest > lambda_home_equal_rest


def test_offense_and_defense_affect_lambdas(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A clearly stronger offensive team facing a weaker defense
    should get a higher lambda than its opponent.
    """

    monkeypatch.setattr(lambda_engine.np.random, "normal", lambda loc, scale: 1.0)

    schedule = pd.DataFrame(
        {
            "Home Team": ["High"],
            "Away Team": ["Low"],
        }
    )

    # Home team: better offense, better defense
    team_stats = pd.DataFrame(
        {
            "Team": ["High", "Low"],
            "ORtg": [120.0, 100.0],
            "DRtg": [100.0, 120.0],
            "Pace": [100.0, 100.0],
            "Volatility": [12.0, 12.0],
        }
    )

    result = lambda_engine.compute_lambda(schedule, team_stats, include_situational=False)

    lambda_home = result.loc[0, "lambda_home"]
    lambda_away = result.loc[0, "lambda_away"]

    assert lambda_home > lambda_away
