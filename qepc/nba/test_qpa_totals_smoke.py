from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - optional in minimal envs
    np = None
    pd = None

from qepc.nba.qpa_totals import (
    TotalsConfig,
    enrich_games_with_config,
    estimate_score_correlation,
    predict_totals,
    evaluate_predictions,
)


class TestQpaTotalsSmoke(unittest.TestCase):
    def test_small_synthetic_run(self) -> None:
        if np is None or pd is None:
            self.skipTest("numpy/pandas not installed in this environment")
        dates = pd.date_range("2023-10-01", periods=4, freq="D")
        team_boxes = pd.DataFrame(
            {
                "game_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "team_id": [100, 200, 100, 200, 100, 200, 100, 200],
                "opp_team_id": [200, 100, 200, 100, 200, 100, 200, 100],
                "game_date": list(dates.date) * 2,
                "teamscore": [110, 105, 111, 108, 115, 102, 107, 112],
                "opponentscore": [105, 110, 108, 111, 102, 115, 112, 107],
            }
        )

        games = pd.DataFrame(
            {
                "game_id": [1, 2, 3, 4],
                "game_date": dates.date,
                "home_team_id": [100, 200, 100, 200],
                "away_team_id": [200, 100, 200, 100],
                "home_score": [110, 108, 115, 107],
                "away_score": [105, 111, 102, 112],
            }
        )
        config = TotalsConfig(sample_size=32)
        _, enriched = enrich_games_with_config(games, team_boxes, config)
        corr_stats = estimate_score_correlation(enriched, config.corr_shrink)
        rng = np.random.default_rng(42)
        preds = predict_totals(enriched, config, corr_stats, rng)
        metrics = evaluate_predictions(preds)

        self.assertIn("pred_total_qpa", preds.columns)
        self.assertFalse(np.isnan(preds["pred_total_qpa"]).all())
        self.assertIn("mae", metrics)
        self.assertIn("bias", metrics)


if __name__ == "__main__":
    unittest.main()
