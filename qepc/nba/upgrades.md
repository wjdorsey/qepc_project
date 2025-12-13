Yeah, there’s some good stuff in there we can absolutely cannibalize for QEPC. Think of this GitHub project as a fairly vanilla “feature kitchen” we can raid, then feed those ingredients into our much fancier quantum-flavored λ engine.

Let me summarize what it’s doing and where it helps us.

---

## 1. What their setup is doing (in human terms)

Across `get_player_data.py` + `predict_next_game.py`, they basically:

### a) Build a rolling player history dataset

From `get_player_data.py`:

* Uses `nba_api`:

  * `players.get_players()` to map names → player IDs
  * `playergamelog.PlayerGameLog` to pull game logs for a given season
* Then per player, they:

  * Sort games by `GAME_DATE`

  * Add rolling averages over a short window (e.g. last 5 games):

    ```python
    # conceptually
    for stat in ["PTS", "REB", "AST", ...]:
        stat_avg_last_5 = stat shifted(1).rolling(5).mean()
    ```

  * Add game features (you can see `add_game_features`):

    * Home vs away
    * Days rest between games
    * Back-to-back flags, etc.

  * Optionally merge team defense ratings into the rows.

  * Optionally add an injury_status column via an injury scraper.

All of that is saved into a CSV like `player_24-25_stats.csv`.

### b) For prediction, they do:

In `predict_next_game.py`:

1. Load that history CSV (`history_df`).

2. Build `latest_stats` = last row per player with those rolling averages.

3. Get **upcoming schedule** via `nba_api.ScoreboardV2`:

   ```python
   def get_upcoming_schedule(days_ahead=7):
       for each date:
           call ScoreboardV2(game_date=date)
           extract HOME_TEAM_ID / VISITOR_TEAM_ID
           map IDs → full names
   ```

4. Pull **team defense** and **SRS** from Basketball-Reference using `pd.read_html`:

   * `get_team_defense_ratings()` → DRtg per team
   * `get_team_srs_ratings()` → SRS per team

5. For each player, in `build_predictions` they compute:

   * Upcoming opponent (from schedule)

   * Whether the player is home/away

   * Historic splits vs that opponent:

     ```python
     past_vs_opp = history_df[
         (history_df["player_name"] == player) &
         (history_df["team"] == team_abbr) &
         (history_df["opponent"] == opponent_abbr)
     ]

     avg_pts_vs_opp, avg_reb_vs_opp, avg_ast_vs_opp, etc.
     ```

   * “Blowout risk” using SRS difference (and maybe spread/line if present).

   * Final prediction: **basically just** the rolling 5-game averages:

     ```python
     def predict_stats(latest_row):
         return {
             "predicted_pts": latest_row["pts_avg_last_5"],
             "predicted_reb": latest_row["reb_avg_last_5"],
             "predicted_ast": latest_row["ast_avg_last_5"],
         }
     ```

   * Attach everything into one row per player and save to `next_game_predictions.csv`.

It’s simple, but structurally it has some things we like.

---

## 2. What’s actually useful for QEPC?

The model itself is not magical (it’s basically “last 5 games” dressed nicely), but several **feature ideas** are directly useful:

### 2.1. Rolling recency windows (with shift)

They use a clean pattern: for each stat, “average of the last N games” **excluding** the current game (with `shift(1)`).

For QEPC we can:

* Add proper recency features into our Eoin-based `player_usage_eoin` pipeline:

  * `pts_avg_last_5`, `reb_avg_last_5`, `ast_avg_last_5`, maybe `last_10` too.
* Then, in the λ engine:

  * Use a blend like:

    [
    \lambda_{\text{player, pts}} =
    w_{\text{season}} \cdot \lambda_{\text{season}} +
    w_{\text{recent}} \cdot \lambda_{\text{last5}} +
    w_{\text{opp}} \cdot \lambda_{\text{vs_opp}}
    ]

  instead of pure season-long averages.

You already *feel* recency matters; this gives us a concrete rolling-window pattern to steal.

---

### 2.2. Opponent-specific splits

`compute_opponent_averages` / their logic in `build_predictions`:

* For a player + opponent, compute mean pts/reb/ast/3PM/etc **vs that opponent**.
* Use that as extra context.

QEPC upgrade ideas:

* Extend our Eoin `player_boxes_qepc` to build a **vs-team** summary table:

  * `player_id, opp_team_id, games_played_vs_opp, pts_per_game_vs_opp, ...`

* In λ-builder for props, treat `λ_vs_opp` as a “small correction” that nudges the baseline:

  ```python
  lambda_final = (
      0.7 * lambda_season
    + 0.2 * lambda_recent
    + 0.1 * lambda_vs_opp
  )
  ```

* That slots really nicely into the “entangled universes” idea: certain player–defense matchups shift the entire multiverse slightly.

---

### 2.3. Team defense + SRS as external context

Their `get_team_defense_ratings` and `get_team_srs_ratings` pull:

* **Defensive rating (DRtg)** per team.
* **SRS** (Simple Rating System: point diff + strength of schedule).

We already built our own **team strength** scores from Eoin (`strength_score`, `off_ppg`, `def_ppg`), but:

* DRtg is a **league-standard defensive feature** we can:

  * Use to cross-check whether our def_ppg + strength_score are aligned.
  * Potentially blend in for current season only (e.g. Eoin might lag a bit).

* SRS is basically:

  * A single scalar that says “this team is +X points better than league average” on a neutral floor.

QEPC upgrade ideas:

* When computing **expected team points** for a matchup:

  * Use our strengths as the primary, but adjust by SRS difference:

    * Higher SRS → slightly higher offensive λ, or more blowout risk.

* “Blowout risk” concept:

  * If SRS_home – SRS_away is large, weight some universes where starters lose minutes.
  * This fits *perfectly* with your multiverse theme:

    * In some universes, game stays close → stars hit ceiling.
    * In blowout universes, rotations shrink for the trailing team, or stars sit.

---

### 2.4. Schedule / upcoming games via `ScoreboardV2`

Their `get_upcoming_schedule` uses:

* `ScoreboardV2(game_date=...)` for the next X days.
* Maps `HOME_TEAM_ID` / `VISITOR_TEAM_ID` → full team names.

This is useful for QEPC’s **“tonight” mode**:

* Right now, our matchups are built from Eoin historical `Games.csv`.
* For **future**/live runs, we’ll want a `schedule_nba_api.py` that:

  * Pulls tonight’s games from `ScoreboardV2` or `LeagueSchedule`.
  * Maps them to our team IDs / names.
  * Produces the same `matchups` schema we used in the backtest notebook.

Basically: use their schedule pattern as the live-data counterpart to `matchups_eoin`.

---

### 2.5. Injury handling

They have:

* `get_current_injuries()` and `add_injury_status(df, injury_df)` that:

  * Scrape some injury source,
  * Attach `injury_status` per player,
  * Skip players marked “OUT” in `build_predictions`.

For QEPC:

* We don’t necessarily want to copy their exact scraping (terms of service, reliability, etc.), but structurally this is important:

  * Have a **clean “player availability” layer**:

    * `status` ∈ {active, out, doubtful, questionable, etc.}
    * Possibly a “expected minutes multiplier” per status.

* In QEPC’s λ engine:

  * For `OUT`: λ = 0, remove from rotations.
  * For `QUESTIONABLE`: reduce minutes/usage in a subset of universes (some universes he plays, others he doesn’t).

    * That’s very on-brand for us: “injury Schrödinger’s Cat”.

---

### 2.6. Simple baseline predictor as sanity check

Their actual prediction is:

* “Use rolling averages of the last 5 games as the predicted points/rebounds/assists.”

That’s not sophisticated, but it *is* a **baseline**.

We can:

* Implement the same simple predictor internally and use it in backtests as:

  * Baseline A: “Last-5 average”.
  * Baseline B: “Season average”.
  * QEPC: full Poisson cascade with usage + entanglement + schedule features.

Then we can compare:

* QEPC MAE/RMSE vs:

  * Vegas lines,
  * Last-5 naive model,
  * Season-only naive model.

That tells us whether the extra quantum bells and whistles are actually doing something.

---

## 3. How I’d plug this into QEPC (concretely)

If we translate their ideas into QEPC modules, something like this:

1. **Enhance `player_usage_eoin.py`**:

   * Add rolling windows:

     * `pts_avg_last_5`, `reb_avg_last_5`, `ast_avg_last_5`
   * Add opponent-split summaries:

     * `vs_opp_pts_pg`, `vs_opp_reb_pg`, etc.

2. **Add a `schedule_live_nba.py` module**:

   * Wrapper around `ScoreboardV2` to produce `matchups` with:

     * `game_date`, `home_team_id`, `away_team_id`, `home_team_name`, `away_team_name`.

3. **Add a `defense_context.py` module**:

   * Option 1: use Basketball-Reference DRtg/SRS like their code (via `pd.read_html`).
   * Option 2: derive “QEPC SRS” directly from Eoin data, keeping it self-contained.
   * Provide `get_team_defense_features(team_id)` that returns:

     * `def_rating`, `srs`, maybe `pace`, etc.

4. **Add a small `injury_layer.py`**:

   * For now, maybe simplify:

     * A CSV or manual file mapping players → status.
   * Later, maybe integrate real injury scraping / API.

5. **In the λ-builder for team totals and player props**:

   * Incorporate:

     * Recency (last-N),
     * Opponent splits,
     * Defense/SRS,
     * Blowout risk from SRS + spread.

6. **In the multiverse sim**:

   * Use blowout risk + injury uncertainty to diversify universes:

     * Universe group A: close game → starters heavy minutes.
     * Group B: blowout → bench gets more volume.
     * Group C: star questionable → branch on “plays” vs “ruled out” scenarios.

---

Short version:
Their model itself is relatively basic, but the **feature engineering** and **workflow pattern** are very aligned with where we’re steering QEPC:

* Recency windows,
* Opponent splits,
* Defense/SRS context,
* Live schedule,
* Injury layer,
* Blowout risk.

The nice part is: we don’t have to abandon the quantum / Poisson / entanglement vision to use any of this. We just feed these as better inputs and priors into the λs that power your multiverse.

Next natural move, if you’re up for it, is to pick one of these and wire it in properly—my vote would be: **rolling recency + vs-opponent splits inside `player_usage_eoin`**, because that’ll immediately improve the player-level λs we’re using everywhere else.
