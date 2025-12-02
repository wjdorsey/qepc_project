# QEPC Data Model

This document defines the **canonical data tables** used by QEPC.

The goal is:

- One clean, canonical source of truth per concept.
- QEPC code reads **only** from these canonical tables.
- All other CSVs / legacy extracts are treated as **raw inputs** or **archives**, not model inputs.

---

## 0. Directory layout

By convention:

- Canonical tables live in:  
  `data/raw/`

- Legacy / archive / experimental files live in (examples):  
  `data/legacy/`  
  `data/_archive/`  
  `experimental/`  

QEPC modules should **only** assume the canonical files exist. Everything else is optional.

---

## 1. NBA Team Logs – `NBA_Team_Logs_All_Seasons.csv`

**Purpose:**  
One row = **one team in one game** (team-game log).  
This is the backbone for **team strengths**, **λ (Poisson means)**, and **game-level backtests**.

**Location:**  
`data/raw/NBA_Team_Logs_All_Seasons.csv`

### 1.1 Keys

- `gameId` (string or int-like) – NBA game id (e.g. `21401227`, `0022300001`)
- `teamId` (int) – NBA team id
- Together, `(gameId, teamId)` are **unique**.

### 1.2 Core columns (required)

Minimum set QEPC assumes:

- `gameDate` – datetime (no timezone)  
- `Season` – string, e.g. `"2014-15"`, `"2023-24"`, `"2025-26"`
- `teamId` – int64
- `teamCity` – string (e.g. `"Boston"`)
- `teamName` – string (e.g. `"Celtics"`)
- `opponentTeamCity` – string
- `opponentTeamName` – string
- `teamScore` – int (points scored by this team)
- `opponentScore` – int
- `home` – 1 if this row is the home team, 0 if away
- `win` – 1 if `teamScore > opponentScore`, else 0

### 1.3 Box score / pace fields (recommended)

- `reboundsTotal`
- `assists`
- `steals`
- `blocks`
- `turnovers`
- `fieldGoalsAttempted`
- `fieldGoalsMade`
- `fieldGoalsPercentage`
- `threePointersAttempted`
- `threePointersMade`
- `threePointersPercentage`
- `freeThrowsAttempted`
- `freeThrowsMade`
- `freeThrowsPercentage`
- `plusMinusPoints`
- `numMinutes` (typically 240 for regulation, >240 for OT)

These support pace estimation, offensive/defensive ratings, and volatility.

### 1.4 Advanced team stats (enrichment targets)

These may start partially missing for older seasons but are **ideal** for recent seasons:

- `pointsInThePaint`
- `benchPoints`
- `pointsFastBreak`
- `pointsFromTurnovers`
- `pointsSecondChance`
- `biggestLead`
- `biggestScoringRun`
- `leadChanges`
- `timesTied`
- `timeoutsRemaining`

These are derived primarily from the **boxscore TEAM_STATS** / live boxscore endpoints and are used to model game “script” and chaos.

---

## 2. NBA Player Logs – `NBA_Player_Logs_All_Seasons.csv`

**Purpose:**  
One row = **one player in one game** (player-game log).  
This table is the foundation for **player props**, **usage modeling**, and **recent form**.

**Location:**  
`data/raw/NBA_Player_Logs_All_Seasons.csv`

### 2.1 Keys

- `gameId` – same semantics as team logs
- `playerId` – NBA player id

Together, `(gameId, playerId)` are **unique**.

### 2.2 Core columns (required)

- `gameId`
- `gameDate` – datetime (no timezone)
- `Season` – `"2014-15"`, `"2023-24"`, etc.
- `playerId`
- `playerName`
- `teamId`
- `teamAbbrev` – e.g. `"BOS"`
- `teamName`
- `opponentTeamAbbrev` – parsed from matchup
- `home` – 1 if player’s team is home, else 0
- `win` – 1 if player’s team won, else 0

### 2.3 Box score stats (required for props)

These are mapped from NBA API columns like `FGM`, `REB`, etc.:

- `minutes` – float or int (total minutes played; can be from `MIN` or `MIN_SEC`)
- `fgm`, `fga`, `fg_pct`
- `fg3m`, `fg3a`, `fg3_pct`
- `ftm`, `fta`, `ft_pct`
- `oreb`, `dreb`, `reb`
- `ast`, `stl`, `blk`
- `tov` (turnovers)
- `pf` (personal fouls)
- `pts`
- `plus_minus`

### 2.4 Contextual / derived fields

Optionally computed during canonicalization:

- `matchup` – original NBA API string like `"BOS @ LAL"` / `"NOP vs. SAS"`
- `opponentTeamId` – if available or derivable
- `starter` – boolean if we can detect from lineup
- `usage_estimate` – can be derived later; not required in this table

---

## 3. NBA Schedule – `NBA_Schedule_All_Seasons.csv`

**Purpose:**  
One row = **one game** (not team-game).  
Used for rest calculations, back-to-backs, and schedule-aware features.

**Location:**  
`data/raw/NBA_Schedule_All_Seasons.csv`

### 3.1 Keys

- `gameId` – same as above (unique per game)

### 3.2 Core columns

- `gameId`
- `Season`
- `gameDate` – datetime
- `gameTimeUTC` – datetime or ISO string
- `homeTeamId`, `homeTeamAbbrev`, `homeTeamName`
- `awayTeamId`, `awayTeamAbbrev`, `awayTeamName`
- `arenaName` (optional)
- `arenaCity` (optional)

### 3.3 Result fields (joined from logs)

For convenience, QEPC may join or store:

- `homeScore`
- `awayScore`
- `homeWin` – 1 if homeScore > awayScore

These can be derived from team logs if needed.

---

## 4. Non-goals & principles

- We do **not** maintain lots of separate CSVs like:
  - `Player_Recent_Form_L5.csv`
  - `Player_Averages_With_CI.csv`
  - `TeamForm.csv`
- Those become **derived tables** created in notebooks, using the canonical logs above.
- If a column isn’t clearly needed for QEPC, it **stays in raw/legacy** or gets computed on the fly.
- QEPC code imports from:
  - `NBA_Team_Logs_All_Seasons.csv`
  - `NBA_Player_Logs_All_Seasons.csv`
  - `NBA_Schedule_All_Seasons.csv`
  and treats everything else as optional.
