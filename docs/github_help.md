This is a practical cheat sheet for Git + GitHub + common terminal errors.
Works in **JupyterLab Terminal** on:
- **Windows PowerShell**
- **macOS/Linux bash/zsh**

---

### What shell am I in?

#### Windows (PowerShell)
```powershell
$PSVersionTable.PSVersion
$env:ComSpec
````

#### macOS/Linux (bash/zsh)

```bash
echo $SHELL
ps -p $$ -o comm=
```

**Rule of thumb:**

* PowerShell uses `Remove-Item`, `Get-ChildItem`, etc. (aliases exist, but flags differ).
* bash/zsh uses `rm -rf`, `ls -la`, etc.

---

#### 1) Git “greatest hits” (daily drivers)

#### Where am I? What’s changed?

```bash
git status
git log --oneline --graph --decorate -n 20
git branch
git branch -a
git remote -v
```

#### Get updates safely

Recommended habit: **fetch first**, then decide what to do.

```bash
git fetch --all --prune
git switch main
git pull --rebase
```

#### Normal work flow

```bash
git switch -c feature/my-thing
git add -p
git commit -m "Explain what changed"
git push -u origin feature/my-thing
```

#### Helpful diffs

```bash
git diff
git diff --staged
git diff main...HEAD
```

#### Stash (temporary “put it in a drawer”)

```bash
git stash push -m "wip"
git stash list
git stash pop
```

---

### 2) Branch / merge / rebase basics (a sane mental model)

#### Merge (simpler history, can create merge commits)

```bash
git switch feature/my-thing
git merge main
```

### Rebase (cleaner history; learn it slowly, it’s worth it)

```bash
git switch feature/my-thing
git rebase main
# if conflicts:
git status
# fix files, then:
git add <fixed-file>
git rebase --continue
# bail out if needed:
git rebase --abort
```

**Avoid rebasing commits you already pushed** (unless you know how to force-push responsibly).

---

## 3) “Undo” toolkit (save yourself from yourself)

#### Unstage a file

```bash
git restore --staged path/to/file
```

#### Discard local edits to a file (⚠️ destructive)

```bash
git restore path/to/file
```

#### Undo last commit but keep changes (good for “oops”)

```bash
git reset --soft HEAD~1
```

#### Nuke local changes (⚠️ truly destructive)

```bash
git reset --hard HEAD
```

#### Fix the last commit message / add forgotten files (be careful if pushed)

```bash
git commit --amend
```

---

### 4) Cleaning files (PowerShell vs bash/zsh)

#### Delete a folder (PowerShell)

```powershell
Remove-Item -Recurse -Force .\folder_name
```

#### Delete a folder (bash/zsh)

```bash
rm -rf ./folder_name
```

#### Show hidden files

PowerShell:

```powershell
Get-ChildItem -Force
```

bash/zsh:

```bash
ls -la
```

---

### 5) “My pull failed” — common Git errors & fixes

#### A) Untracked files would be overwritten by merge

**Error:**
`error: The following untracked working tree files would be overwritten by merge`

**Fix options:**

1. Move the file out of the way (PowerShell):

```powershell
mkdir .\_backup
Move-Item .\path\file.ipynb .\_backup\
git pull
```

2. Or stash including untracked:

```bash
git stash -u
git pull --rebase
git stash pop
```

3. Or delete it (if safe):

* PowerShell:

```powershell
Remove-Item .\path\file.ipynb
```

* bash/zsh:

```bash
rm ./path/file.ipynb
```

---

### B) Diverged branch / messy pull

If `git pull` creates chaos, do:

```bash
git fetch --all --prune
git status
```

Often the clean move is:

```bash
git pull --rebase
```

If you’re in conflict hell and want out:

```bash
git merge --abort   # if you were merging
git rebase --abort  # if you were rebasing
```

---

### C) “not a git repository”

You’re not in the repo root, or `.git` is missing.

Check:

```bash
pwd
```

PowerShell list including hidden:

```powershell
Get-ChildItem -Force
```

Make sure you’re in your project folder that contains `.git`.

---

### D) “dubious ownership”

Fix:

```bash
git config --global --add safe.directory "C:/path/to/repo"
```

---

## 6) Git LFS (Large File Storage) essentials

### Install / verify LFS

```bash
git lfs install
git lfs version
```

### See what files are in LFS

```bash
git lfs ls-files
```

### On a new machine after clone/pull (when big files are missing)

```bash
git lfs pull
```

### Track new large file patterns (example)

```bash
git lfs track "*.parquet"
git add .gitattributes
git commit -m "Track parquet with Git LFS"
git push
```

---

## 7) PowerShell “classic errors” (and how to win)

### A) “The term 'git' is not recognized…”

Git isn’t installed or not on PATH.

PowerShell:

```powershell
where.exe git
git --version
```

Fix: install Git for Windows, restart terminal.

---

### B) “Cannot remove item… being used by another process”

Close the program holding it (VS Code, Jupyter kernel, Explorer preview).

Then:

```powershell
Remove-Item -Recurse -Force .\folder
```

---

### C) “Access to the path is denied”

Try:

```powershell
Remove-Item -Recurse -Force .\folder
```

If still blocked, open PowerShell as Administrator.

---

### D) Flags don’t work like bash

PowerShell aliases exist (`rm`, `ls`, `cat`) but **flags differ**.
Use native PowerShell commands when in doubt:

* `Remove-Item -Recurse -Force`
* `Get-ChildItem -Force`
* `Get-Content file.txt`

---

## 8) Handy “repo hygiene” commands

### See file changes since last commit

```bash
git status
git diff
```

### Remove untracked junk (dry run first!)

```bash
git clean -n
git clean -fd   # ⚠️ deletes untracked files/folders
```

### Quick “I want my local main to match remote exactly” (⚠️ destructive)

This discards local changes on main:

```bash
git switch main
git fetch origin
git reset --hard origin/main
```

---

## 9) Recommended habits (keeps QEPC sane)

* Run `git status` before and after you do anything.
* Prefer `git pull --rebase` on feature branches (cleaner history).
* Tag stable versions (perfect for QEPC versioning):

```bash
git tag -a v0.7.0 -m "QEPC v0.7.0 stable"
git push origin v0.7.0
```

* For notebooks: consider clearing outputs before committing (keeps diffs clean).

---

## 10) Notebook output clearing (quick note)

Get-ChildItem -Recurse -Filter *.ipynb | ForEach-Object {
    python -m jupyter nbconvert --clear-output --inplace $_.FullName
}
```

If you want it even more “drop-in,” I can also give you a **PowerShell one-liner** to create that file directly from JupyterLab Terminal (so you don’t have to copy/paste into an editor).
```



Absolutely. Below is a **full rewrite of `DEV_NOTES.md`** that is:

* Written for a **true beginner**
* Explicitly compatible with **JupyterLab Terminal**
* Assumes **Windows + PowerShell first**, with Linux/macOS notes where helpful
* Explains **what each command does in plain English**
* Clearly marks **safe vs destructive** commands
* Designed to be something you can *actually read end-to-end* without getting lost

You can copy this **as-is** into your project root as `DEV_NOTES.md`.

---

### QEPC Beginner Guide to Git, GitHub, and JupyterLab Terminal

This file explains the tools used to manage this project in **simple, human terms**.
You do NOT need prior Git or terminal experience to use this guide.

If something here feels confusing, that’s normal — Git is powerful but weird.
This guide exists so you don’t have to memorize anything.

---

### PART 1 — What Git Actually Is (and is not)

#### Git ≠ GitHub

- **Git** is a *local version history system* on your computer.
- **GitHub** is a website that stores copies of Git projects online.

If your folder contains a hidden `.git` directory, Git is watching it.

Git lets you:
- Save snapshots of your project
- Go back in time
- Experiment safely
- Sync work between computers

---

### PART 2 — The Four States of a Git Project

Think of Git like a camera with a staging table.

#### 1) Working Directory
Your normal files.
You edit these in notebooks, scripts, etc.

#### 2) Staging Area
A holding area.
You choose which changes you want to save.

#### 3) Commit
A permanent snapshot of staged files, with a message.

#### 4) Remote (GitHub)
A copy of commits stored online.

Nothing goes to GitHub unless you explicitly send it.

---

### PART 3 — The Most Important Command (Always Safe)

`git status`

Shows:
- Which files changed
- Which files are staged
- Which files are untracked
- Which branch you are on

This command **never changes anything**.

Use it:
- Before doing anything
- After doing anything
- When confused

If you only remember ONE command, remember this one.

---

### PART 4 — Seeing Project History

`git log`

Shows a list of saved snapshots (commits).

Each commit has:
- A unique ID
- A date
- A message explaining what changed

`git log --oneline`

Same information, but compact.

These commands are **read-only** and always safe.

---

### PART 5 — Branches (Safe Parallel Experiments)

#### What is a branch?

A branch is a **parallel version of the project**.

- `main` = primary timeline
- Other branches = experiments

You can switch branches at any time.

---

`git branch`

Lists all local branches.

`git switch branch_name`

Moves your project into the state of that branch.
Your files on disk will change to match it.

`git switch -c new_branch_name`

Creates a new branch AND switches to it.
This is the safest way to experiment.

---

### PART 6 — Seeing What Changed

`git diff`

Shows how your files differ from the last commit.
Nothing is changed.

`git diff --staged`

Shows what will be saved in the next commit.

---

### PART 7 — Choosing What to Save (Staging)

`git add filename`

Marks a file to be included in the next snapshot.

`git add .`

Stages **all changes** in the current folder.

After running this, always run:
```

git status

```

`git add -p`

Shows changes one piece at a time and asks if you want to stage them.
Advanced, but very useful later.

---

### PART 8 — Saving a Snapshot

`git commit -m "message"`

Creates a snapshot of staged files.

Important:
- This does NOT upload anything
- It only saves history locally

Think:
“Save checkpoint with explanation.”

---

### PART 9 — Connecting to GitHub

## `git remote -v`

Shows where your project syncs online.

## `git push`

Uploads local commits to GitHub.

## `git pull`

Downloads commits from GitHub and merges them.

## `git pull --rebase`

Downloads commits and replays your work on top.
Usually cleaner for personal projects.

---

### PART 10 — Temporarily Hiding Changes

## `git stash`

Temporarily hides uncommitted changes.

Use when:
“I need to switch branches but I’m not ready to commit.”

## `git stash pop`

Restores stashed changes.

---

### PART 11 — Undo Tools (Read Carefully)

## `git restore filename`

⚠️ Discards edits in that file.

## `git restore --staged filename`

Un-stages a file but keeps edits.

## `git reset --soft HEAD~1`

Undoes the last commit but keeps changes.

## `git reset --hard HEAD`

⚠️ Deletes all uncommitted changes.

Only use when absolutely sure.

---

### PART 12 — JupyterLab Terminal Basics (PowerShell)

Most Windows JupyterLab terminals use **PowerShell**.

#### List files
```

Get-ChildItem

```

#### Show hidden files
```

Get-ChildItem -Force

```

#### Delete a folder
```

Remove-Item -Recurse -Force folder_name

```

⚠️ This permanently deletes files.

---

### PART 13 — Common Errors (and what they mean)

#### “not a git repository”
You are in the wrong folder.
Move to the project root.

## “untracked files would be overwritten”
You have local files blocking a pull.

Fix:
- Move the file
- Delete it
- Or stash with:
```

git stash -u

```

## “file is being used by another process”
Close Jupyter notebooks, VS Code, or kernels using it.

---

# PART 14 — Git LFS (Large Files)

Used for large data files (parquet, csv, etc).

## See tracked large files
```

git lfs ls-files

```

## On a new machine
```

git lfs pull

```

---

# PART 15 — A Safe Daily Workflow

1) `git status`
2) Edit files
3) `git add .`
4) `git status`
5) `git commit -m "clear message"`
6) `git push`

Repeat.

---

# PART 16 — Final Mental Model

- `status` = what’s happening
- `add` = choose what to save
- `commit` = save snapshot
- `push` = upload
- `pull` = download
- branches = parallel universes
- commits = time travel

Git is not magic.
It is a very organized notebook with undo powers.
```

---

Excellent choice. Let’s do **all three**, cleanly and calmly, in a way that actually *sticks*.
What follows is something you can keep open next to JupyterLab and survive real mistakes with.

---

# 1️⃣ A Real Step-by-Step Example (QEPC-style, no shortcuts)

We’ll simulate a **very normal day** working in your QEPC repo.

### Scenario

You:

* open JupyterLab
* edit a notebook
* save your work
* commit it
* push it to GitHub

No branches yet. No fancy tricks.

---

## Step 0 — Open JupyterLab Terminal

You should be **inside your project root**, something like:

```
C:\Users\Will\qepc_project
```

Check where you are:

```
pwd
```

If you don’t see your project files, `cd` into the right folder.

---

## Step 1 — Ask Git what’s going on (always first)

```
git status
```

You might see:

```
On branch main
nothing to commit, working tree clean
```

This means:

* You’re on `main`
* Git sees no changes
* Everything is calm

---

## Step 2 — Edit something

Now:

* Open a notebook (e.g. `notebooks/brain/11_player_points_lambda.ipynb`)
* Change code, markdown, whatever
* Save the notebook

Back in the terminal:

```
git status
```

Now you’ll see something like:

```
modified: notebooks/brain/11_player_points_lambda.ipynb
```

Translation:

> “You changed this file, but I’m not saving it yet.”

---

## Step 3 — Stage the change (choose what to save)

```
git add notebooks/brain/11_player_points_lambda.ipynb
```

Then check again:

```
git status
```

Now you’ll see:

```
Changes to be committed:
  modified: notebooks/brain/11_player_points_lambda.ipynb
```

Translation:

> “This file will be included in the next snapshot.”

---

## Step 4 — Commit (create a snapshot)

``` 
git commit -m "Improve player points lambda calibration notes"
```

What just happened:

* Git took a snapshot of the staged file
* You can always return to this point

Nothing went to GitHub yet.

---

## Step 5 — Push to GitHub

```
git push
```

Now:

* Your work is online
* Another computer can pull it
* You have a backup

🎉 You just completed a full Git cycle.

---

# 2️⃣ Visual Diagrams (How Git *Actually* Thinks)

These are worth slowing down for.

---

## Diagram A — The Git Flow (Most Important One)

```
┌──────────────────┐
│ Working Directory│  ← you edit files here
└─────────┬────────┘
          │ git add
          ▼
┌──────────────────┐
│  Staging Area    │  ← “this will be saved”
└─────────┬────────┘
          │ git commit
          ▼
┌──────────────────┐
│     Commit       │  ← permanent snapshot
└─────────┬────────┘
          │ git push
          ▼
┌──────────────────┐
│   GitHub (remote)│
└──────────────────┘
```

**Key insight:**
Git does *nothing* automatically.
You explicitly move changes through each layer.

---

## Diagram B — Branches as Parallel Universes

```
main ──●──●──●──────────●───▶
            \
             feature-x ─●──●──▶
```

* `main` keeps moving forward
* `feature-x` explores ideas
* You can merge them later

Nothing explodes unless you force it to.

---

## Diagram C — Stash (Temporary Drawer)

```
Before stash:
Working Directory = messy

git stash

Drawer 🗄️ ← changes stored here

Working Directory = clean

git stash pop

Drawer emptied → changes restored
```

Stash is **not a commit**.
It’s just a pause button.

---

# 3️⃣ The Panic Checklist (When Git Yells at You)

Bookmark this mentally.

---

## 🛑 FIRST RULE: STOP

Do **not** type random commands.

Run:

```
git status
```

This almost always tells you what’s wrong.

---

## Panic Case 1 — “I don’t know what I did”

Safe commands only:

```
git status
git log --oneline -5
git diff
```

These **cannot** damage anything.

---

## Panic Case 2 — “I edited stuff but want to switch branches”

```
git stash
git switch other_branch
```

Later:

```
git stash pop
```

---

## Panic Case 3 — “Git says files will be overwritten”

This means:

> “You have local files that conflict with incoming ones.”

Options (choose ONE):

### Option A — Keep your work

```
git stash -u
git pull --rebase
git stash pop
```

### Option B — You don’t care about the file

```
Remove-Item path\to\file.ipynb
git pull
```

---

## Panic Case 4 — “Everything is broken and I just want GitHub’s version”

⚠️ Destructive but sometimes necessary:

```
git fetch origin
git reset --hard origin/main
```

This makes your local `main` **exactly match GitHub**.

Only do this if you are okay losing local changes.

---

## Panic Case 5 — “PowerShell won’t delete a folder”

Usually means:

* Jupyter kernel
* VS Code
* Explorer preview

Fix:

* Close notebooks
* Shut down kernels
* Then:

```
Remove-Item -Recurse -Force folder_name
```

---

# Final Reassurance (Important)

You are **not bad at Git**.
Git is just:

* extremely literal
* extremely cautious
* extremely bad at explaining itself

If you remember nothing else, remember this:

> **`git status` first.
> Think.
> Then act.**

If you want next, we can:

* Practice **making and fixing a mistake on purpose**
* Add a **“QEPC-specific Git rules” section**
* Create a **one-page printable Git survival card**

Just tell me which direction you want to go.



---

# ✅ ADDITION 1: QEPC-Specific Git Rules

*(Append this section to `DEV_NOTES.md`)*


# PART 1— QEPC-Specific Git Rules (Project Survival Guide)

These rules exist because QEPC:
- uses large datasets
- uses Jupyter notebooks
- evolves rapidly
- is easy to accidentally break with Git mistakes

Follow these and you will avoid 90% of Git pain.

---

## RULE 1 — Never work directly on `main` for experiments

`main` should represent:
> “The last version that worked.”

For experiments, refactors, or risky changes:

```

git switch -c experiment/short-description

```

Only merge back into `main` when:
- notebooks run
- no obvious errors appear
- you understand what changed

---

## RULE 2 — Commit small, commit often

Bad:
- “Huge commit with 20 unrelated changes”

Good:
- “Update lambda calibration logic”
- “Fix minutes parsing edge case”
- “Add documentation notes”

Small commits mean:
- easier debugging
- safer rollbacks
- clearer history

---

## RULE 3 — Always run `git status` before and after changes

This is not optional.

**Before editing**
```

git status

```

**Before committing**
```

git status

```

**After pulling**
```

git status

```

If something looks wrong, stop and investigate.

---

## RULE 4 — Jupyter notebooks require extra care

Notebooks change even when you don’t edit code (outputs, metadata).

Before committing notebooks:
- Make sure the changes are intentional
- Prefer clearing outputs when possible
- Avoid committing temporary debug cells

If a notebook diff looks strange:
```

git diff

```

---

## RULE 5 — Never panic-delete `.git`

If Git behaves strangely:
- You are almost always in the wrong folder
- Or dealing with a merge/conflict

Deleting `.git` removes ALL history.
Only do this if you intend to completely start over.

---

## RULE 6 — Large data files belong in Git LFS or ignored

QEPC data rules:
- Large `.csv`, `.parquet`, `.pkl` → Git LFS
- Generated caches → ignored via `.gitignore`

Check LFS files:
```

git lfs ls-files

```

If a pull is missing data:
```

git lfs pull

```

---

## RULE 7 — One machine at a time per branch

If you:
- edit `main` on Laptop A
- edit `main` on Desktop B

You will get conflicts.

Safer pattern:
- Pull before starting work
- Push when done
- Switch branches for parallel work

---

## RULE 8 — When in doubt, do NOT force-push

Avoid:
```

git push --force

```

Unless you explicitly understand:
- rebasing
- rewritten history
- upstream consequences

If unsure, ask or stop.

---

## RULE 9 — Backups are not failure, they are wisdom

QEPC is complex.

Regularly:
- tag stable versions
- zip backups
- push to GitHub

Git is powerful, but redundancy is stronger.

---

## RULE 10 — If Git surprises you, slow down

Git is deterministic.
Nothing happens without a command.

If something unexpected occurs:
1) Stop
2) `git status`
3) Read carefully
4) Act intentionally


---

# 🧠 ADDITION 2: One-Page Printable Git Survival Card

This is designed to fit on **one screen or one printed page**.
You can copy this into a file called `GIT_SURVIVAL_CARD.md`.


# GIT SURVIVAL CARD (QEPC Edition)

## FIRST RULE
When confused:
```

git status

```

---

## DAILY WORKFLOW
```

git status
(edit files)
git add .
git status
git commit -m "clear message"
git push

```

---

## SAFE COMMANDS (READ-ONLY)
```

git status
git log --oneline -5
git diff
git branch
git remote -v

```

---

## COMMON TASKS

### Start safe experiment
```

git switch -c experiment/name

```

### Save work
```

git add .
git commit -m "message"

```

### Upload work
```

git push

```

### Get latest changes
```

git pull --rebase

```

---

## STASH (TEMPORARY HIDE)
```

git stash
git stash pop

```

---

## FIXES

### Undo last commit (keep work)
```

git reset --soft HEAD~1

```

### Discard file edits (⚠️)
```

git restore filename

```

### Make local match GitHub (⚠️)
```

git fetch origin
git reset --hard origin/main

```

---

## POWERHELL FILE COMMANDS

### Delete folder
```

Remove-Item -Recurse -Force folder

```

### List files
```

Get-ChildItem -Force

```

---

## LFS FILES
```

git lfs ls-files
git lfs pull

```

---

## PANIC MODE
1) STOP
2) `git status`
3) Do nothing destructive
4) Ask or think
```

---

# FINAL TRUTH

Git is:

* cautious
* literal
* reversible

Mistakes are normal.
Panicking is optional.

```

---

## What we’ve built now

You now have:
- ✅ A **beginner-first DEV_NOTES**
- ✅ QEPC-specific Git rules
- ✅ A one-page Git survival card
- ✅ A mental model that scales as QEPC grows

Next logical upgrades (your choice):
- 🔹 “QEPC branching strategy” (naming conventions + workflow)
- 🔹 Notebook-specific Git hygiene automation
- 🔹 A **Git recovery drill** (practice breaking + fixing safely)

Just tell me which one you want to tackle next.
```

