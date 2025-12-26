# 📄 `DEV_NOTES.md`

## DEV_NOTES.md
### QEPC — Git, GitHub, and JupyterLab Terminal Guide (Beginner-Safe)

This document is the **single source of truth** for working on the QEPC project.
It assumes **no prior Git experience** and prioritizes clarity over cleverness.

Git is powerful, literal, and unforgiving — but also extremely safe when used
intentionally. This guide exists so you never have to guess.

---

### 1. What Git Is (and Is Not)

#### Git ≠ GitHub

- **Git** is a version-control system that runs on your computer.
- **GitHub** is a website that stores copies of Git repositories online.

If your project folder contains a hidden `.git` directory, Git is active.

Git allows you to:
- Save snapshots of your work
- Go back in time
- Experiment safely
- Sync work between machines

Nothing is uploaded unless **you explicitly push it**.

---

### 2. The Four States of a Git Project

Think of Git like a camera with a table next to it.

#### 1) Working Directory
Files you edit normally (notebooks, scripts, docs).

#### 2) Staging Area
A holding area where you choose what will be saved.

#### 3) Commit
A permanent snapshot with a message (stored locally).

#### 4) Remote (GitHub)
Copies of commits stored online.

Changes only move forward when **you tell Git to move them**.

---

### 3. The Most Important Command

`git status` 

Shows:
- What files changed
- What is staged
- What is untracked
- What branch you are on

This command **never changes anything**.

Use it:
- Before editing
- Before committing
- After pulling
- Whenever confused

If you remember only one command, remember this one.

---

### 4. Viewing History and Changes

#### View Commit History
`
git log
git log --oneline
`

#### View File Differences

`git diff
git diff --staged
`

All of these commands are **read-only** and safe.

---

### 5. Staging and Committing Changes

#### Stage Files

`
git add filename
git add .
`

Always check:

`
git status
`

#### Commit (Save a Snapshot)

`
git commit -m "clear message"
`

* Creates a snapshot locally
* Does **not** upload anything

Think: *“Save checkpoint with explanation.”*

---

#### 6. Branches (Parallel Work)

A branch is a **parallel version** of the project.

* `main` = last known working version
* Other branches = experiments or fixes

#### Branch Commands

`
git branch
git switch branch_name
git switch -c new_branch_name
`

`git switch -c` is the safest way to experiment.

---

### 7. Syncing with GitHub

#### View Remote

`
git remote -v
`

#### Upload Work

`
git push
`

#### Download Work

`
git pull
git pull --rebase
`

`--rebase` keeps history cleaner for personal projects.

---

### 8. Temporarily Hiding Work

#### Stash (Pause Button)

`
git stash
git stash pop
`

Stash is **not a commit** — it’s a temporary drawer.

---

### 9. Undo Tools (Use Carefully)

#### Undo File Changes

`
git restore filename
`

⚠️ Discards edits in that file.

#### Unstage a File

`
git restore --staged filename
`

#### Undo Last Commit (Keep Changes)

`
git reset --soft HEAD~1
`

#### Hard Reset (Very Destructive)

`
git reset --hard HEAD
`

---

### 10. QEPC-Specific Git Rules

#### Rule 1 — `main` Is Sacred

`main` should always represent a working state.

#### Rule 2 — Use Branches for Experiments

`
git switch -c experiment/short-name
`

#### Rule 3 — Commit Small and Often

Small commits are easier to undo and debug.

#### Rule 4 — Jupyter Notebooks Are Noisy

Outputs and metadata change easily. Always review diffs.

#### Rule 5 — Never Delete `.git`

Deleting `.git` deletes all history.

#### Rule 6 — Large Data Files

Large `.csv`, `.parquet`, `.pkl` files belong in Git LFS or `.gitignore`.

`
git lfs ls-files
git lfs pull
`

#### Rule 7 — One Machine per Branch

Pull before starting work. Push when finished.

#### Rule 8 — Avoid Force Push

Do **not** use `git push --force` unless you fully understand it.

---

### 11. JupyterLab Terminal (PowerShell Basics)

#### List Files

`Get-ChildItem`


#### Show Hidden Files


`Get-ChildItem -Force`

#### Clear Notebook Outputs

`Get-ChildItem -Recurse -Filter *.ipynb | ForEach-Object {
    python -m jupyter nbconvert --clear-output --inplace $_.FullName
} `

#### Delete Folder (Destructive)

`
Remove-Item -Recurse -Force folder_name
`

---

### 12. Common Errors (Plain English)

 `not a git repository`

You are in the wrong folder.

`untracked files would be overwritten`

Local files conflict with incoming changes.
Move, delete, or stash them.

`file is being used by another process`

Close Jupyter kernels, VS Code, or Explorer previews.

---

### 13. Daily Safe Workflow

`
git status
(edit files)
git add .
git status
git commit -m "clear message"
git push
`

---

### Final Mental Model

* `status` = what’s happening
* `add` = choose what to save
* `commit` = save snapshot
* `push` = upload
* `pull` = download
* branches = parallel timelines
* commits = time travel

Git is literal, not malicious.
Slow down and it will do exactly what you ask.

```

---


```
