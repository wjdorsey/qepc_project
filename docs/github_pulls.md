Nice — since you pushed the branch, getting it onto your other computer is just “fetch + checkout.”

On the **other computer**, inside the repo folder:

## 1) Fetch the latest branches from GitHub

```powershell
git fetch --all
```

## 2) See the branch name (optional but helpful)

```powershell
git branch -r
```

Look for something like:
`origin/feat/leakage-free-backtest` (or whatever you named it).

## 3) Check out that branch locally (and track the remote)

Use your actual branch name in place of `BRANCH_NAME`:

```powershell
git checkout -b BRANCH_NAME origin/BRANCH_NAME
```

Example (if your branch is `feat/leakage-free-backtest`):

```powershell
git checkout -b feat/leakage-free-backtest origin/feat/leakage-free-backtest
```

## 4) Pull normally from then on

```powershell
git pull
```

---

### If you already created a local branch with the same name on that computer

Then just do:

```powershell
git checkout BRANCH_NAME
git pull
```

### Quick “am I on the right branch?” check

```powershell
git status
```

It should say `On branch <your branch>`.

That’s it — once it’s checked out, it behaves like `main`: edit → commit → push, and the other machine can `pull` that branch too.
