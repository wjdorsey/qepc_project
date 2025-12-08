Short version: yes, but it’s mostly a **one-time setup per machine**.

On any *new* or *stale* machine where you haven’t used Git LFS with that repo before:

1. **Make sure Git LFS is installed**

   * If you haven’t already: install Git LFS from its installer (GitHub for Windows usually has it, but just in case).

2. **Initialize Git LFS on that machine (once)**
   In a terminal (PowerShell) on that machine:

   ```powershell
   git lfs install
   ```

3. **Go to your repo and pull like normal**

   ```powershell
   cd C:\Users\wdors\qepc_project   # or wherever it is on that machine
   git pull
   ```

   If LFS is set up correctly, this usually downloads the big files automatically (via the “smudge” filter).

4. **If you want to be explicit, run an LFS pull too**

   Sometimes, especially if the clone is older or filters were off, you’ll see tiny pointer files instead of real data. In that case:

   ```powershell
   git lfs pull
   ```

   That forces Git LFS to download the actual large content for the current checkout.

---

### Quick sanity checks

* See which files are tracked by LFS:

  ```powershell
  git lfs ls-files
  ```

* Confirm they’re real files (not tiny text pointers) by checking their sizes in Explorer or with:

  ```powershell
  dir path\to\your\lfsy\file
  ```

---

So the “special” part is really:

* **Once per machine:** `git lfs install`
* **When you update the repo:** `git pull` (and if needed) `git lfs pull`

If you want, you can paste the output of `git lfs ls-files` from one machine and we can double-check that your big CSVs and notebooks are really going through LFS the way you expect.
Nice, flipping to the *“how do I do this on purpose?”* chapter.

To add a file to Git LFS **the right way (for future files)**, the flow is:

---

## 1. Decide *what* should go in LFS

Pick either:

* A **file pattern** (recommended): e.g. all `.csv` in a folder

  ```bash
  git lfs track "data/raw/*.csv"
  ```

* Or a **single file**:

  ```bash
  git lfs track "data/raw/PlayerStatistics.csv"
  ```

That command:

* Updates/creates a `.gitattributes` file with the LFS rules.
* Doesn’t touch history yet; it just says “from now on, these will be LFS files.”

---

## 2. Commit the `.gitattributes` change

```bash
git add .gitattributes
git commit -m "Track large data files with Git LFS"
```

This step is important; LFS only really “takes effect” once the attributes are in a committed state.

---

## 3. Add and commit the actual file(s)

Now put your big file in place (or if it’s already there but untracked, that’s fine) and:

```bash
git add path/to/your/file.ext
git commit -m "Add big file via LFS"
```

Because it matches the pattern in `.gitattributes`, Git will store it as an LFS object.

---

## 4. Push as usual

```bash
git push origin main
# or whatever your branch is
```

Git will upload:

* The normal Git commits
* The new LFS objects behind the scenes

---

## 5. Verify it’s really in LFS

After committing (before or after push), run:

```bash
git lfs ls-files
```

You should see your file listed there. That’s your “yes, LFS has adopted this file” check.

---

### Common gotchas (aka how to avoid pain later)

* **You must track *before* adding/committing**
  If you commit a big file *before* telling LFS about it, it goes into regular Git history, and later cleanup requires `git lfs migrate` (history rewrite).

* **Don’t LFS-ify your backups**
  For your `_backups/` zips: better to `.gitignore` them and keep backups out of Git entirely. LFS is great for **data used by the project**, not giant archives of the project itself.

* **Patterns are usually better**
  For QEPC, something like:

  ```bash
  git lfs track "data/raw/*.csv"
  ```

  makes your life easier than doing one file at a time.

---

So in short, future recipe in your repo:

```bash
cd C:\Users\wdors\qepc_project

# 1) Tell LFS which files to manage
git lfs track "data/raw/*.csv"

# 2) Commit the tracking rule
git add .gitattributes
git commit -m "Track raw data csvs with LFS"

# 3) Add the actual file(s)
git add data/raw/NewHugeDataset.csv
git commit -m "Add new dataset"

# 4) Push
git push
```

That’s the clean “no drama” path. Treat LFS rules like traffic signs: put them up *before* you let the big trucks onto the road.



In a **JupyterLab terminal** that’s running a normal Linux-style shell (bash/zsh), the “force delete a folder and everything in it” command is:

```bash
rm -rf folder_name
```

Breakdown:

* `rm` = remove
* `-r` = recursive (go into subfolders)
* `-f` = force (don’t prompt, ignore read-only)

Example:

```bash
rm -rf notebooks/01_core
```

If the folder name has spaces:

```bash
rm -rf "some folder name"
```

If the folder is empty and you want the “nice” version instead of the nuke:

```bash
rmdir folder_name
```

But 99% of the time people just use `rm -rf`.
As always, double-check with `ls` first that you’re in the right directory, because `rm -rf` does not come with an undo button.

