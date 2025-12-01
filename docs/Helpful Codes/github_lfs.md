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
