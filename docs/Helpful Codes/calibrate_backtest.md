Cool — this is your **baseline** run (no affine calibration applied yet), and the bucket labels are now showing properly. ✅

A quick read of what you’ve got:

* **Overall MAE 4.936**, **Bias -0.452** → you’re **under-predicting** overall a bit.
* **By ACTUAL minutes:** over in low minutes, under in high minutes (classic “role/minutes” mismatch).
* **By PREDICTED minutes:** **big under** in predicted 0–20 bucket (**-1.986**), but **over** in predicted 34+ bucket (**+1.338**) → this is that “wrong shape” we fix with the affine step.

## Next step: fit the affine calibration on the *earlier* chunk

Run this (this is the real “Step 1” we meant):

```powershell
python -m qepc.nba.backtest_player_points --start 2022-10-01 --end 2023-06-22 --min-minutes 10 --progress --apply-insample-affine --save-json --save-log
```

What you’re looking for in the output:

* It will print something like: `slope=... | intercept=...`
* It will also print **Calibrated MAE** / **Calibrated Bias**

## Then Step 2: apply those exact numbers to the later chunk

Take the slope/intercept it prints (example numbers shown here are placeholders) and run:

```powershell
python -m qepc.nba.backtest_player_points --start 2023-10-01 --end 2024-06-22 --min-minutes 10 --progress --affine-slope 0.XXX --affine-intercept 1.XXX --save-json --save-log
```

That’s the “fair test” version (learn correction on earlier games → apply to later games).

Once you paste the output from those two runs, I’ll tell you whether the affine is enough… or whether we should go straight after **minutes_coherent** (which is usually the true final boss).
