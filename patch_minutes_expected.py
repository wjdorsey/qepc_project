from pathlib import Path

path = Path("qepc/nba/player_points_model.py")
lines = path.read_text(encoding="utf-8").splitlines(True)

def leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip())]

def is_minutes_expected_block_start(line: str) -> bool:
    # Were looking for the V2-style block: df["minutes_expected"] = (
    return 'df["minutes_expected"]' in line and "=" in line and "(" in line

replaced = False
i = 0
while i < len(lines):
    if is_minutes_expected_block_start(lines[i]):
        indent = leading_ws(lines[i])

        # Find the end of the parenthesized block (a line that is just ")")
        j = i + 1
        while j < len(lines) and lines[j].strip() != ")":
            j += 1

        if j < len(lines) and lines[j].strip() == ")":
            j += 1  # include the closing ")"

            block_text = "".join(lines[i:j])
            # Only replace if this minutes_expected block looks like the reactive blend
            if ("minutes_mean3" in block_text) and ("minutes_mean10" in block_text) and (".fillna" in block_text):
                new_block = [
                    f'{indent}# minutes expected (prior)\n',
                    f'{indent}df["minutes_expected"] = _prior_group_rolling_mean(\n',
                    f'{indent}    df, "player_id", "minutes_actual", window=int(cfg.recent_window)\n',
                    f'{indent})\n',
                ]
                lines = lines[:i] + new_block + lines[j:]
                replaced = True
                break

    i += 1

if not replaced:
    raise SystemExit(
        "Patch failed: couldn't find the minutes_expected reactive-blend block. "
        "Paste the minutes_expected section here and Ill tailor the patch."
    )

path.write_text("".join(lines), encoding="utf-8")
print("OK: reverted minutes_expected to leakage-safe rolling mean.")
