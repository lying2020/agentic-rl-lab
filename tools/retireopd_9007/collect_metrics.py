"""Extract per-step console metrics from verl logs into CSV.

Usage: python3 collect_metrics.py OUT.csv LOG [LOG ...]
Logs are read in order; a step that appears in a later log (after a resume) replaces the earlier one.
For val-only logs, the "Initial validation metrics" dict is written as step 0.
"""

import csv
import re
import sys

STEP_LINE = re.compile(r"step:(\d+) - (.*)$")
PAIR = re.compile(r"([\w/@.\-]+):(-?[0-9.]+(?:e[+-]?\d+)?|nan|inf)")
INITIAL_VAL = re.compile(r"'([\w/@.\-]+)': np\.float64\((-?[0-9.]+(?:e[+-]?\d+)?|nan)\)")


def parse(paths):
    rows = {}
    for path in paths:
        with open(path, errors="replace") as f:
            text = f.read()
        for line in text.splitlines():
            m = STEP_LINE.search(line)
            if m:
                rows[int(m.group(1))] = {k: float(v) for k, v in PAIR.findall(m.group(2))}
        if "Initial validation metrics" in text:
            tail = text[text.index("Initial validation metrics"):]
            rows[0] = {k: float(v) for k, v in INITIAL_VAL.findall(tail)}
    return rows


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rows = parse(paths)
    keys = sorted({k for r in rows.values() for k in r})
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + keys)
        for step in sorted(rows):
            w.writerow([step] + [rows[step].get(k, "") for k in keys])
    print(f"{out}: {len(rows)} steps, {len(keys)} metrics")


if __name__ == "__main__":
    main()
