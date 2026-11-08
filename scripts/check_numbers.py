"""Fail if any tagged number in README.md / paper/paper.md disagrees with outputs/."""
import sys
from pathlib import Path

from ssd.claims import check_numbers

if __name__ == "__main__":
    errs = check_numbers(Path("."))
    for e in errs:
        print("MISMATCH", e)
    print(f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)
