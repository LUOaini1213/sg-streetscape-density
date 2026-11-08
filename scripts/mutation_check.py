"""Plant known bugs one at a time and require the test suite to catch each one.

Each mutant is an exact (file, original, replacement) substitution. The original
text must occur exactly once; the file is always restored. Exit code 1 if any
mutant survives or cannot be applied.
"""
import subprocess
import sys
from pathlib import Path

MUTANTS = [
    ("ssd/grid.py", "np.floor(x / size)", "np.round(x / size)"),
    ("ssd/indices.py", '"vegetation": {"vegetation"}', '"vegetation": {"vegetation", "terrain"}'),
    ("ssd/indices.py", '"building": {"building", "wall"}', '"building": {"building"}'),
    ("ssd/indices.py", "total = counts.sum() - excl", "total = counts.sum()"),
    ("ssd/indices.py", 'name.split("--")[-1]', 'name.split("--")[0]'),
    ("ssd/agreement.py", "(va + vb + (ma - mb) ** 2)", "(va + vb)"),
    ("ssd/agreement.py", "bias - 1.96 * sd", "bias - 1.645 * sd"),
    ("ssd/agreement.py", '"mae": float(np.abs(a - b).mean())', '"mae": float((a - b).mean())'),
    ("ssd/spatial.py", "(n / s0) * (z @ w @ z)", "(1 / s0) * (z @ w @ z)"),
    ("ssd/spatial.py", "w[i, nb] = 1.0 / k", "w[i, nb] = 1.0"),
    ("ssd/spatial.py", "return np.array([fold_of[b] for b in blocks])",
     "return rng.integers(0, n_folds, len(blocks))"),
    ("ssd/spatial.py", "tr, te = folds != f, folds == f", "tr, te = folds == folds, folds == f"),
    ("ssd/density.py", "0 < f < 30", "0 <= f < 30"),
    ("ssd/density.py", '"gpr_mean": g.f.sum() / g.a.sum()', '"gpr_mean": g.f.sum() / area'),
    ("ssd/density.py", 'return (pieces.groupby("cell").a.sum() / area)',
     'return (pieces.groupby("cell").a.count() / area)'),
    ("ssd/density.py", 'str(v).strip().upper() == "LND"', 'str(v).strip().upper() == "EVA"'),
    ("ssd/pairwise.py", "winner = np.where(y == 1, i, j)", "winner = np.where(y == 1, j, i)"),
    ("ssd/pairwise.py", "mu[l] -= var[l] / c * v", "mu[l] += var[l] / c * v"),
    ("ssd/pairwise.py", "(latent[i] - latent[j])", "(latent[j] - latent[i])"),
    ("ssd/pairwise.py", "hits += (p < alpha) and (b > 0)", "hits += (p < 0.5) and (b > 0)"),
    ("ssd/claims.py", "elif val != reg[key]:", "elif False:"),
    ("ssd/sampling.py", "ids = sorted(image_ids)", "ids = list(image_ids)"),
    ("ssd/sampling.py", "while k < len(order) and order[k] in done:", "while k < len(order):"),
    ("ssd/claims.py", 'r[f"agree_{k}_{pair}_ccc"] = f3(ag[k][pair]["ccc"])',
     'r[f"agree_{k}_{pair}_ccc"] = f3(ag[k][pair]["pearson_r"])'),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"],
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    if not run_tests():
        print("baseline test suite fails; fix it first")
        return 1
    survived, broken = [], []
    for n, (f, old, new) in enumerate(MUTANTS, 1):
        p = Path(f)
        raw = p.read_bytes()                      # restore byte-for-byte afterwards
        src = raw.decode("utf-8")
        if src.count(old) != 1:
            broken.append((n, f, old))
            print(f"[{n:02d}] CANNOT APPLY {f}: {old!r} occurs {src.count(old)}x")
            continue
        try:
            p.write_bytes(src.replace(old, new).encode("utf-8"))
            passed = run_tests()
        finally:
            p.write_bytes(raw)
        print(f"[{n:02d}] {'SURVIVED' if passed else 'killed  '} {f}: {old!r} -> {new!r}", flush=True)
        if passed:
            survived.append(n)
    print(f"{len(MUTANTS) - len(survived) - len(broken)}/{len(MUTANTS)} mutants killed; "
          f"{len(survived)} survived; {len(broken)} not applicable")
    return 1 if survived or broken else 0


if __name__ == "__main__":
    sys.exit(main())
