#!/usr/bin/env python3
"""
Find regression test directories that are byte-identical duplicates.

Two tests are considered trivial duplicates when they share:
  - the source file content (sha1),
  - the exact arg line on line 3 of test.desc,
  - the full expected-output regex block on line 4+.

These tests give ESBMC identical inputs and assert identical verdicts, so
running both yields no additional signal. The dominant cause is a copy-paste
of a test directory that was never differentiated.

Usage:
    find_duplicate_tests.py [--root regression] [--json out.json]

Prints one cluster per line:
    <count>  <args>  <dir1> <dir2> ...

The first directory in each cluster is the suggested KEEP; the rest are
suggested DELETE. The choice of keep is heuristic (shortest path / earliest
glob order); a human should curate before mass-deleting.

Not addressed here: tests that touch the same source lines in ESBMC itself
but exercise different code paths (semantic duplicates). For that, build
ESBMC with -DENABLE_COVERAGE=On, run each test with a per-test
LLVM_PROFILE_FILE, and compare line-coverage vectors. That analysis is
left to a follow-up tool.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict


def cluster_tests(root):
    groups = defaultdict(list)
    for desc in sorted(glob.glob(os.path.join(root, "*/*/test.desc"))):
        tdir = os.path.dirname(desc)
        try:
            with open(desc) as f:
                lines = f.readlines()
        except OSError:
            continue
        if len(lines) < 4:
            continue
        src_name = lines[1].strip()
        args = lines[2].strip()
        regex = "\n".join(ln.strip() for ln in lines[3:] if ln.strip())
        src_path = os.path.join(tdir, src_name)
        if not os.path.exists(src_path):
            continue
        with open(src_path, "rb") as sf:
            src_hash = hashlib.sha1(sf.read()).hexdigest()
        groups[(src_hash, args, regex)].append(tdir)
    return {k: v for k, v in groups.items() if len(v) > 1}


def pick_keep(dirs):
    """Heuristic: keep the shortest path, tiebreak alphabetically."""
    return sorted(dirs, key=lambda d: (len(d), d))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="regression")
    ap.add_argument("--json", help="Write structured output to this JSON file")
    args = ap.parse_args()

    clusters = cluster_tests(args.root)
    print(f"# Clusters: {len(clusters)}")
    redundant = sum(len(v) - 1 for v in clusters.values())
    print(f"# Redundant test dirs: {redundant}")
    print()

    rows = []
    for key, dirs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        cmd = key[1]
        keep = pick_keep(dirs)
        drops = [d for d in dirs if d != keep]
        rows.append({"args": cmd, "keep": keep, "drop": drops})
        print(f"{len(dirs)}\t{cmd[:60]!r}")
        print(f"  KEEP  {keep}")
        for d in drops:
            print(f"  DROP  {d}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
