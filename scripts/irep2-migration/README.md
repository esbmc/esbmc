# IREP2 goto-program migration — differential harness (Phase 0)

Tooling for the legacy-IREP → IREP2 migration of the goto-program pipeline.
This was **Phase 0** of the migration recorded in
[`docs/irep2-migration.md`](../../docs/irep2-migration.md); the migration
itself completed under tracking issue
[esbmc/esbmc#4715](https://github.com/esbmc/esbmc/issues/4715) (closed).
The harness is retained as the differential goto-binary diff and surface-ratio
census tooling for any future incremental work.

Phase 0 changes **no** ESBMC behaviour — it adds the safety net that every later
migration PR is gated on: a way to prove a patch leaves the generated goto
program **bit-for-bit identical** (after canonicalisation), plus a scoreboard
that measures the legacy region shrinking.

## Why

The migration's hard constraint is behavioural equivalence. Pass/fail verdicts
alone miss silent drift (reordered instructions, changed canonicalisation, lost
guards). The strongest cheap guard is to diff the **canonical goto dump**
(`esbmc --goto-functions-only`) before and after a change: any difference is a
hard failure unless explicitly justified.

## Contents

| Script | Purpose |
|--------|---------|
| `lib.sh` | Shared helpers (canonicalisation, test.desc-faithful argument construction). Sourced by the others. |
| `capture_goto_baseline.sh` | Capture golden canonical dumps for a corpus. Run on a **clean pre-migration build**. |
| `diff_goto_baseline.sh` | Re-derive dumps and diff against the baseline. Exit 0 = identical, 1 = drift. |
| `migrate_census.sh` | Count `migrate_expr`/`migrate_type`(`_back`) calls per file — the migration scoreboard. |

## Usage

Build a clean `esbmc` first (the binary lives at `build/src/esbmc/esbmc`).
Corpus arguments are directories searched recursively for `test.desc`; paths are
interpreted relative to the repository root.

```sh
# 1. Capture the golden baseline on the current (pre-migration) build.
#    Scope it to the subsystems an upcoming PR touches (here: concurrency).
scripts/irep2-migration/capture_goto_baseline.sh \
    build/src/esbmc/esbmc /tmp/irep2-baseline \
    regression/parallel-solving regression/esbmc-unix

# 2. Apply a migration PR, rebuild, then check for drift.
scripts/irep2-migration/diff_goto_baseline.sh \
    build/src/esbmc/esbmc /tmp/irep2-baseline \
    regression/parallel-solving regression/esbmc-unix
#    exit 0 => goto program unchanged (the gate passes)
#    exit 1 => drift; the per-test unified diff is printed

# 3. Track the legacy region shrinking (default scope: src/goto-programs).
scripts/irep2-migration/migrate_census.sh
scripts/irep2-migration/migrate_census.sh src/goto-programs/rw_set.cpp
```

A migration PR's checklist (per the plan):
1. `migrate_census.sh <target>` strictly decreases in the target region, nowhere increases.
2. `diff_goto_baseline.sh` exits 0 on the relevant corpus.
3. Regression verdicts + counterexamples agree under both Bitwuzla and Z3.

## Canonicalisation

`irep2_canon` (in `lib.sh`) strips only non-deterministic / environment-specific
noise, preserving everything semantically meaningful (instruction kinds,
operands, guards, goto targets, file/line/column locations):

- the `ESBMC version …` banner;
- timing lines (`GOTO program creation time: …s`, etc.);
- the program-global instruction index in `// N file …` comments (a change
  anywhere renumbers everything downstream, so the bare index is dropped while
  the `file/line/column/function` location is kept);
- the build-tree absolute path → `REPO_ROOT`;
- the per-run temporary header directory (`/tmp/esbmc-…` on Linux,
  `/var/folders/…/T/esbmc[._-]…` on macOS, including occurrences embedded inside
  generated type names) → `HDR`.

If a future ESBMC change introduces a new non-deterministic line, add a rule to
`irep2_canon` (and note it here) rather than weakening the diff.

## Notes

- Argument construction mirrors `regression/testing_tool.py` exactly, so each
  test is driven with the same flags the real suite uses (including
  goto-affecting ones like `--data-races-check`, `--k-induction`).
- esbmc's exit status is intentionally ignored; the deterministic dumped text is
  what is compared, so descriptors with no flags line (whose regex line is
  mis-parsed as bogus args, exactly as the real suite does) still yield a stable,
  comparable digest.
- Both scripts clean up `/tmp/esbmc-headers-*` on exit (per the repo's /tmp
  hygiene rule).
- macOS `bash` 3.2 compatible (no associative arrays / `mapfile`).
