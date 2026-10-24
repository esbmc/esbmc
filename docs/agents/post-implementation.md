# Post-implementation

## Post-implementation Pass

After implementing any non-trivial coding task, before committing:

1. **Simplify aggressively.** Remove unnecessary conditional checks, dead code,
   redundant abstractions, duplicate logic. Re-verify the code still works
   correctly. Apply the same pass to test code.
2. **Verify with ESBMC** when the task touches C/C++ code or ESBMC's own
   headers/frontends. Use the `esbmc-verifier` agent to confirm the patch works and introduces no new errors. For non-ESBMC tasks (e.g. Python frontend,
   build scripts), run the project's normal lint/typecheck/test commands. A
   patch that adds or removes a branch in ESBMC's own source also owes an
   `esbmc-verifier` Mode C proof: C-Live for an added branch, C-Dead for a
   removed one.
3. **Code review.** Use the `code-reviewer` agent on the diff. Apply
   high-confidence findings; explain anything you skip.
4. **Coverage gate.** Run the `esbmc-coverage-pr` agent in Mode B on the diff
   before opening or updating any PR. Every executable line the diff adds must
   either be covered by a test in the same PR or triaged with a stated reason
   (vendored / dead / defensive / broken feature). A BLOCKED verdict stops the PR
   — add the missing tests or re-scope.

## Available Subagents

These specialised agents are defined in `docs/agents/subagents/` and should be
preferred over ad-hoc shell commands when their description fits the task.

- **`esbmc-verifier`** — Recommended formal-verification tool for this repo. Two
  modes: (A) bug-fixing inside ESBMC's own codebase — inspects GOTO IR
  (`--goto-functions-only`), VCCs (`--show-vcc`), and the symbol table; applies
  minimal patches; re-runs ESBMC to confirm `VERIFICATION SUCCESSFUL`; produces
  a two-tier harness package under `regression/<suite>/github_<N>/` (literal
  repro), `regression/<suite>/github_<N>-nondet/` (nondet generalisation), and
  an optional `_fail/` negative variant when the patch shifts a checker
  boundary. (B) Any external C/C++ codebase (application, library, firmware) —
  three-phase strategy (language-level safety → functional contracts via
  k-induction → bug-specific negative proofs) with stub-shadowing for whatever
  the module depends on (DBs, network, filesystem, hardware/RTOS, vendor SDKs).
  Invoke for the post-implementation ESBMC step (§Post-implementation Pass #2),
  for deterministic witnesses when sanitizers cannot reproduce a memory/UB bug
  (`docs/agents/bug-fixing.md`), and when diagnosing unexpected ESBMC
  results (`docs/agents/debugging.md`). Defaults to bitwuzla; honours
  `test.desc` flags when present. For one-shot sanity checks
  (`esbmc file.c --incremental-bmc`), call `esbmc` directly instead.
- **`code-reviewer`** — Diff review against the priorities in §Code Review
  Priorities. Invoke for the post-implementation review step
  (§Post-implementation Pass #3).
- **`creduce-reducer`** — Reduces C/C++ programs that trigger an ESBMC bug to a
  minimal reproducer using C-Reduce with property-preserving interestingness
  scripts. Use when filing or investigating ESBMC bug reports against large
  inputs.
- **`esbmc-coverage-pr`** — Codecov line coverage of ESBMC's own sources (distinct
  from `--branch-coverage`/`--cov-report-json`, which measure the program under
  verification). Mode B is the mandatory per-PR gate of §Post-implementation
  Pass #4: it scopes to the diff, requires each added executable line to be
  covered by a test in the same PR or triaged as
  vendored/dead/defensive/unwired, mutation-checks the PR's new tests, and
  returns PASS / SHORT / BLOCKED without forcing an instrumented
  rebuild. Mode A runs coverage campaigns — it pulls the per-line uncovered map
  from the public Codecov API, triages gaps, adds regression and Catch2 tests,
  and proves the delta with `llvm-cov`. Uncovered lines are also the best source
  of dead-code candidates for `esbmc-verifier` Mode C.

## Code Review Priorities

1. **Critical**: Verification soundness, memory safety, undefined behavior
2. **High**: Logic errors in SMT encoding/symbolic execution, performance
   regressions, missing tests
3. **Medium**: Code quality, API consistency, documentation gaps
4. **Low**: Minor style if matching surrounding code
