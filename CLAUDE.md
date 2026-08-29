# AGENTS.md

This file provides guidance to coding agents working with this repository. The same workflow rules also live in `CLAUDE.md` (which Claude Code loads automatically); update both files together when changing build, test, style, or post-implementation rules.

## Project Overview

ESBMC (Efficient SMT-based Context-Bounded Model Checker) is a software model checker that detects bugs or proves their absence in C, C++, CUDA, CHERI-C, Python, Solidity, Java, and Kotlin programs. It works by parsing source → building AST → converting to GOTO program → symbolic execution (SSA) → encoding as SMT formula → solving with SMT solvers.

## Build Commands

**NEVER run cmake in the repo root (e.g., `cmake .` or `cmake -B. -S.`).** Always use `build/` or a subdirectory of it as the build directory (e.g., `-Bbuild`). The `.gitignore` only covers `build/` — in-tree builds pollute the source tree with hundreds of untracked artifacts.

```sh
# Minimal build with Z3 solver (at least one solver must be enabled for regression tests)
cmake -GNinja -Bbuild -S . \
  -DDOWNLOAD_DEPENDENCIES=On \
  -DENABLE_PYTHON_FRONTEND=On \
  -DENABLE_Z3=On \
  -DBUILD_TESTING=On \
  -DENABLE_REGRESSION=On \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build (uses Ninja)
ninja -C build

# Install
ninja -C build install
```

Additional optional CMake flags:
- `-DENABLE_SOLIDITY_FRONTEND=On` — Solidity smart contract frontend
- `-DENABLE_JIMPLE_FRONTEND=On` — Java/Kotlin frontend (requires JDK 11+)
- `-DENABLE_BITWUZLA=On` — Bitwuzla solver backend
- `-DENABLE_BOOLECTOR=On` — Boolector solver backend
- Quality: `-DENABLE_WERROR=On`, `-DENABLE_CLANG_TIDY=On`, `-DENABLE_COVERAGE=On`

See `scripts/build.sh` for full platform-specific dependency setup and solver configuration.

Requires: CMake 3.18+, Ninja, Boost (date_time, program_options, iostreams, system, filesystem), LLVM 11+ (tested up to 21), Bison, Flex, Z3 (or another SMT solver).

## Testing

Regression tests require at least one solver backend (e.g., Z3). All commands run from the `build/` directory.

```sh
# Run unit tests only (fast, excludes regression-labeled tests)
ctest -j$(nproc) -LE regression --timeout 60

# Run all regression tests (slow, creates temp dirs in /tmp — see note below)
ctest -j$(nproc) -L regression --timeout 120

# Run a specific regression suite by label
ctest -j$(nproc) -L esbmc --timeout 120          # core C tests
ctest -j$(nproc) -L python --timeout 120          # Python tests
ctest -j$(nproc) -L "esbmc-cpp/cpp" --timeout 120 # C++ tests
ctest -j$(nproc) -L floats --timeout 120          # floating-point tests

# List all available test labels
ctest --print-labels

# Run a single named test
ctest -R "regression/esbmc/00_big_endian_01" --output-on-failure
```

**Important: the Python frontend needs `python3` on `PATH`.** ESBMC's Python frontend invokes `python3` to run `parser.py`. `ast2json` is vendored in the source tree (`src/python-frontend/libs/ast2json`), so it no longer needs to be installed separately for Python regression tests. (`mypy` is an optional extra for type checking.)

**Note: /tmp disk space.** C and C++ runs write nothing to `/tmp`: bundled clang headers, the C++ operational models and the internal libc are registered with `file_operations::filesystemt` and served to clang out of `.rodata` via `esbmc_clang_vfs()` (`src/clang-c-frontend/AST/vfs_adapter.h`). The Python and Solidity frontends extract to `/tmp`, because they fork `python3`/`solc` and a separate process cannot read ESBMC's memory. Clean up after large runs of those suites: `rm -rf /tmp/esbmc*`

`regression/esbmc/bundled_headers_from_vfs` and `regression/esbmc-cpp/cpp/om_source_from_vfs` pin this: the first asserts clang is handed `-isystem /esbmc-vfs/libc/headers` and `-resource-dir /esbmc-vfs/clang`, the second that an OM source location in a counterexample reads `/esbmc-vfs/cpp/vector`. Reintroducing extraction turns those paths back into a temp directory and both fail. Note that asserting the temp directory is *empty* after a run would not work: `tmp_path`'s destructor removes what it created, so a run that extracts and cleans up is indistinguishable from one that never extracted.

Regression test format (`test.desc`): line 1 is `CORE`/`KNOWNBUG`/`FUTURE`/`THOROUGH` (THOROUGH is Linux-only), line 2 is the source file, line 3 is ESBMC flags, line 4+ are expected output regexes. Every PR adds a *pair* of regression tests — see the "Regression tests come in pairs" bullet below.

**Before committing:**

- Always run the project's test suite. If tests fail, fix the failures before committing — never commit broken or untested code.
- **Regression suite cap.** When running the full regression suite, cap the run at **10 minutes** (600000 ms) — pass the timeout to the `Bash` tool's `timeout` parameter, or wrap the invocation with `timeout 10m …`. If the suite cannot complete in 10 minutes, narrow the scope (e.g. run only the affected subset) or ask the user before extending the limit.
- **Regression tests come in pairs, and both must bite.** A PR that changes
  verification behaviour adds **two** regression tests over the same construct:
  one pinning `^VERIFICATION SUCCESSFUL$` and one pinning
  `^VERIFICATION FAILED$`. Adding only the passing half is the single most
  common review rejection on this repo — write the failing counterpart at the
  same time, not after a reviewer asks.
- **Mutation-check every regression test you add.** Revert the source fix, re-run
  each new test, and confirm it *changes verdict*; then restore the fix. A test
  that passes both before and after pins nothing. This bites most often when the
  property never reaches the code you changed — clang already inserted the cast,
  the assertion was constant-folded away, the claim was never generated. Check
  `--show-claims` output before and after: if it is byte-identical, the test is
  not a gate. If no end-to-end test can be made to bite, say so explicitly in the
  PR and name what does pin the change (e.g. a unit test).
- **Lint and typecheck.** Run lint and typecheckers and fix any errors. For Python code, use `pylint`. For C++ code, ensure clang-format compliance (CI enforces this).
- **Cyclomatic complexity.** `python3 scripts/complexity/ccn_report.py --gate` reports what the branch adds against its merge base, the same check the Complexity workflow runs on the PR (needs `pip install lizard==1.23.0`). It is advisory while the thresholds are being calibrated.

## Branching

Before implementing any feature or bug fix, always work on a dedicated branch:

1. Check the current branch — never work directly on `master`.
2. Create a branch with a descriptive name (e.g. `feat/short-description` or `fix/short-description`).
3. Confirm the branch is active before making any changes.

## Code Style

- **C++**: Clang-format (Clang 11), Allman braces, 80-col limit, 2-space indent, no tabs. Config in `.clang-format`.
- **Python**: YAPF, PEP 8 based, 100-col limit. Config in `.style.yapf`.
- Prefer modern C++ idioms (C++11+). Use const-correctness throughout. Prefer stack allocation over heap when possible. Follow existing patterns in the file being modified.
- CI enforces formatting on PRs via GitHub Actions.

## Coding Guidelines

- Write simple, clean, and readable code with minimal indirection.
- Each function should do one thing well. No redundant abstractions or duplicate code.
- Check the entire codebase to reuse existing methods before writing new ones.
- Tests MUST NOT use mocks, patches, or any form of test doubles. Integration tests are preferred.
- After implementation, simplify and clean up the code aggressively — remove unnecessary conditional checks while ensuring correctness.
- Run ESBMC over your solution to formally check that it works and does not introduce new errors.

## Code Comments

Write few comments — favour self-explanatory code (clear names, small functions) over narration. Keep added comments to a minimum in PRs; excess comments are noise reviewers must wade through.

- **Do not** restate what the code plainly does (`i++; // increment i`), label structure (`// constructor`, `// helpers`), narrate the change or its history (`// added null check`, `// was: foo()`), or echo a function/variable name in prose.
- **Do** comment only when it adds what the code cannot convey: a non-obvious *why* (rationale, trade-off, workaround with an issue/PR link), a caveat or invariant a caller must respect, a citation to the C/C++ standard or a solver/algorithm detail, or genuinely subtle logic. One line beats a paragraph.
- Preserve existing meaningful comments and the file's established doc conventions (e.g. Doxygen-style headers where already used). Match the surrounding comment density rather than exceeding it.

## Post-implementation Pass

After implementing any non-trivial coding task, before committing:

1. **Simplify aggressively.** Remove unnecessary conditional checks, dead code, redundant abstractions, duplicate logic. Re-verify the code still works correctly. Apply the same pass to test code.
2. **Verify with ESBMC** when the task touches C/C++ code or ESBMC's own headers/frontends. Use the `esbmc-verifier` agent to confirm the patch works and introduces no new errors. For non-ESBMC tasks (e.g. Python frontend, build scripts), run the project's normal lint/typecheck/test commands.
3. **Code review.** Use the `code-reviewer` agent on the diff. Apply high-confidence findings; explain anything you skip.
4. **Coverage gate.** Run the `esbmc-coverage` agent in Mode B on the diff before opening or updating any PR. Every executable line the diff adds must either be covered by a test in the same PR or triaged with a stated reason (vendored / dead / defensive / broken feature). A BLOCK verdict stops the PR — add the missing tests or re-scope.

## Available Subagents

These specialised agents are configured in `~/.claude/agents/` and should be preferred over ad-hoc Bash invocations when their description fits the task.

- **`esbmc-verifier`** — Recommended formal-verification tool for this repo. Two modes: (A) bug-fixing inside ESBMC's own codebase — inspects GOTO IR (`--goto-functions-only`), VCCs (`--show-vcc`), and the symbol table; applies minimal patches; re-runs ESBMC to confirm `VERIFICATION SUCCESSFUL`; produces a two-tier harness package under `regression/<suite>/github_<N>/` (literal repro), `regression/<suite>/github_<N>-nondet/` (nondet generalisation), and an optional `_fail/` negative variant when the patch shifts a checker boundary. (B) Any external C/C++ codebase (application, library, firmware) — three-phase strategy (language-level safety → functional contracts via k-induction → bug-specific negative proofs) with stub-shadowing for whatever the module depends on (DBs, network, filesystem, hardware/RTOS, vendor SDKs). Invoke for the post-implementation ESBMC step (§Post-implementation Pass #2), for deterministic witnesses when sanitizers cannot reproduce a memory/UB bug (§Regression Tests for Memory/UB Bugs), and when diagnosing unexpected ESBMC results (§Debugging Verification Issues). Defaults to bitwuzla; honours `test.desc` flags when present. For one-shot sanity checks (`esbmc file.c --incremental-bmc`), call `esbmc` directly via Bash instead.
- **`code-reviewer`** — Diff review against the priorities in §Code Review Priorities. Invoke for the post-implementation review step (§Post-implementation Pass #3).
- **`creduce-reducer`** — Reduces C/C++ programs that trigger an ESBMC bug to a minimal reproducer using C-Reduce with property-preserving interestingness scripts. Use when filing or investigating ESBMC bug reports against large inputs.
- **`esbmc-coverage`** — Codecov line coverage of ESBMC's own sources (distinct from `--branch-coverage`/`--cov-report-json`, which measure the program under verification). Mode B is the mandatory per-PR gate of §Post-implementation Pass #4: it scopes to the diff, requires each added executable line to be covered by a test in the same PR or triaged as vendored/dead/defensive/unwired, mutation-checks the PR's new tests, and returns PASS / PASS WITH NOTES / BLOCK without forcing an instrumented rebuild. Mode A runs coverage campaigns — it pulls the per-line uncovered map from the public Codecov API, triages gaps, adds regression and Catch2 tests, and proves the delta with `llvm-cov`. Uncovered lines are also the best source of dead-code candidates for `esbmc-verifier` Mode C.

## Regression Tests for Memory/UB Bugs

When fixing a memory-safety or undefined-behaviour bug in C/C++ code:

1. Before applying the fix, write a regression test that reproduces the bug under sanitizers (ASan, UBSan, or MSan as appropriate; TSan for data races).
2. Compile and run the regression test, and confirm it fails on the unfixed code — either via a clear sanitizer diagnostic or by tripping an embedded `assert` — so the failure mode is reproducible end-to-end, not just inferred.
3. Apply the fix and re-run the compiled test; confirm it now passes cleanly (assertion holds and no sanitizer diagnostic).
4. Skip this step for pure logic bugs, build/config issues, or non-C/C++ work — sanitizers do not apply.

If sanitizers do not reproduce the bug (e.g. timing-dependent races, allocator-dependent use-after-free, MSan without instrumented dependencies, optimisation-dependent UB, or input coverage gaps):

1. Try a different sanitizer (ASan ↔ TSan ↔ MSan ↔ UBSan) and vary build flags (`-O0` vs `-O2`, `_GLIBCXX_DEBUG`, `MALLOC_PERTURB_`, `ASAN_OPTIONS=detect_stack_use_after_return=1`).
2. If still not reproducible under sanitizers, fall back to ESBMC (`esbmc-verifier` agent) to obtain a deterministic witness.
3. As a last resort, write a regression test that reproduces the observable symptom (wrong output, assertion, crash) without relying on a sanitizer diagnostic, and note in the commit message why sanitizer-based reproduction was not feasible.

## Consulting the C/C++ Standard

When a C/C++ change concerns standard-defined semantics — undefined behaviour, implicit conversions, object lifetime, name lookup, overload resolution, constant evaluation, or similar — consult the relevant standard draft (e.g. the latest C or C++ working draft on open-std.org, or cppreference for a digestible summary) before implementing. Cite the section in the commit message or code comment when it clarifies a non-obvious choice. Skip for routine edits that do not depend on standard semantics.

**Use the standard and the compiler together — they answer different
questions.** When a change to an operational model under `src/cpp/library/` or
`src/c2goto/library/` adds or moves a version gate (`#if __cplusplus >= …`, a
`constexpr`/`noexcept` qualifier, a conditionally-declared member), consult
both:

- **The standard** says what the rule is, when it changed, and why — the
  semantics, the paper number, the wording worth citing. Read it first; it is
  what goes in the commit message.
- **The compiler and its C++ library** (`clang++` with libc++ here, libstdc++
  on the Linux CI runners) say what is actually available in a given `-std` mode
  on this target. That is what a user sees when they compile a regression test
  by hand, and what the OM has to reproduce.

They usually agree. Where they do not:

- The implementation offers **more** than the standard requires — libc++ exposes
  `<string_view>` in C++11 — follow the implementation. Code built with that
  toolchain compiles, so rejecting it would be a false `PARSING ERROR` on valid
  input. ESBMC's C++11 `<string_view>` gate is exactly this call (#3387).
- The implementation is **non-conforming** — follow the standard, and leave a
  one-line comment naming the divergence.

Establish the boundary by measurement, never by recall:

1. Write one small probe per behaviour in question.
2. Run each probe through `clang++ -std=<mode> -fsyntax-only` for every mode the
   gate spans — the real library, not the OM.
3. Run the same probes through `esbmc --std <mode>` and diff accept/reject.
   Any cell where they disagree is the defect.
4. When changing an OM header, A/B the header trees directly
   (`clang++ -I <tree>`) — it is seconds, against a full OM rebuild.

Mirror the host library's *shape*, not just its version number: libc++ spells
these `_LIBCPP_CONSTEXPR_SINCE_CXX17`, `_LIBCPP_STD_VER >= N`, and where it
declares a member unconditionally `constexpr` and gates only its callee, do the
same — the accept/reject behaviour and the diagnostic both depend on it. Cite
the paper (e.g. P0426R1) in the commit message; take the boundary from the
implementation.

**Pin the mode in every test that depends on it.** A `test.desc` with a blank
flags line inherits whatever LLVM ESBMC was built against — `gnu++17` for
ESBMC's bundled clang, `gnu++14` for Apple clang. A test relying on that is not
pinning anything. Give it `--std c++NN` and mutation-check the pin: change it to
an older mode and confirm the test *fails*.

## Incremental Patch Testing

When a fix involves multiple patches (e.g. N1, N2), apply and test them one at a time:

1. Apply patch N1, then run the relevant tests to check whether the bug is fixed.
2. If fixed, stop — do not apply further patches.
3. If not fixed, apply patch N2 and test again. Repeat until the bug is resolved or all patches are exhausted.
4. Do not apply all patches at once before testing.

## Code Review Priorities

1. **Critical**: Verification soundness, memory safety, undefined behavior
2. **High**: Logic errors in SMT encoding/symbolic execution, performance regressions, missing tests
3. **Medium**: Code quality, API consistency, documentation gaps
4. **Low**: Minor style if matching surrounding code

## Source Architecture

Key directories under `src/`:

- `esbmc/` — Main entry point and CLI driver
- `irep2/` — Internal representation (IRep2), the core data structure for expressions/types
- `goto-programs/` — GOTO intermediate representation and transformations
- `goto-symex/` — Symbolic execution engine (core verification logic)
- `solvers/` — SMT solver backends (z3, bitwuzla, boolector, cvc4, cvc5, yices, mathsat, smtlib)
- `langapi/` — Language API abstractions shared across frontends
- `pointer-analysis/` — Memory model and pointer safety analysis
- `util/` — Shared utilities and data structures

Frontends (each parses a language into the shared GOTO representation):
- `clang-c-frontend/` — C, CHERI-C, CUDA (via Clang)
- `clang-cpp-frontend/` — C++ (via Clang)
- `python-frontend/` — Python 3.10+ (AST→JSON→IRep2)
- `jimple-frontend/` — Java/Kotlin (via Soot/Jimple)
- `solidity-frontend/` — Solidity smart contracts

Tools:
- `c2goto/` — Converts C operational models to GOTO binaries
- `goto2c/` — Converts GOTO programs back to C

Other top-level directories:
- `unit/` — Catch2 unit tests
- `regression/` — regression test suites (60+ categories)
- `scripts/` — build scripts and CMake modules (`scripts/cmake/`)
- `docs/` — generated documentation
- `website/` — Hugo-based project website

## Debugging Verification Issues

When ESBMC produces an unexpected VERIFICATION FAILED or SUCCESSFUL result, use these techniques:

**1. Inspect the GOTO program** — Use `--goto-functions-only` to dump the intermediate GOTO representation. This reveals exactly what code ESBMC is verifying, including how frontend constructs are lowered:
```sh
esbmc test.py --unwind 9 --goto-functions-only 2>&1 | grep -A50 "python_user_main"
```
Look for the `python_user_main` function to see how Python source maps to GOTO instructions (ASSIGN, FUNCTION_CALL, ASSERT). This is especially useful for catching compile-time optimizations that incorrectly pre-resolve values.

**2. Bisect with simpler test cases** — When a test fails, create variants that isolate the problem.

**3. Read the counterexample trace** — ESBMC's `[Counterexample]` section shows the state at each step. Track field assignments in structs (e.g., `PyObject`'s `.value`, `.type_id`, `.size`) through the trace.

**4. Key files for Python frontend debugging:**
- `src/python-frontend/python_converter.cpp` — Main expression/statement conversion
- `src/python-frontend/python-list/` — List operations (split by concern: construction, mutation, access, query, string ops, comprehension, set ops, type map, type inference)
- `src/python-frontend/function_call_expr.cpp` — Method call handling
- `src/c2goto/library/python/list.c` — C operational model for list operations

**5. Hypothesis tests** — Property-based tests in `unit/python-frontend/` test ESBMC's models against CPython. Run with: `uv run python -m pytest unit/python-frontend/ -v`

## SV-COMP Benchmarking

The `Run Benchexec` workflow is the only measurement of ESBMC's competition score. A number from it is easy to read as a verdict on the PR in front of you when it is nothing of the sort.

**Label the PR.** When a change can move SV-COMP verdicts — anything under `goto-symex/`, `solvers/`, `pointer-analysis/`, `goto-programs/`, a frontend's semantics, an operational model, or `scripts/competitions/svcomp/` — add `needs-svcomp-run` alongside the area label, so it is not merged on the regression suite alone. Add `SV-COMP` when the competition setup itself is what changed.

**Check the provenance of both runs before attributing a score move.** `gh run list --workflow "Run Benchexec"` prints each run's `headBranch`: runs described as "master" are routinely another PR's branch. The timeout and `ESBMC_OPTS` are `workflow_dispatch` inputs, recorded together with the CPU and RAM in the `<result …>` element of every `*.results.*.xml.bz2` in the `esbmc-result` artifact — a 30s run and a 900s run are not comparable, and neither are two different strategies. Master moves daily, so a baseline more than a few days older than the PR run measures master's drift rather than the PR.

**Attribute per task, not per total.** Download both `esbmc-result` artifacts, parse the per-task `status` and `category` out of the XML, and diff the task sets. If the tasks a PR appears to lose are the same ones another branch's run already lost, the PR is not the cause; a third run from the same week on an unrelated branch settles it.

**Read a per-task log before concluding the verifier regressed.** `*.logfiles.zip` in the artifact holds ESBMC's full stdout per task. `VERIFICATION FAILED` followed by `Unknown` is `esbmc-wrapper.py` failing to classify the output, not ESBMC failing to find the bug.

**Changing what ESBMC prints is an interface change.** `parse_result()` in `scripts/competitions/svcomp/esbmc-wrapper.py` classifies each task by matching substrings of ESBMC's output, so a PR that adds, renames, or reformats a verdict line, a property comment, or a summary block must be checked against it — `python3 scripts/competitions/svcomp/test_esbmc_wrapper.py` covers the parsing — and carries `needs-svcomp-run`. PR #7064 added a per-property table listing every property including the unchecked ones; the wrapper read those rows and turned ~2600 correct-false verdicts into `Unknown` (#7250).

## Commit Conventions

Prefix commits with a category tag in brackets, e.g., `[python]`, `[build]`, `[solver]`, `[om]` (operational model). Title: one line, imperative mood, <72 chars. Description: 2–4 lines explaining what changed and why. Reference the relevant issue/PR with `Fixes #N` when applicable.

**Never squash commits.** Always preserve the full commit history — every individual commit must remain intact. Do not use `git merge --squash`, `git rebase` to squash, or any PR merge strategy that collapses commits.

## PR Conventions

- Branch from `master` (the default branch)
- Target PRs to `master`
- Check formatting with clang-format before submitting

## Issue and PR Labels

Always apply at least one label when creating an issue or PR. Pick the label that matches the affected area — e.g. `python`, `clang-c-frontend`, `solver`, `build`, `docs`. Use `gh label list --repo esbmc/esbmc` to see the available labels, then `gh issue edit <N> --add-label <label>` or `gh pr edit <N> --add-label <label>`. If no existing label fits, ask the user rather than creating a new one.

Add `needs-svcomp-run` on top of the area label whenever the change can move competition verdicts — see *SV-COMP Benchmarking* for what qualifies.

For module-specific instructions, subdirectory CLAUDE.md files can be added (they load automatically when working in those directories).
