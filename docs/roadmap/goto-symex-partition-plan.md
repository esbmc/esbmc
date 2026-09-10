# Partitioning `src/goto-symex/` into subsystem directories

`src/goto-symex/` holds **51 flat top-level source files** (21 headers, 30
`.cpp`) plus the one existing subdirectory `builtin_functions/` (11 `.cpp`) —
**62 files, 29,677 lines**. This plan partitions the flat level into **seven
named subdirectories**, on the model of
[#6381](https://github.com/esbmc/esbmc/pull/6381), which did the same for
`src/util/` (151 → 9 directories).

The change is a **pure reorganisation**: files move, and the only content edits
anywhere are `#include` path updates. It is not a refactor, and it must not
become one — see §11.

Baseline for every measurement below: `ddc686db93`, macOS/arm64, Ninja,
`RelWithDebInfo`.

---

## 1. Why the flat level is a problem here specifically

`goto-symex` is where verdicts are produced. Three concerns are currently
indistinguishable by location:

* **the symbolic execution engine** (`goto_symext`, ~12.9k lines across 22
  files) — soundness-critical;
* **the SSA equation and the passes over it** — soundness-critical;
* **everything downstream of a verdict** — counterexample rendering, SARIF,
  HTML/JSON reports, GraphML witnesses, CTest/pytest generation (~7.9k lines,
  19 files) — presentation, not soundness.

A reviewer opening a diff cannot tell from the path which of the three they are
in. That is the concrete cost. Directory names fix it at zero runtime risk.

## 2. The proposed partition

| Directory | Contents | Files | Lines |
|---|---|---:|---:|
| `engine/` | `goto_symext` itself: the driver (`symex_main`), the statement handlers (`symex_assign`, `symex_goto`, `symex_function`, `symex_other`, `symex_dereference`, `symex_stack`, `symex_valid_object`), `dynamic_allocation`, `goto_symex.h`, and `builtin_functions/` moved underneath it. | 22 | 12,866 |
| `state/` | The per-thread symbolic state and the SSA naming layer it is built on: `goto_symex_state`, `renaming`. | 4 | 2,167 |
| `scheduler/` | Thread interleaving and the exploration tree: `execution_state`, `reachability_tree`, `reachability_tree_cin`. | 5 | 4,114 |
| `equation/` | The SSA formula and every pass over it: `symex_target`, `symex_target_equation`, `slice`, `symex_symmetry`, `features`. | 10 | 2,539 |
| `trace/` | Counterexample construction and rendering: `goto_trace`, `build_goto_trace`, `printf_formatter`, `xml_goto_trace`, `html`, `json`, `sarif`. | 12 | 3,474 |
| `witness/` | SV-COMP GraphML/YAML witness emission and violation-witness replay: `witnesses`. | 2 | 1,768 |
| `testgen/` | Test-case generation: `ctest`, `pytest`, `test_gen_guard`. | 5 | 2,701 |
| *(top level)* | `symex_invariant.{h,cpp}` — see §3. | 2 | 48 |

`src/goto-symex/` then holds `CMakeLists.txt`, `.gitignore`, and
`symex_invariant.{h,cpp}`.

## 3. Boundary rules

Each rule is objective — checkable by grep, not by taste. That is what makes the
partition reviewable and what keeps it from decaying.

1. **`engine/`** — a file belongs here iff it defines member functions of
   `goto_symext` or a helper called only from them. Every one of the 11
   `builtin_functions/*.cpp` includes `goto-symex/goto_symex.h` and defines
   `goto_symext::` members; it is a sub-partition of `engine/`, so it nests
   there.
2. **`state/`** — the data a single thread's execution carries, and its naming.
   No file here may know about scheduling or about more than one thread.
3. **`scheduler/`** — the only place that knows there is more than one thread.
4. **`equation/`** — a file belongs here iff it produces or transforms
   `symex_target_equationt::SSA_stepst`. `slice`, `symex_symmetry` and
   `features` are all `ssa_step_algorithm` subclasses; this is a mechanical
   test, not a judgement.
5. **`trace/`, `witness/`, `testgen/`** — delimited by a hard rule, the way
   `util/base/` was: **nothing in these three may be reachable from
   `goto_symext::symex_step`.** They run after a verdict exists. A future
   include that violates this is a design error visible at review time.
6. **top level** — `symex_invariant` is the component-wide release-checked
   invariant primitive. It depends on nothing and is depended on by `state/`,
   `equation/` and `src/esbmc/bmc.cpp`. Putting it inside any one subdirectory
   would misstate its scope, so it stays at the top. *(Alternative, if the
   reviewer prefers an empty top level: a `support/` directory holding these two
   files. Either is defensible; the plan does not depend on the choice.)*

## 4. Measured dependency structure

Header include graph over the 21 headers, Tarjan SCC:

**Exactly one cycle exists today: `witnesses.h` ↔ `ctest.h`.**
`witnesses.h:9-10` includes `pytest.h` and `ctest.h`; `ctest.h:9` includes
`"witnesses.h"` (quoted, resolved via the including file's own directory —
which is precisely the include form that breaks silently when a file moves).

Group-level edges **after** the §5 prep (measured on `6103d9ea96`), computed
from headers only:

```
engine    -> state, equation          state     -> equation, witness, <top>
scheduler -> engine, state, equation  testgen   -> equation
trace     -> equation                 witness   -> equation, trace
```

One non-trivial SCC survives: **`equation` ↔ `trace`**. Cause, exactly:
`symex_target_equation.h:111` declares `SSA_stept::type` as
`goto_trace_stept::typet` — the SSA step-kind enum lives in the *trace* header,
so the equation cannot be compiled without it. See §11 for the follow-up.

Two further inversions worth naming, neither introduced here:

* **`state/` → `witness/`.** `goto_symex_state.h:22` includes `witnesses.h` for
  the `waypoint` type (violation-witness replay). The engine's state header
  therefore drags in the witness emitter. `witnesses.h` is three headers in one
  — replay input types, GraphML/YAML output, and (until §5) the test
  generators. Splitting it is a follow-up, not part of this move.
* **`src/util/` → `goto-symex/`, twice.** `util/ssa/algorithms.h` and
  `util/ssa/goto_expr_factory.h` include `symex_target_equation.h`, and
  `util/base/yaml_parser.h:3` includes `<goto-symex/witnesses.h>` — the base
  layer depending upward on this component. #6381 named the `ssa/` half and
  left it; the `yaml_parser.h` half is the reason `yaml_parser.cpp` sits in the
  affected-TU set of any change to `witnesses.h`. Still out of scope.

## 5. Step 0 — prerequisite include repair — **DONE**

Landed as `6103d9ea96`, `[symex] Break the witnesses/ctest header cycle in
goto-symex`, on branch `fix/goto-symex-include-cycle`. Five files, +4/-5.

Three header edges removed, each unjustified by any symbol at the site that
declared it:

* `witnesses.h` included `pytest.h` and `ctest.h` and referenced no symbol from
  either. Removed.
* `ctest.h` included `"witnesses.h"` (quoted, unqualified) and referenced no
  symbol from it. **Removed outright** rather than canonicalised — only
  `ctest.cpp` uses `collect_nondet_values`, so the include moved down to the
  `.cpp`. This is stronger than this plan originally proposed: it deletes the
  `testgen` → `witness` group edge instead of merely respelling it.
* `build_goto_trace.h` included `goto_symex_state.h` and referenced no symbol
  from it. `build_goto_trace.cpp` uses `renaming::renaming_levelt::get_original_name`
  and `symbol_renaming_level::level0`, so `renaming.h` was added there.
  (Deleting the include outright fails to compile — two
  `use of undeclared identifier 'renaming'` errors — so the include moved down,
  it did not disappear.)

Plus two hygiene edits in `witnesses.h`: a mid-file `#include` at global scope
that had split a doc comment from the function it documents was hoisted to the
top, and a missing `<vector>` was added (the header uses `std::vector` at six
sites and had been reaching it only transitively — a margin this patch thins).

**Result, measured:** `src/` now has **no multi-node header include cycle at
all** — the `ctest.h` ↔ `witnesses.h` pair was the only one in the whole tree,
not just in `goto-symex/`. 44 translation units stop parsing `ctest.h` and
`pytest.h`.

**Gates discharged.** Full `ninja` green; unit tests 835/836; regression
`-L esbmc/ -j2` 2145/2146 (both failures — *"an exhausted stack still reports
itself"* and `bundled_headers_from_vfs` — pre-existing on clean `master` on this
host); `drift_check.py` exit 0; complexity gate +0; clang-format 11 clean vs
`origin/master`. Independently reviewed: a per-TU preprocessed header-closure
diff over all 719 compilable TUs found **46 TUs changed, 0 gaining any header, 0
losing any system, boost, yaml-cpp or libc++ header**, and confirmed that
neither `ctest.h` nor `pytest.h` declares a template, specialization, free
operator, or anything in `namespace std` — so no silent ODR exposure hides
behind the green build.

No Mode C obligation arises: the change adds and removes zero branches.

## 6. Execution order

Seven move commits, in dependency order so each one leaves the tree buildable,
plus the prep. **Every commit builds green before the next begins** — this is
what made #6381 reviewable, and it is not optional here.

| # | Commit | Rationale for the position |
|---|---|---|
| 0 | Prep: the §5 include repair | **Done** — `6103d9ea96`. |
| 1 | `testgen/` | Leaf after step 0; nothing in `goto-symex` depends on it. |
| 2 | `witness/` | Depends only on `trace`/`equation`, unmoved at this point. |
| 3 | `trace/` | — |
| 4 | `equation/` | — |
| 5 | `state/` | — |
| 6 | `scheduler/` | — |
| 7 | `engine/` (incl. `builtin_functions/` → `engine/builtin_functions/`) | Largest; last, so a conflict here costs the least re-work. |
| 8 | clang-format 11 reflow of the moved files | **Not foreseen when this plan was written** — see §6.1. |

Each commit is `git mv` + include-path rewrite + `CMakeLists.txt` path update.
No commit changes a declaration, a definition, a build flag, or a compiler
option.

**Executed** on branch `refactor/goto-symex-partition`: commits 1–7 as tabled,
each building green before the next began, plus commit 8 below. The measured
post-move group graph is exactly the one §4 predicts — `engine -> state,
equation`; `scheduler -> engine, state, equation`; `state -> equation, witness,
<top>`; `equation -> trace`; `trace -> equation`; `witness -> equation, trace`;
`testgen -> equation` — so the two pre-existing inversions §11.1 and §11.2 name
are the only edges that cross the §3 rule-5 boundary, and this PR adds neither.

### 6.1 Commit 8, and why the plan needed it

The plan asserted the change would be include-path edits alone. It is not,
because of how the code-style gate measures a move.
`.github/workflows/ci-pull-request.yml`'s `code-style` job runs
`git-clang-format --diff origin/<base>` over the PR's changed files, and
`git-clang-format` computes its changed-line ranges from a diff taken **without
rename detection**. A moved file is therefore an added file, every line of it is
in range, and clang-format 11 is asked to format the whole thing — surfacing
formatting the file had drifted into under a *newer* clang-format and that no
previous PR had touched a changed line of.

27 of the 62 moved files are affected. Commit 8 applies exactly that reflow,
kept separate so commits 1–7 remain reviewable as pure renames. G2 below is
stated over tokens rather than lines for the same reason.

#6381 did not hit this only because none of the 151 files it moved had drifted.
Any future partition PR should expect the extra commit.

## 7. The mechanical rewrite, and its three traps

For a group with files `F` moving to `goto-symex/<G>/`:

```sh
git mv <files> src/goto-symex/<G>/
# rewrite every include of the moved headers, repo-wide
git ls-files -- '*.cpp' '*.h' '*.hpp' | xargs sed -i '' \
  -e 's|goto-symex/<name>\.h|goto-symex/<G>/<name>.h|g'
```

Three ways this goes wrong silently:

1. **Quoted, path-qualified includes.** `src/util/ssa/goto_expr_factory.h:3`
   spells it `#include "goto-symex/symex_target_equation.h"` — quotes, not angle
   brackets. A rewrite keyed on `include <goto-symex/` misses it. Match on the
   *path*, never on the bracket form. (This is the same class of defect #6381
   found four instances of in `src/util`.)
2. **Quoted, unqualified includes.** `ctest.h:9`'s `#include "witnesses.h"`
   resolves through the including file's own directory and breaks the moment the
   two files stop being siblings. Step 0 removes the only instance; the
   post-move gate re-checks that none has reappeared.
3. **`goto-symex/` is not on the include path.** Only `${CMAKE_SOURCE_DIR}/src`
   and `${CMAKE_BINARY_DIR}/src` are (`src/CMakeLists.txt:5-6`), so
   `<goto-symex/sub/x.h>` is the sole route to a moved header and no basename
   collision with LLVM or Boost is possible. This is the property that makes the
   rewrite safe; it must be re-asserted, not assumed, if the include dirs ever
   change.

## 8. Verification gates

A commit lands only with all of these discharged. Gates G1 and G2 are the ones
that make this a *proof* rather than a green build.

* **G1 — include-resolution equality.** `ninja -t deps` exposes the fully
  resolved (translation unit → header) graph. Snapshot it before the first move
  (844,488 pairs over 769 TUs as measured on this branch), apply the rename map
  — to the object paths as well as the header paths, since a moved source moves
  its object — to the snapshot, and diff against the post-move graph. The
  required result is **0 lost, 0 gained**, and it held at every one of the eight
  commits: every include in the project resolves to the same header *content*
  as on `master`. This is the only check that catches an include quietly
  resolving to a different file.
* **G2 — content proof.** A scripted assertion that, for every C/C++ file the
  branch touches, the comment-free **token stream** — with the inserted
  `goto-symex/<group>/` path segment undone — is identical to the same file on
  the base revision. Run per commit; a single differing token fails the gate.
  Tokens rather than lines because commit 8 (§6.1) reflows, and comment-free
  because path citations inside comments move with the files. As run on this
  branch: 96 files proved identical, and the gate was mutation-checked by
  flipping one `==` to `!=` in `equation/slice.cpp` and confirming it fails.
* **G3 — unit tests.** `ctest -LE regression --timeout 60`, all green. The 16
  `unit/goto-symex/*.test.cpp` plus `symex_run.h` and `ssa_validator.h` include
  these headers directly and are the closest thing to a compile-level contract
  test for the component.
* **G4 — regression subset.** `ctest -L 'esbmc/' -j2 --timeout 45`, capped at
  10 minutes, run **alone** (concurrent ctest runs share `/tmp/esbmc*` and
  invent failures). Compare against a same-tree `master` run, not against
  memory: `bundled_headers_from_vfs`, `cbmc_fpclassify`, `ra-log-nan`,
  `ra-pow-nan` and the two host-`libstdc++` C++ tests fail on clean `master`
  here.
* **G5 — harness drift.** `python3 scripts/verification/symex/drift_check.py`
  must exit 0. This is a **required CI job**
  (`.github/workflows/pull_request.yml:225`) and it *will* fail on the `state/`
  commit: `regression/esbmc/symex_ssa_00/symex_ssa_00.c:4,6` pin
  `src/goto-symex/renaming.cpp::renaming::level2t::{make_assignment,coveredinbees}`
  by path, and `region_digest()` raises `cited file does not exist` when the
  path is stale. Fix by editing the two `SYMEX-HARNESS-TARGET:` lines. **Do not
  run `--update`**: the recorded sha256 is taken over the definition text alone,
  so a pure move leaves it unchanged, and refreshing it would mask a real drift
  if one were present.
* **G6 — formatting.** `scripts/install-format-hook.sh --fix` (clang-format 11;
  a Homebrew clang-format disagrees and misses CI failures).
* **G7 — complexity gate.** `python3 scripts/complexity/ccn_report.py --gate`.
  No function changes, so the delta must be empty. `ccn_report.py` classifies by
  `rel.startswith("src/")`, so the moved files stay in the `core` bucket — no
  script change needed.
* **G8 — SV-COMP.** Not applicable. Nothing here changes what ESBMC prints, so
  `parse_result()` in `esbmc-wrapper.py` is unaffected and the PR does not carry
  `needs-svcomp-run`. State this explicitly in the PR rather than leaving it
  inferred.

**As run on `refactor/goto-symex-partition`.** G1 0 lost / 0 gained at all eight
commits; G2 96 files token-identical, mutation-checked; G3 849/850 (the failure
is *"an exhausted stack still reports itself"*, red on clean `master` on this
host); G4 2164 tests, one failure — `bundled_headers_from_vfs`, likewise red on
clean `master` here; G5 exit 0 with the recorded sha256 untouched; G6 reports
*"clang-format did not modify any files"* after commit 8; G7 delta `+0` in all three
partitions; G8 not applicable. No Mode C obligation: the change adds and removes
zero branches.

## 9. References to update outside `src/goto-symex/`

**Includes (must change, or the build breaks):** `src/esbmc/bmc.{h,cpp}`,
`src/esbmc/parseoptions/{bmc_strategy,command_line_options,driver,goto_program,k_induction,process_goto_program}.cpp`,
`src/util/base/yaml_parser.h`, `src/util/ssa/algorithms.h`,
`src/util/ssa/goto_expr_factory.h` *(quoted form — see §7)*, and the 16
`unit/goto-symex/` files including `symex_run.h` and `ssa_validator.h`.

**Path citations in non-code (must change, or a tool breaks):**
`regression/esbmc/symex_ssa_00/symex_ssa_00.c` (G5).

**Prose citations (should change):** `website/content/docs/theory/LTL.md:222,243`
— these cite `file:line`, so the line numbers are stale regardless; update the
paths and re-derive the lines. `src/goto-programs/goto_check_excessive_alloc.cpp:32`,
`src/goto-programs/contracts/contracts.cpp:2646`,
`src/pointer-analysis/dereference.cpp:2503`, `src/irep2/irep2_expr.h:52`,
`src/irep2/README.md`, `src/solvers/README.md`, `CLAUDE.md`/`AGENTS.md`.

**Deliberately left stale:** `docs/roadmap/*.md` other than this file. Those
records describe the tree as it was when written; rewriting them would falsify
the record. Same policy as #6381.

## 10. Known limitations

* **`__FILE__` changes in every moved `.cpp`**, so `log_*` output at debug
  verbosity and `assert()` messages show the new paths. Measured: **zero**
  `regression/**/test.desc` expectations and zero `unit/` assertions match a
  `goto-symex` source path or basename, so nothing pins them. The binary is
  therefore **not** byte-identical and this plan does not claim it is.
* **Codecov per-file history resets for moved files.** Codecov keys coverage by
  path. The `esbmc-coverage-pr` Mode B bar "no touched file's coverage falling
  below its pre-PR value" has no pre-PR value to compare against for 62 renamed
  paths. Handle it explicitly: quote the *repo total* before and after (it must
  not move — no executable line changes), and say in the PR that per-file
  baselines are re-established rather than regressed. Do not report the gate as
  passing on an unmeasured file.
* **This conflicts aggressively with any in-flight PR touching
  `src/goto-symex/`.** As of 2026-09-09 the open set is #7669, #7668 and #7665
  (`goto_symex_state.{h,cpp}`, `renaming.h` — commit 5), #7505
  (`goto_symex_state.cpp`, `unit/goto-symex/`) and #6687
  (`builtin_functions/object_size.cpp` — commit 7); #7661-#7663 restructure
  `src/esbmc/`, which is the largest external consumer of these headers.
  Re-derive the list with `gh pr list --repo esbmc/esbmc` immediately before
  starting, and check for a concurrent session on the same branches.
* **`esbmc/esbmc` merges by squash**, so the seven-commit history documented in
  §6 survives on the branch and in review, but not on `master`. The staged
  commits are for reviewability and bisection during development; the PR
  description must therefore carry the group table itself.

## 11. Out of scope — named, not fixed

Each of these is a real defect found while measuring. None belongs in a move
PR; each is a separate, small, testable change.

1. **`SSA_stept::type` is `goto_trace_stept::typet`.** The SSA step-kind enum
   belongs in `symex_target.h`, which both `equation/` and `trace/` already
   depend on and which depends on nothing. Hoisting it removes the last group
   cycle. Mechanical but wide: every `goto_trace_stept::ASSERT`-style reference
   changes.
2. **`witnesses.h` is two headers.** Splitting the violation-witness replay
   types (`waypoint`) from the GraphML/YAML emitters would remove
   `state/` → `witness/` and stop the engine's state header from pulling in
   `boost/property_tree` and `yaml-cpp` into 44 translation units. Step 0
   removed the third role (the test generators); this is the remainder.
3. **`src/util/` depends upward on `goto-symex/`.** `util/ssa/{algorithms,
   cache, goto_expr_factory}` are not utilities; #6381 named this and deferred
   it because it changes CMake target topology. `util/base/yaml_parser.h`'s
   include of `witnesses.h` is the same inversion in a second place. The
   natural sequel to this PR, not part of it.
4. **`goto_symex.h` is 66 KB in one file.** Splitting the `goto_symext`
   declaration is a real improvement and is emphatically *not* a move — it
   changes what each translation unit sees. Separate PR, separate risk.
