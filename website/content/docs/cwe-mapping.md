---
title: CWE Mapping
weight: 12
---

ESBMC annotates every reported property violation with the matching Common
Weakness Enumeration (CWE) identifiers. The mapping is pinned to **MITRE CWE
4.20** (published 2024-11-19, <https://cwe.mitre.org/data/index.html>) and only
retains ids whose Vulnerability Mapping Usage is `ALLOWED` or
`ALLOWED-WITH-REVIEW`.

ESBMC currently distinguishes **36 unique CWE identifiers** across **30
violation kinds**: CWE-120, 121, 122, 125, 129, 131, 190, 191, 193, 252, 362,
366, 369, 401, 415, 416, 457, 469, 476, 562, 563, 590, 617, 674, 681, 761, 787,
789, 822, 823, 824, 825, 833, 835, 908, 1335. One of these — CWE-563 — is an
*advisory* rather than a property violation; see [Advisories](#advisories)
below.

In addition, one **advisory** identifier — CWE-561 (Dead Code) — is emitted
only under `--dead-code-check` (see [Dead code](#dead-code-cwe-561-advisory)
below). Advisory findings are informational: they never flip the verdict to
`FAILED`.

The CWE ids appear in:

- the textual counterexample, on a `CWE: CWE-476, CWE-125` line immediately
  after the violated-property comment;
- the JSON trace (`--output-json`), as an `assertion.cwe` array of integers;
- the GraphML violation witness, as a `<data key="cwe">` node on the violation
  node;
- the SARIF report (`--sarif-output <path|->`), as `result.taxa[]` references
  into a `runs[].taxonomies` block whose `name` is `CWE` and `version` is
  `4.20`. Violations are emitted at `result.level = "error"`; advisories at
  `result.level = "note"`.

## Mapping table

The mapping is implemented in `src/util/cwe_mapping.cpp` as a first-match-wins
substring table ordered longest-substring-first.

| ESBMC violation comment substring                            | CWE ids                                     |
| ------------------------------------------------------------ | ------------------------------------------- |
| `dereference failure: NULL pointer`                          | 476                                         |
| `dereference failure: invalid pointer freed`                 | 415, 416, 590, 761, 825                     |
| `dereference failure: invalidated dynamic object freed`      | 415, 416, 590, 761, 825                     |
| `dereference failure: invalidated dynamic object`            | 416, 825                                    |
| `dereference failure: accessed expired variable pointer`     | 416, 562, 825                               |
| `dereference failure: invalid pointer`                       | 416, 822, 824, 908                          |
| `dereference failure: free() of non-dynamic memory`          | 590, 761                                    |
| `Operand of free must have zero pointer offset`              | 590, 761                                    |
| `dereference failure: forgotten memory`                      | 401                                         |
| `array bounds violated: heap object`                         | 122, 125, 129, 131, 193, 787                |
| `array bounds violated`                                      | 121, 125, 129, 131, 193, 787                |
| `Access to object out of bounds: heap object`                | 122, 125, 787, 823                          |
| `Access to object out of bounds`                             | 125, 787, 823                               |
| `dereference failure: memset of memory segment`              | 120, 125, 787                               |
| `dereference failure on memcpy: reading memory segment`      | 120, 125, 787                               |
| `Relational comparison between pointers is only valid for pointers to the same object` | 469 |
| `division by zero`                                           | 369                                         |
| `NaN on`                                                     | 681                                         |
| `arithmetic overflow`                                        | 190, 191                                    |
| `Cast arithmetic overflow`                                   | 190, 191                                    |
| `undefined behavior on shift operation`                      | 1335                                        |
| `atomicity violation`                                        | 362, 366                                    |
| `data race on`                                               | 362, 366                                    |
| `Deadlocked state`                                           | 833                                         |
| `use of uninitialized variable`                              | 457                                         |
| `unchecked return value`                                     | 252                                         |
| `excessive allocation size`                                  | 789                                         |
| `unreachable code reached`                                   | 617                                         |
| `non-terminating execution`                                  | 835                                         |
| `dead store` _(advisory; `--dead-store-check`)_              | 563                                         |
| `uncontrolled recursion in <function>`                       | 674                                         |
| `recursion unwinding assertion` / `unwinding assertion loop` | _(none — k-bound exceeded, not a weakness)_ |

CWE-561 (Dead Code) is **not** in this substring table. It is advisory and
emitted only by the `--dead-code-check` reporter through a dedicated rule
(`dead_code_cwe_rule()`), so an ordinary violation whose comment happens to
contain the text "dead code" is never mislabelled as CWE-561. See
[Dead code](#dead-code-cwe-561-advisory) below.

The last two rows distinguish two different recursion outcomes:

- **`uncontrolled recursion in <function>`** (CWE-674) is emitted when symex
  proves a recursive function has *no reachable base case* — every path to a
  return goes through a recursive self-call, so the recursion is genuinely
  unbounded. This is a real weakness. The analysis is structural (it treats a
  direct self-call as an impassable edge in the function CFG) and therefore
  sound but incomplete: it never relabels a recursion that has any
  non-recursive return path, so terminating recursion is never mislabelled.
- **`recursion unwinding assertion`** (unmapped) is the pre-existing outcome
  when the recursion merely exceeds the unwind bound. A recursion that
  terminates but is deeper than the bound keeps this comment and no CWE, since
  it signals insufficient unwinding rather than a weakness.

### Heap vs. stack out-of-bounds

When the overflowed object is a `malloc`/`calloc`/`realloc` allocation, the
symbolic-execution dereference code (`src/pointer-analysis/dereference.cpp`)
appends `: heap object` to the bounds-violation comment. The heap variants swap
**CWE-121** (Stack-based Buffer Overflow) for **CWE-122** (Heap-based Buffer
Overflow) and, being strict superstrings of the generic comments, win the
longest-substring-first match. `alloca`, which lives on the stack, keeps the
generic (stack) mapping. Compile-time array bounds checks
(`src/goto-programs/goto_check.cpp`) only fire on lexical arrays and never see
heap objects, so they are unchanged.

As with the generic bounds entries, the heap variants do not distinguish reads
from writes — the CWE list keeps both CWE-125 (Out-of-bounds Read) and CWE-787
(Out-of-bounds Write), so a heap OOB read is also annotated with CWE-122.

### Non-termination (CWE-835)

The `--termination` strategy refutes the termination property by proving a
loop's exit condition unreachable (via k-induction or a recurrent set). This
is [CWE-835](https://cwe.mitre.org/data/definitions/835.html), "Loop with
Unreachable Exit Condition ('Infinite Loop')". Unlike the property violations
above, a non-termination verdict is proven by UNSAT and therefore has **no
counterexample trace**, so ESBMC anchors the CWE annotation to the loop's exit
marker. Markers inside ESBMC's own library helpers — such as the
`while (atexit_count > 0)` loop in `__ESBMC_atexit_handler`, which is linked
into every program — rank below markers in user code, so the reported location
never points into ESBMC's installed sources. The annotation still reaches the
text output (the `CWE: CWE-835` line
follows the `... non-terminating execution` verdict) and the SARIF, JSON and
GraphML outputs, exactly as it does for any other violation kind. (The YAML
witness format has no CWE field, so it is unaffected.) Unwinding-assertion
failures remain intentionally unmapped — they signal an insufficient k-bound,
not a weakness.

### Excessive allocation size (CWE-789)

The `excessive allocation size` row is produced only by the opt-in
`--excessive-alloc-check[=K]` flag, which inserts an
`ASSERT(size <= K)` before every `malloc`/`calloc`/`realloc`/`operator new[]`
(default `K` = 1 MiB). The bound `K` is a **policy** choice, not a soundness
property: a violation proves "a path reaches an allocation with size > K", not
"memory can be exhausted" (undecidable in general). The assertion precedes the
allocation, so an excessive size is still reported under
`--force-malloc-success`. CWE-770 is the class-level entry for unbounded
allocation; ESBMC maps to the CWE-789 variant.

A typed request such as `malloc(sizeof(T))` is lowered to an element count of
one with element type `T`, so the check scales by `sizeof(T)`; a byte-count
request (`malloc(n)`, `malloc(n * sizeof(T))`) keeps a `char` element type and
is compared directly. `calloc` and other allocators modelled by an operational
model (e.g. `strdup`) are covered by instrumenting their model bodies, so a
violation there is reported at the model's internal `malloc` site rather than
the user call site, and a library routine that allocates an input-sized buffer
can raise a genuine but library-located CWE-789.

## Advisories

Some CWEs describe code-quality signals rather than property violations. ESBMC
surfaces these as **advisories**: they are opt-in, note-level, and never change
the verification verdict.

### CWE-563 — Assignment to Variable without Use (`--dead-store-check`)

With `--dead-store-check`, ESBMC runs an intra-procedural backward
live-variable analysis over each function's GOTO control-flow graph and reports
every plain assignment to a scalar local whose written value is never read on
any subsequent path — a *dead store*. For example, in

```c
int x = 5;
x = 6;
return x;
```

the `x = 5` store is dead. ESBMC emits:

```
file.c:1: dead store: assignment to x never read
  CWE: CWE-563
```

Only automatic-storage, non-`extern`, non-address-taken, non-`volatile` scalar
locals are considered; excluding address-taken variables keeps the analysis
sound without an alias analysis, and a `volatile` write is an observable side
effect (C11 §5.1.2.3), never a dead store. Advisories are additive: the verdict,
existing regressions, and the SV-COMP wrapper are unaffected when the flag is
off. In SARIF the dead store appears as a `result.level = "note"` under rule id
`dead-store`, with CWE-563 in `taxa`; it is emitted from a verdict-independent
point so it reaches SARIF even under `--result-only` / a suppressed
counterexample. Functions that use exceptions (`throw`/`catch`) are skipped:
this pass runs before exception lowering, so the handler edges are not yet in
the CFG and a value read only in a `catch` would be misreported. Reporting is
restricted to user source (system-header and operational-model locations are
excluded by a best-effort path heuristic). Inter-procedural dead-store
detection is a future extension.

## Dead code (CWE-561, advisory)

[CWE-561](https://cwe.mitre.org/data/definitions/561.html) (Dead Code) is
`ALLOWED` for vulnerability mapping in CWE 4.20, but ESBMC treats it as an
**advisory** rather than a violation: it is the dual of the CWE-617
reachability check. Where CWE-617 reports that an error location *is* reachable,
dead-code detection reports that a statement is provably *unreachable* under all
inputs. Unlike a compiler's `-Wunreachable-code`, BMC-based detection is sound
under non-trivial guards.

Detection is off by default and enabled with `--dead-code-check`. It reuses the
branch-coverage instrumentation: each conditional branch is probed with a
reachability assertion `assert(c)` per direction, and a probe the solver never
violates marks the *opposite* direction, `!c`, as dead — that is the guard
named in the finding. Because it issues one solver query per branch probe, it
can be slow on large programs — hence the default-off gate.

Findings are reported as:

- a `[Dead code]` section in the textual output, one `CWE: CWE-561` line per
  dead branch;
- SARIF results with `result.level == "note"` (advisory, not `error`) and a
  `result.taxa[]` reference to CWE-561, under the same `CWE` taxonomy.

The verdict is **not** affected: a run with `--dead-code-check` reports
`VERIFICATION SUCCESSFUL` regardless of how many dead branches are found, so the
`FALSE_*` / `TRUE` classification used by the SV-COMP wrapper is preserved.
Soundness is bounded by the unwinding depth, as with all BMC results; use a
sufficient `--unwind` for programs with loops. The mode is a standalone
base-case analysis and cannot be combined with the k-induction / incremental
strategies (`--k-induction`, `--incremental-bmc`, `--falsification`,
`--termination`, `--loop-invariant`) or `--multi-fail-fast`. As with the other
coverage modes, instrumentation is keyed off the source locations of the input
translation units, so it has no effect on a pre-compiled `.goto` binary.

```
$ cat dead.c
int f(void) { return 1; }

int main(void)
{
  int x = 5;
  if (x > 10)
    f();
  return 0;
}

$ esbmc dead.c --dead-code-check

[Dead code]

The following branches are unreachable up to the current unwinding bound:
dead.c:6: dead code: unreachable branch [guard: x > 10]
  CWE: CWE-561

VERIFICATION SUCCESSFUL
```

### Scope and limitations

- **Branch directions only.** The analysis probes the two directions of each
  `if`/loop guard. Canonical CWE-561 shapes with no branch guard — statements
  after an unconditional `return`/`abort`, or entirely unreferenced functions —
  are *not* detected; they simply report no dead code.
- **Only user-written branches are probed.** A guard already folded to a
  constant (`if (1)`) is not probed, and neither are the branches the Python
  frontend inserts on its own behalf — the `IndexError` / `KeyError` /
  `ValueError` / `ZeroDivisionError` guards and the exception-propagation edge
  after each call — so a clean program reports no dead code. The
  constant-evaluation fold that replaces a provable Python `assert` with
  `assert True` is disabled under this flag, so code reachable only through a
  call inside that assertion is still explored.
- **Bounded, and beyond-bound branches read as dead.** A branch that becomes
  reachable only past the unwinding bound is cut like any other beyond-bound
  path and is therefore listed as unreachable — the report is explicitly scoped
  "up to the current unwinding bound", not an absolute proof. Raise `--unwind`
  for loop-bearing programs.
- **Completeness depends on every probe solving.** A finding is "this branch's
  probe was never satisfiable". A solver *error* aborts the run (non-zero exit,
  not a false SUCCESSFUL). A per-claim `--timeout`, however, leaves a probe
  unsolved and that branch would then be listed as dead; do not pair
  `--dead-code-check` with a per-claim timeout if the finding set must be exact.
- **Concurrent programs are explored exhaustively.** A branch can be reachable
  under one thread interleaving and not another, so on concurrent input the mode
  explores *every* interleaving before reporting, rather than stopping at the
  first one as a normal run does. Probe reachability accumulates across them and
  a single advisory is emitted at the end. This is what keeps a branch that only
  a later ordering reaches from being reported as dead, but it does mean a
  `--dead-code-check` run on a heavily concurrent program costs considerably more
  than a plain verification run.
- **The verdict is not a safety verdict.** The branch-coverage instrumentation
  this mode reuses replaces every pre-existing assertion with `assert(true)`
  before symbolic execution, so ESBMC's ordinary checks — including the bounds
  and division-by-zero checks that are on by default — are neutralised. A
  program with a real array-bounds violation reports `VERIFICATION SUCCESSFUL`
  and exits 0 under `--dead-code-check`, where a plain run reports
  `VERIFICATION FAILED`. This is inherited from the coverage modes
  (`--branch-coverage` behaves identically). Run `--dead-code-check` as a
  separate advisory pass, never as a substitute for a verification run. The
  combinations that would instead inject *new* claims during symbolic execution
  (`--memory-leak-check`, `--deadlock-check`, `--data-races-check[-only]`) are
  rejected up front rather than silently masked.

## Ids dropped vs. published mappings

The mapping derives from Table 4 of Sousa et al., _"Finding Software
Vulnerabilities in Open-Source C Projects via Bounded Model Checking"_,
arxiv:2311.05281, with the following adjustments to comply with CWE 4.20's
Vulnerability Mapping Notes:

| Dropped | Status      | Why                                                |
| ------- | ----------- | -------------------------------------------------- |
| 391     | PROHIBITED  | Slated for deprecation; CWE-476 is sufficient.     |
| 119     | DISCOURAGED | Class-level; use children 125 / 787 instead.       |
| 788     | DISCOURAGED | Slated for deprecation; use 125 / 787.             |
| 690     | DISCOURAGED | Chain entry; map to 252 + 476 separately.          |
| 20      | DISCOURAGED | Class-level; ESBMC has no taint signal to back it. |
| 682     | DISCOURAGED | Pillar-level; too abstract.                        |
| 755     | DISCOURAGED | Class-level.                                       |
| 664     | DISCOURAGED | Pillar-level.                                      |

## SARIF output

`--sarif-output <path>` writes a SARIF 2.1.0 document. `-` writes to stdout. The
schema reference is

```
https://docs.oasis-open.org/sarif/sarif/v2.1.0/cs01/schemas/sarif-schema-2.1.0.json
```

A minimal example for a NULL-pointer dereference:

```json
{
  "version": "2.1.0",
  "runs": [{
    "tool": { "driver": {
      "name": "ESBMC", "version": "8.2.0",
      "rules": [{
        "id": "null-pointer-dereference",
        "name": "NULL pointer dereference",
        "properties": { "tags": ["external/cwe/cwe-476"] }
      }]
    }},
    "taxonomies": [{
      "name": "CWE", "organization": "MITRE", "version": "4.20",
      "informationUri": "https://cwe.mitre.org/",
      "taxa": [{
        "id": "476", "name": "NULL Pointer Dereference",
        "helpUri": "https://cwe.mitre.org/data/definitions/476.html"
      }]
    }],
    "results": [{
      "ruleId": "null-pointer-dereference",
      "level": "error",
      "message": { "text": "dereference failure: NULL pointer" },
      "locations": [...],
      "taxa": [{ "id": "476", "toolComponent": { "name": "CWE" } }]
    }]
  }]
}
```

## SV-COMP wrapper compatibility

The CWE annotations are additive: the freeform violation comment strings (e.g.
`dereference failure: NULL pointer`) are unchanged, so
`scripts/competitions/svcomp/esbmc-wrapper.py` continues to classify results by
substring as before.
