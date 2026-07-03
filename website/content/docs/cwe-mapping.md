---
title: CWE Mapping
weight: 12
---

ESBMC annotates every reported property violation with the matching Common
Weakness Enumeration (CWE) identifiers. The mapping is pinned to **MITRE CWE
4.20** (published 2024-11-19, <https://cwe.mitre.org/data/index.html>) and only
retains ids whose Vulnerability Mapping Usage is `ALLOWED` or
`ALLOWED-WITH-REVIEW`.

ESBMC currently distinguishes **31 unique CWE identifiers** across **26
violation kinds**: CWE-120, 121, 125, 129, 131, 190, 191, 193, 252, 362, 366,
369, 401, 415, 416, 457, 469, 476, 562, 590, 617, 681, 761, 787, 822, 823, 824,
825, 833, 908, 1335.

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
  `4.20`.

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
| `array bounds violated`                                      | 121, 125, 129, 131, 193, 787                |
| `Access to object out of bounds`                             | 125, 787, 823                               |
| `dereference failure: memset of memory segment`              | 120, 125, 787                               |
| `dereference failure on memcpy: reading memory segment`      | 120, 125, 787                               |
| `Same object violation`                                      | 469                                         |
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
| `unreachable code reached`                                   | 617                                         |
| `dead code` _(advisory; `--dead-code-check` only)_           | 561                                         |
| `recursion unwinding assertion` / `unwinding assertion loop` | _(none — k-bound exceeded, not a weakness)_ |

## Dead code (CWE-561, advisory)

[CWE-561](https://cwe.mitre.org/data/definitions/561.html) (Dead Code) is
`ALLOWED` for vulnerability mapping in CWE 4.20, but ESBMC treats it as an
**advisory** rather than a violation: it is the dual of the CWE-617
reachability check. Where CWE-617 reports that an error location *is* reachable,
dead-code detection reports that a statement is provably *unreachable* under all
inputs. Unlike a compiler's `-Wunreachable-code`, BMC-based detection is sound
under non-trivial guards.

Detection is off by default and enabled with `--dead-code-check`. It reuses the
branch-coverage instrumentation: every conditional branch is probed with a
reachability assertion, and any probe the solver proves unsatisfiable marks that
branch direction as dead. Because it issues one solver query per branch probe,
it can be slow on large programs — hence the default-off gate.

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
$ esbmc dead.c --dead-code-check

[Dead code]

dead.c:9: dead code: unreachable branch [guard: x > 5]
  CWE: CWE-561

VERIFICATION SUCCESSFUL
```

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
