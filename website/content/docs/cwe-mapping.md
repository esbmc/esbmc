---
title: CWE Mapping
weight: 12
---

ESBMC annotates every reported property violation with the matching
Common Weakness Enumeration (CWE) identifiers. The mapping is pinned
to **MITRE CWE 4.20** (published 2024-11-19,
<https://cwe.mitre.org/data/index.html>) and only retains ids whose
Vulnerability Mapping Usage is `ALLOWED` or `ALLOWED-WITH-REVIEW`.

ESBMC currently distinguishes **30 unique CWE identifiers** across
**25 violation kinds**: CWE-120, 121, 125, 129, 131, 190, 191, 193,
362, 366, 369, 401, 415, 416, 457, 469, 476, 562, 590, 617, 681,
761, 787, 822, 823, 824, 825, 833, 908, 1335.

The CWE ids appear in:

- the textual counterexample, on a `CWE: CWE-476, CWE-125` line
  immediately after the violated-property comment;
- the JSON trace (`--output-json`), as an `assertion.cwe` array of
  integers;
- the GraphML violation witness, as a `<data key="cwe">` node on the
  violation node;
- the SARIF report (`--sarif-output <path|->`), as `result.taxa[]`
  references into a `runs[].taxonomies` block whose `name` is `CWE`
  and `version` is `4.20`.

## Mapping table

The mapping is implemented in `src/util/cwe_mapping.cpp` as a
first-match-wins substring table ordered longest-substring-first.

| ESBMC violation comment substring                              | CWE ids                           |
|----------------------------------------------------------------|-----------------------------------|
| `dereference failure: NULL pointer`                            | 476                               |
| `dereference failure: invalid pointer freed`                   | 415, 416, 590, 761, 825           |
| `dereference failure: invalidated dynamic object freed`        | 415, 416, 590, 761, 825           |
| `dereference failure: invalidated dynamic object`              | 416, 825                          |
| `dereference failure: accessed expired variable pointer`       | 416, 562, 825                     |
| `dereference failure: invalid pointer`                         | 416, 822, 824, 908                |
| `dereference failure: free() of non-dynamic memory`            | 590, 761                          |
| `Operand of free must have zero pointer offset`                | 590, 761                          |
| `dereference failure: forgotten memory`                        | 401                               |
| `array bounds violated`                                        | 121, 125, 129, 131, 193, 787      |
| `Access to object out of bounds`                               | 125, 787, 823                     |
| `dereference failure: memset of memory segment`                | 120, 125, 787                     |
| `dereference failure on memcpy: reading memory segment`        | 120, 125, 787                     |
| `Same object violation`                                        | 469                               |
| `division by zero`                                             | 369                               |
| `NaN on`                                                       | 681                               |
| `arithmetic overflow`                                          | 190, 191                          |
| `Cast arithmetic overflow`                                     | 190, 191                          |
| `undefined behavior on shift operation`                        | 1335                              |
| `atomicity violation`                                          | 362, 366                          |
| `data race on`                                                 | 362, 366                          |
| `Deadlocked state`                                             | 833                               |
| `use of uninitialized variable`                                | 457                               |
| `unreachable code reached`                                     | 617                               |
| `recursion unwinding assertion` / `unwinding assertion loop`   | *(none — k-bound exceeded, not a weakness)* |

## Ids dropped vs. published mappings

The mapping derives from Table 4 of Sousa et al.,
*"Finding Software Vulnerabilities in Open-Source C Projects via
Bounded Model Checking"*, arxiv:2311.05281, with the following
adjustments to comply with CWE 4.20's Vulnerability Mapping Notes:

| Dropped | Status      | Why                                                |
|---------|-------------|----------------------------------------------------|
| 391     | PROHIBITED  | Slated for deprecation; CWE-476 is sufficient.     |
| 119     | DISCOURAGED | Class-level; use children 125 / 787 instead.       |
| 788     | DISCOURAGED | Slated for deprecation; use 125 / 787.             |
| 690     | DISCOURAGED | Chain entry; map to 252 + 476 separately.          |
| 20      | DISCOURAGED | Class-level; ESBMC has no taint signal to back it. |
| 682     | DISCOURAGED | Pillar-level; too abstract.                        |
| 755     | DISCOURAGED | Class-level.                                       |
| 664     | DISCOURAGED | Pillar-level.                                      |

## SARIF output

`--sarif-output <path>` writes a SARIF 2.1.0 document. `-` writes to
stdout. The schema reference is

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

The CWE annotations are additive: the freeform violation comment
strings (e.g. `dereference failure: NULL pointer`) are unchanged, so
`scripts/competitions/svcomp/esbmc-wrapper.py` continues to classify
results by substring as before.
