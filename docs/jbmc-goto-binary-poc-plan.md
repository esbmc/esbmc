# PoC: Verifying JBMC-produced GOTO Programs with ESBMC

**Status:** PROPOSED (plan only — no implementation yet)
**Date:** 2026-07-18
**Related:** [`docs/cprover-support-roadmap.md`](cprover-support-roadmap.md) (the CBMC
goto-binary ingestion effort this builds on), `src/jimple-frontend/` (an unrelated,
pre-existing Java route)

---

## 1. Question

Can ESBMC verify Java programs by consuming the GOTO programs that JBMC produces from
Java bytecode, reusing the CBMC goto-binary reader that already exists in the tree?

This is a **feasibility PoC**, not a Java frontend. Success means: a handful of small
Java programs are verified end-to-end by ESBMC via JBMC, with verdicts matching JBMC's
own, and a measured, itemised list of what blocks the rest.

### Why this is worth a PoC

ESBMC's Java story today is `src/jimple-frontend/` — a source-language frontend for
Soot's Jimple IR, fed by JSON produced by an external tool. Its Java type handling is
shallow (`src/jimple-frontend/AST/jimple_type.h:59-69` maps `java.lang.String` and
`java.lang.AssertionError` to `INT` with `// TODO: handle this properly`), and it is
gated behind an optional build flag with 10 regression tests.

Meanwhile ESBMC has, over roughly 30 merged PRs, built a mature reader for CBMC's
goto-binary format (`read_cbmc_goto_object.cpp`, `cbmc_adapter.cpp`, ~2000 lines, 131
regression fixtures under `regression/goto-transcoder/`). JBMC is built on the same
CPROVER codebase and emits the same IR. If that reader can be pointed at Java, ESBMC
gets a Java frontend largely for free — and reuses the JVM semantics, class loading,
and Java runtime models that the CPROVER project already maintains.

---

## 2. What is already established

The probes below were run against `jbmc`/`cbmc` **6.5.0** and ESBMC at `c3deaba546`,
on OpenJDK 21. They are cheap to reproduce; §7 records the exact commands.

### 2.1 JBMC cannot write a goto binary, but a route exists

`jbmc` has no `--write-goto-binary` / `--outfile` for GOTO; its `--gb` flag *reads* one.
The C-mode `goto-instrument` in the same distribution cannot read `.class` files
(`not a goto binary`). However, the CPROVER distribution ships **`symtab2gb`**, which
compiles a JSON symbol table into a goto binary. That yields a working pipeline:

```
Test.java ──javac──> Test.class
   └─ jbmc Test --no-lazy-methods --show-symbol-table --json-ui  ──> symtab.json
   └─ extract the {"symbolTable": …} element                     ──> st.json
   └─ symtab2gb st.json --out jbmc.goto                          ──> 0x7f 'G' 'B' 'F', v6
```

The output header is byte-for-byte the format ESBMC's reader targets:

```
00000000: 7f47 4246 06a9 0100 3e65 6d70 7479 004e   .GBF....>empty.N
```

`goto-instrument --show-goto-functions jbmc.goto` reads it back cleanly, so the binary
is well-formed and not a JBMC-private dialect.

**`--no-lazy-methods` is load-bearing.** JBMC loads classes on demand; without it the
symbol table is incomplete and the resulting binary would silently omit method bodies.

### 2.2 ESBMC already reads a JBMC binary — structurally

Feeding the above to ESBMC:

```
$ esbmc --binary jbmc.goto --goto-functions-only
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
CBMC goto-binary detected: linking ESBMC additions automatically
...
Reading GOTO program from file
ERROR: Unexpected side-effect statement: java_new_array
```

This is a far better starting position than a blank sheet. The magic-byte dispatch
(`goto_binary_reader.cpp:22-47`), the varint irep grammar, the symbol table, the
function/instruction tables, and most of `cbmc_adapter`'s type and expression rewriting
all worked on Java input without modification. Execution reaches
`src/util/migrate.cpp:2098` — i.e. the failure is in *Java-specific irep vocabulary*,
not in format or structure.

Two defects observed in passing, both worth fixing regardless of this PoC:

- ESBMC **crashes (SIGABRT/core dump)** on the unsupported construct rather than
  exiting cleanly. The tree already has the right pattern for this — `expand_anon_struct`
  throws a `std::string` caught in `goto_program.cpp:283-287`, pinned by the
  `regression/goto-transcoder/cbmc_anon_aggregate` test.
- ESBMC also core-dumps *after* correctly printing `ERROR: 'x.goto' is not a
  goto-binary.` — a clean-error path that still aborts.

### 2.3 The Java-specific surface is small and measurable

Walking every irep in the symbol table of a Java program exercising inheritance,
virtual dispatch, arrays, `instanceof`, and `try`/`catch`:

| Construct | Kind | Occurrences |
|---|---|---|
| `java_new_array` | side effect (array allocation) | 11 |
| `java_new` | side effect (object allocation) | 2 |
| `virtual_function` | virtual dispatch | 2 |
| `java_instanceof` | type test | 1 |
| `push_catch` / `pop_catch` | exception region scoping | 2 / 2 |
| `@class_identifier` | class-tag component | 14 |
| `string_constant` | Java string literal | 3 |
| `java::array[T]`, `java::@inflight_exception` | struct-tag types / global | — |

Everything else in the binary is standard CPROVER vocabulary that `cbmc_adapter`
already handles: `signedbv`, `pointer`, `struct_tag`, `member`, `dereference`, `assign`,
`block`, `code`, `constant`, `floatbv`. **No new number theory, no new memory model.**

### 2.4 The decisive finding: JBMC's own pipeline lowers most of it

`jbmc --show-goto-functions` reveals that JBMC's internal pipeline runs
`remove_virtual_functions`, `remove_instanceof`, and `remove_exceptions` before symex.
Comparing what survives in each route:

| Construct | Route A (`symtab2gb`) | Route B (JBMC-internal, post-lowering) |
|---|---|---|
| `java_new_array` | 11 | 11 |
| `java_instanceof` | 1 | **0** |
| `virtual_function` | 2 | **0** |
| `CATCH` (`push`/`pop_catch`) | 4 | **0** |

`symtab2gb` performs goto-conversion only — it does **not** run the Java lowering
passes, because those live in JBMC's driver, not in the symbol-table-to-GOTO tool.

So the residual Java surface after JBMC's own lowering is essentially just
**`java_new` and `java_new_array`** — two allocation side effects, both of which map
naturally onto ESBMC's existing `sideeffect` allocation machinery (the same machinery
`cbmc_adapter.cpp:1040-1067` already uses to turn CBMC's `malloc`/`alloca` into
`sideeffect`). Class identifiers and Java array types are then plain structs.

This reframes the whole PoC: **the cheap win is getting the lowered model out of JBMC,
not teaching ESBMC to lower Java.**

---

## 3. Two routes, and the recommendation

### Route A — `symtab2gb` today, lower inside ESBMC

Works this instant with zero changes outside ESBMC. But ESBMC must then implement
virtual dispatch resolution, `instanceof` against a class hierarchy, and Java exception
region semantics — reimplementing three CPROVER passes, in an adapter that has no class
hierarchy available (the hierarchy is a JBMC-side artefact). High cost, high risk of
subtly diverging from JBMC's semantics, which defeats the "reuse CPROVER's JVM
semantics" rationale.

### Route B — teach JBMC to write a goto binary after lowering (**recommended**)

Add a `--write-goto-binary <file>` to `jbmc`, dumping the goto model after its existing
lowering passes. CBMC's `write_goto_binary()` is already linked into every CPROVER
tool; this is plumbing, not new semantics — plausibly a few dozen lines upstream. ESBMC
then only needs `java_new` / `java_new_array` plus Java struct-layout conventions.

**Recommendation: pursue B, with A as the fallback and as the *immediate* unblocked
path for Phase 1.** Phase 1 below is deliberately route-agnostic so it delivers value
before any upstream dependency resolves.

A cheap hedge exists if upstream is slow: JBMC's lowering passes are ordinary CPROVER
library calls, so a small out-of-tree driver (or a patched local JBMC build) can do
load → lower → `write_goto_binary` without waiting for a merged flag.

---

## 4. Phased plan

Each phase ends in a decision, not just an artefact. **Any phase may end the PoC**;
that is a successful outcome if the blocker is named and evidenced.

### Phase 0 — Reproducible harness (0.5 day)

Script the §2.1 pipeline (`javac` → `jbmc` → extract → `symtab2gb`) plus the ESBMC
invocation, so every later result is one command. Pin `jbmc`/`cbmc` version in the
output. Add a corpus of ~8 Java programs in tiers: (T1) integer/boolean arithmetic and
asserts, no allocation; (T2) arrays; (T3) objects and fields; (T4) inheritance and
virtual calls; (T5) `try`/`catch`.

*Exit:* one command produces a goto binary + an ESBMC verdict per corpus program.

### Phase 1 — Fail gracefully, then measure (1 day)

Convert the `abort()` on unknown Java side effects into the existing throw/catch clean-exit
pattern (`cbmc_adapter.cpp:791-793` + `goto_program.cpp:283-287`), and fix the
core-dump-after-clean-error path from §2.2. Then run the whole corpus and record, per
program, the *first* unsupported construct.

This is the single highest-value phase: it converts "ESBMC crashes on Java" into a
ranked, evidence-backed work-list, and it is useful to the CBMC roadmap independently
of Java.

*Exit:* a table of constructs ranked by how many corpus programs each blocks.
**Decision point:** if the list is dominated by constructs outside §2.3 — i.e. the
simple corpus was misleading — reassess scope before writing any adapter code.

### Phase 2 — `java_new` / `java_new_array` (2–3 days)

Map both onto ESBMC `sideeffect` allocation, following the `malloc`/`alloca` precedent
in `cbmc_adapter.cpp:1040-1067`. `java_new_array` additionally carries a length operand
and JBMC's own `array-create-negative-size` check (already present as a plain assertion
in the binary — ESBMC should inherit it, not re-derive it).

*Exit:* T1–T3 corpus programs verify, verdicts matching JBMC.
**This is the PoC's minimum publishable result:** it demonstrates the whole chain works
on real Java.

### Phase 3 — Route B upstream (parallel, 2–3 days + review latency)

Prototype `jbmc --write-goto-binary`, confirm the emitted binary is lowered as §2.4
predicts, and re-run the corpus through it. Open an upstream PR against `diffblue/cbmc`.

*Exit:* T4/T5 (virtual dispatch, exceptions) verify through the lowered route with no
further ESBMC adapter work — or a concrete statement of what still fails.

### Phase 4 — Java runtime models (spike only, 1 day)

JBMC's semantics depend on its Java core models (`java.lang.String`, boxed types, etc.).
Even the trivial program in §2.1 pulled in 169 symbols, mostly models. Determine
empirically whether these arrive inside the goto binary (in which case ESBMC gets them
free) or are resolved lazily by JBMC at symex time (in which case ESBMC needs a linking
story analogous to `link_cbmc_libc_bodies`, `goto_program.cpp:349`).

*Exit:* a yes/no on whether a linking mechanism is needed, sized if yes. **Do not build
it in this PoC.**

### Phase 5 — Report and regression fixtures (1 day)

Land whatever works as `regression/goto-transcoder/`-style fixtures — checked-in `.goto`
binaries with `test.desc`, matching the 131 existing CBMC fixtures, including negative
`_fail` variants and an unsupported-construct test pinning the graceful-error message
(the `cbmc_anon_aggregate` pattern). Write up the verdict.

**Budget: ~7 working days plus upstream review latency**, with a go/no-go after Phase 1.

---

## 5. What would make this a "no"

Stated in advance so the PoC can honestly fail:

- **Lazy loading is unavoidable.** If `--no-lazy-methods` still yields incomplete symbol
  tables for non-trivial programs, every binary is silently partial and any verdict is
  unsound. This is the most serious risk and Phase 0 should probe it directly.
- **Model dependence is deep.** If Phase 4 finds JBMC resolves core models at symex time,
  ESBMC inherits a Java runtime-library problem — a much larger project than IR ingestion.
- **Java struct conventions leak everywhere.** If `@class_identifier` / `java::array[T]`
  layouts turn out to need special handling across the adapter rather than being plain
  structs, the "reuse the CBMC reader" premise weakens.
- **Verdict divergence.** If ESBMC and JBMC disagree on corpus programs, the *reason*
  matters more than the count: a JVM-semantics gap is a no; a known ESBMC pointer-model
  difference is a tracked bug.

---

## 6. Explicit non-goals

Not a Java frontend. No `.class`/`.jar` parsing in ESBMC. No replacement of, or
integration with, `src/jimple-frontend/`. No JVM concurrency, reflection, generics, or
`invokedynamic`/lambdas. No performance work.

---

## 7. Reproducing the §2 probes

```sh
javac Test.java
jbmc Test --no-lazy-methods --show-symbol-table --json-ui > symtab.json
python3 -c "import json;d=json.load(open('symtab.json'));\
json.dump([e for e in d if 'symbolTable' in e][0], open('st.json','w'))"
symtab2gb st.json --out jbmc.goto
xxd jbmc.goto | head -1                       # expect: 7f47 4246 06…
goto-instrument --show-goto-functions jbmc.goto | head    # round-trip sanity
esbmc --binary jbmc.goto --goto-functions-only

# Route A vs Route B construct survival (§2.4)
jbmc Rich --no-lazy-methods --show-goto-functions > routeB.txt
goto-instrument --show-goto-functions rich.goto > routeA.txt
for k in java_new_array java_instanceof virtual_function CATCH; do
  printf '%-18s A=%s B=%s\n' "$k" "$(grep -c $k routeA.txt)" "$(grep -c $k routeB.txt)"
done
```

---

## 8. Open questions for upstream

1. Is there an existing supported way to obtain a *lowered* JBMC goto model as a binary
   that this plan has missed?
2. Would `diffblue/cbmc` accept a `jbmc --write-goto-binary` flag?
3. Is `symtab2gb` on JBMC symbol tables a supported use, or an accident that may break?
