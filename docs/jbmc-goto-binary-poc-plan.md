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
   └─ jbmc Test -cp .:/usr/lib/core-models.jar --no-lazy-methods \
                --show-symbol-table --json-ui                    ──> symtab.json
   └─ extract the {"symbolTable": …} element                     ──> st.json
   └─ symtab2gb st.json --out jbmc.goto                          ──> 0x7f 'G' 'B' 'F', v6
```

The output header is byte-for-byte the format ESBMC's reader targets:

```
00000000: 7f47 4246 06a9 0100 3e65 6d70 7479 004e   .GBF....>empty.N
```

`goto-instrument --show-goto-functions jbmc.goto` reads it back cleanly, and
`goto-instrument --drop-unused-functions` rewrites a still-valid binary — so this is a
genuinely shared format, not a JBMC-private dialect. Note the converse does **not** hold:
`cbmc` in C mode on the same file dies with an invariant failure in `goto_symex.cpp`.
Format-compatible is not the same as interchangeable.

**Both `--no-lazy-methods` and an explicit models classpath are load-bearing**, and the
classpath matters more. JBMC loads classes on demand, and `core-models.jar` is *not*
auto-loaded (there is no `--java-core-models` flag; the Ubuntu package merely drops it at
`/usr/lib/core-models.jar`). Measured function counts for the same Java program:

| configuration | functions |
|---|---|
| `-cp .` | 50 |
| `-cp .` `--no-lazy-methods` | 52 |
| `-cp .:core-models.jar` | 72 |
| `-cp .:core-models.jar` `--no-lazy-methods` | **2500** |

A ~48x spread. Any binary built without both flags silently omits method bodies, and any
verdict from it is meaningful only relative to whatever happened to be on the classpath.

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

**Treat this table as a lower bound.** It was measured on a binary built *without*
`core-models.jar` — i.e. the ~52-function configuration from §2.1, not the 2500-function
one. A full-models binary may well contain constructs this corpus never reached (JVM
concurrency instrumentation, `java_super_method_call`, generics attributes, lambda method
handles). Phase 1 exists precisely to replace this estimate with a measurement over the
real configuration.

Additional Java-specific ids exist in `src/util/irep_ids.def` upstream that the corpus did
not exercise: `java_super_method_call`, `exception_list`, `exception_id`,
`exception_landingpad`, `throw_decl`, the `C_java_generic*` family, and
`java_lambda_method_handle*`. (`java_new_array_data` is not in this list — see §2.4,
where it is the one construct that survives JBMC's lowering.)

### 2.4 The decisive finding: JBMC's own pipeline lowers most of it

`jbmc --show-goto-functions` reveals that JBMC's driver (`jbmc_parse_options.cpp`)
unconditionally runs `remove_instanceof`, `remove_virtual_functions`, `remove_exceptions`,
and `remove_java_new` before symex. Comparing what survives in each route (counts are
word-boundary matched — a naive `grep -c java_new` substring-matches `java_new_array` and
inflates the result):

| Construct | Route A (`symtab2gb`) | Route B (JBMC-internal, post-lowering) |
|---|---|---|
| `java_new` | 2 | **0** |
| `java_new_array` | 11 | **0** |
| `java_instanceof` | 1 | **0** |
| `virtual_function` | 2 | **0** |
| `CATCH` (`push`/`pop_catch`) | 4 | **0** |
| `java_new_array_data` | 0 | 11 |
| `side_effect statement="allocate"` | 2 | 15 |

`symtab2gb` performs goto-conversion only — it does **not** run the Java lowering passes,
because those live in JBMC's driver, not in the symbol-table-to-GOTO tool.

So JBMC's own lowering is more thorough than expected: it eliminates object *and* array
allocation too, rewriting `java_new`/`java_new_array` into CBMC's generic
`side_effect statement="allocate"`. The residual Java-specific surface is a single
construct, **`java_new_array_data`**, plus the `allocate` side effect itself — which,
despite being generic CPROVER vocabulary, ESBMC does **not** currently handle:
`src/util/migrate.cpp:1937-1961` recognises `malloc`, `realloc`, `alloca`, `cpp_new`,
`va_arg` but not `allocate`, and `cbmc_adapter.cpp` has no mapping for it (CBMC C-mode
binaries reach ESBMC as `FUNCTION_CALL malloc`, which the adapter rewrites at
`cbmc_adapter.cpp:1300-1303`). Class identifiers and Java array types are then plain
structs.

This reframes the whole PoC: **the cheap win is getting the lowered model out of JBMC,
not teaching ESBMC to lower Java.** Two constructs, not seven.

---

## 3. Two routes, and the recommendation

### Route A — `symtab2gb` today, lower inside ESBMC

Works this instant with zero changes outside ESBMC. But ESBMC must then implement object
and array allocation, virtual dispatch resolution, `instanceof` against a class hierarchy,
and Java exception region semantics — reimplementing four CPROVER passes, in an adapter
that has no class hierarchy available (the hierarchy is a JBMC-side artefact). High cost,
high risk of subtly diverging from JBMC's semantics, which defeats the "reuse CPROVER's
JVM semantics" rationale.

### Route B — teach JBMC to write a goto binary after lowering (**recommended**)

Add a `--write-goto-binary <file>` to `jbmc`, dumping the goto model after its existing
lowering passes. `write_goto_binary()` lives in *shared* `src/goto-programs/`, and
`GOTO_BINARY_VERSION 6` is defined there once for all CPROVER tools — there is no
Java-specific writer, and `grep -rn write_goto_binary jbmc/src/` returns nothing. So this
is plumbing, not new semantics — plausibly a few dozen lines upstream. ESBMC then needs
only `allocate` + `java_new_array_data` plus Java struct-layout conventions.

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

### Phase 2 — allocation side effects (2–3 days)

Map CBMC's `side_effect statement="allocate"` and Java's `java_new_array_data` onto
ESBMC `sideeffect` allocation, following the `malloc`/`alloca` precedent in
`cbmc_adapter.cpp:1040-1067` and extending the statement dispatch at
`src/util/migrate.cpp:1937-1961`. Array allocation additionally carries a length operand
and JBMC's own `array-create-negative-size` check (already present as a plain assertion
in the binary — ESBMC should inherit it, not re-derive it).

`allocate` is worth doing on its own merits: it is generic CPROVER vocabulary that any
`goto-instrument`-lowered binary may contain, so this benefits the CBMC roadmap too.

On Route A, this phase instead targets `java_new`/`java_new_array` directly — which is
why Route B is preferred: same effort, but Route B's two constructs also cover virtual
dispatch, `instanceof`, and exceptions for free.

*Exit:* T1–T3 corpus programs verify, verdicts matching JBMC.
**This is the PoC's minimum publishable result:** it demonstrates the whole chain works
on real Java.

### Phase 3 — Route B upstream (parallel, 2–3 days + review latency)

Prototype `jbmc --write-goto-binary`, confirm the emitted binary is lowered as §2.4
predicts, and re-run the corpus through it. Open an upstream PR against `diffblue/cbmc`.

*Exit:* T4/T5 (virtual dispatch, exceptions) verify through the lowered route with no
further ESBMC adapter work — or a concrete statement of what still fails.

### Phase 4 — Java runtime models (spike only, 1 day)

Partly answered already by §2.1: models arrive in the binary **only** if `core-models.jar`
is put on the classpath explicitly, and doing so with `--no-lazy-methods` inflates the
model to ~2500 functions. Two things remain to determine:

- **Scale.** Is a 2500-function goto binary tractable for ESBMC's symex at all? This is
  the practical gate on non-trivial Java, and it is cheap to measure.
- **Coverage.** `core-models.jar` is narrower than its name suggests — 91 classes, mostly
  `java.lang`; `java.util` collections are essentially unmodelled. Unmodelled library
  methods become bodyless stubs, so a corpus touching collections is verifying against
  nothing.

**`java.lang.String` is a special case and a genuine risk.** It is not modelled by the JAR
at all — JBMC handles it via `java_string_library_preprocess.cpp` and CBMC's **string
refinement solver**. ESBMC has no string-refinement backend, so string-heavy Java may be
out of reach through this route regardless of how well IR ingestion works. Establish
whether the corpus can avoid strings, and if not, say so plainly.

*Exit:* a scale number, a coverage statement, and a yes/no on strings. **Do not build a
linking mechanism in this PoC.**

### Phase 5 — Report and regression fixtures (1 day)

Land whatever works as `regression/goto-transcoder/`-style fixtures — checked-in `.goto`
binaries with `test.desc`, matching the 131 existing CBMC fixtures, including negative
`_fail` variants and an unsupported-construct test pinning the graceful-error message
(the `cbmc_anon_aggregate` pattern). Write up the verdict.

**Budget: ~7 working days plus upstream review latency**, with a go/no-go after Phase 1.

---

## 5. What would make this a "no"

Stated in advance so the PoC can honestly fail:

- **Model completeness cannot be established.** §2.1 shows the model size swings 50→2500
  functions on flags alone. If we cannot state confidently that a given binary contains
  every method the program reaches, every verdict is relative to an unknown classpath and
  the whole exercise is unsound. This is the most serious risk.
- **Scale.** If ESBMC's symex cannot handle a 2500-function model in reasonable time, the
  route works only for programs that touch almost no library code.
- **Strings.** JBMC delegates `java.lang.String` to CBMC's string refinement solver, which
  ESBMC does not have. If the corpus cannot avoid strings, this is a hard stop for a large
  class of real Java, independent of everything else in this plan.
- **`java.util` is unmodelled.** Collections-using code verifies against bodyless stubs.
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
MODELS=/usr/lib/core-models.jar
javac Test.java
jbmc Test -cp .:$MODELS --no-lazy-methods --show-symbol-table --json-ui > symtab.json
python3 -c "import json;d=json.load(open('symtab.json'));\
json.dump([e for e in d if 'symbolTable' in e][0], open('st.json','w'))"
symtab2gb st.json --out jbmc.goto
xxd jbmc.goto | head -1                       # expect: 7f47 4246 06…
goto-instrument --show-goto-functions jbmc.goto | head    # round-trip sanity
esbmc --binary jbmc.goto --goto-functions-only

# Model size vs flags (§2.1) — expect roughly 50 / 52 / 72 / 2500
for cp in "-cp ." "-cp .:$MODELS"; do for lz in "" "--no-lazy-methods"; do
  printf '%-42s %s\n' "$cp $lz" "$(jbmc Rich $cp $lz --list-goto-functions 2>/dev/null | wc -l)"
done; done

# Route A vs Route B construct survival (§2.4).
# -oE with \b is required: a plain `grep -c java_new` also matches java_new_array.
jbmc Rich --no-lazy-methods --show-goto-functions > routeB.txt
goto-instrument --show-goto-functions rich.goto > routeA.txt
for k in 'java_new\b' 'java_new_array\b' 'java_new_array_data\b' \
         'java_instanceof\b' 'virtual_function\b' 'CATCH' 'statement="allocate"'; do
  printf '%-26s A=%-3s B=%s\n' "$k" \
    "$(grep -oE "$k" routeA.txt | wc -l)" "$(grep -oE "$k" routeB.txt | wc -l)"
done
```

---

## 8. Open questions for upstream

1. Would `diffblue/cbmc` accept a `jbmc --write-goto-binary` flag? (No such writer exists
   in `jbmc/src/` today, and `--outfile` is the SMT/DIMACS formula, not GOTO — an easy trap.)
2. Is `symtab2gb` on JBMC symbol tables a supported use, or an accident that may break?
3. Is there a recommended way to state, for a given binary, that class loading was
   complete for the program's reachable set — or is classpath discipline the only answer?

### Version note

Probes were run against locally installed `jbmc`/`cbmc` **6.5.0**; upstream `develop` is at
**6.10.0**. `GOTO_BINARY_VERSION` is **6** in both, so the format conclusions hold, but
Phase 0 should re-confirm the construct inventory against a current build before any
upstream PR is opened.

---

## 9. References

- L. C. Cordeiro, P. Kesseli, D. Kroening, P. Schrammel, M. Trtík, "JBMC: A Bounded Model
  Checking Tool for Verifying Java Bytecode", CAV 2018, LNCS 10981, pp. 183–190
  ([doi:10.1007/978-3-319-96145-3_10](https://doi.org/10.1007/978-3-319-96145-3_10)).
- L. Cordeiro, D. Kroening, P. Schrammel, "JBMC: Bounded Model Checking for Java Bytecode
  (Competition Contribution)", TACAS 2019, LNCS 11429, pp. 219–223
  ([doi:10.1007/978-3-030-17502-3_17](https://doi.org/10.1007/978-3-030-17502-3_17)).

Both describe JBMC's architecture and Java operational model. Neither documents lazy class
loading — for that behaviour the source is authoritative
(`jbmc/src/java_bytecode/lazy_goto_model.h`), as are `jbmc/src/java_bytecode/README.md` for
exception lowering and [diffblue/java-models-library](https://github.com/diffblue/java-models-library)
for `core-models.jar`. Note that <https://diffblue.github.io/cbmc/> documents CBMC/C only
and has no Java section; JBMC documentation is effectively in-repo.
