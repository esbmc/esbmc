# PoC: Verifying JBMC-produced GOTO Programs with ESBMC

**Status:** PROPOSED (plan only — no implementation yet)
**Date:** 2026-07-18 (revised after technical review, 2026-07-18)
**Related:** [`docs/cprover-support-roadmap.md`](cprover-support-roadmap.md) (the CBMC
goto-binary ingestion effort this builds on), `src/jimple-frontend/` (an unrelated,
pre-existing Java route)

---

## 1. Question

Can ESBMC verify Java programs by consuming the GOTO programs that JBMC produces from
Java bytecode, reusing the CBMC goto-binary reader that already exists in the tree?

This is a **feasibility PoC**, not a Java frontend. Success means: a handful of small
Java programs are verified end-to-end by ESBMC via JBMC, with verdicts matching JBMC's
own, and a measured, itemised list of what blocks the rest. §4.1 states the success
criteria precisely.

### Why this is worth a PoC

ESBMC's Java story today is `src/jimple-frontend/` — a source-language frontend for
Soot's Jimple IR, fed by JSON produced by an external tool. Its Java type handling is
shallow (`src/jimple-frontend/AST/jimple_type.h` maps `java.lang.String`,
`java.lang.AssertionError`, `java.lang.Runtime` and `java.lang.Class` to `INT`, each
with `// TODO: handle this properly`), and it is gated behind the optional
`ENABLE_JIMPLE_FRONTEND` build flag with 15 regression tests under `regression/jimple/`.

Meanwhile ESBMC has, over roughly 30 merged PRs, built a mature reader for CBMC's
goto-binary format (`src/goto-programs/read_cbmc_goto_object.cpp` +
`src/goto-programs/cbmc_adapter.cpp`, ~2000 lines, 131 regression fixtures under
`regression/goto-transcoder/`). JBMC is built on the same CPROVER codebase and emits the
same IR. If that reader can be pointed at Java, ESBMC gets a Java frontend largely for
free — and reuses the JVM semantics, class loading, and Java runtime models that the
CPROVER project already maintains.

**Scope of that "for free" claim.** It applies to *IR ingestion*, not to verification
capability. Java semantics that JBMC discharges in its solver rather than in its GOTO
program — string refinement above all (§5) — do not transfer with the binary. This plan
does not assume otherwise.

---

## 2. What is already established

### 2.1 Provenance of the measurements

Two independent probe runs are reported below. Distinguishing them matters, because the
absolute counts in §2.5 and §2.6 are corpus-dependent and only the *directional*
conclusions replicate.

| | Run 1 (original) | Run 2 (review replication) | Run 3 (first-blocker triage) |
|---|---|---|---|
| `jbmc`/`cbmc` | 6.5.0 (Ubuntu, Diffblue `.deb`) | 6.8.0 (macOS, Homebrew) | 6.8.0 (macOS, Homebrew) |
| ESBMC | `c3deaba546` | `8.4.0`, feature branch, aarch64 | `9bac4d40c7` (`master`), aarch64 |
| JDK | OpenJDK 21 | none — corpus taken from `core-models.jar` | none |
| Corpus | hand-written `Rich.java` | `java.lang.Integer` + transitive models | `java.lang.Integer` + transitive models |

All ESBMC line references in this document are as they appear at **`c3deaba546`**, the
commit Run 1 was measured against. Several have since drifted on `master`; §2.4 records
the drift explicitly rather than silently restating master's numbers.

Upstream CPROVER references are to `diffblue/cbmc` `develop` at
`ff8e1122feffbe0c067b88a9ce44840a99a99428` (2026-07-17, version 6.10.0).

### 2.2 JBMC cannot write a goto binary, but a route exists

`jbmc` has no `--write-goto-binary`, and no goto-binary output path of any kind: its
`--gb` flag *reads* one (`OPT_JAVA_GOTO_BINARY`,
`jbmc/src/java_bytecode/java_bytecode_language.h:159-165`, help text "goto-binary file
to be checked"), and `grep -rn write_goto_binary jbmc/` returns nothing.

> **Trap.** `--outfile` *does* exist for `jbmc` — it arrives via `OPT_SOLVER`
> (`src/goto-checker/solver_factory.h:120`) and writes the **SAT/SMT formula**, not a
> GOTO program. It is not an escape hatch.

The C-mode `goto-instrument` in the same distribution cannot read `.class` files
(`not a goto binary`); it registers only C and C++ frontends
(`src/goto-instrument/goto_instrument_languages.cpp:19-23`). However, the CPROVER
distribution ships **`symtab2gb`**, which compiles a JSON symbol table into a goto
binary. That yields a working pipeline:

```
Test.java ──javac──> Test.class
   └─ jbmc Test -cp .:<core-models.jar> --no-lazy-methods \
                --show-symbol-table --json-ui                    ──> symtab.json
   └─ extract the {"symbolTable": …} element                     ──> st.json
   └─ symtab2gb st.json --out jbmc.goto                          ──> 0x7f 'G' 'B' 'F', v6
```

**This pipeline was independently re-run in Run 2** on cbmc 6.8.0/macOS with a different
corpus, and works end to end: `symtab2gb` exits 0 and emits a 768 KiB binary whose header
is

```
00000000: 7f47 4246 068c 0c00 8c01 7374 7275 6374  .GBF......struct
```

i.e. `0x7f 'G' 'B' 'F'`, version **6** — byte-for-byte the format ESBMC's reader targets.
`goto-instrument --show-goto-functions jbmc.goto` reads it back cleanly, and
`goto-instrument --drop-unused-functions` rewrites a still-valid binary — so this is a
genuinely shared format, not a JBMC-private dialect.

`GOTO_BINARY_VERSION` is defined once, in shared code
(`src/goto-programs/write_goto_binary.h:15`), and was checked to be **6** at every 6.x
release tag from 6.0.0 through 6.10.0. The format conclusion is therefore stable across
the whole 6.x line, not just the two versions probed.

Note the converse does **not** hold: `cbmc` in C mode on the same file dies with an
invariant failure in `goto_symex.cpp`. This is expected rather than surprising — `cbmc`
registers `ansi_c`, `statement_list`, `cpp` and `json_symtab`, but not `java_bytecode`
(`src/cbmc/cbmc_languages.cpp:21-27`), so symbols carrying `mode == "java"` have no
registered language handler (`src/langapi/mode.cpp:51`). **Format-compatible is not the
same as interchangeable**, and ESBMC will need its own answer to the same `mode` question.

**Both `--no-lazy-methods` and an explicit models classpath are load-bearing**, and the
classpath matters more. JBMC loads classes on demand — lazy is the default,
`options.set_option("lazy-methods", !cmd.isset("no-lazy-methods"))`
(`jbmc/src/java_bytecode/java_bytecode_language.cpp:109`) — and `core-models.jar` is *not*
auto-loaded. The classpath is built solely from `--classpath`/`--cp`/`-jar`/JSON config
(`java_bytecode_language.cpp:296-299`): there is no default entry, no `CLASSPATH`
environment fallback, no hardcoded models path, and no `--java-core-models` flag.
(`--java-cp-include-files` is a regex *filter* over files to load, not a classpath adder.)
Measured function counts for the same Java program (Run 1):

| configuration | functions |
|---|---|
| `-cp .` | 50 |
| `-cp .` `--no-lazy-methods` | 52 |
| `-cp .:core-models.jar` | 72 |
| `-cp .:core-models.jar` `--no-lazy-methods` | **2500** |

A ~48x spread. Any binary built without both flags silently omits method bodies, and any
verdict from it is meaningful only relative to whatever happened to be on the classpath.

> **`core-models.jar` is not at a portable path.** Verified locations:
> `/usr/lib/core-models.jar` in **Diffblue's release `.deb`** (`jbmc/CMakeLists.txt:32-38`
> installs to `${CMAKE_INSTALL_LIBDIR}`); `/opt/homebrew/Cellar/cbmc/<version>/libexec/lib/core-models.jar`
> on **Homebrew/macOS**. The **Ubuntu archive `jbmc` package is different from Diffblue's
> `.deb` and ships no jar at all** — only `/usr/bin/jbmc` and a man page. The harness must
> locate the jar, not assume it (§7).

### 2.3 ESBMC already reads a JBMC binary — structurally

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
(`src/goto-programs/goto_binary_reader.cpp:22-47`), the varint irep grammar, the symbol
table, the function/instruction tables, and most of `cbmc_adapter`'s type and expression
rewriting all worked on Java input without modification. Execution reaches
`src/util/migrate.cpp:2098` — i.e. the failure is in *Java-specific irep vocabulary*,
not in format or structure.

Two defects observed in passing, both worth fixing regardless of this PoC, and **both
reproduced in Run 2** (exit code 134 = SIGABRT in each case):

- ESBMC **crashes (SIGABRT/core dump)** on the unsupported construct rather than
  exiting cleanly — `migrate.cpp:2098` logs the error and `migrate.cpp:2099` calls
  `abort()`. The tree already has the right pattern for this: `expand_anon_struct`
  throws a `std::string` (`cbmc_adapter.cpp:791-793` at `c3deaba546`) which is caught in
  **`src/esbmc/parseoptions/goto_program.cpp:283-287`** — note the path, there is also an
  unrelated `src/goto-programs/goto_program.cpp` whose lines 283-287 are successor
  computation.
- ESBMC also core-dumps *after* correctly printing `ERROR: 'x.goto' is not a
  goto-binary.` — a clean-error path that still aborts.

> **Run 2 caveat on the diagnostic — now resolved by Run 3.** With a different corpus and
> a newer ESBMC, the first blocker surfaced as a bare `ERROR: string` before the abort,
> rather than a message naming the construct. Run 3 traced it; see §2.3.1. It is neither a
> regression in error reporting nor corpus noise — it is a *different and earlier* abort
> site than the one §2.3 reports, and the message is uninformative because that site logs
> a bare type dump.

#### 2.3.1 The real first blocker: `@class_identifier` is typed `string` (Run 3)

Run 3 attached a debugger to the `ERROR: string` abort on `master` (`9bac4d40c7`). The
abort is **not** `migrate.cpp:2099` (`Unexpected side-effect statement`). It is
`migrate.cpp:379`, the fall-through of `migrate_type0`:

```cpp
log_error("{}", type);   // a bare typet("string") pretty-prints as exactly: string
abort();
```

The offending type is reached through two nested `struct` migrations from an instruction
expression, and the enclosing struct is `java.lang.Object`:

```
java::java.lang.Object = struct {
  @class_identifier   : string          <-- ID_string; migrate_type0 has no case for it
  cproverMonitorCount : signedbv[32]
}
```

**Consequence: this blocks 100% of Java input, unconditionally.** Every Java class embeds
`java.lang.Object`, so ingestion fails during *type* migration, before symbol values,
instructions, or any side effect are reached. It is reachable on Route A and Route B
alike — lowering does not touch it, because it is a field type, not a statement.

This corrects three claims elsewhere in this document:

- **§2.3's transcript is not representative.** Run 1's `ERROR: Unexpected side-effect
  statement: java_new_array` came from a binary built *without* `core-models.jar` (the
  ~52-function configuration §2.2 warns against). It is a real blocker, but it is not the
  first one in the load-bearing configuration.
- **§2.5 under-rates `@class_identifier`.** The table lists it as a "class-tag component"
  with 14 occurrences; §2.6 then asserts class identifiers "are then plain structs". The
  component is plain; **its type is not** — `ID_string` has no ESBMC representation at
  all. It ranks first, not last.
- **§7 assumption 2 is false as stated.** The diagnostic does *not* name the offending
  construct at this site: `migrate_type0`'s fall-through has no descriptive prefix, unlike
  `migrate_expr`'s. Phase 1 must therefore repair **both** abort sites, and the
  `migrate_type0` one needs a message prefix added, not merely a `throw` in place of the
  `abort()`.

§2.6's core conclusions are unaffected: the Route A → Route B lowering direction stands,
and no new number theory or memory model is implied.

### 2.4 Drift between `c3deaba546` and current `master`

Recorded so a later reader does not mistake stale-but-correct citations for errors:

- `regression/goto-transcoder/` held **131** fixtures at `c3deaba546`; **137** on master.
- **The `cbmc_anon_aggregate` precedent has changed.** At `c3deaba546` its `test.desc`
  expected `^ERROR: CBMC adapter: anonymous aggregate types are not yet supported` — it
  genuinely pinned the graceful-error path. On master it expects
  `^VERIFICATION SUCCESSFUL$`, i.e. anonymous aggregates are now *supported* and the test
  no longer pins graceful failure. **Phase 1 and Phase 5 must therefore create their own
  unsupported-construct fixture rather than copying a test that no longer demonstrates
  the pattern.**

### 2.5 The Java-specific surface is small and measurable

Walking every irep in the symbol table of a Java program exercising inheritance,
virtual dispatch, arrays, `instanceof`, and `try`/`catch` (Run 1):

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
`core-models.jar` — i.e. the ~52-function configuration from §2.2, not the 2500-function
one. A full-models binary may well contain constructs this corpus never reached (JVM
concurrency instrumentation, `java_super_method_call`, generics attributes, lambda method
handles). Phase 1 exists precisely to replace this estimate with a measurement over the
real configuration.

Additional Java-specific ids exist in `src/util/irep_ids.def` upstream that the corpus did
not exercise; all spellings verified against `develop`: `java_super_method_call` (:612),
`exception_list` (:543), `exception_id` (:544), `exception_landingpad` (:616),
`throw_decl` (:765), the four-member `C_java_generic*` family —
`C_java_generic_parameter` (:692), `C_java_generics_class_type` (:693),
`C_java_implicitly_generic_class_type` (:694), `C_java_generic_symbol` (:695) — and
`java_lambda_method_handle{,_index,s}` (:700-702).

### 2.6 The decisive finding: JBMC's own pipeline lowers most of it

`jbmc --show-goto-functions` reveals that JBMC's driver (`jbmc_parse_options.cpp`) runs
`remove_instanceof` (:691), `remove_virtual_functions` (:698) and `remove_java_new` (:730)
unconditionally in `process_goto_function`, alongside `remove_returns` (:705),
`replace_java_nondet` (:707) and `convert_nondet` (:710).

**`remove_exceptions` is the exception to "unconditional", though the net effect is the
same.** It runs at `:719` *only* when `using_symex_driven_loading`
(`--symex-driven-lazy-loading`, set at `:688-689`); otherwise it runs in
`process_goto_functions` at `:829`, which itself early-returns at `:826` when
symex-driven loading is on. So it always runs exactly once, but **in one of two places,
with different fidelity** — the in-function variant leaves dead catch sites behind
(comment at `:712-715`). A PoC that relies on exception lowering must state which mode it
measured.

Comparing what survives in each route (counts are word-boundary matched — a naive
`grep -c java_new` substring-matches `java_new_array` and inflates the result):

| Construct | Route A (`symtab2gb`)<br>Run 1 / **Run 2** | Route B (post-lowering)<br>Run 1 / **Run 2** |
|---|---|---|
| `java_new` | 2 / **28** | 0 / **0** |
| `java_new_array` | 11 / **11** | 0 / **0** |
| `java_instanceof` | 1 / **38** | 0 / **0** |
| `virtual_function` | 2 / **77** | 0 / **0** |
| `CATCH` (`push`/`pop_catch`) | 4 / **2** | 0 / **0** |
| `java_new_array_data` | 0 / **0** | 11 / **11** |
| `side_effect statement="allocate"` | 2 / **24** | 15 / **68** |

Absolute counts differ because the corpora differ; **the directional result replicates
exactly across both runs, two cbmc versions, and two platforms**: every Java-specific
construct in Route A goes to zero in Route B.

`symtab2gb` performs goto-conversion only — it does **not** run the Java lowering passes,
because those live in JBMC's driver, not in the symbol-table-to-GOTO tool. This is
structural, not incidental: `symtab2gb`'s `register_languages()`
(`src/symtab2gb/symtab2gb_parse_options.cpp:146-153`) registers only `json_symtab_language`
and `ansi_c_language` — **the Java frontend is not linked into the binary at all**, so no
amount of flag-tuning will make it lower Java.

So JBMC's own lowering is more thorough than expected: it eliminates object *and* array
allocation too, rewriting `java_new`/`java_new_array` into CBMC's generic
`side_effect_exprt(ID_allocate, ...)` (`remove_java_new.cpp:100-105`, `:154-160`; the
in-source comment reads "we produce a malloc side-effect, which stays").

The residual Java-specific surface is a single construct, **`java_new_array_data`**, plus
the `allocate` side effect itself.

> **Framing correction.** `java_new_array_data` does not *survive* lowering — it is
> **introduced by** it: `remove_java_new.cpp:240-241` constructs
> `side_effect_exprt(ID_java_new_array_data, allocate_data_type, location)` for the array
> payload. It then reaches symex and is interpreted natively there
> (`src/goto-symex/goto_symex.cpp:76`, `src/goto-symex/symex_builtin_functions.cpp:502,522`).
> The practical consequence is *stronger* than "one leftover construct": lowering does not
> eliminate Java-specific vocabulary, it **substitutes** a construct whose semantics live
> in CPROVER's symex, which ESBMC must therefore reimplement rather than inherit.

`allocate` — despite being generic CPROVER vocabulary — ESBMC does **not** currently
handle: the string `"allocate"` appears nowhere in `src/`. `src/util/migrate.cpp`
dispatches side-effect statements in two places: a `#size` guard at :1936-1938 covering
`malloc`, `realloc`, `alloca`, `va_arg`; and an `allockind` dispatch from :1953 covering
`malloc`, `realloc`, `alloca`, `cpp_new`, `cpp_new[]`, `nondet`, `va_arg`,
`function_call`, `printf`, `printf2`. Neither recognises `allocate`. `cbmc_adapter.cpp`
has no mapping for it either — CBMC C-mode binaries instead reach ESBMC as
`FUNCTION_CALL malloc`, which the adapter rewrites at `cbmc_adapter.cpp:1300-1303` via
`build_mem_rhs` (`:1040-1067`). Class identifiers and Java array types are then plain
structs.

This reframes the PoC: **the cheap win is getting the lowered model out of JBMC, not
teaching ESBMC to lower Java.** Two constructs, not seven — with the caveat above that
one of the two carries symex-level semantics.

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
Java-specific writer, and no `jbmc/` source file calls `write_goto_binary`. So this
is plumbing, not new semantics — plausibly a few dozen lines upstream. ESBMC then needs
only `allocate` + `java_new_array_data` plus Java struct-layout conventions.

**Recommendation: pursue B, with A as the fallback and as the *immediate* unblocked
path for Phase 1.** Phase 1 below is deliberately route-agnostic so it delivers value
before any upstream dependency resolves.

**Upstream dependency is the schedule risk, not the technical risk.** Route B's ESBMC-side
work is small; its exposure is that a third-party maintainer must accept a flag on their
schedule, and §8 Q1 is genuinely open. The hedge: JBMC's lowering passes are ordinary
CPROVER library calls, so a small out-of-tree driver (or a patched local JBMC build) can
do load → lower → `write_goto_binary` without waiting for a merged flag. **Phase 3 should
build that driver first and open the upstream PR second**, so no phase of this PoC blocks
on review latency. A permanently-unmerged flag degrades Route B to "requires a patched
JBMC" — usable for a PoC, not for a shipped feature; that trade-off should be decided
before any user-facing commitment.

---

## 4. Phased plan

Each phase ends in a decision, not just an artefact. **Any phase may end the PoC**;
that is a successful outcome if the blocker is named and evidenced.

### 4.1 Success criteria

The PoC is a **success** if Phase 2 exits green: T1–T3 corpus programs verify under
`esbmc --binary`, with verdicts (SUCCESSFUL/FAILED per property) matching JBMC's on the
same programs, and every divergence classified per §5.

The PoC is an **informative failure** — still a valid outcome — if it terminates at any
phase with a named, evidenced blocker plus a reproducer.

It is a **failure** only if it terminates without either.

**Soundness bar.** No verdict produced by this PoC may be reported as a verification
result for a Java program unless the model-completeness question in §5 is answered for
that program. Until then, all results are "ESBMC agrees with JBMC on this binary" — a
statement about IR ingestion, not about Java. This distinction must appear in the
Phase 5 write-up.

### Phase 0 — Reproducible harness (0.5 day)

Script the §2.2 pipeline (`javac` → `jbmc` → extract → `symtab2gb`) plus the ESBMC
invocation, so every later result is one command. The harness must:

- **Locate `core-models.jar` rather than hardcode it** (§2.2), and fail loudly if absent —
  a missing jar silently produces a 50-function model, not an error.
- Record `jbmc --version`, `cbmc --version`, the ESBMC commit, the JDK version, and the
  resolved jar path into every output artefact.
- **Confirm the §2.3 diagnostic still names the offending construct** on `master`/x86-64
  before Phase 1 depends on it.

Add a corpus of ~8 Java programs in tiers: (T1) integer/boolean arithmetic and asserts, no
allocation; (T2) arrays; (T3) objects and fields; (T4) inheritance and virtual calls;
(T5) `try`/`catch`. **Every tier must be string-free** (§5), and each program must carry
both a passing and a failing property so verdict matching is falsifiable in both
directions.

*Exit:* one command produces a goto binary + an ESBMC verdict per corpus program.

**Status: harness landed, corpus outstanding.** `scripts/jbmc-poc-pipeline.sh`
implements the pipeline half — jar discovery (fatal when absent or mis-set),
`--no-lazy-methods`, symbol-table extraction, `symtab2gb`, the ESBMC run, and a
`manifest.txt` recording jbmc/cbmc/ESBMC versions, the ESBMC source commit, the
resolved jar, the goto header bytes and the verdict. It accepts either a class
from `core-models.jar` (the no-JDK route) or a `--source File.java` (needs a
JDK).

The ~8-program tiered corpus is **not** landed: it cannot be compiled, and
therefore cannot be validated, without a JDK. Checking in unvalidated `.java`
files — or opaque `.class` files — was rejected in favour of leaving the gap
explicit. Note that `command -v javac` is not a usable JDK probe on macOS,
which ships a stub that exists on PATH and exits 0 while reporting no runtime;
the harness matches the version string instead.

### Phase 1 — Fail gracefully, then measure (1 day)

Convert **both** abort sites into the existing throw/catch clean-exit pattern
(`cbmc_adapter.cpp:791-793` → `src/esbmc/parseoptions/goto_program.cpp:283-287`):

1. `migrate.cpp:379` — `migrate_type0`'s fall-through, the *actual* first blocker (§2.3.1).
   This one additionally needs a descriptive prefix; today it logs a bare type dump, which
   is why Run 2 saw `ERROR: string`.
2. `migrate.cpp:2098-2099` — `migrate_expr`'s unknown-side-effect arm, already descriptive.

Then fix the core-dump-after-clean-error path from §2.3. Add a fixture pinning each new
graceful-error message — **do not reuse `cbmc_anon_aggregate`, which no longer pins that
behaviour** (§2.4). Then run the whole corpus and record, per program, the *first*
unsupported construct.

**Phase 1 no longer needs to discover its own first entry.** §2.3.1 already supplies it:
`@class_identifier : string` on `java.lang.Object`, blocking every program in the corpus.
Deciding how to represent `ID_string` — most plausibly as the class-tag integer or opaque
pointer that ESBMC's struct model can carry — is the first Phase 2 task, ahead of
`java_new_array_data`.

This is the single highest-value phase: it converts "ESBMC crashes on Java" into a
ranked, evidence-backed work-list, and it is useful to the CBMC roadmap independently
of Java.

*Exit:* a table of constructs ranked by how many corpus programs each blocks.
**Decision point:** if the list is dominated by constructs outside §2.5 — i.e. the
simple corpus was misleading — reassess scope before writing any adapter code.

### Phase 2 — allocation side effects (2–3 days)

Map CBMC's `side_effect statement="allocate"` and Java's `java_new_array_data` onto
ESBMC `sideeffect` allocation, following the `malloc`/`alloca` precedent in
`cbmc_adapter.cpp:1040-1067` and extending the statement dispatch at
`src/util/migrate.cpp:1953+`. Array allocation additionally carries a length operand
and JBMC's own `array-create-negative-size` check (already present as a plain assertion
in the binary — ESBMC should inherit it, not re-derive it).

**`java_new_array_data` is the harder half** and should be scheduled first, not second:
per §2.6 its semantics live in CPROVER's `symex_builtin_functions.cpp`, so this is not a
vocabulary mapping but a behavioural port. If it does not fit the `sideeffect` model in
~1 day, that is a Phase 2 decision point, not an overrun.

`allocate` is worth doing on its own merits: it is generic CPROVER vocabulary that any
`goto-instrument`-lowered binary may contain, so this benefits the CBMC roadmap too.

On Route A, this phase instead targets `java_new`/`java_new_array` directly — which is
why Route B is preferred: same effort, but Route B's two constructs also cover virtual
dispatch, `instanceof`, and exceptions for free.

*Exit:* T1–T3 corpus programs verify, verdicts matching JBMC.
**This is the PoC's minimum publishable result:** it demonstrates the whole chain works
on real Java.

### Phase 3 — Route B (parallel, 2–3 days + review latency)

Build the out-of-tree lowering driver **first** (load → lower → `write_goto_binary`),
confirm the emitted binary is lowered as §2.6 predicts, and re-run the corpus through it.
*Then* open an upstream PR against `diffblue/cbmc` for `jbmc --write-goto-binary`. Record
which `remove_exceptions` mode was measured (§2.6).

*Exit:* T4/T5 (virtual dispatch, exceptions) verify through the lowered route with no
further ESBMC adapter work — or a concrete statement of what still fails. Upstream merge
is explicitly **not** an exit criterion.

### Phase 4 — Java runtime models (spike only, 1 day)

Partly answered already by §2.2: models arrive in the binary **only** if `core-models.jar`
is put on the classpath explicitly, and doing so with `--no-lazy-methods` inflates the
model to ~2500 functions. Two things remain to determine:

- **Scale.** Is a 2500-function goto binary tractable for ESBMC's symex at all? This is
  the practical gate on non-trivial Java, and it is cheap to measure. Record wall-clock
  and peak RSS, not just pass/fail.
- **Coverage.** `core-models.jar` is narrower than its name suggests. The shipped jar
  contains **exactly 91 class entries** (85 top-level, 6 inner): 77 in `java/lang/`, 10 in
  `java/lang/reflect/`, 3 `sun/misc/`, 3 `org/sosy_lab/sv_benchmarks/`, 3
  `java/util/regex/`, and **exactly two in `java/util/` — of which the only
  collection-adjacent class is `Random`**. No `ArrayList`, `HashMap`, `LinkedList`,
  `List`, `Map`, `Set`, `Collection`, or `Iterator`. Unmodelled library methods become
  bodyless stubs, so a corpus touching collections is verifying against nothing.

  > **Do not check this against the `java-models-library` repo's current HEAD.** That HEAD
  > *does* contain `ArrayList.java`, `HashMap.java`, `LinkedList.java` and more. CBMC pins
  > a 2022 submodule commit (`c7835345`) which does not, and the **shipped jar is what
  > matters**. Inspect the jar, not the repo.

**`java.lang.String` is a special case and a genuine risk.**

> **Corrected from the original plan, which stated String is "not modelled by the JAR at
> all". That is false as literally written.** `java/lang/String.class` *is* in
> `core-models.jar` (22,469 bytes), alongside `StringBuilder`, `StringBuffer`,
> `AbstractStringBuilder` and `CharSequence`. But it is a **thin shim, not an
> implementation**: the 4,401-line source references `org.cprover.CProverString` 71 times.
> Real semantics come from `jbmc/src/java_bytecode/java_string_library_preprocess.cpp`,
> which maps `CProverString.*` signatures onto `ID_cprover_string_*` primitives, and from
> CBMC's **string refinement solver** — which `jbmc` enables **by default**
> (`jbmc_parse_options.cpp:98` sets `refine-strings` true; the opt-*out* is
> `--no-refine-strings`).

The risk is unchanged and arguably sharper: the presence of a String model in the jar is
misleading, because the model delegates to a solver ESBMC does not have. ESBMC has no
string-refinement backend (`string_refinement`, `string_constraint` and `refine_string`
have zero hits in `src/`), so string-heavy Java is out of reach through this route
regardless of how well IR ingestion works. **A binary built with string models present
will ingest and produce verdicts — those verdicts are not trustworthy.** Establish
whether the corpus can avoid strings, and if not, say so plainly.

*Exit:* a scale number, a coverage statement, and a yes/no on strings. **Do not build a
linking mechanism in this PoC.**

### Phase 5 — Report and regression fixtures (1 day)

Land whatever works as `regression/goto-transcoder/`-style fixtures — checked-in `.goto`
binaries with `test.desc`, matching the existing CBMC fixtures, including negative
`_fail` variants and a **new** unsupported-construct test pinning the graceful-error
message (§2.4). Write up the verdict, including the soundness statement from §4.1.

**Budget: ~7 working days plus upstream review latency**, with a go/no-go after Phase 1.
The estimate assumes no Phase-2 overrun on `java_new_array_data`; §7 records the
assumptions the budget rests on.

### 4.2 Testing and maintenance

- **Fixture durability.** Checked-in `.goto` binaries are opaque and version-locked. They
  are pinned to `GOTO_BINARY_VERSION 6`, which has held across 6.0.0–6.10.0 (§2.2), so the
  near-term risk is low — but a version 7 would invalidate every fixture at once. Each
  fixture must record the `jbmc` version that produced it, and the harness (Phase 0) must
  be able to regenerate them from source, so a bump is a re-run rather than an
  archaeology exercise.
- **CI cost.** Java fixtures require a JDK and a `core-models.jar` on the runner. If Java
  fixtures are gated behind a build flag as the jimple frontend is, state which flag; if
  they run unconditionally, the JDK becomes a CI dependency. Decide in Phase 5, not later.
- **Relationship to `src/jimple-frontend/`.** This route does not replace it, and no
  deprecation is proposed here. If the PoC succeeds, "two Java routes, both partial" is a
  maintenance question that needs an owner and an explicit decision.
- **Upstream tracking.** Route B ties ESBMC to CPROVER internals (`remove_java_new`'s
  choice of `ID_allocate`, `java_new_array_data`'s symex semantics). These are not a
  stable API. Breakage will show up as ingestion failures on a new `jbmc` release; the
  Phase 0 harness is what makes that cheap to diagnose.

---

## 5. What would make this a "no"

Stated in advance so the PoC can honestly fail:

- **Model completeness cannot be established.** §2.2 shows the model size swings 50→2500
  functions on flags alone. If we cannot state confidently that a given binary contains
  every method the program reaches, every verdict is relative to an unknown classpath and
  the whole exercise is unsound. This is the most serious risk, because it fails
  *silently* — a short model produces a clean SUCCESSFUL, not an error.
- **Scale.** If ESBMC's symex cannot handle a 2500-function model in reasonable time, the
  route works only for programs that touch almost no library code.
- **Strings.** JBMC delegates `java.lang.String` to CBMC's string refinement solver, on by
  default, which ESBMC does not have. If the corpus cannot avoid strings, this is a hard
  stop for a large class of real Java, independent of everything else in this plan.
- **`java.util` is unmodelled.** Collections-using code verifies against bodyless stubs.
- **`java_new_array_data` needs symex-level work.** If porting CPROVER's symex handling
  (§2.6) rather than mapping a side effect turns out to be the real Phase 2 cost, the
  "two constructs" framing understates the work.
- **Java struct conventions leak everywhere.** **Partly realised already** (§2.3.1):
  `@class_identifier` is a plain component, but carries type `ID_string`, which
  `migrate_type0` cannot represent. That is one type mapping, not a cross-adapter leak, so
  the premise holds for now — but it is the first evidence on this axis, and a second such
  finding (e.g. `java::array[T]` needing bespoke layout) would be a genuine warning.
- **`mode == "java"` symbols.** `cbmc` fails on JBMC binaries partly because it has no
  handler for Java-mode symbols (§2.2). ESBMC must answer the same question; if the answer
  requires a language handler, "no Java frontend" is harder to hold.
- **Verdict divergence.** If ESBMC and JBMC disagree on corpus programs, the *reason*
  matters more than the count: a JVM-semantics gap is a no; a known ESBMC pointer-model
  difference is a tracked bug.

### Risks accepted without mitigation in this PoC

Named so they are not mistaken for oversights: JVM concurrency, reflection, generics,
`invokedynamic`/lambdas, and performance are all out of scope (§6). Any of them appearing
in the Phase 1 work-list is a scope signal, not a task.

---

## 6. Explicit non-goals

Not a Java frontend. No `.class`/`.jar` parsing in ESBMC. No replacement of, or
integration with, `src/jimple-frontend/`. No JVM concurrency, reflection, generics, or
`invokedynamic`/lambdas. No performance work. No string-refinement backend.

---

## 7. Reproducing the §2 probes

Portable across the distributions in §2.2. Run 2 executed the first block on cbmc 6.8.0 /
macOS / aarch64.

```sh
# Locate the models jar rather than assuming a path (see §2.2).
MODELS=$(ls /usr/lib/core-models.jar \
            /opt/homebrew/Cellar/cbmc/*/libexec/lib/core-models.jar 2>/dev/null | head -1)
[ -n "$MODELS" ] || { echo "core-models.jar not found; results would be meaningless"; exit 1; }
jbmc --version; cbmc --version; esbmc --version

javac Test.java
jbmc Test -cp .:$MODELS --no-lazy-methods --show-symbol-table --json-ui > symtab.json
python3 -c "import json;d=json.load(open('symtab.json'));\
json.dump([e for e in d if 'symbolTable' in e][0], open('st.json','w'))"
symtab2gb st.json --out jbmc.goto
xxd jbmc.goto | head -1                       # expect: 7f47 4246 06…
goto-instrument --show-goto-functions jbmc.goto | head    # round-trip sanity
esbmc --binary jbmc.goto --goto-functions-only; echo "exit=$?"   # expect 134 (SIGABRT) today

# Model size vs flags (§2.2) — expect roughly 50 / 52 / 72 / 2500
for cp in "-cp ." "-cp .:$MODELS"; do for lz in "" "--no-lazy-methods"; do
  printf '%-42s %s\n' "$cp $lz" "$(jbmc Rich $cp $lz --list-goto-functions 2>/dev/null | wc -l)"
done; done

# Route A vs Route B construct survival (§2.6).
# -oE with \b is required: a plain `grep -c java_new` also matches java_new_array.
jbmc Rich -cp .:$MODELS --no-lazy-methods --show-goto-functions > routeB.txt
goto-instrument --show-goto-functions jbmc.goto > routeA.txt
for k in 'java_new\b' 'java_new_array\b' 'java_new_array_data\b' \
         'java_instanceof\b' 'virtual_function\b' 'CATCH' 'statement="allocate"'; do
  printf '%-26s A=%-3s B=%s\n' "$k" \
    "$(grep -oE "$k" routeA.txt | wc -l)" "$(grep -oE "$k" routeB.txt | wc -l)"
done
```

`Rich.java` is the §2.5 corpus program: a single file exercising inheritance, virtual
dispatch, arrays, `instanceof`, and `try`/`catch`, and **no strings**. Phase 0 checks it
in alongside the harness; until then the absolute counts above are not reproducible from
this document alone — only the Route A → Route B *direction* is.

**Run 2 alternative with no JDK.** The pipeline can be exercised without compiling
anything by using a class already inside the models jar, which is how the replication in
§2.2/§2.6 was performed:

```sh
jbmc java.lang.Integer -cp $MODELS --show-symbol-table --json-ui > symtab.json
```

**Run 3: locating the first blocker (§2.3.1).** The abort site is not visible from the
message, so attach a debugger rather than reading stderr:

```sh
lldb -b -o "b migrate.cpp:379" -o "run" \
     -o 'expr (const char*)type.id().c_str()' -o "up 2" \
     -o 'expr (const char*)type.get("tag").c_str()' \
     -- esbmc --binary jbmc.goto --goto-functions-only
# expect: $0 = "string"   (the unrepresentable type)
#         $1 = "java.lang.Object"   (the struct carrying it)

# Confirm the component name straight from the symbol table:
python3 -c "import json;t=json.load(open('st.json'))['symbolTable']['java::java.lang.Object']['type'];\
print([(c['namedSub']['name']['id'], c['namedSub']['type']['id']) \
       for c in t['namedSub']['components']['sub']])"
# expect: [('@class_identifier', 'string'), ('cproverMonitorCount', 'signedbv')]
```

### Assumptions the ~7-day budget rests on

1. Phase 2's `java_new_array_data` work fits the existing `sideeffect` model (§2.6 flags
   this as the likeliest overrun).
2. ~~The Phase 1 diagnostic names the offending construct (§2.3 Run 2 caveat).~~
   **Falsified by Run 3 (§2.3.1)**: at `migrate_type0`'s abort site it does not. Phase 1
   absorbs the fix (add a prefix); the budget impact is hours, not days, but the
   assumption should not be carried forward silently.
3. A string-free corpus is achievable at every tier (§5).
4. Phase 3's out-of-tree driver is written before, and independently of, upstream review.

---

## 8. Open questions for upstream

1. Would `diffblue/cbmc` accept a `jbmc --write-goto-binary` flag? (No such writer exists
   in `jbmc/` today, and `--outfile` is the SMT/DIMACS formula, not GOTO — an easy trap.)
2. Is `symtab2gb` on JBMC symbol tables a supported use, or an accident that may break?
   It works empirically on 6.5.0 and 6.8.0, but `symtab2gb` does not link the Java
   frontend (§2.6), so the input is arguably outside its intended domain.
3. Is there a recommended way to state, for a given binary, that class loading was
   complete for the program's reachable set — or is classpath discipline the only answer?
   (§5 treats this as the PoC's most serious soundness risk.)
4. Is `remove_java_new`'s lowering to `ID_allocate` + `ID_java_new_array_data` a stable
   contract, or an implementation detail that may change? Route B's ESBMC-side work
   depends on it.

### Version note

Run 1 probed `jbmc`/`cbmc` **6.5.0**; Run 2 replicated on **6.8.0**; upstream `develop` is
at **6.10.0**. `GOTO_BINARY_VERSION` is **6** at every 6.x release tag from 6.0.0 through
6.10.0, so the format conclusions hold across the line. Phase 0 should still re-confirm
the construct inventory against a current build before any upstream PR is opened.

---

## 9. References

- L. C. Cordeiro, P. Kesseli, D. Kroening, P. Schrammel, M. Trtík, "JBMC: A Bounded Model
  Checking Tool for Verifying Java Bytecode", CAV 2018, LNCS 10981, pp. 183–190
  ([doi:10.1007/978-3-319-96145-3_10](https://doi.org/10.1007/978-3-319-96145-3_10)).
- L. Cordeiro, D. Kroening, P. Schrammel, "JBMC: Bounded Model Checking for Java Bytecode
  (Competition Contribution)", TACAS 2019, LNCS 11429, pp. 219–223
  ([doi:10.1007/978-3-030-17502-3_17](https://doi.org/10.1007/978-3-030-17502-3_17)).
- R. Brenguier, L. Cordeiro, D. Kroening, P. Schrammel, "JBMC: A Bounded Model Checking
  Tool for Java Bytecode", arXiv:2302.02381 (2023)
  ([doi:10.48550/arXiv.2302.02381](https://doi.org/10.48550/arXiv.2302.02381)). An
  extended technical report; **arXiv preprint, not peer-reviewed** — no DBLP record beyond
  the CoRR entry. Note the author list differs from the CAV 2018 paper (adds Brenguier,
  drops Kesseli and Trtík); it is not a substitute citation for it.

All three describe JBMC's architecture and Java operational model. **None of the three
documents lazy class loading or goto-binary output** — verified by full-text search of
each. For those behaviours the source is authoritative
(`jbmc/src/java_bytecode/lazy_goto_model.h`, `java_bytecode_language.cpp:109`), as is
`jbmc/src/java_bytecode/README.md` (887 lines; exception lowering at :467-510,
`java_new_array_data` example at :439) and
[diffblue/java-models-library](https://github.com/diffblue/java-models-library) for
`core-models.jar` — subject to the pinned-commit caveat in Phase 4. Note that
<https://diffblue.github.io/cbmc/> documents CBMC/C only and has no Java section; JBMC
documentation is effectively in-repo.
