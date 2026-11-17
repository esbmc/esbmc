# Scope: field-sensitive SSA in goto-symex (#7521)

> **Status at 2026-11-17: Phase 0 merged (#10773); Phase 1 merged (#10778)
> before its trace gate was met. #10802 makes a split whole-struct write print
> its right-hand side, as master does, and hides that step where the slicer
> dropped a definition the right-hand side reads; with it, every remaining
> trace change is one §8.6 accepts (§8.6). SV-COMP not yet run.** The corpus is in `regression/field-sensitivity/`
> with baselines, and the census has been run and reverted. Phase 1 adds
> `--field-sensitive-ssa` (default off): symex gives each field of an
> eligible local, parameter or return slot its own SSA name. It meets the 2x
> bar on the Phase 1 corpus rows and changes no verdict on the C, C++ (Z3 and Bitwuzla) and
> k-induction suites or three Python samples (3132 tests), and no correctness witness, but it
> still changes some trace text and violation witnesses (§8.5).
> §8.6 traces the remaining changes to §3's own design; the Phase 1 trace gate
> is now Invariant 4 (same values), with those changes accepted. Two
> mitigations are merged: #10037 stopped phi merges of wide structs from
> scaling with the square of the width, and #10527 added the opt-in GOTO pass
> `--scalarize-structs`. Line numbers are from master at `d3104e99c2`.

## 1. What is left after #10037 and #10527

`goto_symext::symex_assign_member` (`src/goto-symex/engine/symex_assign.cpp:1059`)
turns `a.c = e` into `a' == a WITH [c := e]` (`:1108`), and
`goto_symex_statet::assignment` (`src/goto-symex/state/goto_symex_state.cpp:895`)
asserts that the left-hand side is a whole symbol. Every field write therefore
emits an SSA value as wide as the struct.

Measured on 2026-11-14 (master `bbdd80dfc1`, symex time only):

| Program | 8 fields | 2048 fields | `--scalarize-structs`, 2048 |
|---|---|---|---|
| #7521 reproducer (field written once, read in a loop) | 0.35 s | 0.29 s (500 fields) | 0.63 s |
| local struct, 8 field writes x 200 cycles | 0.013 s | 0.71 s | 0.013 s |
| same, address of the struct taken once | 0.012 s | 0.68 s | 0.71 s |
| global struct written through `db->f` in a callee | 0.16 s | 2.2 s | 2.1 s |

The first row is fixed: #7521 measured 53 s at 501 fields on that program on
8.5.0, and master after #10037 takes 0.29 s. The remaining cost is field writes, which still scale
with `writes x width`. `--scalarize-structs` removes it only for structs whose
address is never taken (`src/goto-programs/scalarize_structs.cpp:82-83`). The
PLC models that motivated #7521 pass each data block to the cycle function by
pointer, which is the last row, so the pass does not reach them.

On about 850 Python regression tests, with and without `--k-induction`, the
pass changes no verdict; total CPU time moves between -0.1 % and +4.8 %, and
single tests swing both ways through solver heuristics. Python is not a target
of this scope.

## 2. Goal and non-goals

**Goal.** Give each field of an eligible struct object its own L1/L2 name, so
that a field write costs one scalar SSA step and a phi merges only the fields
that differ. Equation size becomes O(writes), not O(writes x width), for:

1. structs whose address is never taken, including struct-typed parameters and
   return values, which `--scalarize-structs` excludes (`:146`, `:119-120`);
2. structs reached through pointers whose every dereference resolves to a
   constant field path (the PLC `cycle(&db)` shape).

**Non-goals.**
- Unions, and any struct accessed at byte level (`memcpy`, `memset`, a cast of
  its address, a dynamic offset into it). These keep one SSA name.
- Changing the value-set representation. Pointer targets stay
  `(object, offset)` pairs (`src/pointer-analysis/value_set.cpp:1376-1419`).
- Making the pass default-on before Phase 4's gates are met.
- #10731 (a loop writing through a pointer into one member havocs the whole
  struct under k-induction). That is a havoc-target problem in
  `goto_k_induction.cpp:194`, fixable without this scope. Phase 3 makes the
  per-field havoc it needs cheaper.

**Success bar.** On the Phase 0 corpus, symex time and formula size at 2048
fields are within 2x of 8 fields for every eligible shape, with no verdict
change on the C, C++ and Python suites and on an SV-COMP run.

## 3. Design

A field symbol is a new context symbol per field of an eligible object, named
`<object id>::<field index>`. The name must not add a `.` (value-set entries
use `.member` suffixes, `value_set.cpp:1823-1891`) or any substring that
k-induction's havoc filters match; `scalarize_structs.cpp:169-170` uses the
same scheme for that reason.

Symex rewrites expressions before renaming:
- `member2t(s, f)` of an eligible `s`, with a constant field path, becomes the
  field symbol.
- A read of the whole `s` becomes `constant_struct2t` of its field symbols.
- A write of the whole `s` becomes a write to that `constant_struct2t`, which
  `symex_assign_structure` (`symex_assign.cpp:738`) already splits into
  member-wise assignments.

Everything keyed on symbol names then works per field without change:
L2 renaming (`renaming.cpp:155-227`), phi merges (`symex_goto.cpp:744-835`),
the slicer (`slice.cpp:167`), and equation renumbering
(`symex_target_equation.cpp:546`).

Places that need work:

| Place | Change |
|---|---|
| `symex_decl` (`symex_other.cpp:72`) | declare every field symbol and add it to `new_context` |
| `symex_dead`, `pop_frame` (`symex_function.cpp:1091-1144`) | retire every field; `ns.lookup` must find each one |
| argument binding, return slot (`symex_function.cpp:309-320`, `:1151-1181`) | treat struct parameters and the return value as eligible objects |
| `constant_propagation` struct arm (`goto_symex_state.cpp:448-504`) | bypassed for eligible objects: each field propagates on its own |
| dereference (`symex_dereference.cpp:226-258`) | constant-offset results (`dereference.cpp:2791`) map to field symbols; dynamic-offset and byte paths make the object ineligible up front |
| trace (`build_goto_trace.cpp:93-189`) | reassemble whole-struct values from field values where a step prints a struct |
| interval domain havoc in `pop_frame` (`:1109-1111`) | havoc per field |

## 4. Phases

Each phase lands behind `--field-sensitive-ssa` (experimental, default off)
and ends at a gate. A phase that misses its gate is not merged.

### Phase 0: corpus and census

- Add a benchmark corpus under `regression/field-sensitivity/`: the four
  shapes in §1 at widths 8, 512 and 2048; the #7521 reproducer; a PLC-shaped
  model with a data block passed by pointer; and struct-heavy SV-COMP tasks.
- Add a census (debug option, reverted after measuring) that classifies every
  struct object symex sees: never address-taken; address-taken with only
  constant field paths; byte-level; union. Run it over the C, C++ and Python
  suites and the SV-COMP set.

**Gate:** corpus committed with recorded baselines for symex time, formula
size and solver time under Bitwuzla and Z3; census numbers recorded in §8.

### Phase 1: structs whose address is never taken

- Eligibility: locals, parameters and return slots of fixed-size struct type,
  with no `address_of` anywhere. Reuse `is_address_taken_local`
  (`reachability_tree.h:503`, filled by `scan_address_taken`,
  `reachability_tree.cpp:317-329`).
- Implement the rewrite, DECL/DEAD/pop_frame handling and trace reassembly.
- Disable for multithreaded programs in this phase, as `scalarize_structs.cpp:335`
  does.

**Gate:** the flag forced on for the C, C++ and Python suites changes no
verdict and keeps Invariant 4; the trace text may change only in the four
ways §8.6 accepts, each pinned by a test; the local-struct rows of the corpus
meet the 2x bar;
every new branch proved live (Mode C-Live), dual solver.

### Phase 2: structs reached through pointers

- Eligibility is a whole-program property of the object: every dereference
  whose points-to set includes it has a constant field path of the matching
  type. Compute it before symex with the Andersen analysis
  (`src/pointer-analysis/andersen.h`), which is sound and flow-insensitive, so
  one imprecise pointer makes the object ineligible rather than unsound.
- Escapes that keep the object whole: a dynamic offset, `byte_extract` or
  `byte_update` on it, a cast of its address to another type, a `memcpy` or
  `memset` over it, a union overlay, and pointer arithmetic that leaves a field.
- Global objects: L2 versions of globals carry the thread number
  (`renaming.h:146-227`). Keep this phase single-threaded until Phase 4.

**Gate:** as Phase 1, plus the address-taken and pointer rows of the corpus and
the PLC model meet the 2x bar; an SV-COMP run (`needs-svcomp-run`) shows no
lost correct verdicts against a same-week master run with the same timeout and
options.

### Phase 3: nested structs, arrays of structs, k-induction

- Nested structs: flatten field paths (`s.a.b`), bounded by a field-count cap
  per object (start at 4096, measured in Phase 0) that mirrors
  `multidim_propagation_bound` (`goto_symex_state.cpp:102`).
- Arrays of structs with constant indices only; a symbolic index keeps the
  array whole.
- k-induction and loop-invariant havoc: `intrinsic_havoc_object`
  (`memory_ops.cpp:1865-1886`) and `havoc_targets` (`goto_k_induction.cpp:194`)
  havoc field symbols, not the whole object.

**Gate:** as Phase 2, plus the k-induction suite forced on with the flag.

### Phase 4: default on

- Lift the single-thread restriction or keep it as an explicit exclusion,
  decided on Phase 2's census of multithreaded struct use.
- Make `--field-sensitive-ssa` the default and retire `--scalarize-structs`
  once its cases are a subset.

**Gate:** full suites and a 900 s SV-COMP run with no lost correct verdicts;
witnesses validate as before (`witnesses_validate` suite).

## 5. Invariants every phase keeps

1. Each field symbol has exactly one context symbol of the right type; phi
   merges (`symex_goto.cpp:786`), `pop_frame` (`symex_function.cpp:1117`) and
   `is_live_variable` (`symex_dereference.cpp:179`) look it up.
2. Eligibility is decided before symex and never changes during it. An object
   is either always split or never split.
3. No byte-level access reaches a split object. A path that would need one is
   an eligibility bug, and debug builds assert on it.
4. Traces and witnesses print the same values with and without the flag:
   every step both runs print shows the same value for every lvalue the run
   without the flag pins down. Steps may be added or dropped, and a struct may
   print field by field (§8.6).

## 6. Risks

- **Symbol names.** A field name containing `.` or a havoc-filter substring
  breaks the value set or k-induction silently. Covered by the naming rule in
  §3 and by a regression that runs k-induction on a split struct.
- **Name count.** 2048 fields x many frames multiplies L1/L2 entries. The
  field-count cap bounds it; Phase 0 measures memory.
- **Trace output.** Traces and witnesses key on `original_lhs`
  (`goto_trace.cpp:946-952`, `witnesses.cpp`, `json.cpp`, `pytest.cpp`). A
  printing change is an interface change for `esbmc-wrapper.py`, so each phase
  checks it against `parse_result()`.
- **Overlap with #9785.** #10541 says the rest of that task's cost is symex's
  per-member store encoding. Phase 2 removes that encoding for eligible objects
  only; integer-derived pointers stay whole.
- **Duplicate work.** `--scalarize-structs` and Phase 1 cover the same
  never-address-taken case. Keep both until Phase 4 so either can be bisected.

## 7. Related

#7521 (this scope), #1344 (merge overhead, fixed), #9785 and #10541 (store
encoding through integer-derived pointers), #10037 (phi merges of wide
structs), #10527 (`--scalarize-structs`), #10731 (k-induction havoc of a
pointer-written member).

## 8. Census and measurements

### 8.1 Corpus

`regression/field-sensitivity/` holds five shapes at 8, 512 and 2048 fields,
each as a pair: a safe program expecting `VERIFICATION SUCCESSFUL` and a
`_fail` variant that changes one constant and expects `VERIFICATION FAILED`.
Every row pins `--z3` and unwinds its loops fully.

| Shape | Program |
|---|---|
| `repro7521` | the #7521 reproducer: one field written, read in a 100 x 20 x 10 loop |
| `local_writes` | a local struct, 8 field writes x 200 cycles, nondet start value |
| `local_addr_once` | the same, with the first write made through `&s` |
| `global_callee` | a global struct written through `d->f` in a callee, 200 calls |
| `plc_db` | a PLC data block (`_Bool`, `short`, `int` tags) passed by pointer to the cycle function, 20 cycles, with branches |

Z3 does not finish `plc_db` at 512 or 2048 fields; after 30 s it holds
1 GB and 4 GB. `test.desc` cannot set a timeout, so those four rows run with
`--smtlib --output /dev/null` and pin only that symex finishes and emits a
non-trivial formula. The bench script removes those flags and measures their
verdicts. A phase that brings them within budget turns them into verdict
tests. The struct-heavy SV-COMP tasks are covered by the census
in §8.3 rather than copied into the corpus.

### 8.2 Baselines

Master `a1d98c9a1f`, RelWithDebInfo, Z3 4.13.3, Bitwuzla 0.10.0 (PyPI
bindings), measured on 2026-11-15 on one 4-core container with
`scripts/field_sensitivity_bench.py`. Symex time and SSA steps are from the
Z3 run. "SMT-LIB bytes" is the size of the `--smtlib --output` dump, and the
"Bitwuzla s" column is the time to parse and solve that dump; §8.2.1 times
ESBMC's own Bitwuzla backend. The SMT-LIB backend encodes tuples differently from
Z3's native API, so on `plc_db` the dump stays the same size at every width
while Z3 times out.

| Row | Symex s | SSA steps | After slicing | SMT-LIB bytes | Z3 s | Bitwuzla s | Peak MiB |
|---|---|---|---|---|---|---|---|
| repro7521_w8 | 0.725 | 24415 | 1 | no VCC | - | - | 122 |
| repro7521_w512 | 0.686 | 24415 | 1 | no VCC | - | - | 122 |
| repro7521_w2048 | 0.685 | 24415 | 1 | no VCC | - | - | 124 |
| repro7521_w2048_fail | 0.675 | 26415 | 2 | 2411 | 0.000 | 0.001 | 143 |
| local_writes_w8 | 0.544 | 1821 | 1606 | 108814 | 0.016 | 0.004 | 365 |
| local_writes_w512 | 1.418 | 1821 | 1606 | 147166 | 0.708 | 0.004 | 561 |
| local_writes_w2048 | 4.514 | 1821 | 1606 | 267998 | 4.995 | 0.005 | 1261 |
| local_addr_once_w8 | 0.602 | 1822 | 1606 | 108814 | 0.017 | 0.003 | 365 |
| local_addr_once_w512 | 1.390 | 1822 | 1606 | 147166 | 0.775 | 0.004 | 562 |
| local_addr_once_w2048 | 4.242 | 1822 | 1606 | 267998 | 5.079 | 0.004 | 1262 |
| global_callee_w8 | 0.162 | 2021 | 1605 | 108254 | 0.014 | 0.003 | 136 |
| global_callee_w512 | 0.713 | 2021 | 1605 | 108254 | 0.131 | 0.003 | 170 |
| global_callee_w2048 | 2.601 | 2021 | 1605 | 108254 | 0.461 | 0.003 | 284 |
| plc_db_w8 | 0.021 | 455 | 276 | 220423 | 6.393 | 0.024 | 151 |
| plc_db_w512 | 0.131 | 455 | 276 | 220423 | >120 | 0.024 | 1155 |
| plc_db_w2048 | 0.430 | 455 | 276 | 220423 | >120 (>900 once) | 0.024 | 4057 |

The `_fail` rows differ from their safe rows by at most 22 % in symex time.
Their Z3 times are up to 57 % higher (`global_callee_w2048_fail`: 0.72 s),
except `plc_db_w8_fail`, which takes 3.1 s. Against the 2x bar (2048 fields vs 8 fields),
symex time is 0.94x on `repro7521`, which already passes, 8.3x on
`local_writes`, 7.0x on `local_addr_once`, 16x on `global_callee` and 20x on
`plc_db`. Z3 time grows 312x on `local_writes`, 33x on `global_callee`, and
from 6.4 s to over 120 s on `plc_db`. Peak memory grows 3.5x on
`local_writes` and 27x on `plc_db`.

#### 8.2.1 ESBMC's Bitwuzla backend

Same binary and container, built with ESBMC's Bitwuzla backend (Bitwuzla
0.9.0, the version `DOWNLOAD_DEPENDENCIES` pins, built from git with
CaDiCaL 2.1.2), measured on 2026-11-15 with
`scripts/field_sensitivity_bench.py --solver bitwuzla`. All 30 rows give
their expected verdict.

| Row | Symex s | Bitwuzla s | Peak MiB |
|---|---|---|---|
| repro7521_w8 / w512 / w2048 | 0.705 / 0.641 / 0.687 | no VCC | 122 / 122 / 125 |
| local_writes_w8 / w512 / w2048 | 0.565 / 1.369 / 4.717 | 0.002 / 0.002 / 0.001 | 344 / 345 / 371 |
| local_addr_once_w8 / w512 / w2048 | 1.026 / 1.300 / 4.617 | 0.001 / 0.001 / 0.002 | 345 / 345 / 371 |
| global_callee_w8 / w512 / w2048 | 0.161 / 0.821 / 2.516 | 0.001 / 0.003 / 0.002 | 122 / 132 / 196 |
| plc_db_w8 / w512 / w2048 | 0.021 / 0.117 / 0.422 | 0.028 / 0.017 / 0.021 | 122 / 128 / 204 |

Bitwuzla's solve time stays flat with width on every row, and it decides
the `plc_db` rows at 512 and 2048 fields, including their `_fail` variants,
in under 0.03 s, where Z3 runs past 120 s and 4 GB. Symex does not depend on the
solver and its times are close to the Z3 run, so the 2x bar is missed on the same rows under both
solvers, and the gap there is symex, not solving. The `plc_db` rows at 512
and 2048 fields stay pinned to formula generation because the regression
tests pin Z3.

### 8.3 Census

A debug option, `--field-sensitivity-census` (added and reverted in the
Phase 0 PR), classified each struct or union object (by L0 name) that symex
reached: never address-taken; address-taken with every dereference resolving
to a constant offset and building no `byte_extract`, `byte_update` or
`bitcast`; byte-level (a dynamic offset, a reinterpreting access, a byte
operation in the program, or `memcpy`/`memmove`/`memcmp`/`memchr`/`memset`
over it); union. Leaves count flattened members, including padding members.
Each test ran with its `test.desc` flags under a 30 s limit, and each suite
under a 570 s budget. Counts are summed over tests, so an object in the
libc or C++ operational models counts once per test that reaches it.

| Set | Tests with a census | Objects | Never address-taken | Constant field paths | Byte-level | Union | Max leaves |
|---|---|---|---|---|---|---|---|
| C (`regression/esbmc`) | 3308 / 3439 | 2018 | 1064 (53 %) | 585 (29 %) | 180 (9 %) | 189 | 201 |
| C++ (`esbmc-cpp/cpp`) | 1437 / 1466 | 37608 | 22581 (60 %) | 11660 (31 %) | 3349 (9 %) | 18 | 49 |
| Python (`regression/python`) | 9042 / 9739 | 55678 | 22293 (40 %) | 31663 (57 %) | 1722 (3 %) | 0 | 21 |
| SV-COMP sample (259 tasks: ldv-memsafety, ldv-regression, list-properties, forester-heap, heap-manipulation, ntdrivers-simplified; `--unwind 3`) | 257 / 259 | 263 | 13 (5 %) | 209 (79 %) | 33 (13 %) | 8 | 18 |
| Corpus | 30 / 30 | 30 | 12 | 18 | 0 | 0 | 2049 |

Further counts. Struct objects that are globals: 886 in C, 32414 in C++,
32999 in Python and 109 in SV-COMP. Heap objects: 182, 75, 12566 and 89.
Objects with a nested struct or union member: 142, 4936, 340 and 23. In C,
294 of the 732 tests that reach a struct reach one that is address-taken with
constant field paths only, and 151 reach a byte-level one. No object in any
set has more than 4096 leaves, and outside the corpus none has more than 201,
so the Phase 3 cap of 4096 binds only synthetic inputs.

The Python row was measured on 2026-11-15 with Python 3.11, 4 tests in
parallel, each under a 30 s limit, in six 570 s batches. 697 tests printed
no census, because they stop in the frontend or hit the limit (not split).
In this run the first 1192 tests give 8267 objects (never address-taken
4026, constant field paths 4073, byte-level 168), against 8762 (4023, 4570,
169) in the earlier partial run; the difference is in the constant field
path and heap counts.

On the corpus, the census put `repro7521` and `local_writes` in Phase 1
(never address-taken), and `local_addr_once`, `global_callee` and `plc_db` in
Phase 2 (constant field paths), as designed.

### 8.4 Not measured in Phase 0

- The census on the full SV-COMP set.

### 8.5 Phase 1 measurements

Branch `feat/field-sensitive-ssa-phase1` on master `f1c5ee1b4`, RelWithDebInfo,
Z3 4.13.3 and ESBMC's Bitwuzla 0.9.0 backend, one 4-core container,
2026-11-15. Symex seconds from `scripts/field_sensitivity_bench.py
--flags=--field-sensitive-ssa`; the flag changes no SSA step count on these rows.

| Row | 8 fields | 512 | 2048 | 2048 / 8 | Without the flag, 2048 |
|---|---|---|---|---|---|
| `local_writes` (Z3) | 0.017 | 0.018 | 0.019 | 1.1x | 6.39 |
| `local_writes` (Bitwuzla) | 0.019 | 0.018 | 0.021 | 1.1x | 6.28 |
| `repro7521` (Z3) | 0.794 | 0.931 | 1.249 | 1.6x | 0.83 |
| `repro7521` (Bitwuzla) | 0.792 | 0.875 | 1.146 | 1.4x | 0.84 |

Peak memory on `local_writes_w2048` falls from 1262 MiB to 135 MiB (Z3) and
from 370 MiB to 123 MiB (Bitwuzla); wall time from 19.1 s to 0.30 s with Z3.
`local_addr_once`, `global_callee` and `plc_db` are Phase 2 rows: nothing in
them is split and their times match the unflagged run.

The rest of `repro7521`'s cost is per call: `fun` is called 100 times and
each call declares, initialises and retires 2048 fields. A nondet write of a
whole split object gives each field narrower than a pointer a fresh L2 name
instead of an assignment step, and `pop_frame` skips the value-set erase for
those fields; before both, the row took 13.9 s at 2048 fields.

Flag forced on against off, each `test.desc` command run twice, 30 s per run:

| Suite | Tests | With a split object | Verdict changes | Output changes |
|---|---|---|---|---|
| C (`regression/esbmc`) | 3443 | 590 | 0 | 49 |
| C++ (`esbmc-cpp/cpp`) | 1466 | 823 | 0 | 9 |
| Python (first 726 of 9749) | 726 | 24 | 0 | 0 |

In C++, 7 of the 9 output changes are k-induction or parallel-solving output
order and one (`ch16_4`) is a counterexample that violates one of two
assertions on the same line instead of both. Of the 49 C output changes, 19 are timing
lines the comparison did not filter. The remaining 30, all with the same
verdict:

- 7 print the member written (`v.un.i = 9`) where master prints the whole
  struct that symex rewrote (`v.un.i = { .un=9, .t=0 }`).
- 7 lose a whole-struct step because the slicer keeps the fields a property
  needs, not the whole object; with `--no-slice` the step comes back.
- 6 print other values: a pointer member of a printed struct reads `0`
  where master prints `{ .pointer_object=nil, .pointer_offset=nil }`, and an
  uninitialised struct prints `{ .a = 0 }` where master prints every member.
  Master itself prints these differently under Z3 and Bitwuzla.
- 3 (`github_4309-struct`, `github_4312-*`) enumerated fewer
  `--all-witnesses` inputs; the flag is now ignored with `--all-witnesses`
  and the test-generation options, which read inputs off SSA steps.
- The rest are k-induction or parallel-solving output order. In C++,
  `github_7797_flat_set` unwinds a loop past iterations master rules out by
  constant propagation; the extra unwinding assertions pass.

Witnesses, k-induction and more Python, measured on 2026-11-16 at
`34ecc9021` (master `f1c5ee1b4`), Z3 4.13.3 only (Bitwuzla's dependencies
could not be downloaded in that container), `CORE` tests, flag forced on
against off, 30 s per run. Output is compared after dropping timings, temp
paths and the "is ignored" warning.

| Suite | Tests run | With a split object | Verdict changes | Witness file changes | Other output changes |
|---|---|---|---|---|---|
| `k-induction`, `k-induction-parallel` | 277 | 78 | 0 | not written | 0 (38 line order only, all in `-parallel`, none split) |
| C, expects `VERIFICATION FAILED`, GraphML + YAML | 1516 | 242 | 0 | 12 GraphML, 2 YAML | only §8.6 tests and `github_1087`, `github_1093` |
| C, expects `VERIFICATION SUCCESSFUL`, GraphML + YAML | 1617 | 282 | 0 | 0 | `github_5868_div`, `github_7597-literal` (§8.6) |
| Python, every 8th test after the first 726 | 706 | 60 | 0 | not written | 0 |

Every violation-witness change is in a test that also prints a different
trace, so the witnesses follow §8.6 and add no new cause. Two tests are new
to the list, both with a different counterexample but the same property
verdicts: `github_1087` picks another value for `num_players`, and
`github_1093` reports the NULL dereference in `strncpy` before the invalid
free at line 47. With `--multi-property`, `github_1093` gives the same 5
FAILED and 28 PASSED claims with and without the flag. One Python output
difference (`github_5444_dictcomp_param_fail`, `ESBMC_compvar_0_v` against
`_1_v`) also shows up between two runs without the flag.
`github_2153_5` took 32.5 s without the flag and 32.1 s with it, nothing
split; it times out at 30 s either way.

Bitwuzla, measured on 2026-11-16 with the branch merged into master
`6c46544cd`, ESBMC's Bitwuzla 0.9.0 backend as the default solver (tests that
pin `--z3` keep it), `CORE` tests, flag forced on against off, 30 s per run.
Output is compared after dropping timings, temp paths, the "splits" notice and
the assignment counts in the VCC summary lines.

| Suite | Tests run | With a split object | Verdict changes | Output changes |
|---|---|---|---|---|
| `k-induction`, `k-induction-parallel` | 277 | 79 | 0 | 0 outside `-parallel`; 41 in `-parallel`, none split, and two runs without the flag differ in 42 of 42 |
| Python, the next 1700 tests after the first 726, skipping every 8th | 1700 | 236 | 0 | 2 split, 5 not split |

The 5 unsplit Python changes are names built from heap addresses
(`ESBMC_unpack_temp_<address>`) and temp directories, which also differ
between two runs without the flag. The 2 split ones,
`github_10323_list_reversed_fail` and `github_10493_ir_fail`, print the same
steps and violate the same property, but the solver gives another value to
a nondet list element (`*(unsigned long int *)copied = 0` against `-1`), as
`github_1087` does in the C witness run. 4 Python tests time out both ways.

C++ under Bitwuzla, measured on 2026-11-16 with the branch merged into master
`de72a905e` (LLVM 18), each `CORE` test of `regression/esbmc-cpp/cpp` run
with and without the flag, 30 s per run, Bitwuzla unless the test pins a
solver: 1355 tests, 715 split, 26 ignore the flag (threads), 0 verdict
changes, 9 fail to parse or error out both ways. Apart from the flag's own
status line, 41 differ in output and none in a counterexample trace:

- 37 print another assignment count in `Generated N VCC(s)`.
- `github_7797_flat_set`: two loop guards that master folds stay symbolic, so
  two loops unwind to the `--unwind 6` bound; the unwinding assertions hold
  and symex plus solving takes 0.91 s against 1.03 s. Same cause as the second
  row of §8.6.
- `cpp_sum_class` (`--k-induction-parallel` order),
  `tuple_structured_binding` (`--incremental-bmc` folds a claim at another
  step) and `intrinsic_unroll_while_decl_true` (temporary directory in
  `--verbosity 9` output).

Flag off, `ctest -L esbmc-cpp/cpp` on the same build: 1458 of 1466 pass; the
8 failures (`ch8_5`, five `github_7433*`, two `github_8698_*_uncaught_fail`)
are the 8 reported above, and `github_7433_throw_site` prints `my_error`
where the test expects `struct my_error` under LLVM 18.

Still not measured in Phase 1: the other 6395 Python tests, the C and
witness suites again under Bitwuzla on this merge, and an SV-COMP run.

The `static llvm-22 DebugOpt` job on `2fb2595b6` failed on
`regression/consensus/integer_squareroot` (loop invariant base case) and
`regression/python/dict23` (120 s timeout). That job tested the PR merged into
master `f1c5ee1b4`. Merged into master `6c46544cd`, both pass with and without
the flag, under Z3 and Bitwuzla; `dict23` takes 4.7 s. With the flag off the
branch splits nothing, and the branch was merged with master to re-run CI.

Patch coverage, measured on 2026-11-16 with an `ENABLE_COVERAGE` build at
`e66e161ba`: 287 of the 320 instrumented lines the branch adds under `src/`
run in the branch's own three tests (Codecov reported 23 %, because the
coverage job runs only the tests listed in `regression/COV_TESTS.d`, and these
were not listed). Four more tests (`field_sensitive_ssa_members`, `_threads`,
`_loop_clauses`, `_all_witnesses`) and two existing witness-validation tests
bring it to 320 of 320. Each new test fails under a mutation of the code it
pins; with the thread exclusion removed, `_threads` reports a false
VERIFICATION FAILED.

### 8.6 Phase 1 trace gate

Re-measured on 2026-11-16 at `a60262602` (master `f1c5ee1b4`), Bitwuzla,
each `CORE` test of `regression/esbmc` run with and without the flag, 30 s
per run. Of 3442 tests, none changes verdict and 5 time out both ways. Output
differs in 84; 57 differ only in line order (parallel solving, k-induction,
multi-property), 4 only in timings, AST addresses or quoting, and 3 are this
PR's own tests. The other 20:

| Cause | Tests |
|---|---|
| A member write prints the member's value where master prints the whole struct, because master's model read of `member(s#k, f)` is not a value for a pointer, bit-field or union member | `counterexample_aggregate_component_fail`, `github_10135_fail`, `github_10135_irep2_only_fail`, `github_1093` |
| A step is dropped or added. Master's struct arm of `constant_propagation` folds `s WITH [f := e]` while `e` is symbolic (`goto_symex_state.cpp:438-500`), so a reader reads either `s#k` or nothing of `s`; per-field propagation folds a different set, and the slicer then keeps different steps | `github_1476_fail`, `github_1590-local`, `github_5868_div_fail`, `github_7597-literal_fail`, `github_7521_uintptr_fail`, `overflow-result3`, `overflow-result4`, `overflow-result-all-fail` |
| A struct the model leaves free prints other free values (Z3) or `nil` per element instead of one `nil` (Bitwuzla) | `struct_array_ite_new_index_fail` |
| `--incremental-bmc` calls the solver one time more or fewer, since a claim folds to a constant at a different bound | `github_5868_div`, `github_7597-literal` |

Tried and reverted: slicing a split object at object granularity (a kept
field step keeps the object's earlier field steps, back to its last whole
write, as `s' == s WITH [f := v]` does). It removes `overflow-result3` and
`overflow-result4` and adds no new difference in the suite, but on

```c
struct S s; s.b = 2; s.a = nondet_int(); assert(s.a != 5);
```

it prints three steps where master prints none, because master folds the
nondet member write into the assertion. It trades one difference for another.

Every row comes from a choice §3 makes: fields propagate on their own, the
slicer works per field, and a struct is read back from its fields. Matching
master's text byte for byte means copying master's struct-level propagation,
slicing and tuple model read onto the fields, which is most of the cost the
split removes. The gate needs one of:

1. Keep "no trace text change" and accept that the per-field propagation and
   slicing in §3 go: fields propagate and slice as their whole object would.
   The equation stays O(writes), but slicing removes less.
2. Replace it with Invariant 4 as written: every printed step that both runs
   print shows the same value for every lvalue master pins down, verdicts are
   unchanged, and the four rows above are the accepted differences, each
   pinned by a test.

Decided on 2026-11-16: option 2. Byte-identical traces are no longer a gate
for any phase; Invariant 4 is. The four rows above are accepted. Each still
needs a test pinning one example of it, so that a further change to how a
split object prints fails a test instead of passing unnoticed.

Pinned on 2026-11-16 at `70d965f06`, each test failing when
`--field-sensitive-ssa` is dropped from it and passing under Z3 and Bitwuzla:

| Row | Test |
|---|---|
| member value | `field_sensitive_ssa_print_member_fail` (`v.un.i = 9`) |
| step dropped | `field_sensitive_ssa_print_slice_fail` (no step for an uninitialised struct) |
| free struct | none, the values are free; `field_sensitive_ssa_print_array_fail` and `_print_pointer_fail` pin master's values for a copied struct |
| `--incremental-bmc` | `field_sensitive_ssa_print_incremental` (both claims fold) |

Until 2026-11-16 the free-struct row also held five tests where the flag printed a
pointer member as `0` (`github_7965_fail`: `.next=( struct node *)8` on
master), breaking Invariant 4. The print step rebuilt the struct from its field
symbols after the write, so a field the slicer dropped had no model value and
an integer-to-pointer cast read back as `0`. It now prints the right-hand side,
renamed before any field is written, as an unsplit struct does; a nondet
initialisation still prints its fields. Re-measured with the fix, Bitwuzla, 30 s
per run: of 3331 `CORE` tests, none changes verdict, the five tests print what
master prints, and the 28 that still differ print no other value: a member
value, a step added or dropped, an assignment count, line order, AST addresses,
or `struct_array_ite_new_index_fail`'s `nil`. The fix adds no difference.
Like master, it also prints an unevaluated right-hand side such as
`s = dynamic_1_value.in`.

The right-hand side can also read a symbol defined only by a field step the
slicer drops, and the model then leaves it free: for
`s = {nondet_int() | 1, nondet_int()}` with only `s.b` read, the flag printed
`s = { .a=0, .b=5 }` (`.a=1` under `--no-slice`), where master prints no step
for `s`. Under the flag the slicer now hides a kept step whose printed
right-hand side reads a definition it removed, as it already drops a branching
step that reads one (#9884). A symbol with no defining step, such as a nondet
field, is as free in the full formula and does not hide the step. Keeping the
definitions instead would grow the formula. `field_sensitive_ssa_print_sliced_rhs_fail`
pins it and fails without the change under Z3 and Bitwuzla.

Measured on 2026-11-17, flag on, this change against #10802 without it,
Bitwuzla, 30 s per run, output compared after dropping timings and witness
uuids:

| Suite | Tests with a verdict | Verdict changes | Output changes |
|---|---|---|---|
| C (`regression/esbmc`, `CORE`) | 3209 of 3332 | 0 | 5 hide a step: the new test, `github_1093` (`node = { .tag={ .pointer_object=nil, ... }, ... }`), `github_5393_array_member_fail` (`tmpl = { .a={ .pointer_object=nil, ... }, ... }`), `github_5393_array_member_dyn_offset_fail` (`tmpl = { .a = 0 }`, master `.a=5`); 3 timing or line order |
| C++ (`esbmc-cpp/cpp`) | 1327 of 1355 | 0 | 1, k-induction-parallel line order |
| Python, every 10th test | 870 of 927 | 0 | 2, pointer addresses in temporary names |
| C, flag off | 3209 of 3332 | 0 | only the new test, which passes the flag |

`repro7521`, 2048 vs 8 fields, symex seconds: 1.25-1.31 vs 0.84-0.85 (1.5x)
under Z3 and 1.21-1.31 vs 0.84-0.86 (1.5x) under Bitwuzla, unchanged from
before the change; slicing takes 0.01 s more. `local_writes` stays about 1.1x.

