# Scope: field-sensitive SSA in goto-symex (#7521)

> **Status at 2026-11-15: Phase 0 in review, gate partly met (see §8).** The
> corpus is in `regression/field-sensitivity/` with baselines, and the census
> has been run and reverted. Baselines cover Z3 and ESBMC's Bitwuzla backend;
> the census covers the full C, C++ and Python suites and two SV-COMP samples
> (259 tasks, then 1215 drawn from every category). Two
> mitigations are merged: #10037 stopped phi merges of wide structs from
> scaling with the square of the width, and #10527 added the opt-in GOTO pass
> `--scalarize-structs`. Symex itself still gives a struct one SSA name, so a
> field write makes a new value of the whole struct. This document plans the
> symex change. Line numbers are from master at `d3104e99c2`.

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
verdict and no trace text; the local-struct rows of the corpus meet the 2x bar;
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
4. Traces and witnesses print the same values with and without the flag.

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
| SV-COMP stratified sample (1215 tasks, up to 40 per category from all 36 C categories; `--unwind 3`) | 1153 / 1215 | 6058 | 4143 (68 %) | 1096 (18 %) | 99 (2 %) | 720 | 243 |
| Corpus | 30 / 30 | 30 | 12 | 18 | 0 | 0 | 2049 |

Further counts. Struct objects that are globals: 886 in C, 32414 in C++,
32999 in Python and 109 in SV-COMP. Heap objects: 182, 75, 12566 and 89.
Objects with a nested struct or union member: 142, 4936, 340 and 23. In C,
294 of the 732 tests that reach a struct reach one that is address-taken with
constant field paths only, and 151 reach a byte-level one. No object in any
set has more than 4096 leaves, and outside the corpus none has more than 243,
so the Phase 3 cap of 4096 binds only synthetic inputs.

The stratified SV-COMP sample was measured on 2026-11-15 with the census
commit, Z3, sv-benchmarks `main` at that date, each task's data model, 4
tasks in parallel, each under a 30 s limit (1871 s in all). Tasks were drawn
with seed 7521 from every C `.set` file except `Unused_*` and the witness
sets. 60 tasks hit the limit before symex finished and 2 aborted; 39 of the
62 are in `DeviceDriversLinux64`, `DeviceDriversLinux64Large` and
`Sequentialized`. In this sample, struct globals number 1734, heap objects
112 and objects with a nested member 826. The largest object outside the
corpus has 243 leaves, in `DeviceDriversLinux64`. Most of the 720 unions
(555) are in `Intel-TDX-Module`, and `uthash` is the only category where
byte-level objects are the majority (29 of 31). The two concurrency
categories reach 371 objects: 319 never address-taken, 45 with
constant field paths and 7 unions, which is the input to Phase 4's
multithreading decision. 15 of the 36 categories reach no struct or union
object, 2 of them because no sampled task finished symex.

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

- The census on the full SV-COMP set (about 33000 C tasks). At the rate of
  the stratified sample (1.5 s of wall time per task, 4 in parallel) that is
  about 14 hours on one 4-core container. §8.3 covers every category through
  the sample instead.
