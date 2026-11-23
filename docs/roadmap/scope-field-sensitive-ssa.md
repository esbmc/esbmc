# Scope: field-sensitive SSA in goto-symex (#7521)

> **Status at 2026-11-21: Phase 0 merged (#10773); Phase 1 merged (#10778),
> with the trace fixes #10799 (§8.7) and #10808 (§8.8) and the whole-read fix
> #10839 (§8.13). On master `48d5f770e` with the flag forced on, the full
> Python `CORE` suite (9279 tests, Bitwuzla), the C and C++ `CORE` suites
> (Bitwuzla) and `tagged_attr_store` (Z3 and Bitwuzla) keep every verdict, and
> the Phase 1 corpus rows meet the 2x bar under both solvers (§8.14-§8.17);
> so does the whole Python suite under Z3 (§8.16). The Phase 1 fixes for a
> split copy into a split destination (#10862, §8.18) and for a re-split L2
> name and `github_7797_flat_set` (#10864, §8.19-§8.24) are merged; with
> them the full Python `CORE` suite keeps every verdict with the flag forced
> on (§8.20). Phase 1 still lacks an SV-COMP run. Phase 2's first part
> (#10876) splits address-taken locals that every pointer access reaches as a
> whole or through one member, as Andersen's points-to sets show (§8.25);
> `local_addr_once` now meets the 2x bar. An assert-enabled build of it runs
> every C and C++ `CORE` test with the flag forced on without tripping
> Invariant 3's assert (§8.26), and a phi no longer reads a copy carried over
> the object's own name from the other path's value set (§8.27). Its merge
> with master `aabf837b1` keeps those results under Z3 (§8.27). Phase 2's
> second part (#10886), stacked on the first, splits static objects as well;
> `plc_db` with the flag now solves at 512 and 2048 fields, and a split
> global copied from a split value loses its trace step as a split local
> already does (§8.28); the slicer now moves such a print to a kept field
> step, which fixes both (§8.30). The dereference-size memo (#10894) is
> stacked on it; with it, every corpus row meets the 2x bar under Z3 and
> Bitwuzla (§8.29). The stack's head, built with asserts, runs every C and
> C++ `CORE` test with the flag forced on under Bitwuzla without tripping
> Invariant 3's assert (§8.31). The stack merged with master `a6615b898`
keeps the 2x bar on every corpus row and every verdict under Z3 and
Bitwuzla (§8.32), and so does its merge with master `30592f8a7`, which adds
#10895's rebuild of a struct read over differently named members (§8.33).
Its merge with master `6c3c2cf9d` keeps them too, and the whole `esbmc/`
ctest label under Bitwuzla fails only the tests that fail on master
(§8.35). So does its merge with master `3194304fb`, which adds #10901's
member lookup through a `byte_extract` (§8.36). The SV-COMP run is still
open.**
> The corpus is in `regression/field-sensitivity/`
> with baselines, and the census has been run and reverted. Phase 1 adds
> `--field-sensitive-ssa` (default off): symex gives each field of an
> eligible local, parameter or return slot its own SSA name. The Phase 1
> trace gate is Invariant 4 (same values), with the §8.6 changes accepted.
> Two mitigations are merged: #10037 stopped phi merges of wide structs from
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
| A reassembled struct prints differently: master reads the struct from the solver as one tuple, the flag reads each field, so a pointer member prints `0` instead of `{ .pointer_object=nil, ... }`, an array `{ 6, 7 }` instead of `(signed long int [2])0x7...6` | `github_5393_array_member_fail`, `github_5393_array_member_dyn_offset_fail`, `github_7855-tie-hides-deref`, `github_7965_fail`, `pointer_repr_branch_flow_fail`, `struct_array_ite_new_index_fail` |
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
| reassembled struct | `field_sensitive_ssa_print_array_fail` (`.pair={ 6, 7 }`) |
| `--incremental-bmc` | `field_sensitive_ssa_print_incremental` (both claims fold) |

The third row is not only formatting. Master prints the correct value of a
pointer member in five of its six tests, and the flag prints another value
under both solvers:

| Test | Master | Flag, Z3 | Flag, Bitwuzla |
|---|---|---|---|
| `github_5393_array_member_fail` | `.a=(void *)4660` | `.a=0` | `.a={ .pointer_object=nil, ... }` |
| `github_5393_array_member_dyn_offset_fail` | `.a=5` | `.a=0` | `.a=0` |
| `github_7855-tie-hides-deref`, `github_7965_fail`, `pointer_repr_branch_flow_fail` | `.next=( struct node *)8` | `{ .next = 0 }` | `{ .next = 0 }` |

With `--no-slice`, `dyn_offset` prints `.a=5` again, so there the slicer drops
the field's step and the model leaves the field free. `github_7965_fail`
still prints `0`, and `github_5393_array_member_fail` prints
`.a=&__memset_impl`: the print step reads the pointer field symbol from the
model, while master evaluates the right-hand side (`(void *)CONCAT(...)`,
`&container->node` with `container` NULL). That breaks Invariant 4, so the
pointer half of the third row is not accepted, and the Phase 1 trace gate is
unmet until a split pointer field prints the value master prints. The array
half (`{ 6, 7 }` for the same bytes) keeps the value and stays accepted.
`struct_array_ite_new_index_fail` prints other nondet values under Z3 and
`nil` per element under Bitwuzla, where master prints `.e=nil`.


### 8.7 Phase 1 trace gate: split pointer fields

The step for a whole write of a split struct printed the struct of its field
symbols, read back from the model one field at a time. A pointer field's
model value is an object and an offset, and `pointer_logict::pointer_expr`
prints any pointer into the NULL object as `NULL`, so `&container->node` with
`container` NULL printed `0`. Master prints the right-hand side evaluated in
the model, which keeps `(struct node *)8`. `symex_assign_split` now hands the
printing step the right-hand side, renamed before any field is written, as a
whole write of an unsplit struct does. A nondet write has no right-hand side
to show and still prints its fields.

Measured on 2026-11-16 at `ebadfe075` (master `1a3fba6c7`), RelWithDebInfo,
LLVM 18, Z3 4.13.3 and ESBMC's Bitwuzla 0.9.0 backend, one 4-core container.

The five pointer tests of §8.6 now print master's trace, line for line, under
both solvers: `github_5393_array_member_fail`,
`github_5393_array_member_dyn_offset_fail`, `github_7855-tie-hides-deref`,
`github_7965_fail` and `pointer_repr_branch_flow_fail`. So does
`field_sensitive_ssa_print_array_fail` (`.pair=(signed long int [2])0x7...6`),
whose test now expects master's text. `struct_array_ite_new_index_fail` still
prints other values for `w1`, which the program never initialises, so master
does not pin them either.

Each `CORE` test of `regression/esbmc` with and without the flag, default
solver, 30 s per run: 3330 tests, 0 verdict changes (11 tests that already
pass the flag report an option error when it is given twice). After dropping
timings, witness UUIDs and the "splits" notice, 20 differ in output: 4 only in
line order, and 16 in the remaining §8.6 rows (member value, step dropped or
added, `--incremental-bmc`). One test is new to that list:
`github_10770_initialised_struct_fail` gains a step `t.f = 7` that master
folds into its assertion. `github_7521_uintptr_fail` prints `*p = 0` where
master prints `*p = 5`: the write `*p = 5` is two SSA steps, one per object
`p` may point to, and with the flag the value set lists those objects in the
other order, so the trace prints the other step. Master prints `*p = 0` itself
on a variant of the program; its verdicts do not change with the flag.

Flag off, `ctest -L esbmc/`: 3474 tests, 17 failures, the same 17 failing on
master's code in this container. `ctest -L esbmc-cpp/cpp`: 1466 tests, the 8
failures §8.5 lists. Python, every 12th test: 814 tests, one failure
(`github_7559_literal`, the parser rejects it under this container's Python).

`field_sensitive_ssa_print_pointer_fail` pins the pointer value and a
self-reading whole write (`t = (struct pair){t.b, t.a}`); with the change
reverted it prints `.next = 0` and fails under both solvers, as does
`field_sensitive_ssa_print_array_fail`. `field_sensitive_ssa_print_pointer` is
its SUCCESSFUL counterpart and cannot fail on a print-only change. The new
branch and the three values the printing step can take (none, the fields, the
right-hand side) were each hit by small C programs, with the same counts under
Z3 and Bitwuzla, using temporary instrumentation.

Each `CORE` test of `regression/esbmc-cpp/cpp` with and without the flag,
Bitwuzla unless the test pins a solver: 1355 tests, 0 verdict changes, and the
only output changes are the two §8.5 lists (`cpp_sum_class` in
`--k-induction-parallel` order, `github_7797_flat_set` unwinding to the bound).
Every 30th test of `regression/python`: 300 tests, 0 verdict changes, one
output change: `github_9185_raise_local_fail` gains a step
`err = { .message = 0 }`; with `--no-slice` master prints
`.message={ .pointer_object=nil, .pointer_offset=nil }` there, a value the
model leaves free, and the same `.v`.

Symex seconds on `repro7521` with the flag, three runs each, medians, against
the same build with the change reverted:

| Solver | 8 fields | 2048 fields | 2048 / 8 | Reverted, 2048 / 8 |
|---|---|---|---|---|
| Z3 | 0.764 | 1.282 | 1.68x | 1.64x |
| Bitwuzla | 0.753 | 1.267 | 1.68x | 1.67x |

`local_writes` stays at 0.016 to 0.035 s at every width (about 1.0x) under
both solvers.

Not measured here: the rest of the Python suite with the flag, witness files,
and an SV-COMP run.

### 8.8 Phase 1 trace gate: printed right-hand sides of sliced definitions

§8.7 prints a whole write of a split struct as its right-hand side. The
slicer keeps only the field steps a property reads, so that right-hand side
can read a symbol whose defining step was sliced; the model leaves it free
and the trace prints a value the program cannot produce. On

```c
int v = nondet_int() | 1;
struct S s = {v, nondet_int()};
assert(s.b != 5);
```

master without the flag prints no step for `s`; with the flag, master
`d93aabc32` prints `s = { .a=0, .b=5 }` although `v` is odd. Under the flag
the slicer now records the left-hand sides of the assignments it removes and
hides a kept step whose printed right-hand side reads one of them, as it
already drops a branching step that reads a sliced value (#9884). A symbol
with no defining step (a fresh L2 name of a nondet write) is as free in the
sliced formula as in the full one and does not hide a step.

Measured on 2026-11-17 on master `d93aabc32`, RelWithDebInfo, LLVM 18, Z3
4.13.3, one 4-core container, flag forced on, the change against the same
build without it, 30 s per run. The Bitwuzla re-run is below.

| Suite | Tests | Verdict changes | Output changes |
|---|---|---|---|
| C (`regression/esbmc`, `CORE`) | 3348 | 0 | 4 hidden steps: `field_sensitive_ssa_print_sliced_rhs_fail`, `github_1093` (`node = { .tag=0, .value=invalid-object }`), `github_5393_array_member_fail` and `_dyn_offset_fail` (`tmpl = { .a = 0 }`) |
| C++ (`esbmc-cpp/cpp`, `CORE`) | 1359 | 0 | 0 (`cpp_sum_class` differs in `--k-induction-parallel` line order only) |
| Python, every 10th `CORE` test | 923 | 0 | 0 |

`time_h_difftime_antisymmetry` (C) and `github_8971_is_payload_unknown`
(Python) run close to 30 s and timed out in one run each; re-run, both give
the same verdict with and without the change (31.3 s and 29.6 s; 25.8 s and
27.2 s). `github_2630_1` changes claim order under `--parallel-solving`
between two runs of the same binary.

The four hidden steps fall under the "step dropped" row of §8.6: in
`github_1093` and both `github_5393` tests master without the flag prints
the declaration of an uninitialised struct, which the flag printed with
other values (§8.6, "reassembled struct"). `field_sensitive_ssa_print_sliced_rhs_fail`
pins the change: without it the test prints `s = { .a=0, .b=5 }` and fails.

Each new branch was reached with temporary instrumentation, flag forced on
over the C suite: a removed definition recorded (3106 tests), a step hidden
(6), the walk finding a removed definition (6) or none (97), and the walk
skipping a shared subterm (`struct_array_ite_new_index` and its `_fail`).
Symex seconds with the flag, Z3, three runs each, medians:

| Row | Without the change, 8 / 2048 fields | With it, 8 / 2048 fields |
|---|---|---|
| `repro7521` | 0.822 / 1.216 (1.48x) | 0.822 / 1.271 (1.55x) |
| `local_writes` | 0.019 / 0.023 (1.2x) | 0.019 / 0.021 (1.1x) |

Re-run with Bitwuzla on 2026-11-17 at `512c450bb`, the same build setup
with ESBMC's Bitwuzla 0.9.0 backend (CaDiCaL 2.1.2 cloned with git, as the
tarball download is refused), flag forced on, `--bitwuzla` replacing any
solver option of the test, 30 s per run:

| Suite | Tests | Verdict changes | Output changes |
|---|---|---|---|
| C (`regression/esbmc`, `CORE`) | 3348 | 0 | the same 4 hidden steps as with Z3; under Bitwuzla master prints `node = { .tag={ .pointer_object=nil, .pointer_offset=nil }, ... }` in `github_1093` and `tmpl = { .a={ .pointer_object=nil, .pointer_offset=nil }, .pair={ 0, 0 } }` in `github_5393_array_member_fail` |
| C++ (`esbmc-cpp/cpp`, `CORE`) | 1359 | 0 | 0 (`cpp_sum_class` line order only) |
| Python, every 20th `CORE` test from the 6th | 461 | 0 | 0 |

62 C tests use `--ir`, which Bitwuzla rejects; both builds error out on
them, and the Z3 run above covers them. `github_2630_1` again changes claim
order under `--parallel-solving`.

Symex seconds, both solvers, the same binaries, three runs each, medians:

| Row | Solver | Without the change, 8 / 2048 fields | With it, 8 / 2048 fields |
|---|---|---|---|
| `repro7521` | Z3 | 0.747 / 1.240 (1.66x) | 0.759 / 1.260 (1.66x) |
| `repro7521` | Bitwuzla | 0.736 / 1.262 (1.71x) | 0.770 / 1.353 (1.76x) |
| `local_writes` | Z3 | 0.016 / 0.019 (1.2x) | 0.016 / 0.022 (1.4x) |
| `local_writes` | Bitwuzla | 0.017 / 0.016 (0.9x) | 0.016 / 0.018 (1.1x) |

Codecov reported two added lines uncovered: `return true` in
`reads_sliced_definition` and `step.hidden = true` in `symex_slicet::run`.
The coverage job runs only the tests in `regression/COV_TESTS.d`, and
`field_sensitive_ssa_print_sliced_rhs_fail` was not listed; it now is. With
`return true` changed to `return false` the test prints `s = { .a=0, .b=5 }`
and fails under Z3 and Bitwuzla, so it runs both lines.

Measured on 2026-11-17 with the branch merged into master `dc68c7e1f`,
RelWithDebInfo, LLVM 18, Z3 4.13.3 and ESBMC's Bitwuzla 0.9.0 backend:

- Python, `CORE` tests 2700 to 3669 in name order (970 tests, mostly not in
  the earlier samples), flag forced on against off, default solver, 30 s per
  run: 0 verdict changes. After dropping timings and the "splits" notice, 6
  differ in output: `github_4345` and its `_fail` only in the symex
  assignment count (389 against 398), and 4 (`github_5444_dictcomp_*_fail`,
  `github_7434_yaml_drop_fail`) in compiler-variable numbering or witness
  UUIDs, which also differ between two runs without the flag.
- Witness files for the 4 tests whose trace changes, flag on, with the change
  against the same build with `reads_sliced_definition` never matching: each
  GraphML witness loses one edge, the hidden step's, and keeps the same
  assumptions in the same order; the YAML witnesses are identical apart from
  UUIDs and timestamps.

Measured on 2026-11-17 at the PR head `e9787f1f`, RelWithDebInfo, LLVM 18,
Z3 4.8.12, flag forced on against off, 30 s per run:

- Python, `CORE` tests 3670 to 4639 in name order (`LC_ALL=C`, 0-based;
  970 tests, from `github_7552_index_fail`): 0 verdict changes. Three tests
  (`github_8283_dict_zip_nondet`, `github_8418`,
  `github_8488_functools_reduce`) time out with and without the flag. After
  dropping timings, the "splits" notice, temporary model paths and
  address-derived temporary names, 3 differ in output, each only in the
  symex assignment count: `github_8130_reverse_fail` (836 against 839),
  `github_8404_alias_parameter_refused` (166 against 175) and
  `github_8404_passed_to_imported_refused` (159 against 168).

Measured on 2026-11-17 at the PR head `f5e84b28c`, RelWithDebInfo, LLVM 18,
Z3 4.13.3, flag forced on against off, 4 runs in parallel, 30 s per run;
tests that timed out either way, or had no result because a batch was cut
off, were re-run at 90 s:

- Python, `CORE` tests 0 to 2699 in name order (`LC_ALL=C`, 0-based; 2700
  tests): 0 verdict changes. 18 time out at 90 s with and without the flag.
  `dynamic_type_scalar_tag_add_str_dyn` succeeded without the flag and timed
  out with it at 90 s; run alone, without the flag it does not finish in
  500 s, and with the flag it succeeds in 114 s, so its solver time varies
  by run, not by flag. After dropping timings, the "splits" notice and
  temporary model paths, 39 differ in output. 30 differ only in the symex
  or slicing assignment count, and `github_10322_width` only in the bytes of
  a temporary model path. In 5 (`arith_lost_type_default_fail`,
  `arith_lost_type_fail`, `github_10365_fail`, `github_10365_type_fail`,
  `github_10673_tagged_fail`), master prints a pointer field
  of a whole struct write as a name (`.value=tmp$3.value`,
  `.value=return_value$.value`) and the flag prints its value
  (`&dynamic_1_array[0]`), the §8.7 right-hand side; master pins no value
  there, so Invariant 4 holds. In 3 the counterexample chooses other values
  for nondet inputs: `github_10679_def_alias_nested_call_fail` (`x`, from
  `nondet_int()`, which the assertion leaves free), `github_10323_list_reversed_fail`
  (the bytes of the nondet list elements) and `github_10493_ir_fail`, whose
  printed value also changes between two runs without the flag.

Measured on 2026-11-18 at the PR head `ca9c9ed17`, RelWithDebInfo, LLVM 18,
Z3 4.13.3, flag forced on against off, 4 runs in parallel, 30 s per run;
tests that timed out either way were re-run at 90 s:

- Python, `CORE` tests 4640 to 9227 in name order (`LC_ALL=C`, 0-based;
  4588 tests, from `github_8502_ternary_ctor`). 25 time out at 90 s with and
  without the flag. Two change verdict at 30 s and at 90 s, in opposite
  directions; run alone:

  | Test | Without the flag | With the flag |
  |---|---|---|
  | `tagged_attr_store` | SUCCESSFUL, 25 s | no result in 1800 s |
  | `tagged_attr_store_nested` | SUCCESSFUL, 156 s | SUCCESSFUL, 28 s |

  The two programs differ only in class and field names. With the flag
  the sliced formula is 5 assignments smaller in both (16527 against 16532,
  16542 against 16547). With `--no-slice` the flag makes `tagged_attr_store`
  faster (24 s against 34 s). The build with this PR's slicer change
  reverted (master `dc68c7e1f`'s `slice.cpp`) also gives no result in 300 s
  with the flag, so the slowdown comes from Phase 1 as merged, not from
  §8.8. The suite's per-test budget is 1200 s, so with the flag forced on
  this is a lost verdict, and the Phase 1 gate is unmet on it.
- After dropping timings, the "splits" notice and temporary model paths,
  101 differ in output. 60 differ only in
  the symex, slicing or VCC assignment count. 28 are threaded programs
  where the flag prints `WARNING: --field-sensitive-ssa is ignored: the
  program may create threads`. In 4 (`github_8970_alias_tagged_fail`,
  `github_9030_call_param_fail`, `tagged_dict_param_fail`,
  `github_9135_unrelated_cb_fail`) master prints a pointer field as a name
  (`tmp$3.value`) and the flag as a value, as in the earlier samples.
  `github_9185_raise_local_fail` gains the step §8.7 lists.
  `github_9799_none_rebind_tagged_fail` differs only in a state number,
  `github_9799_tagged_attr_fail` in an address-derived temporary name and
  `typed_counter_gate_lowered` in a temporary path. In
  `github_9135_unrelated_cb_fail` the flag picks `c = 0, d = 0` where master
  picks `d = 1`; the violated assertion (`3 + 3 == 7`) reads neither, and
  `r = "b"` matches `k2(False)`. `github_9314_fail`, `os1_fail`,
  `github_8971_pass_static_fail` and `nondet_list21` report another failing
  claim or other nondet values, and each varies between three runs of the
  same binary (or, for `github_9314_fail`, prints the same trace in both
  modes when run alone). `github_9075_lambda` aborts both ways, as its test
  expects.

Measured on 2026-11-18 at the PR head `d9e07db7`, RelWithDebInfo, LLVM 18,
Z3 4.13.3 and ESBMC's Bitwuzla 0.9.0 backend, run alone, 300 s:

| `tagged_attr_store` | Without the flag | With the flag |
|---|---|---|
| `--z3` | SUCCESSFUL, 30 s | no result in 300 s |
| `--bitwuzla` | SUCCESSFUL, 7.8 s | SUCCESSFUL, 8.3 s |

CI builds Bitwuzla, the default solver when it is built in, so the loss is
specific to Z3. The flag splits one object, `result` in
`__python_scalar_add_obj_dyn` (`src/c2goto/library/python/scalar.c:490`),
called three times. The two SSA programs differ only there, apart from
address-derived temporary names. Without the flag `r` is assigned the symbol
`result#9 == result#1 WITH [float_idx:=0] ... WITH [size:=...]`, with
`result#1` nondet. With the flag `r` is assigned the struct literal
`{ .value=&dynamic_4_array[0], ..., .size=result::3#4 }`, inlined into both
arms of the `if`. The two formulas are equivalent.

The `--smt-formula-only` output of each, solved by the standalone Z3 4.13.3
binary with `smt.random_seed` and `sat.random_seed` set to N, 4 runs in
parallel. "Unsolved" means a timeout, or a run the 15 GB memory limit
killed at about 4 GB:

| Formula | Seeds | Solved (unsat) | Unsolved |
|---|---|---|---|
| `tagged_attr_store`, without the flag | 0-7 (150 s for 0-3, 240 s for 4-7) | 7, in 58-127 s | 1 (seed 0, 150 s) |
| `tagged_attr_store`, with the flag | 0-7 (same limits) | 1 (seed 4, 69 s) | 7 |
| `tagged_attr_store_nested`, without the flag | 0-5, 240 s | 0 | 6 |
| `tagged_attr_store_nested`, with the flag | 0-5, 240 s | 1 (seed 1, 147 s) | 5 |

So the slowdown on `tagged_attr_store` is not a single unlucky seed: Z3
finds the reassembled-literal encoding harder on this formula at almost
every seed. `tagged_attr_store_nested` is hard for Z3 both ways. A fix
would change how a whole read of a split struct reaches the formula (a
named symbol instead of an inlined literal), which §3 does not yet specify.

Still not measured: an SV-COMP run, and the full Python suite under
Bitwuzla.

### 8.9 Phase 1 Python gate: the carried whole-read literal

A whole read of a split struct renames to `constant_struct2t` of its field
symbols. A field whose value constant propagation refuses keeps its L2 name,
and `aggregate_literal_may_propagate` accepts an L2 name as immutable
(#7597), so the literal is carried into every reader. Without the flag the
object is one symbol whose `with` chain is refused, and readers read the
symbol. In `tagged_attr_store` the carried literal is `result`'s, returned by
`__python_scalar_add_obj_dyn`.

Two rules were tried in `literal_element_may_propagate`
(`goto_symex_state.cpp`), both treating a split field's L2 name as refused
inside a literal. They differ in whether #7597's pinning (`pin_refused_elements`)
then replaces that element by a read of the assigned name.

Measured on 2026-11-18, branch `feat/field-sensitive-ssa-phase1-whole-read`
on #10808's head `5c823fff`, RelWithDebInfo, LLVM 18, Z3 4.13.3, `--z3`, run
alone, wall time:

| Build | `tagged_attr_store` | `tagged_attr_store_nested` | `field_sensitive_ssa_copy_pinned` |
|---|---|---|---|
| without the flag | 25 s, 25 s, 27 s | 188 s, 182 s, 190 s | 0 VCCs left |
| flag, parent | no result in 200 s (300 s earlier) | 29 s | 0 VCCs left |
| flag, refuse and pin | 35 s, 33 s, 36 s | no result in 400 s (3 runs) | 1 VCC left |
| flag, refuse without pinning | 23 s | 30 s | never stops unwinding |

The pinned rule fixes `tagged_attr_store` but turns `_nested` from 29 s into
a timeout. Without pinning both pass, but a constant field of a copy no
longer bounds a loop: `field_sensitive_ssa_copy_pinned` (`t = s` into an
address-taken `t`, then `while (i < t.n)`) unwinds without end, where master
and the parent stop at 3. Neither rule is acceptable. The branch keeps the
pinned rule, which loses no termination; the test pins it (the parent leaves
0 VCCs, the no-pin rule never terminates).

Not measured on this branch: the C, C++ and Python suites, the corpus rows,
and anything under Bitwuzla (the Bitwuzla sources cannot be downloaded in
this container). A next step is to find what in `_nested` the pinned literal
slows down, starting from the two `--smt-formula-only` outputs.

Re-measured on 2026-11-18 on the same branch at `caa88902`, RelWithDebInfo,
LLVM 18, Z3 4.13.3, 4-core container:

- `_nested` with the flag, run alone with ESBMC's default seed: 347 s, within
  the 1200 s suite budget, so no verdict is lost (without the flag: 161 s,
  with one other job running). The earlier "no result in 400 s" was a 400 s
  cap, not the suite budget.
- On the two `--smt-formula-only` outputs, standalone Z3 4.13.3 with
  `smt.random_seed`/`sat.random_seed` 1-6, 600 s cap, 4 jobs at a time:
  without the flag 4 of 6 seeds solve (76-213 s); with the flag 5 of 6
  (51-347 s). The formulas differ only in the names of `result`'s fields and
  two unpack temporaries. The pinned formula is not harder than master's, and
  the 29 s of the parent is a fast case of a high-variance instance.
- C (`regression/esbmc`, `CORE`, 3427 tests) and C++ (`esbmc-cpp/cpp`,
  `CORE`, 1361 tests), flag forced on, this branch against its parent
  `5c823fff`, 30 s per run: 0 verdict changes. The C output differences (9)
  are parallel-solving line order, Clang AST addresses and witness timestamps
  that match on a re-run. In C++, 8 of 10 differences are assignment counts
  and one (`cpp_sum_class`) is line order.
- **Lost propagation:** `filesystem_path_decompose` (`--std c++17 --unwind
  12`). Without the flag and with the flag on the parent, symex proves it
  alone: 0 VCCs left after simplification, 320 unwind steps. With this
  branch, 360 VCCs remain and the `find_last_of`, `substr` and `basic_string`
  loops of the string model unwind to the bound, 609 unwind steps. The
  verdict holds at this bound, but a test without one, or with a lower one,
  would lose it. The rule refuses more than the unsplit object refuses: a
  next step is to find which split field's L2 name in a literal master
  carries, and refuse only names whose unsplit read master also refuses.

Not measured in this run: the Python suite, the corpus 2x bar, and anything
under Bitwuzla.


### 8.10 Phase 1 Python gate: refusing only opaque field names

The §8.9 rule refuses every split field's L2 name inside a literal. A field
whose value was refused but is an immutable read (an L2 symbol, a member of
one) is what master carries as a pinned read of the unsplit object, so
refusing it refuses more than master. The diagnostic on
`filesystem_path_decompose` showed only such fields refused: `_size` (field 1)
of the `basic_string` temporaries in `substr` and the copy constructor.

The rule tried here marks a split field's L2 name opaque when it is assigned a
value that is neither carried nor an immutable read, and refuses only opaque
names (`split_structst::mark_opaque`, `literal_element_may_propagate`).

Measured on 2026-11-18, PR #10839's head `cb01b995` plus this rule,
RelWithDebInfo, LLVM 18, Z3 4.13.3, `--z3` where stated, 4-core container,
wall time. The §8.9 rule and the parent's (master's) rule were measured on
the same binary through a temporary switch, since removed:

| Test, flag on | parent (master) | §8.9 rule (#10839 head) | opaque rule |
|---|---|---|---|
| `tagged_attr_store`, Z3 | no result in 200 s (§8.9) | 35 s, 33 s, 36 s (§8.9) | 35 s, 36 s, 36 s |
| `tagged_attr_store_nested`, Z3 | 29 s (§8.9) | 347 s (§8.9) | 375 s (without the flag 168 s) |
| `filesystem_path_decompose` | 0 VCCs left | 360 VCCs left | 0 VCCs left |
| `field_sensitive_ssa_nested_copy`, `--unwind 6` | 0 VCCs left | 3 VCCs left, loop unwound to the bound | 0 VCCs left |
| `field_sensitive_ssa_opaque_field`, `--unwind 6` | 0 VCCs left | 3 VCCs left | 3 VCCs left |

Without `--unwind`, the two "3 VCCs left" rows never stop unwinding: the loop
bound `o.in.a[3]` is a constant element of a struct copied into a member of
an unsplit, address-taken struct, and refusing one element of the copied
literal pins the whole member update, so the constant sibling is lost. Master
without the flag folds both programs in 1 assignment. The §8.9 rule as
pushed in #10839 therefore loses termination on `nested_copy`, a 34-line C
program. The opaque rule fixes that and `filesystem_path_decompose`, but
still loses it on `opaque_field`, where the field is written `x + 1`.

C (`regression/esbmc`, `CORE`, 3435 tests) and C++ (`esbmc-cpp/cpp`, `CORE`,
1361 tests), flag forced on, opaque rule against the parent, default solver,
30 s per run, 4 at a time: 0 verdict changes. C: 0 differences in the VCCs
left after simplification. C++: `assoc_emplace` and `cpp_sum_class` differ
only in the assignment count.

The new tests: `nested_copy` fails under the §8.9 rule (3 VCCs left).
`opaque_field` fails when `pin_chain_updates` pins a zero instead of the read
of the assigned name. `opaque_field_fail` checks that the copied opaque value
is the written one. `copy_pinned` now expects 0 VCCs left, as the parent
does. The new branches (`mark_opaque` and the refusal of an opaque name) were
reached on `tagged_attr_store` (field 3, `size`, of `result` in
`__python_scalar_add_obj_dyn`) and on `opaque_field`, where the opaque rule
and the parent differ.

The Phase 1 Python gate is still unmet: no rule tried so far fixes
`tagged_attr_store` without losing a propagation master keeps. Master pins
the opaque `size` update to `member(result#n, size)` and still carries the
object; the split L2 name is the same value. So the next step is to find why
master's carried `result` is cheap for Z3 and the split one is not, by
diffing the two `--smt-formula-only` outputs around `result`, instead of
refusing literals. Not measured in this run: the Python suite, the corpus 2x
bar, Bitwuzla, and SV-COMP.

### 8.11 Pinning a copied literal element by element

In §8.10's `opaque_field`, `o.in = tmp` copies a split struct into a member
of an unsplit one. `pin_chain_updates` refuses the copied literal because
one element, `tmp.n`'s L2 name, is opaque, and pins the whole update to
`member(o#2, in)`, so the constant `a[3]` that bounds the loop is lost. The
change here pins only the refused elements of a struct literal, to reads of
`member(o#2, in)` (`pin_struct_literal` applied to that read), and keeps the
rest.

Measured on 2026-11-18, PR #10839's head `cfce509d` with and without the
change, RelWithDebInfo built with clang 18 (GCC 13 hit internal compiler
errors in this container), LLVM 18, Z3 4.13.3, 4-core container, flag
forced on in both binaries, default solver, 30 s per run, 4 at a time:

| Test or suite | `cfce509d` | with the change |
|---|---|---|
| `opaque_field`, `--unwind 6` | 3 VCCs, 3 left | 2 VCCs, 1 left |
| `opaque_field`, no `--unwind` | never stops unwinding (§8.10) | stops; 2 VCCs, 1 left |
| `opaque_field`, no flag | 0 left (master) | 0 left (master) |
| C `CORE` (3438) | | 0 verdict changes, 0 differences in VCCs left |
| C++ `CORE` (1361) | | 0 verdict changes; `cpp_sum_class` differs only in its incremental step count |
| Python `CORE`, 920 of a shuffled list | | 0 verdict changes; `class_copy_none_attr_nondet_fail` leaves 4 VCCs instead of 5 |
| `tagged_attr_store`, `--z3` | no result in 300 s (90 s run: symex 1.4 s, 11828 VCCs, then Z3) | no result in 1800 s |
| `tagged_attr_store`, no flag, `--z3` | 26 s | |

The VCC left on `opaque_field` is `o.in.n == tmp.n`: master folds it because
both sides are `member(tmp#5, n)`; here one side is `tmp`'s field name and
the other a read of `o`. Its verdict is unchanged.

`opaque_field`'s `test.desc` now expects 2 VCCs, which fails on
`cfce509d` (3 VCCs). The new branch was reached on `opaque_field` with
temporary logging under Z3; this container has no Bitwuzla build.

`tagged_attr_store` does not reproduce §8.10's 36 s here: the `cfce509d`
binary, built with clang instead of the GCC used before, gives no result in
300 s with the flag. Symex output is the same size as without the flag
(11828 VCCs left, 16530 against 16532 assignments), so the opaque rule's
result there depends on the build, in line with the seed sensitivity in
§8.9. The Python gate is therefore unmet whichever rule is used; the
`--smt-formula-only` diff §8.10 proposes is still the next step.

### 8.12 The SSA difference behind `tagged_attr_store`

Measured on 2026-11-18 at PR #10839's head `9a5160a9`, RelWithDebInfo built
with clang 18, LLVM 18, Z3 4.13.3, `--z3`, 4-core container, run alone, wall
time:

| `tagged_attr_store` | Result |
|---|---|
| without the flag | SUCCESSFUL, 25 s |
| with the flag | no result in 600 s |
| with the flag, return of a split whole read through a named temporary (prototype, reverted) | no result in 600 s |

`--program-only` with and without the flag, with SSA numbers, temporary
addresses and step numbers stripped, differs only in `result` of
`__python_scalar_add_obj_dyn` (25528 against 25507 assignments, 7938 against
7915 sliced). Without the flag, `result` is declared nondet, and its
`both_str` path is the symbol `result#k == result#1 WITH [float_idx:=0] WITH
[value:=...] WITH [type_id:=...] WITH [size:=lhs_len + rhs_len + 1]`; the
callers' stores (`r := ...` in `dynamic_2_value`, the `ESBMC_unpack_temp`
merges) read that symbol in the `both_str` arm and an inlined `WITH` chain on
`result#1` in the `both_num` arm. With the flag, `size` is the field symbol
`result::3`, both arms are struct literals, and the `both_str` arm of each
merge pins `size` to the previous version of the store target
(`.size=dynamic_2_value#n.r.size`, `.size=ESBMC_unpack_temp#n.size`), as
§8.10's opaque rule intends.

The prototype tested §8.8's proposal of a named symbol instead of an inlined
literal: `make_return_assignment` assigned the renamed literal to a fresh
symbol with the value set updated, and returned that symbol. The callers then
read one symbol per call, as without the flag, and the formula still gives no
result in 600 s. Without the value-set update the prototype reported
`VERIFICATION FAILED` (the assertion at `main.py:21`), since the
dereference of `.value` through an untracked symbol is unconstrained; any
such rewrite must keep the value set. So naming the whole read at the return
is not enough. What remains different from master is that master's `result`
starts from a nondet base and every path updates it with `WITH`, while the
flag gives four independent field chains. The next step is to test that
difference directly: an SMT-level swap of the two encodings of `result` in
the `--smt-formula-only` output, or Z3's statistics (`-st`) on both formulas,
to find which part Z3 searches.

Not measured in this run: any suite, the corpus 2x bar, and Bitwuzla (not
built in this container).

### 8.13 Reading a split struct with an opaque field through its own symbol

§8.10's `--smt-formula-only` diff, on the clang build at `9a5160a9`: with the
flag, `return result;` in `__python_scalar_add_obj_dyn` writes the literal
`{ .value=&dynamic_4_array[0], ..., .size=result::3#4 }` straight into the
caller's `self->r`, so the `with` chain of `dynamic_2_value` holds the
literal, and the phi after the call reads `.size` back out of that chain.
Without the flag the same write stores `result#9`, a symbol defined once by
its own `with` chain.

`goto_symext::split_read` now handles a whole read of a split object (the
right-hand side of an assignment, or a returned value) whose L2 literal holds
an opaque field name (§8.10): it assigns the literal to the object's own
symbol in a hidden step and reads that symbol. #7597's pinning then carries
`{ ..., .size=result#2.size }`, so constant fields still fold, and readers
read `result#2` as they read `result#9` without the flag. A read with no
opaque field is unchanged. The object's symbol is written only by this step
and is never read at byte level.

Measured on 2026-11-18, PR #10839's head `9a5160a9` with the change against
the same build with it switched off through a temporary environment check
(since removed), RelWithDebInfo built with clang 18, LLVM 18, Z3 4.13.3,
4-core container, `--z3`, wall time:

| Test | No flag | Flag, `9a5160a9` | Flag, this change |
|---|---|---|---|
| `tagged_attr_store` | 31 s | no result in 120 s (1800 s in §8.11) | 29 s, 28 s, 27 s, 29 s |
| `tagged_attr_store_nested` | 214 s | 34 s | 35 s |

`tagged_attr_store_nested` crashes in the Python frontend on this clang build
with and without the flag: `python_converter::self_attribute_type` keeps a
pointer into a `typet` local to an `if` block after the block ends. The
`_nested` row was measured with that local hoisted, a change not in this PR.

Standalone Z3 4.13.3 on the three `--smt-formula-only` outputs of
`tagged_attr_store`, `smt.random_seed`/`sat.random_seed` set to N, 240 s
cap, 4 jobs at a time:

| Formula | Seeds | Solved (unsat) | Unsolved |
|---|---|---|---|
| no flag | 0-3 | 3, 107-159 s | 1 |
| flag, `9a5160a9` | 0-3 | 3, 65-160 s | 1 |
| flag, this change | 0-7 | 7, 58-125 s | 1 |

Standalone, the three formulas are about equally hard. The in-process run is
what loses the verdict at `9a5160a9`, and the change restores it there. ESBMC
calls Z3 with a fixed seed, so the in-process runs above repeat one sample.

§8.12's prototype assigned the literal to a fresh temporary at the return.
This change differs in two ways: it writes the object's own symbol through
`symex_assign_symbol`, so `goto_symex_statet::assignment` updates the value
set and #7597's pinning carries `{ ..., .size=result#2.size }` to readers
with the other fields constant; and it also covers the right-hand side of an
assignment.

Flag forced on, this change against `9a5160a9`, default solver (Z3), 30 s per
run, 4 at a time; differences after dropping timings:

| Suite | Tests | Verdict changes | Output changes |
|---|---|---|---|
| C (`regression/esbmc`, `CORE`) | 3440 | 0 | 0 beyond run-to-run noise: `--parallel-solving` line order (`github_2619`, `github_2630_1`), the shared proof cache (`proof_cache_*`), uninitialised bytes (`github_8524_unterminated_fail`) |
| C++ (`esbmc-cpp/cpp`, `CORE`) | 1361 | 0 | `cpp_sum_class`: `--k-induction-parallel` line order |
| Python, `CORE` tests among 920 of a shuffled list | 860 | 0 | `arith_lost_type_fail` prints `x = { .value=tmp$3.value, ... }`, as master without the flag does, where `9a5160a9` printed `.value=&dynamic_1_array[0]`; `github_5444_dictcomp_return_fail` in compiler-variable numbering |

Four tests timed out at 30 s in both builds.

Phase 1 corpus rows, flag on, `--z3`, symex seconds, three runs each,
medians: `repro7521` 0.865 / 1.325 (8 / 2048 fields, 1.53x);
`local_writes` 0.017 / 0.018 (1.06x).

`field_sensitive_ssa_opaque_whole_read` checks the hidden step
`r#N == { .a=1, .b=r::1#M }` in `--program-only` output and fails with the
change switched off. `_fail` checks that the value read through the symbol
is the written one. All three branches of `split_read` were reached under
Z3, with temporary logging: the opaque branch on `opaque_whole_read`,
`opaque_field` and `tagged_attr_store`, the unchanged split read on
`copy_pinned`. In `opaque_whole_read_fail` the trace no longer prints
`s = { .a=1, .b=0 }`, as master without the flag prints no step for `s`
(§8.6, step dropped).

Not measured: Bitwuzla (not buildable in this container), the full Python
suite with the flag forced on, the full C and C++ suites with the flag off
under ctest (each needs more than 10 minutes on 4 cores; the change is not
reached without the flag), and an SV-COMP run.

### 8.14 Phase 1 Python gate on merged master

Measured on 2026-11-18 at master `02f76186e` (#10839 merged),
RelWithDebInfo built with clang 18, LLVM 18, Z3 4.13.3 and ESBMC's
Bitwuzla 0.9.0 backend, 4-core container, run alone, 300 s cap, wall time:

| `tagged_attr_store` | Without the flag | With the flag |
|---|---|---|
| `--z3` | SUCCESSFUL, 25.8 s | SUCCESSFUL, 23.7 s, 24.4 s, 23.9 s |
| `--bitwuzla` | SUCCESSFUL, 7.0 s | SUCCESSFUL, 6.8 s |

So §8.13's result holds on a second build: the flag no longer loses this
verdict under Z3, and Bitwuzla is unaffected. `tagged_attr_store_nested`
still crashes in the Python frontend on this clang build within 1 s, with
and without the flag and under both solvers (§8.13).

### 8.15 Phase 1 suites and corpus on merged master

Measured on 2026-11-18 at master `02f76186e`, RelWithDebInfo built with
gcc 13.3, LLVM 18, Z3 4.13.3 and ESBMC's Bitwuzla 0.9.0 backend (CaDiCaL
2.1.2 cloned with git), 4-core container. Each `CORE` test command run with
and without the flag, Bitwuzla as the default solver (tests that pin `--z3`
keep it), 30 s per run, 4 at a time. Output is compared after dropping
timings, temp paths, UUIDs, timestamps, heap addresses and the flag's own
notices.

| Suite | Tests | With a split object | Verdict changes | Output changes in split tests |
|---|---|---|---|---|
| C (`regression/esbmc`) | 3442 | 541 | 0 | 23, beyond the flag's own tests |
| C++ (`esbmc-cpp/cpp`) | 1361 | 717 | 0 | 4 |
| Python (`regression/python`) | 9279 | 737 | 0 | 4 |

The flag's own `field_sensitive_ssa_*` tests are run with the flag removed
for the comparison. In C, 19 of the 23 are tests §8.6 lists, in its four
rows; `github_10770_initialised_struct_fail` gains the step §8.7 lists, and
`github_2630_1` changes claim order under `--parallel-solving` (§8.13). The
other two run under `--incremental-bmc` and fold claims at another bound
(§8.6's fourth row): `github_1476` leaves 1 of 4 VCCs where master leaves 4,
and `github_6464_struct_field_pointer` generates 7 VCCs where master
generates 12, with the same 5 properties passed. In C++,
`cpp_sum_class` (`--k-induction-parallel` order), `github_7797_flat_set`
(unwinding to the bound) and `tuple_structured_binding` (`--incremental-bmc`)
are listed in §8.6; `github_5868_const_iterator_deref` leaves 34 VCCs after
simplification where master leaves 33, with the same verdict. In Python,
`github_10323_list_reversed_fail` and `github_10493_ir_fail` print the
values §8.6 lists, `github_9185_raise_local_fail` gains the step §8.7 lists,
and `nondet_list21` (nothing split) prints a different counterexample in two
of three runs without the flag.

`queue_fifo` timed out without the flag and passed with it; run alone, it
takes 31.2 s both ways, so the 30 s cap decides it. 5 C tests and 8 Python
tests time out both ways.

Phase 1 corpus rows, symex seconds, three runs each, medians, 8 / 2048
fields:

| Row | Solver | Without the flag | With the flag |
|---|---|---|---|
| `repro7521` | Z3 | 0.749 / 0.762 (1.02x) | 0.766 / 1.052 (1.37x) |
| `repro7521` | Bitwuzla | 0.675 / 0.772 (1.14x) | 0.738 / 1.196 (1.62x) |
| `local_writes` | Z3 | 0.702 / 4.666 (6.65x) | 0.022 / 0.020 (0.91x) |
| `local_writes` | Bitwuzla | 0.649 / 4.573 (7.05x) | 0.018 / 0.017 (0.94x) |

Every run reports `VERIFICATION SUCCESSFUL`. Not measured: the full Python
suite under Z3 on merged master, and an SV-COMP run.

### 8.16 Phase 1 Python suite under Z3 on merged master

Measured on 2026-11-18 at master `02f76186e`, the clang 18 build of §8.14.
Whole Python suite under ctest (`ctest -L python`, 9823 tests including
`KNOWNBUG` and `THOROUGH`), the flag forced on through a wrapper that also
replaces any solver option with `--z3`, `ESBMC_REGRESS_TIMEOUT_MAX=60`, 3
jobs, in 10-minute runs. Every failure was re-run without the flag. During
the runs `timeout` left orphaned ESBMC processes that competed for memory,
so some timeouts and OOM kills below are inflated; the tests run alone are
not affected.

- 9726 pass with the flag. 97 fail; 96 of these also fail without the flag:
  - 86 hit the 60 s cap both ways under Z3. Re-run under Bitwuzla with
    the flag, 85 pass; `dict_splat_symbolic_key_fail` gives `VERIFICATION
    FAILED`, as expected, in 63 s without the flag and 65 s with it, run
    alone.
  - Five `complex_*` tests (`--fp2bv`, no solver pinned) are OOM-killed
    under Z3 both ways, even run alone; under Bitwuzla all five pass both
    ways in the same time (3.6-20.8 s).
  - Five fail both ways for reasons unrelated to the flag: `math27_nondet`
    pins `--bitwuzla`, which the wrapper replaced (5.0 s and 4.9 s
    SUCCESSFUL under Bitwuzla); `ir-ieee-no-solver` expects the "defaulting
    to Z3" notice that the forced `--z3` suppresses; `github_7559_literal`
    is a Python syntax error under this container's `python3`;
    `elif_divergent_types_prior_float_fail` is `KNOWNBUG`;
    `tagged_attr_store_nested` crashes in the frontend (§8.14).
- `nondet_dict14` (`THOROUGH`) passed without the flag in the ctest re-run
  and hit the 60 s cap with it. Run alone under Z3, both give `VERIFICATION
  SUCCESSFUL`: 61, 47 and 47 s without the flag, 99, 91 and 89 s with it
  (solver time 44 s against 87 s). Under Bitwuzla it takes 5.9 s and 6.3 s.
  The flag splits three objects here, but the `--program-only` outputs with
  and without it hold the same steps in a different order: at each join the
  phi of `dynamic_12_inf_array` and that of the split `dynamic_11_value`
  swap places. Z3's solve time depends on that order; the verdict does not,
  and both runs are far inside the 1200 s budget.

Phase 1 corpus rows on this build, symex seconds, three runs each,
medians, 8 / 2048 fields, with the flag: `repro7521` 0.716 / 1.199 (1.67x)
under Z3 and 0.615 / 1.119 (1.82x) under Bitwuzla; `local_writes` 0.017 /
0.017 under Z3 and 0.017 / 0.018 under Bitwuzla. The 2x bar holds on both
builds, with `repro7521` under Bitwuzla closest to it.

Not measured: trace text of the Python suite under Z3 (ctest compares
only the expected patterns), and an SV-COMP run.

### 8.17 Phase 1 gate on master after #10839

Measured on 2026-11-18, master `48d5f770e` (#10839 merged), RelWithDebInfo
built with gcc 13.3, LLVM 18, Z3 4.13.3 and ESBMC's Bitwuzla 0.9.0 backend
(CaDiCaL 2.1.2 cloned with git), 4-core container, run alone, wall time:

| `tagged_attr_store` | Without the flag | With the flag |
|---|---|---|
| `--z3` | SUCCESSFUL, 30.0 s, 29.8 s | SUCCESSFUL, 28.5 s, 29.9 s |
| `--bitwuzla` | SUCCESSFUL, 7.9 s, 8.2 s | SUCCESSFUL, 7.7 s, 8.1 s |

§8.13's result holds on a gcc build and under Bitwuzla: the verdict lost in
§8.8 is restored under both solvers.

Same build and binary, flag forced on against off, 4 runs in parallel, 30 s
per run; tests that timed out either way were re-run at 120 s:

| Suite | Tests | Verdict changes | Output changes beyond assignment counts |
|---|---|---|---|
| Python, every `CORE` test, default solver (Bitwuzla; tests that pin `--z3` keep it) | 9279 | 0 | 15, all listed below |
| C (`regression/esbmc`, `CORE`), `--bitwuzla` replacing any solver option | 3422 | 0 | §8.6's accepted rows; `github_10770_initialised_struct_fail` (§8.7); noise |
| C++ (`esbmc-cpp/cpp`, `CORE`), same | 1361 | 0 | `github_7797_flat_set` (below); `cpp_sum_class` line order |

- Python: 12 tests time out at 30 s both ways; at 120 s all 12 reach the
  same verdict with and without the flag. Of the 15 output differences, 8
  are address-derived temporary names, 2 the bytes of a temporary path and 1
  compiler-variable numbering. `github_9185_raise_local_fail` gains the step
  §8.7 lists. `tagged_attr_store_fail` and `github_9799_tagged_attr_fail`
  differ only in a temporary's name. In `nondet_list21`,
  `github_10323_list_reversed_fail` and `github_10493_ir_fail` the
  counterexample picks other values for nondet list elements that the
  violated assertion leaves free. `nondet_list21` varies between three runs
  without the flag, and the flag's trace matches one of them; the other two
  are stable without the flag, and both traces are valid counterexamples.
  The 5 traces §8.8 listed as printing a pointer field as a value
  (`arith_lost_type_fail` and others) now match master without the flag,
  apart from assignment counts.
- C: the 20 `field_sensitive_ssa_*` tests pass the flag themselves and were
  left to ctest (52/52 pass with the corpus). `github_6464_struct_field_pointer`
  leaves fewer VCCs with the flag (7 against 12). The rest are §8.6's rows,
  Clang AST addresses and `--parallel-solving` line order.

Phase 1 corpus rows, flag on, symex seconds, three runs each, medians:

| Row | Z3, 8 / 2048 fields | Bitwuzla, 8 / 2048 fields |
|---|---|---|
| `repro7521` | 0.822 / 1.308 (1.59x) | 0.749 / 1.218 (1.63x) |
| `local_writes` | 0.017 / 0.019 (1.12x) | 0.017 / 0.023 (1.35x) |

**Lost propagation, `github_7797_flat_set`** (`--std c++23 --unwind 6`).
The verdict holds at the test's bound, but with the flag symex stops
bounding two loops: `__lower_index` in `<flat_set>` and the copy loop of
`vector::operator=` unwind to 6 and leave unwinding assertions, where
without the flag neither does (2110 against 2376 symex assignments). Without
`--unwind`, master without the flag is SUCCESSFUL after 0.09 s of symex,
and with the flag gives no result in 120 s. A build of #10839's parent
`56dd046ae` with the flag loses the same two bounds, so the loss predates
#10839. A 12-line program
(`flat_set<int>`, two inserts, `extract()`, `replace()`) reproduces it. In
the SSA, `extract()` copies `__c` into a split temporary field by field,
reading the buffer as `tmp$4::0 == tmp$2#4.buf`; the field is opaque, so the
whole read into the unsplit `out` goes through #10839's hidden step
(`out#1 == tmp$4#2`), and the loop guard reads `out#1._size` unfolded. The
likely cause, not yet confirmed by instrumentation, is that #7597's pinned
literal for `tmp$4#2` is refused because `immutable_read_type` excludes
pointers (github_5868), so its constant `_size` is dropped with it. Without the flag `tmp$4 == tmp$2#4 WITH [_size:=2] WITH
[_capacity:=2]` keeps the buffer inside the base symbol and propagates.

A second form of the same loss, a split struct with a pointer field copied
into another split struct (`struct V out = s;` then a loop to `out.size`),
was introduced by #10839: its parent bounds the loop and master does not.

### 8.18 A split destination takes a whole read field by field

`goto_symext::symex_assign` now passes the assignment's target to
`split_read`, which skips the hidden step when that target is a split whole
object and returns `split_rewrite`'s literal, as before #10839. `symex_assign_split` takes that read field by field, so no
literal is carried and the hidden step bought nothing, while reading through
the source's symbol lost a constant sibling whenever a pointer field kept
the pinned literal unpropagated. A return still goes through `split_read`,
so `tagged_attr_store`'s path is unchanged: applying the same rule to a
return into a split return slot brought the Z3 timeout back (no result in
300 s).

Measured on 2026-11-18, the change on master `48d5f770e`, the same gcc 13.3
build, Z3 4.13.3 and Bitwuzla 0.9.0:

- `field_sensitive_ssa_split_copy_pointer` (`struct V out = s;` with a
  `malloc`ed `buf`, then a loop to `out.size`, `--unwind 6`): 1 VCC, 0 left
  after simplification, under both solvers. Master's binary unwinds the loop
  to 6 and leaves 2 VCCs, so the test fails on master; `_fail` pins
  `1 remaining` and FAILED, and master leaves 2. Four variants (through a
  callee, `return *c`, and two struct shapes) bound the loop with the change
  and not on master. The C++ `flat_set` reproducer of §8.17 is unchanged:
  its destination is not split.
- `tagged_attr_store`, flag on: Z3 28.4 s, Bitwuzla 7.9 s.
- Flag forced on in both binaries, this change against master, default
  solver (Bitwuzla), 30 s per run, 4 at a time: C `CORE` (3442) and C++
  `CORE` (1361) 0 verdict changes; output differences are assignment counts,
  `--parallel-solving` order, Clang AST addresses and timing fields. Python,
  every 10th `CORE` test plus the `tagged_attr_store` tests (931): 0 verdict
  changes; one temporary name differs.
- ctest `field_sensitive_ssa` and `field-sensitivity`: 52/52 pass.


### 8.19 `github_7797_flat_set`: a carried pointer field and a re-split L2 name

Two defects, found from the 12-line `flat_set` reproducer of §8.17
(`regression/esbmc-cpp/cpp/field_sensitive_ssa_flat_set_extract`).

**Lost propagation.** The whole read of `tmp$4` in `extract()` is assigned
to the object's own symbol (§8.13) as `{ .buf=tmp$4::0#2, ._size=2,
._capacity=2 }`. `buf` is opaque, so #7597's pinning replaces it with
`tmp$4#2.buf`; that read is a pointer, which `immutable_read_type` refuses
(github_5868), so the whole pinned literal is refused and `_size` with it.
§8.17's hypothesis is confirmed. Without the flag the copy is
`tmp$2#4 WITH [_size:=2] WITH [_capacity:=2]`: the pointer stays inside the
base symbol. `pin_symbolic_updates` now falls back, for a split object only,
to that shape: `tmp$4#2 WITH [f := v]` over the members that propagate. The
pointer rule is unchanged, and no pointer value is carried out of another
object.

**Wrong verdict.** That carried value holds `tmp$4#2`, an L2 name of a split
object. Passing it as an argument to a split parameter runs
`split_structs.rewrite` on the renamed argument (`argument_assignments` after
`cur_state->rename`, `symex_function.cpp`), and `rewrite` turned the L2 name
into the field symbols at generation 0, values never assigned. #7597's pinned
literal already carries such names for non-pointer opaque fields, so this is
reachable without the first change. On master `48d5f770e` and on the head
of #10862, this program gives `VERIFICATION FAILED` with the flag and
`SUCCESSFUL` without it (`p::1#1 == s::1#0` in the SSA):

```c
struct S { int a; int b; };
void f(struct S p, int x) { assert(p.b == x + 1); }
int main(void)
{
  int x = nondet_int();
  struct S s;
  s.a = 1;
  s.b = x + 1;
  struct S t = s;
  struct S *q = &t;
  f(t, x);
  return q->a - 1;
}
```

`rewrite` now leaves an L2 name as it is: it names the definition the hidden
step put in the equation. Debug builds would have hit `at_level_of`'s
assertion instead.

Measured on 2026-11-18/19, the head of #10862 (`3f47f12e6`) with and
without this change, RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3,
Bitwuzla 0.9.0, 4-core container:

- Reproducer, no `--unwind`: without the flag SUCCESSFUL in 0.8 s; with the
  flag no result in 120 s before, SUCCESSFUL in 0.75 s after. At
  `--unwind 6` with the flag, the four loops unwind 1/3/2/4 times, as
  without the flag (before: 1/6/5/4, with unwinding assertions).
- Each change alone is not enough: the pinning change alone bounds the copy
  loop but reads `tmp$4::0#0` and keeps `__lower_index` unbounded; the
  rewrite fix alone leaves both bounds lost.
- New tests: `field_sensitive_ssa_carried_read_arg` (FAILED before),
  `field_sensitive_ssa_split_copy_pointer_unsplit` and `_fail` (4 VCCs,
  2 left and 5 iterations before; 3 and 1 after), and
  `field_sensitive_ssa_flat_set_extract` and `_fail` (no result in 60 s
  before). `field_sensitive_ssa_carried_read_arg_fail` pins the real
  violation and passes both ways. All pass with the change.
- Every new branch reached under Z3 and Bitwuzla by a C harness run with
  `--enable-unreachability-intrinsic` and full unwinding, its
  `__ESBMC_unreachable()` unreached (an instrumented build printed each
  branch; a control `__ESBMC_unreachable()` on a reachable line FAILED).
- Flag forced on, this change against #10862's head, default solver, 30 s
  per run, 4 at a time: C `CORE` (3448) 0 verdict changes, and no output
  change beyond witness UUIDs, timing fields and `--parallel-solving` order.
  C++ `CORE` (1363) 0 verdict changes; `github_7797_flat_set` now unwinds
  as it does without the flag (2430 VCCs, 422 left, against 2586 and 511),
  and `cpp_sum_class` differs in k-induction line order. Python, every 10th
  test (940): 0 verdict changes. `github_9185_raise_local_fail` loses the
  step §8.17 listed and now matches the run without the flag;
  `github_10365_type_fail` drops the step `b = { ... }`, which the run
  without the flag prints: `b`'s value is now carried, nothing reads `b#1`,
  and the slicer removes it. The verdict, the violated property and every
  other step are unchanged (Invariant 4).
- ctest without the flag: `field_sensitive_ssa|field-sensitivity|github_7797`
  84/84; `esbmc/` 3576 of 3593, and the 17 failures fail the same way on
  #10862's binary (Boolector not built, clang-18 builtins, 32-bit headers).
- `tagged_attr_store`, wall time, flag on: Z3 25.8 s (25.4 s on #10862's
  head, 27.9 s without the flag), Bitwuzla 7.4 s (7.2 s, 7.2 s).
- Phase 1 corpus rows, flag on, symex seconds, medians of three:
  `repro7521` 0.758 / 1.152 (1.52x) under Z3 and 0.773 / 1.229 (1.59x)
  under Bitwuzla; `local_writes` 0.016 / 0.019 (1.19x) and 0.017 / 0.019
  (1.12x).
- Not measured: the full Python suite with this change, k-induction suites,
  and SV-COMP.

### 8.20 Full Python suite on the `flat_set` fix

Measured on 2026-11-19 at `7e9aef1f0` (the head of #10864), RelWithDebInfo,
gcc 13.3, LLVM 18, Z3 4.13.3, Bitwuzla 0.9.0 (CaDiCaL 2.1.2 cloned with
git), 4-core container. Every Python `CORE` test (9279), flag forced on
against off on the same binary, default solver, 4 at a time, 30 s per run;
the 10 tests that timed out either way were re-run at 120 s:

- Verdicts: 0 changes. At 30 s, `queue_fifo` and `github_9392` time out
  without the flag and pass with it (29.8 s and 29.3 s). At 120 s all 10
  reach the same verdict both ways, each between 29 s and 62 s.
- `tagged_attr_store`, run alone, wall time: Bitwuzla 4.8 s without the
  flag, 5.5 s with it; Z3 21.3 s and 20.3 s.
- After dropping timings, temporary model paths and address-derived names,
  25 tests differ in output. 14 are noise seen before (§8.17): compiler-variable
  numbering, witness UUIDs, the bytes of a temporary path, nondet values
  the assertion leaves free
  (`nondet_list21`, `github_10323_list_reversed_fail`,
  `github_10493_ir_fail`), and the property list of a run that finished
  only on one side of the 30 s cap.
- 11 tests lose one step with the flag: the whole write of a tagged Python
  value (`x = { .value=&dynamic_1_array[0], ... }` or `r = { ... }`) that
  the run without the flag prints. The tests are `arith_lost_type_fail`,
  `arith_lost_type_default_fail`, `github_10365_fail`,
  `github_10365_type_fail`, `github_10673_tagged_fail`,
  `github_8970_alias_tagged_fail`, `github_9030_call_param_fail`,
  `github_9135_unrelated_cb_fail`, `tagged_arg_function_fail`,
  `tagged_arg_staticmethod_fail` and `tagged_dict_param_fail`.
  #10862's head (`3f47f12e6`, same build) prints the step in all 11 with the
  flag, so this change drops it: the carried value leaves the definition
  unread and the slicer removes it, as §8.19 records for
  `github_10365_type_fail`. Invariant 4 allows a dropped step, but
  `arith_lost_type_fail` pins `State N ... main.py line 13` in its
  `test.desc`, so with the flag forced on it fails. That is a lost test, and
  the Phase 1 gate is unmet until the step is kept or the loss is accepted
  in review. Which of §8.19's two changes drops it has not been isolated.

### 8.21 A copied split chain carried over the destination's name

§8.20's 11 lost steps come from the pinning change of §8.19. Disabling
`split_literal_chain` alone brings the step back in `arith_lost_type_fail`.
There the return temporary `tmp$3` is split, its pointer field is a phi of
two addresses, and its `float_idx` is 0 on both paths, so its whole read
carries `tmp$3#1 WITH [float_idx := 0]`. `x = ...` copied that value, so
readers of `x` read `tmp$3#1`, nothing read `x#1`, and the slicer removed
the step that prints `x`. Without the flag `tmp$3` is a phi of two `with`
chains, which is not carried, so `x#1` is read and the step stays.

`goto_symex_statet::assignment` now carries such a value, a `with` chain
over the L2 name of another split object, as the same updates over the
destination's own L2 name: `x#1 WITH [float_idx := 0]`. The equation
defines `x#1` as the chain, so the updates hold on it, and the constants
still fold. The rebuild only runs when something is split.

Measured on 2026-11-19, `80cbf4393` against #10864's previous head
`410e5925e`, RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3, Bitwuzla 0.9.0
(CaDiCaL 2.1.2 cloned with git), 4-core container. `d466dc811` only moves
the rebuild into a helper; its output on the 11 tests is identical and
ctest below ran on it.

- The 11 tests of §8.20 with the flag print the same output as without it,
  after dropping timings and addresses.
- New tests: `field_sensitive_ssa_carried_copy_fail` pins the step
  `g = { .p=..., .k=0 }` of a global copied from a split return value. It
  fails on `410e5925e`, which drops the step, under Z3 and Bitwuzla.
  `field_sensitive_ssa_carried_copy` pins `1 remaining` for the folded
  sibling. It fails on a build without `split_literal_chain` (2 left).
  Both pass with the change under both solvers. Each solver prints the
  same step with and without the flag (Z3 prints `.p=t.p` both ways).
- Flag forced on, the change against `410e5925e`, default solver, 30 s per
  run, 4 at a time: C `CORE` (3450) and C++ `CORE` (1363), 0 verdict
  changes. Output differences are witness timestamps, timing fields,
  Clang AST addresses, and `--parallel-solving` or
  `--k-induction-parallel` line order. Python, every 5th `CORE` test plus
  the 11 and the `tagged_attr_store` tests (1869): 0 verdict changes; the
  11 regain their step; 2 differ in an address-derived temporary name;
  `function_values_fail` times out both ways.
- `tagged_attr_store`, wall time, run alone: Z3 26.7 s without the flag,
  24.7 s with it (25.9 s on `410e5925e`); Bitwuzla 7.6 s and 7.9 s (7.6 s).
- Phase 1 corpus rows, flag on, symex seconds, medians of three, 8 / 2048
  fields: `repro7521` 0.855 / 1.228 (1.44x) under Z3 and 0.805 / 1.248
  (1.55x) under Bitwuzla; `local_writes` 0.021 / 0.022 (1.05x) and
  0.020 / 0.022 (1.10x). All `VERIFICATION SUCCESSFUL`.
- Each outcome of the new decisions is reached under Z3 and Bitwuzla by a
  C harness run with `--enable-unreachability-intrinsic` and full
  unwinding; an instrumented build printed each one, and the harness's
  `__ESBMC_unreachable()` after the copy was reported reached.
- ctest without the flag: `field_sensitive_ssa|field-sensitivity|github_7797`
  86/86; `esbmc/` 3578 of 3595, and the 17 failures fail the same way on
  `410e5925e`'s binary (Boolector not built, clang-18 builtins, 32-bit
  headers).
- Not measured: the full Python suite with this change, k-induction suites,
  and SV-COMP.

### 8.22 A separate build of the stack's head

Measured on 2026-11-19 at `325108b28` (the head of the PR stacked on
#10862) merged with master `08198b792`. RelWithDebInfo, built from scratch
with gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla 0.9.0 (CaDiCaL 2.1.2 cloned
with git), 4-core container.

- ctest `field_sensitive_ssa|field-sensitivity|github_7797|tagged_attr_store|split_struct|scalarize`:
  101/101 after the merge, and 92/92 for the same pattern without the last
  two terms on `325108b28`.
- The ten tests the two open PRs add, each run with `--z3` and with
  `--bitwuzla` added to its `test.desc` options: every pattern matches
  under both solvers.
- `tagged_attr_store`, wall time, run alone: Z3 31.2 s without the flag and
  28.9 s with it; Bitwuzla 7.9 s both ways. All `VERIFICATION SUCCESSFUL`.
- `github_7797_flat_set` with the flag and no `--unwind`:
  `VERIFICATION SUCCESSFUL` in 0.92 s (Z3) and 0.95 s (Bitwuzla).
- Phase 1 corpus rows, flag on, symex seconds, medians of three, 8 / 2048
  fields: `repro7521` 0.762 / 1.283 (1.68x) under Z3 and 0.807 / 1.332
  (1.65x) under Bitwuzla; `local_writes` 0.018 / 0.020 (1.11x) and
  0.016 / 0.020 (1.25x). On this build `repro7521` stays under the 2x bar
  but is the closest of the two rows to it.
- Flag forced on against off, every third directory of `regression/esbmc`
  and `regression/esbmc-cpp/cpp`, `CORE` tests only, default solver, 30 s
  per run, 4 at a time: 1612 tests, 0 verdict changes. Only verdicts were
  compared.
- Not measured here: trace text, Python, k-induction suites and SV-COMP.

### 8.23 Python trace text under Z3 on the stack's head

Measured on 2026-11-19 at `273566c77` (the head of the PR stacked on
#10862, which contains master `08198b792`). RelWithDebInfo built from
scratch with gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla 0.9.0, 4-core
container. Every fourth `regression/python` `CORE` test, starting at the
second (2322 tests), each run with its `test.desc` options, any solver
option replaced by `--z3`, with and without the flag, 60 s per run, 3 tests
at a time. Output is compared after dropping timings, temporary paths,
addresses, UUIDs and the flag's own notices.

- Verdicts: 2321 agree. `github_8947_nondet_dict_fn_fail` gives
  `VERIFICATION FAILED` without the flag and hits the 60 s cap with it. Run
  alone, both give `VERIFICATION FAILED`: Z3 solve time 76.2, 76.6 and
  73.3 s without the flag, 98.1, 96.6 and 93.2 s with it, and 793 VCCs, 182
  after simplification, both ways. The flag splits one object. Under
  Bitwuzla, the solver the test runs with, it takes 24.5 s and 25.6 s.
  18 other tests hit the cap both ways and were not re-run.
- Output: the flag splits an object in 185 tests. Beyond the flag's notice,
  19 differ. 14 print another assignment count after simplification, with
  the same VCC counts. `open_text_read_fail` numbers its violation as state
  49 instead of 45 and prints the same values. `github_10493_ir_fail`
  prints other values for nondet `*cell` reads, as in §8.17.
  `github_10322_width` and `github_7693_fail` differ only in a temporary
  path's bytes and an address-derived name. No printed value that the run
  without the flag pins changes.

### 8.24 The stack merged with #10848

Measured on 2026-11-19 at `36b63072a` (the head of the PR stacked on
#10862) merged with master `1aa90ce5a`, which adds #10848: `symex_dead`
now erases a dead struct's or array's member entries from the value set.
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 and Bitwuzla 0.9.0 (CaDiCaL 2.1.2
cloned with git), 4-core container.

- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_9060|github_9434`:
  92/92.
- `github_9060` with the flag: after `DEAD t`, the value set holds no entry
  for the split field symbols `t::0` and `t::1`, as it holds none for
  `t.p` and `t.q` without the flag. The test's own patterns match the
  unsplit names, so they cannot see split ones.
- Flag forced on against off, every eighth directory of `regression/esbmc`
  and `regression/esbmc-cpp/cpp`, `CORE` tests only, 60 s per run, 4 at a
  time: 607 tests under each of `--z3` and `--bitwuzla`, 0 verdict changes.
  Two tests that already pass the flag reject it a second time and were
  not compared. Only verdicts were compared.
- `tagged_attr_store`, wall time: Z3 26.8 s without the flag and 24.4 s
  with it; Bitwuzla 7.5 s and 7.3 s. All `VERIFICATION SUCCESSFUL`.
- `github_7797_flat_set` with the flag (`--unwind 6`): `VERIFICATION
  SUCCESSFUL` in 0.96 s (Z3) and 1.03 s (Bitwuzla).
- Phase 1 corpus rows, flag on, symex seconds, medians of three, 8 / 2048
  fields: `repro7521` 0.860 / 1.279 (1.49x) under Z3 and 0.869 / 1.271
  (1.46x) under Bitwuzla; `local_writes` 0.021 / 0.022 (1.05x) and
  0.019 / 0.022 (1.16x).
- Not measured here: trace text, Python suites, k-induction and SV-COMP.

### 8.25 Phase 2, part 1: address-taken locals

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2` (stacked on
the `flat_set` PR at `c73e2982a`, which includes master `1aa90ce5a`).
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla 0.9.0, 4-core
container. "Baseline" is the parent commit's binary.

What changes, behind `--field-sensitive-ssa` only:

- The eligibility scan no longer drops a local whose address is taken as
  `&s` of its own type. Such a local stays eligible if, in every function
  reachable from the entry point, each dereference whose Andersen points-to
  set includes it has the object's own type and is read whole or through one
  member. Pointer arithmetic on a pointer that may reach it, a pointer
  passed to a function symex models itself or that has no body, an `OTHER`
  instruction, and a points-to set Andersen cannot bound all drop it.
  Uncalled functions are skipped because Andersen gives their pointer
  parameters TOP.
- Symex rewrites what a dereference resolves to, the pointer it queries
  the value set with, the operands it renames inside a dereference, and the
  guards of the dereference claims and assumptions it emits. `rewrite`
  leaves an `address_of` alone, and a split local whose address is taken
  keeps its own record in `local_variables`, so `is_live_variable` still
  finds it.

Results:

- Harnesses: writes and reads through a pointer, a pointer field read
  through a pointer, a pointer chosen between two objects, a dangling
  pointer, and a reinterpreting `char *`. Same verdicts with and without the
  flag under Z3 and Bitwuzla; the reinterpreted object is not split.
- New tests `field_sensitive_ssa_through_pointer` and `_fail`,
  `field_sensitive_ssa_pointer_dangling_fail`,
  `field_sensitive_ssa_pointer_reinterpreted`. Each mutant fails at least
  one of them: without the post-dereference rewrite, without the liveness
  record, or without the claim guard rewrite, `_through_pointer` fails;
  with the Andersen filter accepting every candidate, the reinterpreted
  object splits and `_pointer_reinterpreted` fails.
  `field_sensitive_ssa_opaque_whole_read` and `_fail` now split 2 objects:
  `s`, reached only through `p`, splits too; the pinned `r` step is
  unchanged.
- Every new branch outcome (20) reached under Z3 and Bitwuzla: an
  instrumented build printing a marker per outcome, 9 C and C++ harnesses.
  An index on a pointer never reaches symex (`p[i]` is lowered to
  `*(p + i)`), so the scan has no case for it.
- Flag forced on against the baseline without it, every third directory of
  `regression/esbmc` and `regression/esbmc-cpp/cpp`, `CORE` tests that do
  not already pass the flag, 30 s per run, 4 at a time: 1626 tests, 218 of
  which split more objects than the baseline with the flag. 0 verdict
  changes: `vector_reserve_realloc` solves only with the flag, and every
  other difference is both runs timing out or printing no verdict. The 218
  re-run on the final binary give the same split counts and verdicts.
- Trace text, the 87 of those 218 that fail: 83 print the same
  counterexample. `github_9409_fail`, `github_9416_fail` and
  `vector_count_value_ctor_fail` print a member write's value where the
  baseline prints the whole struct (§8.6, first row); `github_9416_fail`
  also no longer lists an unwinding assertion the flag lets symex fold.
  `github_5868_is_scalar_const_lookup_fail` lists two more `NOT CHECKED`
  invalid-pointer claims: a pointer field read through a split object has a
  wider value set there. No verdict moves, but this is open.
- Python, every 25th `CORE` test (373), flag forced on against off: 0
  verdict changes. Python objects are never split.
- ctest without the flag: `esbmc/` 3580/3604 and `esbmc-cpp/cpp` 1468/1476,
  each failure failing the same way on the baseline; every 20th `python/`
  test, 491/491.
- Corpus, flag on, symex seconds, medians of three, 8 / 512 / 2048 fields:
  `local_addr_once` 0.016 / 0.017 / 0.015 (0.94x) under Z3 and 0.017 /
  0.014 / 0.017 (1.0x) under Bitwuzla; the baseline with the flag, which
  does not split it, takes 0.611 / 1.636 / 4.664 and 0.611 / 1.723 / 4.272.
  Z3 solving at 2048 fields drops from 5.5 s to 0.010 s. Every other corpus
  row gives its expected verdict under both solvers except `plc_db` at 512
  and 2048 fields, a global, which times out at 300 s as before.
- Still open for the Phase 2 gate: globals (`global_callee`, `plc_db`), the
  wider value set above, and an SV-COMP run.

### 8.26 Phase 2, part 1: assert-enabled run over the C and C++ suites

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2-locals` merged
with the `flat_set` branch, itself merged with master `e0ce5d373`. Built
with `-O2 -g1` and no `-DNDEBUG`, so `assert` is live, including the one in
`split_structst::rewrite` that checks Invariant 3. gcc 13.3, LLVM 18, Z3
4.13.3, 4-core container. Bitwuzla could not be built there (its CaDiCaL
download was refused), so these runs are Z3 only. The Invariant 3 assert
and the eligibility scan run before any solver is called.

- Every `CORE` test of `regression/esbmc` and `regression/esbmc-cpp/cpp`
  that does not already pass the flag (4802), flag forced on and off, 20 s
  per run, 4 at a time. No run prints a failed `assert`. Five runs end in
  `SIGABRT` both ways: each prints the `ERROR:` its test expects
  (`github_7905_ir_refused`, `github_7905_struct_refused`,
  `init_obj_vlas`) or a Z3 sort error it also prints without the flag
  (`github_6310_typeid_dynamic` and `_fail`).
- Five C++ tests lost their verdict with the flag: `github_7797_flat_eq`,
  `github_7797_regex`, `github_9403`, `github_9417` and `github_9419` pass
  without it, and with it end in `uncaught exception [St12out_of_range]:
  unordered_map::at` before symex. §8.25's every-third-directory sample
  did not include them. The pointer scan erased a candidate on the first of
  two identical entries for it in one Andersen points-to set (a `std::map`
  iterator's pointers into the map's arrays), then looked it up again. It
  now skips a name it already dropped, which gives the same set as visiting
  each name once. With the fix all five give `SUCCESSFUL` with the flag
  (12 to 44 objects split), as without it.
- New pair `field_sensitive_ssa_points_to_duplicate` and `_fail` (C++,
  `std::multimap`, reduced from `github_9403`). Both end in the exception
  without the fix and give `SUCCESSFUL` and `FAILED` with it. ctest
  `field_sensitive_ssa|field-sensitivity|github_7797`: 92/92.
- The other verdict differences: `assoc_emplace`, `om_source_from_vfs`,
  `vector_emplace_back` and `_fail`, and `vector_reserve_realloc` time out
  at 20 s without the flag and give their expected verdicts with it.
- Still open for the Phase 2 gate: as §8.25, plus a Bitwuzla run of the
  above.

### 8.27 Phase 2, part 1: a phi over a copy carried over its own name

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2-locals` at
`0a43398b7`. RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla
0.9.0, 4-core container. "Base" is `0a43398b7`'s binary.

- Cause of the open item in §8.25. In `std::map::upper_bound`, `begin()`'s
  return temporary is split once its address may be taken, so the copy
  `it = begin()` is carried as `it#1 WITH [...]` over `it`'s own L2 name
  (§8.21). At the merge after `return it;`, `phi_function` renames the
  other path's value of `it` to that chain and assigns it on the current
  path, whose value set has already dropped `it` at its `DEAD`. Read there,
  `it.it_key` and `it.it_value` have no entry and become unknown, and
  `merge_value_sets` keeps the unknown next to the other path's targets. The
  next `operator++` then emits invalid-pointer claims at `map:110-111`.
  Without the flag the carried value reads `begin`'s temporary instead,
  which both paths still hold.
- Change, with the flag only: when the merged path's value is a `with`
  chain over an L2 name of the object the phi assigns, the phi leaves the
  current path's value-set entries for that object as they are, and
  `merge_value_sets` adds the merged path's entries. This is sound because
  each path's entries describe that path's value, and the merge takes their
  union. `goto_symex_statet::assignment` takes a flag to skip the value-set
  update.
- `github_5868_is_scalar_const_lookup_fail` with the flag: 657 VCCs and 48
  `NOT CHECKED` claims under Z3 and Bitwuzla, as without the flag (base:
  661 and 50).
- New pair `field_sensitive_ssa_phi_own_name` (`--multi-property`,
  `SUCCESSFUL`) and `_fail` (an out-of-bounds read through the iterator,
  `FAILED`), both C++ and reduced from the map. Both also require that no
  `invalid pointer` claim is printed. On the base binary the first lists
  3 such claims and `_fail` lists one, so both fail there; both pass with
  the change.
- Both outcomes of the new decision are reached under Z3 and Bitwuzla, in
  a C harness of the same shape and in the C++ test (instrumented build
  printing a marker per outcome).
- Flag forced on, base against this change, 30 s per run: every fourth
  `regression/esbmc-cpp/cpp` directory (345 `CORE` tests), every eighth
  `regression/esbmc` directory (435), and every 150th `regression/python`
  directory (61). 0 verdict changes. Text changes only in
  `github_5868_is_scalar_const_lookup`, `_fail` and `github_9372`, which
  generate fewer claims; the rest is timings and addresses.
- ctest without the flag: `esbmc-cpp/cpp` 1472/1480 and `esbmc/` 3599/3618,
  and the same 8 and 19 tests fail on the base binary. Under the 10-minute
  cap, 1637 `python` tests ran and all passed.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868`:
  215/215.
- Corpus with the flag, symex seconds, medians of three, 8 / 2048 fields:
  `local_addr_once` 0.014 / 0.016 (1.14x) under Z3 and 0.014 / 0.015
  (1.07x) under Bitwuzla; `repro7521` 0.677 / 0.906 (1.34x) and 0.688 /
  1.032 (1.50x); `local_writes` 0.014 / 0.016 (1.14x) and 0.013 / 0.015
  (1.15x).
- Still open for the Phase 2 gate: globals (`global_callee`, `plc_db`), the
  assert-enabled run under Bitwuzla (§8.26), and an SV-COMP run.

Re-measured on 2026-11-20 after merging master `aabf837b1` into
`feat/field-sensitive-ssa-phase2-locals` (`f212d6ff0`; #10864 was
squash-merged, so its conflicts were resolved in favour of this branch).
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3, 4-core container. Bitwuzla
was not built: its CaDiCaL download is refused by the sandbox proxy.
"Master" is a build of `aabf837b1`.

- Flag forced on against off, same binary, 30 s per run: every fourth
  `regression/esbmc-cpp/cpp` directory (343 `CORE` tests) and every eighth
  `regression/esbmc` directory (437). 0 verdict changes. The only
  differences are 6 tests that already pass `--field-sensitive-ssa` (given
  twice, the option is rejected), `vector_reserve_realloc` and
  `assoc_emplace`, which hit the 30 s cap without the flag (see below), and
  random witness UUIDs; no trace text changes.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868`:
  215/215.
- ctest without the flag, 10-minute cap, `-j4`: `esbmc-cpp/cpp` 627 tests
  finished, 2 failed; `esbmc/` 746 finished, 4 failed; every 20th
  `python/` test, 305 of 493 finished, 1 failed. Each failure prints the
  same output on master: `vector_reserve_realloc` (Z3 does not finish
  within 150 s), `ch8_5` (no `out_of_range` under LLVM 18), `02_eureka23`
  and `gcc_popcount_01` (no Boolector), `builtin_clzg_ctzg` and
  `bundled_headers_from_vfs` (LLVM 18), and `complex_cmath_log`
  (`THOROUGH`, killed for memory on master too).
- Corpus with the flag under Z3, symex seconds, medians of three, 8 / 2048
  fields: `local_addr_once` 0.020 / 0.026 (1.30x); `repro7521` 0.922 /
  1.355 (1.47x); `local_writes` 0.022 / 0.021 (0.95x).
- clang-format 23.1.5 clean; complexity gate: nothing over threshold.
- The merge adds no branch to `src/`.

### 8.28 Phase 2, part 2: static objects

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2-globals`,
stacked on `feat/field-sensitive-ssa-phase2-locals` at `738317be6`.
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla 0.9.0 (CaDiCaL
2.1.2 cloned with git), 4-core container. "Base" is `738317be6`'s binary.

What changes, behind `--field-sensitive-ssa` only: a global or a static
local of struct type is eligible under the same rules as a local. Its field
symbols copy its `static_lifetime`, so they rename to `level1_global` and
are live everywhere. Two kinds stay whole: a `_Thread_local` object, and
every static object when `--memory-leak-check` is set, since the leak check
takes the context's static symbols as roots and reads them by name. On the
harnesses below the leak verdicts did not change with that exclusion
removed, so it is a precaution. A `const` global is not excluded: the
read-only write check does not fire on a const struct on master either
(`const struct S g; ((struct S *)&g)->a = 3;` is `SUCCESSFUL` with and
without the flag).

Results:

- Harnesses, with and without the flag, Z3 and Bitwuzla: a global written
  through a pointer in a callee, a static global with an initialiser, a
  static local, a whole copy into and out of a global, a `const` global, a
  `_Thread_local` global, and three `--memory-leak-check` programs. Same
  verdicts both ways under both solvers.
- New tests: `field_sensitive_ssa_global` and `_fail` (4 objects: a global,
  a static global, a static local and a local); `_thread_local` and
  `_leak_check`, which pin the split count with a local next to a static
  object that stays whole. Each fails on its mutant: allowing every static
  object fails `_global` and `_fail`; dropping the thread-local exclusion
  fails `_thread_local`; dropping the leak-check exclusion fails
  `_leak_check`. `field_sensitive_ssa_members` now splits its global too
  (2 objects, was 1); its verdict is unchanged.
- Every outcome of the new decision (static object split, kept for
  `_Thread_local`, kept for the leak check, field renamed at
  `level1_global`) reached under Z3 and Bitwuzla (instrumented build).
- Flag forced on, base against this change, 20 s per run, every eighth
  directory of `regression/esbmc` and `regression/esbmc-cpp/cpp` (606
  `CORE` tests): 0 verdict changes; 121 split more objects.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868`:
  218/219. `field_sensitive_ssa_carried_copy_fail` fails: it pins the step
  `g = { .p=..., .k=0 }` of a global copied from a split return value, and
  `g` is now split. The print rides on the step of `g`'s last field, which
  nothing reads, so the slicer removes it. With the base binary a split
  *local* in the same place loses the step the same way, so this is the
  §8.21 loss for split destinations, not new to globals. Handing the print
  to the kept `g.p` step is not enough: the printed value reads the whole
  read of the return value's own symbol, which the slicer also removes, so
  §8.8's rule hides it. Fixed in §8.30.
- Corpus, flag on, symex seconds, 8 / 2048 fields. Single runs right after
  the build: `global_callee` 0.330 / 1.971 (6.0x) under Z3 and 0.291 /
  2.019 (6.9x) under Bitwuzla; `plc_db` 0.033 / 0.205 (6.2x) and 0.034 /
  0.204 (6.0x). Medians of three on an idle machine later: `global_callee`
  0.547 / 3.751 (6.9x) and 0.663 / 3.691 (5.6x); `plc_db` 0.086 / 0.455
  (5.3x) and 0.073 / 0.499 (6.8x). Neither row meets the 2x bar. Without
  the flag, `global_callee` at 2048 takes 3.759 s (Z3) and 3.961 s
  (Bitwuzla) of symex, and `plc_db` at 512 and 2048 gives no result in
  300 s under Z3; with the flag both solve in under 1 s (Z3 0.60 s and
  0.83 s wall). Stack samples of `global_callee` at 2048 with the flag sit
  in `dereferencet::check_data_obj_access`, `check_pointer_alignment` and
  `object_base_alignment`, which call `type_byte_size` on the whole struct
  on every dereference of `d`: the remaining width cost is in the pointer
  checks, not in SSA.
- ctest without the flag: the `esbmc` label did not finish within the
  10-minute cap. The change only acts when the flag is set
  (`split_structst::analyse` returns before it otherwise).
- Still open for the Phase 2 gate: the 2x bar on `global_callee` and
  `plc_db` (per-dereference size and alignment work), the trace step above,
  the assert-enabled run under Bitwuzla (§8.26), and an SV-COMP run.

### 8.29 Phase 2, part 3: record sizes in the dereference checks

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2-deref-size`,
stacked on `feat/field-sensitive-ssa-phase2-globals` at `d74461ab2`.
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla 0.9.0, 4-core
container. "Base" is `d74461ab2`'s binary.

Callgrind on `global_callee` at 2048 fields with the flag, base: 71 % of all
instructions are in `type_byte_size_expr`, called from
`dereferencet::object_base_alignment`. `size_bits_expr` builds one
`constant_int2t` per member (13.1 M for 6400 calls), and `check_data_obj_access`
spends another 11 % summing the same members in `type_byte_size_bits`.

What changes: `type_sizet` caches the size of a struct or union per type
(`thread_local`, keyed by `type2tc`), for both `size_bits` and
`size_bits_expr`, but only when sizing it followed no symbol type through the
namespace, so the cached value is a function of the type alone. This is not
behind the flag: it returns the same numbers on every path.

Results:

- Corpus, flag on, symex seconds, median of three, 8 / 2048 fields
  (base, then this change):

  | Row | Z3 base | Z3 new | Bitwuzla base | Bitwuzla new |
  |---|---|---|---|---|
  | `global_callee` | 0.253 / 1.723 (6.8x) | 0.180 / 0.205 (1.14x) | 0.220 / 1.650 (7.5x) | 0.195 / 0.199 (1.02x) |
  | `plc_db` | 0.026 / 0.181 (7.0x) | 0.024 / 0.033 (1.38x) | 0.028 / 0.175 (6.3x) | 0.025 / 0.039 (1.56x) |
  | `local_addr_once` | 0.020 / 0.020 | 0.018 / 0.019 | 0.017 / 0.022 | 0.018 / 0.021 |
  | `local_writes` | 0.019 / 0.020 | 0.024 / 0.020 | 0.021 / 0.019 | 0.018 / 0.020 |
  | `repro7521` | 0.846 / 1.322 | 0.790 / 1.361 (1.72x) | 0.837 / 1.283 | 0.834 / 1.263 (1.51x) |

- Base against this change, 30 s per run, every `test.desc` option kept:
  every fourth C and C++ `CORE` test plus every tenth Python `CORE` test
  without the flag (2142), and every eighth C and C++ plus every twentieth
  Python test with the flag forced on (1071). 0 verdict changes. 32 outputs
  differ, all in witness UUIDs, Clang AST addresses and one Python temporary
  named after an address.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|type_byte_size`:
  218/219. `field_sensitive_ssa_carried_copy_fail` fails as on the base
  (§8.28), with output identical to the base binary's.
- New unit test: a struct and a union holding a symbol type, sized against two
  namespaces that resolve it to 4 and 8 bytes, through both `type_byte_size`
  and `type_byte_size_expr`. It fails when the cache ignores whether a symbol
  was followed.
- Branches (instrumented build, C and C++ harnesses, Z3 and Bitwuzla, same
  counts under both): cache hit and store in `size_bits` and in
  `size_bits_expr`, the struct and union arms, the no-store arm of `size_bits`
  and both symbol-follow arms are reached. The no-store arm of
  `size_bits_expr` was reached by none of the harnesses or by ~700 sampled C,
  C++ and Python tests: a record the dereference sizes this way has already
  been sized by `size_bits`. The unit test reaches it. The `cpp_name` arms,
  which already existed and now also set the flag, were not reached.
- Still open for the Phase 2 gate: the trace step of §8.28, the
  assert-enabled run under Bitwuzla (§8.26), and an SV-COMP run.
- After merging `feat/field-sensitive-ssa-phase2-globals` at `df48c998c`
  (§8.30), Z3 4.13.3 only (Bitwuzla's CaDiCaL download was refused): ctest
  `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|type_byte_size`
  220/220, so `field_sensitive_ssa_carried_copy_fail` now passes here. Corpus,
  flag on, symex seconds, median of three, 8 / 2048 fields:
  `global_callee` 0.213 / 0.217 (1.01x), `plc_db` 0.026 / 0.034 (1.30x),
  `local_addr_once` 0.019 / 0.019, `local_writes` 0.017 / 0.019,
  `repro7521` 0.757 / 1.102 (1.45x); all `VERIFICATION SUCCESSFUL`. Flag
  forced on against off, 20 s per run, every eighth C and C++ `CORE`
  directory and every fortieth Python one (852 tests; 6 already pass the
  flag): 0 verdict changes. `typed_list_eq_sorted_fail` hit the cap only
  without the flag; alone it gives `VERIFICATION FAILED` both ways (25 s
  and 13 s).

### 8.30 Phase 2, part 2: the print of a sliced split write

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2-globals` after
merging `feat/field-sensitive-ssa-phase2-locals` at `9e276a5e5`.
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3 and Bitwuzla 0.9.0 (CaDiCaL
2.1.2 cloned with git), 4-core container. "Base" is the merge's binary.
(§8.29 is on the PR stacked on this one.)

`symex_assign_split` prints a whole write to a split object on the step of
its last field. In `field_sensitive_ssa_carried_copy_fail` that field is
`g`'s padding, which nothing reads, so the slicer removed the step and the
print with it, while the kept `g.p` step stayed hidden. Under the flag the
slicer now moves the print of a removed last-field step to the latest kept
field step of the same write (same instruction, same `original_lhs`) and
keeps what the print reads, as the unsplit object's step keeps its
right-hand side. §8.28's obstacle, a print reading the return value's
removed whole read, goes away because that read is now kept. The test
prints `g = { .p=&b, .k=0 }`; without the flag Z3 prints
`g = { .p=t.p, .k=0 }`.

Results:

- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868`:
  220/220 (219 before, plus the new test). `field_sensitive_ssa_carried_copy_fail`
  and the new `field_sensitive_ssa_carried_copy_local_fail` (the same
  program with a split local) fail on the base binary, which prints no step
  for `g`, and pass under Z3 and Bitwuzla.
- Flag forced on, base against this change, 20 s per run: every fourth
  directory of `regression/esbmc` and `regression/esbmc-cpp/cpp` and every
  twentieth of `regression/python` (1680 `CORE` tests). 0 verdict changes.
  27 outputs differ. 16 Python tests differ only in temporary paths or
  timings, and `github_7905_struct_refused` and `ts_dump_exec_guard` only
  in timings. `github_8777_fail` ran past the 20 s cap both ways; re-run
  alone, both take 27 s and fail the same 6 of 13 properties.
  `assoc_emplace` and `field_sensitive_ssa_copy` keep more assignments.
  `field_sensitive_ssa_carried_copy_fail` prints its step (the fix).
  `overflow-result3` and `overflow-result-all-fail` print their `foo`
  steps again, now the same as without the flag (§8.6 listed them as
  dropped). `field_sensitive_ssa_print_pointer_fail` adds the steps
  `t = { .a=5, .b=2 }` and `t = { .a=2, .b=5 }`, the values the program
  computes from `x = 5`. `os1_fail` reports another uncaught exception from
  a different model, same verdict. In `field_sensitive_ssa_phi_own_name_fail`
  (no `--multi-property`), `assertion.3` is labelled `NOT CHECKED` instead
  of `FAILED`: that mode labels a claim `FAILED` only when the one model
  violates it, and the extra kept steps change the model. With
  `--multi-property` both binaries report it `FAILED`. This run used a
  build that still had the unreached check dropped below; the ctest runs
  and the corpus used the final code.
- Each outcome of `carry_split_print` reached under Z3 and Bitwuzla
  (instrumented build, the new test's program and a variant whose last
  field is kept): not a split field step; a removed printing step; a kept
  printing step; a hidden step with no matching print; a removed sibling;
  the print moved. A check for a sibling kept only for a nondet result was
  reached by no harness: a nondet split write gives its last field a nondet
  right-hand side too, so that step is kept rather than removed. It was
  dropped. The new test's program with `__ESBMC_unreachable()` after the
  copy reports it reached under both solvers.
- Corpus, flag on, symex seconds, median of three, 8 / 2048 fields:

  | Row | Z3 base | Z3 new | Bitwuzla base | Bitwuzla new |
  |---|---|---|---|---|
  | `repro7521` | 0.693 / 0.984 | 0.667 / 0.974 | 0.681 / 1.016 | 0.667 / 1.133 |
  | `local_writes` | 0.014 / 0.017 | 0.014 / 0.017 | 0.014 / 0.016 | 0.016 / 0.016 |
  | `local_addr_once` | 0.015 / 0.018 | 0.014 / 0.017 | 0.014 / 0.019 | 0.015 / 0.016 |
  | `global_callee` | 0.169 / 1.287 | 0.156 / 1.304 | 0.160 / 1.201 | 0.160 / 1.343 |
  | `plc_db` | 0.032 / 0.138 | 0.020 / 0.143 | 0.020 / 0.141 | 0.021 / 0.145 |

  All `VERIFICATION SUCCESSFUL`. `plc_db_w2048` ran with the `w8` row's
  options, since its own test only dumps SMT-LIB. The change runs after
  symex, so the table only bounds the noise. The global rows miss the 2x
  bar on this branch; #10894 brings them under it (§8.29).
- ctest without the flag (the change acts only under it), 10-minute caps:
  `esbmc-cpp/cpp` 627 of 628 run, `esbmc/` 902 of 906 run; the 1 and 4
  failures (`ch8_5`, `02_eureka23`, `builtin_clzg_ctzg`,
  `bundled_headers_from_vfs`, `gcc_popcount_01`) fail the same way with the
  base binary.
- clang-format 23.1.5 clean.
- Still open for the Phase 2 gate: the assert-enabled run under Bitwuzla
  (§8.26) and an SV-COMP run.

### 8.31 Phase 2: assert-enabled run under Bitwuzla

Measured on 2026-11-20 on `feat/field-sensitive-ssa-phase2-deref-size` at
`106575464`, the head of the Phase 2 stack (#10876, #10886, #10894). Built
with `-O2 -g1` and no `-DNDEBUG`, so the Invariant 3 assert in
`split_structst::rewrite` is live (its text is in the binary). gcc 13.3,
LLVM 18, Bitwuzla 0.9.0 (CaDiCaL 2.1.2 cloned with git), 4-core container.

- Every `CORE` test of `regression/esbmc` and `regression/esbmc-cpp/cpp`
  that does not already pass the flag (4838 of 4881), flag forced on and
  off, solver options replaced by `--bitwuzla`, 20 s per run, 4 at a time.
  No run prints a failed `assert`. Two runs end in `SIGABRT` both ways, each
  printing the `ERROR:` its test expects (`github_7905_struct_refused`,
  `init_obj_vlas`).
- Verdict differences: `ch20_4`, `vector_reserve_realloc` and `_fail` hit
  the 20 s cap without the flag and give their expected verdicts with it
  (19.3 s, 0.8 s, 0.8 s). Every other test has the same verdict and exit
  code both ways; 9 hit the cap both ways.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|type_byte_size`
  on this build (default solver Bitwuzla): 220/220.
- This closes §8.26's open Bitwuzla run. Still open for the Phase 2 gate:
  an SV-COMP run.

### 8.32 Phase 2: the stack merged with master `a6615b898`

Measured on 2026-11-21 on `feat/field-sensitive-ssa-phase2-deref-size` at
`f7dc7c07d`: master `a6615b898` merged into #10876 (`f76c06e4d`), then up
the stack into #10886 (`ad66ead87`) and #10894. No conflicts. RelWithDebInfo,
gcc 13.3, LLVM 18, Z3 4.13.3, Bitwuzla 0.9.0 (CaDiCaL 2.1.2 cloned with
git), 4-core container.

- Corpus, flag on, symex seconds, median of three, 8 / 2048 fields:

  | Row | Z3 | Bitwuzla |
  |---|---|---|
  | `global_callee` | 0.215 / 0.223 (1.04x) | 0.227 / 0.274 (1.21x) |
  | `plc_db` | 0.028 / 0.036 (1.29x) | 0.029 / 0.037 (1.28x) |
  | `local_addr_once` | 0.019 / 0.021 (1.11x) | 0.019 / 0.024 (1.26x) |
  | `local_writes` | 0.020 / 0.021 (1.05x) | 0.019 / 0.021 (1.11x) |
  | `repro7521` | 0.800 / 1.205 (1.51x) | 0.818 / 1.204 (1.47x) |

- Every corpus row (30), Z3 and Bitwuzla, flag on and off, 600 s cap: 116
  of 120 runs give the expected verdict, all 60 with the flag among them.
  The other 4 are `plc_db` at 512 and 2048 fields (and `_fail`) under Z3
  without the flag, which hit the cap, as in §8.2.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|type_byte_size`:
  220/220.
- Flag forced on against off, solver options replaced, 20 s per run: every
  eighth C and C++ directory and every fortieth Python one under Bitwuzla
  (853 `CORE` tests), half of those under Z3 (433). No verdict changes.
  The only differing outcomes are tests whose `test.desc` already passes the
  flag (6 and 3), where passing it twice is a command-line error. 3 and 5
  tests hit the cap both ways. `github_6310_typeid_dynamic_fail` aborts with
  the same Z3 sort mismatch both ways.
- Trace text under Bitwuzla: 35 outputs differ. 6 Python ones differ between
  two runs without the flag as well. Of the other 29, 16 differ only by the
  warning that the flag is ignored (threads, `--all-witnesses`,
  `--generate-pytest-testcase`). The other 13 fall into the §8.6 rows or
  differ in the model: `github_1476_fail`, `github_5868_div_fail`,
  `github_7597-literal_fail`, `irep2_constructor_assign_struct_arg_fail` and
  `github_9443_fail` add or drop a step; `github_9375_fail` prints a member
  write's value where the run without the flag prints the whole struct, with
  equal values; `github_7597-cond-write_fail` and `typed_list_eq_sorted_fail`
  pick another value for a nondet input (`IO.in`, `n`), and every value
  derived from it agrees with that input; `github_9416_fail`,
  `container_cbegin_cend2_fail`, `map_find_end_neq` and `assoc_contains_fail`
  list fewer claims, since more fold to constants in simplification, with
  the same failed claim and verdict; `github_9060` (`--show-symex-value-sets`)
  dumps value sets that name field symbols. Under Z3, 14 outputs differ, all
  of those kinds.
- ctest `-L esbmc` without the flag, 10-minute cap: 2336 passed, 7 failed
  before the cap. 6 are environmental: `02_eureka23`, `gcc_popcount_01`,
  `github_1352-32bit` and `github_1352-fail-32bit` need Boolector,
  `builtin_clzg_ctzg` needs a newer clang than 18, and
  `bundled_headers_from_vfs` expects a `-resource-dir` this LLVM 18 install
  does not use. `cwe_dead_code_concurrent` fails
  ctest on its `CHECK_FILE out.sarif`; run by hand it prints the expected
  text, and the stack changes no dead-code or SARIF file.
- Still open for the Phase 2 gate: an SV-COMP run.

### 8.33 Phase 2: the stack merged with master `30592f8a7`

Measured on 2026-11-21 on `feat/field-sensitive-ssa-phase2-deref-size` at
`ae651e171`: master `30592f8a7` merged into #10876 (`0d6c46a0a`), then up
the stack into #10886 (`97074691e`) and #10894. No conflicts. Same toolchain
as §8.32. The merge brings #10895 (`construct_struct_from_members`, which
reads a struct laid over another struct's members, from the bytes where no
member matches) and #10897 (no alignment check for packed reads).

- #10895 with the flag: a local and a global `struct A { char a, b; }` read
  through a `struct B { char c, d; } *` keep the object whole (the cast makes
  it ineligible), so no byte-level read reaches a split object. Under Z3 and
  Bitwuzla, with and without the flag, the `same` claims hold and only the
  `fail` claim is violated.
- Corpus, flag on, symex seconds, median of three, 8 / 2048 fields:

  | Row | Z3 | Bitwuzla |
  |---|---|---|
  | `global_callee` | 0.243 / 0.253 (1.04x) | 0.253 / 0.290 (1.14x) |
  | `plc_db` | 0.032 / 0.038 (1.19x) | 0.031 / 0.038 (1.23x) |
  | `local_addr_once` | 0.021 / 0.023 (1.09x) | 0.021 / 0.022 (1.04x) |
  | `local_writes` | 0.021 / 0.022 (1.04x) | 0.020 / 0.022 (1.10x) |
  | `repro7521` | 0.845 / 1.313 (1.55x) | 0.876 / 1.295 (1.47x) |

  `plc_db` is timed with `--unwind 21` and the solver named, since its
  2048-field `test.desc` only writes SMT-LIB; every width gives `SUCCESSFUL`.
- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|type_byte_size|github_10888`:
  224/224.
- Flag forced on against off, solver options replaced, 20 s per run: every
  eighth C and C++ directory, every fortieth Python one and every test
  master added since `a6615b898`, under Bitwuzla (851 `CORE` tests, 434
  with an object split); half of those under Z3 (432). No verdict changes.
  The one differing outcome is `vector_emplace_back` under Z3, which hits the
  cap without the flag and gives `SUCCESSFUL` with it.
- Trace text, after dropping timings, assignment and VCC counts, witness
  UUIDs and AST addresses: 20 outputs differ under Bitwuzla, 9 under Z3, all
  of §8.32's kinds. The new `github_10888_fail` adds a step for the
  initialised `c`; `github_10135_irep2_only_fail`, `github_1590-local`,
  `github_5393_array_member_dyn_offset_fail` and
  `github_7597-array-member_fail` drop or add a step or print a member write
  with equal values; `vector_emplace_back` and `github_5868_iterator_ops`
  list fewer claims after simplification (192 to 138 and 113 to 84 `PASSED`),
  with the same verdict; `github_10888` sends two claims to the solver that
  fold away without the flag, and both hold; `github_2630_1` prints its
  counterexamples in another order. `python/github_9314_fail` reports
  `global.assertion.5` or `.2` as the failed claim under Z3 from one run to
  the next without the flag as well.
- ctest without the flag, 10-minute cap: `-L esbmc` 1835 passed, 4 failed
  before the cap (`02_eureka23`, `gcc_popcount_01`, `builtin_clzg_ctzg`,
  `bundled_headers_from_vfs`, the environmental failures of §8.32);
  `-L esbmc-cpp/cpp` 1473/1481. The 8 C++ failures (`ch8_5`, the five
  `github_7433*`, `github_8698_array_uncaught_fail`,
  `github_8698_string_view_uncaught_fail`) fail the same way on a master
  `30592f8a7` build in the same container.
- Still open for the Phase 2 gate: an SV-COMP run.

### 8.34 Phase 2: the whole `esbmc/` suite on the stack's head, Z3 only

Measured on 2026-11-21 on `feat/field-sensitive-ssa-phase2-deref-size` at
`b357259ff` (code as §8.33), gcc 13.3, LLVM 18, Z3 4.13.3, without the flag.
Bitwuzla could not be built: the proxy refuses the CaDiCaL archive it
downloads, so tests that name no solver run under Z3. All 3667
`regression/esbmc/` tests, in strided shards of at most 10 minutes, `-j4`,
leftover ESBMC processes killed after each shard:

- 3637 passed, 2 skipped, 20 failed, 8 did not finish.
- 16 failures name `--boolector` or `--bitwuzla`, which this build lacks.
  `builtin_clzg_ctzg` and `irep2_only_va_binds_by_reference_c23` need a
  newer Clang than 18; `bundled_headers_from_vfs` is §8.32's `-resource-dir`
  regex. `github_1172-3` and `github_1352-success-32bit` name no solver and
  fail under Z3 after 62 s and 75 s.
- Not finished under Z3 in 10 minutes, alone on the machine: the CORE tests
  `builtin_overflow_generic` (7 VCCs, the time is all in Z3),
  `github_9785_int_base_stores` and `_fail`, `github_10774_symbolic_fixedbv`,
  and the THOROUGH `github_306`, `github_2513`, `github_302`, `realloc20`.
- No failure is outside these environmental classes, but the two Z3
  failures and the eight unfinished tests were not compared with a master
  build, so the full-label gate stays open until a run with Bitwuzla.

### 8.35 Phase 2: the stack merged with master `6c3c2cf9d`

Measured on 2026-11-21 on `feat/field-sensitive-ssa-phase2-deref-size` at
`c519e3437`: master `6c3c2cf9d` merged into #10876 (`6fe7738aa`), then up
the stack into #10886 (`19e35edb8`) and #10894. No conflicts. The merge
brings #10906 and #10907, which rebuild a struct read from the bytes of a
wider member, of an aggregate or of an unaligned array. RelWithDebInfo,
gcc 13.3, LLVM 18, Z3 4.13.3, Bitwuzla 0.9.0 (CaDiCaL 2.1.2 cloned with
git), 4-core container; master built the same way in the same container.

- The 34 tests master added since `30592f8a7`, flag forced on against off,
  Z3 and Bitwuzla, 20 s each: no verdict changes. In the `github_10888*`
  and `github_10891*` tests the struct is read through a cast pointer, so
  it stays whole. The three Kani `goto-transcoder` `_fail` tests print
  added steps for whole writes of a tuple struct (`var_5 = { .=1, .=0, .=0 }`,
  §8.6 row 2); under Z3, `cbmc_kani_unaligned_struct_store_fail` also
  picks other values for two unconstrained variables.
- Corpus, flag on, symex seconds, median of three, 8 / 2048 fields:

  | Row | Z3 | Bitwuzla |
  |---|---|---|
  | `global_callee` | 0.173 / 0.185 (1.07x) | 0.159 / 0.169 (1.06x) |
  | `plc_db` | 0.021 / 0.030 (1.43x) | 0.020 / 0.029 (1.45x) |
  | `local_addr_once` | 0.014 / 0.016 (1.14x) | 0.016 / 0.017 (1.06x) |
  | `local_writes` | 0.015 / 0.017 (1.13x) | 0.015 / 0.018 (1.20x) |
  | `repro7521` | 0.736 / 1.222 (1.66x) | 0.682 / 1.107 (1.62x) |

- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|github_10888|github_10891`:
  245/245; the `type_byte_size` unit test passes (13 cases).
- Flag forced on against off, solver options replaced, 20 s per run: every
  eighth C and C++ directory and every fortieth Python one under Bitwuzla
  (842 `CORE` tests, 423 with an object split), every other one of those
  under Z3 (417, 218 split). No verdict changes. Trace text, normalised as in
  §8.33: 40 outputs differ under Bitwuzla and 17 under Z3. 31 differ only by
  the warning that the flag is ignored. 6 Python tests,
  `intrinsic_unroll_unwind_assertion_fail` and
  `pragma_unroll_nested_complete_true` print different text from one run
  to the next with the flag and without it; `harness_math_remainder_fail`
  under Z3 picks a different `x` on a second run with the flag, which is
  the value the run without it prints. The rest are the kinds §8.32 lists: `github_1093`,
  `github_9375_fail`, `assoc_contains_fail` and `github_9416_fail` print a
  member write with the value the whole struct holds; `github_5393_array_member_fail`,
  `github_7855-tie-hides-deref` and `struct_array_ite_new_index_fail`
  print the reassembled struct (§8.6 row 3); `container_cbegin_cend2_fail`,
  `irep2_constructor_assign_struct_arg_fail` and `typed_list_concat_fail`
  add or drop a step; `assoc_contains_fail` (190 to 126 claims),
  `github_9416_fail` (103 to 102) and `container_cbegin_cend2_fail` (2 more
  decided) fold more claims, with the same failed claim;
  `map_find_end_neq` and `read_spec_verify_1` print fewer loop-unwinding
  lines, since per-field propagation folds the guards; and
  `typed_list_sorted_untyped_nondet_fail` picks other nondet inputs.
- Whole `esbmc/` ctest label without the flag, default solver Bitwuzla,
  `-j4`: all 3686 tests finished in 452 s. 3666 passed and 20 failed. Run on
  the master build, 16 of the 20 fail as well: Boolector
  (among them `github_1352-32bit` and `_fail`), Clang newer than 18,
  `bundled_headers_from_vfs`'s `-resource-dir` regex, and the
  `tuple_const_array_*`, `math_log-rec*` and `overflow_*_fir*` tests. The other 4
  (`cwe_dead_code_concurrent`, `github_664_fail`, `proof_cache_kinduction`,
  `witness_float_return_nan`) failed because the flag comparison above had
  left their output files (`out.sarif`, `out.txt`, `fp.tsv`, `w.yml`) in
  the test directories. With those files removed, all 4 pass.
- §8.34's ten Z3 failures and unfinished tests run through ctest with
  their own options (Bitwuzla), 280 s cap: all ten pass, on this head and on
  master.
- ctest `esbmc-cpp/cpp/`: 1473/1481; the same 8 as §8.33 fail on the master
  build. ctest `python/` under the 10-minute cap: 1618 run, all passed.
- Still open for the Phase 2 gate: an SV-COMP run.

### 8.36 Phase 2: the stack merged with master `3194304fb`

Measured on 2026-11-21 on `feat/field-sensitive-ssa-phase2-deref-size` at
`23283b102`: master `3194304fb` merged into #10876 (`71466a59a`), then up
the stack into #10886 (`80dc69d74`) and #10894. No conflicts. The merge
brings #10901, which makes the value set of a `byte_extract` also look up
the member paths at its offset. A split object never reaches a
`byte_extract` (Invariant 3), so the new lookup sees only whole objects.
RelWithDebInfo, gcc 13.3, LLVM 18, Z3 4.13.3, Bitwuzla 0.9.0 (CaDiCaL 2.1.2
cloned with git), 4-core container.

- ctest `field_sensitive_ssa|field-sensitivity|github_7797|github_5868|github_10888|github_10891|type_byte_size|cbmc_kani_nonnull_write|cbmc_kani_dyn_size_of_val`:
  250/250.
- Corpus, flag on, symex seconds, median of three, 8 / 2048 fields:

  | Row | Z3 | Bitwuzla |
  |---|---|---|
  | `global_callee` | 0.223 / 0.239 (1.07x) | 0.234 / 0.244 (1.04x) |
  | `plc_db` | 0.029 / 0.038 (1.31x) | 0.030 / 0.038 (1.26x) |
  | `local_addr_once` | 0.019 / 0.021 (1.10x) | 0.021 / 0.022 (1.04x) |
  | `local_writes` | 0.019 / 0.021 (1.10x) | 0.021 / 0.021 (1.00x) |
  | `repro7521` | 0.909 / 1.233 (1.35x) | 0.900 / 1.351 (1.50x) |

- Flag forced on against off, solver options replaced, 20 s per run: every
  eighth C and C++ `CORE` directory, every fortieth Python one and the 4
  tests #10901 added, under Bitwuzla (851 tests, 439 with an object split),
  and every other one of those under Z3 (426, 225 split). No verdict
  changes. Once timings, the split count and the ignored-flag warning are
  removed from the output, 33 outputs differ under Bitwuzla and 18 under
  Z3. 16 of the Bitwuzla ones differ only in witness UUIDs, Python
  temporaries named after an address or a temporary file, operational-model
  load times, or a `TS-CHECK` extraction time. The rest are the kinds §8.32
  lists: a member write printed with its value instead of the whole struct
  (`assoc_emplace_fail`, `github_9372_fail`,
  `field_sensitive_ssa_print_member_fail`), added or dropped steps for a
  whole write (`github_4281_value_propagates`,
  `irep2_constructor_assign_struct_arg_fail`, `github_10888_fail`, whose
  `c` is split while the cast-to `s` stays whole, `github_7597-*_fail`, and
  the two new Kani `_fail` tests), fewer loop-unwinding lines
  (`github_9416_fail`, `github_5868_*`), claims folded or solved in a
  different order (`github_7965_fail`,
  `github_8255_kstep_single_verdict_fail`,
  `field_sensitive_ssa_opaque_field`), and other nondet inputs
  (`nondet_list21`). Under Z3, `github_1087` (single-model
  mode) reports 1 failed claim with the flag and 3 without it;
  `cbmc_kani_nonnull_write_fail` picks other inputs. With
  `--multi-property`, both give every claim the same status with and
  without the flag under Z3 and Bitwuzla (`github_1087`: 5 of 17 failed;
  `cbmc_kani_nonnull_write_fail`: 1 of 8).
- CI on #10876's previous head `6fe7738aa`: "static llvm-22 DebugOpt"
  failed one test, `regression/esbmc/cbrtl_model`, the long-double solve
  that #10894's comments measured near its 120 s cap on other branches. The
  stack touches neither the model nor its solve. The re-run was cancelled
  when the merge was pushed. On this head, without the flag, the test gives
  `VERIFICATION SUCCESSFUL` in 87.4 s, 85.7 s of it in Bitwuzla, against
  the 120 s cap.
- Still open for the Phase 2 gate: an SV-COMP run.
