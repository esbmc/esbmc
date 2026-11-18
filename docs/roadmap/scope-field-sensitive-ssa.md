# Scope: field-sensitive SSA in goto-symex (#7521)

> **Status at 2026-11-18: Phase 0 merged (#10773); Phase 1 merged (#10778)
> with the §8.7 trace fix merged (#10799). §8.8 hides a split struct's
> printed write when its right-hand side reads a definition the slicer
> removed; with it, the Phase 1 trace gate is met on the C and C++ suites and
> a tenth of the Python suite under Z3, and on the same C and C++ suites and
> a twentieth of the Python suite under Bitwuzla. Every Python `CORE` test
> has now been run with the flag forced on under Z3: one verdict is lost,
> `tagged_attr_store` (25 s without the flag, no result in 1800 s with it,
> master's Phase 1 code as well), so the Phase 1 Python gate is unmet
> (§8.8). The loss is Z3-only (Bitwuzla: 8 s both ways) and holds at 7 of
> 8 Z3 seeds. Phase 1 also lacks an SV-COMP run.** The corpus is in `regression/field-sensitivity/`
> with baselines, and the census has been run and reverted. Phase 1 adds
> `--field-sensitive-ssa` (default off): symex gives each field of an
> eligible local, parameter or return slot its own SSA name. It meets the 2x
> bar on the Phase 1 corpus rows and changes no verdict on the C, C++ (Z3 and Bitwuzla) and
> k-induction suites or three Python samples (3132 tests), and no correctness witness.
> The Phase 1 trace gate is Invariant 4 (same values), with the §8.6 changes
> accepted. Two
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
