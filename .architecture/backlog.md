# Architecture deepening backlog

Candidates surfaced by `pm-deepen`, with the status that decides whether a
future run may pick them. Never delete an entry: `landed`, `dropped` and
`rejected` rows are the memory that stops a recurring run re-deriving the same
ideas. See `.architecture/reviews/` for the run that scored each one.

## irep2-adjust-arm-table

- **Status**: proposed
- **Score**: 23/25 (leverage 4, locality 5, blast radius 1, heat 5)
- **Files**: ~4 estimated
- **Modules**: `src/clang-c-frontend/clang_c_adjust_irep2.h`, `src/clang-c-frontend/clang_c_adjust_irep2.cpp`
- **Summary**: The IREP2 adjust pass dispatches its rewrites through a hand-ordered `if`-chain split across two functions; make the arm set an ordered table so an arm's predicate, action and ordering constraint live in one place and a test can read them.
- **First seen**: 2026-08-31

## contracts-single-file-seam

- **Status**: proposed
- **Score**: 15/25 (leverage 3, locality 3, blast radius 3, heat 3)
- **Files**: ~9 estimated
- **Modules**: `src/goto-programs/contracts/contracts.cpp`, `src/goto-programs/contracts/contracts.h`
- **Summary**: `code_contractst` carries 20 public and 55 private methods over one 6125-line implementation; the requires/ensures/assigns/loop-contract concerns share no seam.
- **First seen**: 2026-08-31

## python-list-facade

- **Status**: proposed
- **Score**: 15/25 (leverage 3, locality 3, blast radius 4, heat 4)
- **Files**: ~11 estimated
- **Modules**: `src/python-frontend/python-list/python_list.h` and its ten implementation files
- **Summary**: `python_list` presents a 925-line header over ~9.5k lines of implementation; the interface names the construction strategy (`build_bool_mask_row_select_symbolic`, `try_build_ravel_pointer_view`) rather than the list operation, so a caller must learn the implementation to pick a method.
- **First seen**: 2026-08-31

## function-call-expr-split

- **Status**: proposed
- **Score**: 15/25 (leverage 3, locality 3, blast radius 4, heat 4)
- **Files**: ~8 estimated
- **Modules**: `src/python-frontend/function_call/expr.h`, `src/python-frontend/function_call/expr.cpp`
- **Summary**: 151 declarations in a 795-line header over a 7012-line implementation; builtin lowering, method dispatch and type inference are one module with no internal seam.
- **First seen**: 2026-08-31

## python-converter-god-class

- **Status**: dropped
- **Score**: 18/25 (leverage 5, locality 5, blast radius 5, heat 5)
- **Files**: 40+ estimated for the programme; 3-5 per handler if sliced
- **Modules**: `src/python-frontend/python_converter.h` and the thirteen `src/python-frontend/converter/*.cpp` files
- **Summary**: One class with 152 methods and 65 public members, implemented across thirteen translation units, included by 37 files, and granting `friend` to fifteen handler classes — so there is no seam between the converter and any handler, and each reads the other's private state directly.
- **First seen**: 2026-08-31
- **Reason**: Blast radius 5 — repo-wide for the Python frontend, not one unattended PR. Weakly testable besides: a friend removal is verified only by "still compiles, suites still pass", which is a weaker gate than the other candidates have. Listed under *Too large to automate*; a human should schedule it, plausibly one handler at a time. The cheapest first slice is `python_dict_handler`'s friendship of `python_list`, which overlaps `list-element-types-owned-module`.

## cpp-om-library-headers

- **Status**: dropped
- **Score**: 12/25 (leverage 1, locality 3, blast radius 3, heat 5)
- **Files**: ~12 estimated
- **Modules**: `src/cpp/library/{string,vector,type_traits,map,set,list}`
- **Summary**: The hottest directory in the repo by file touches, but these are operational models that deliberately mirror libc++'s shape.
- **First seen**: 2026-08-31
- **Reason**: Leverage 1 — fails the deletion test. These headers exist to reproduce a published interface (the C++ standard library) verbatim; "deepening" them would make ESBMC diverge from the thing it is modelling. Their shape is a requirement, not a defect.

## address-decomposition-single-walk

- **Status**: proposed
- **Score**: 20/25 (leverage 3, locality 5, blast radius 2, heat 5)
- **Files**: ~4 estimated
- **Modules**: `src/util/expr/expr_simplifier.cpp`, `src/util/expr/type_byte_size.cpp`
- **Summary**: Five functions each walk an `index2t`/`member2t` chain to turn an address into a base plus an offset, with different rules, and no caller can tell which rules it got.
- **First seen**: 2026-08-31
- **Note**: Scored 24/25 on first pass and would have been the pick. Re-scored to
  leverage 3 after reading the five walks: their contracts differ *deliberately*
  along four orthogonal axes — constant-only vs symbolic subscripts, byte vs
  linear-element vs pointee units, member offsets counted vs skipped, unzeroed
  root vs rebuilt zeroed base. `subscript_offset_in_units:3771` skips members
  precisely *because* its caller rebuilds the base keeping the member path,
  while `address_root_and_offset:740` counts them precisely *because* it returns
  an unzeroed root. A single `decompose_address` reconciling these needs a knob
  per axis, which is a union rather than a deepening: the interface would be
  nearly as complex as the implementation. The *friction* is nonetheless the
  best-evidenced in the repo — six PRs in ten days (#7346, #7391, #7392, #7393,
  #7394, #7395) each hand-rolled part of this walk. A human should scope the
  genuinely shared core (typecast peeling, the walk skeleton, `member_offset`)
  rather than unify the contracts. Soundness-critical code; would warrant
  `needs-svcomp-run`.

## property-report-single-render-seam

- **Status**: proposed
- **Score**: 18/25 (leverage 4, locality 4, blast radius 4, heat 3)
- **Files**: ~5 estimated
- **Modules**: `src/esbmc/property_report.{h,cpp}`, `src/esbmc/bmc.cpp`
- **Summary**: The pure row-building half was extracted for testability and then never tested, while the decision that actually breaks — whether to print, which table, which caveats — stayed in `bmct::report_property_verdicts`.
- **First seen**: 2026-08-31
- **Note**: Crosses a published interface: this is the output `scripts/competitions/svcomp/esbmc-wrapper.py::parse_result()` matches on, and PR #7064 breaking it caused #7250 (~2600 verdicts turned `Unknown`). Needs `needs-svcomp-run`. Making that format unit-assertable is the point of the candidate, but the blast radius puts it outside an unattended run.

## list-element-types-owned-module

- **Status**: dropped
- **Score**: 19/25 (leverage 4, locality 4, blast radius 3, heat 4)
- **Files**: ~9 estimated
- **Modules**: `src/python-frontend/python-list/`, `src/python-frontend/python-dict/`
- **Summary**: `list_type_map` is a mutable process-global keyed by symbol-id string with ~59 raw `find`/`[]` sites; `python_dict_handler` is a `friend` of `python_list` only to squat on it under invented string keys.
- **First seen**: 2026-08-31
- **Reason**: Already in flight — PR #7366, "[python] Give per-instance element
  types one owned home", is open against this exact friction. Dropped to avoid a
  competing PR, not on merit: at 19/25 it would otherwise rank second overall.
  `dropped` is reversible by design, so a later run should re-score it once
  #7366 lands and move it back to `proposed` if the friction survives.

## tagged-value-representation-module

- **Status**: proposed
- **Score**: 17/25 (leverage 3, locality 4, blast radius 3, heat 4)
- **Files**: ~7 estimated
- **Modules**: `src/python-frontend/dynamic_type/`, `src/python-frontend/type/type_handler.cpp`
- **Summary**: "Is this value tagged?" is answered two incompatible ways — `dynamic_type_handler::is_tagged(name)` by name and `type_handler::is_tagged_scalar_type(t)` by type — and callers must know which applies where.
- **First seen**: 2026-08-31

## om-feature-gate-single-header

- **Status**: proposed
- **Score**: 17/25 (leverage 3, locality 4, blast radius 2, heat 5)
- **Files**: ~4 estimated if scoped to the three hottest headers
- **Modules**: `src/cpp/library/OM_compiler_defs.h` and the C++ operational-model headers
- **Summary**: 151 raw `__cplusplus >= NNNNNNL` comparisons across 59 headers, 42 of which never include the header that exists to own that decision.
- **First seen**: 2026-08-31
- **Note**: Distinct from the `cpp-om-library-headers` entry below, which is about the modelled interfaces themselves. This one is about the *version-gate* mechanism, which is ESBMC's own and is a legitimate deepening target. Changes what compiles under each `--std` mode, so every touched gate needs CLAUDE.md's probe protocol (`clang++ -std=<mode> -fsyntax-only` vs `esbmc --std <mode>`).

## preprocessor-state-ownership

- **Status**: dropped
- **Score**: 17/25 (leverage 3, locality 5, blast radius 5, heat 4)
- **Files**: ~18 estimated for the full programme
- **Modules**: `src/python-frontend/preprocessor/` — 19 mixins over one ~110-attribute state bag
- **Summary**: `Preprocessor` inherits 19 mixins that all read and write the same flat attribute set, so there is no interface between any two of them.
- **First seen**: 2026-08-31
- **Reason**: Blast radius 5. Also fails the deletion test as posed — collapsing the 19 mixins into one class makes nothing worse, because nothing is hidden today; the file split is cosmetic rather than a seam. The real fix is state ownership, which is a programme, not a PR. The sequence-iterator slice (3 files) is the automatable entry point if a future run wants it.
