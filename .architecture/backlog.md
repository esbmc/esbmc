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
- **Files**: ~40+ estimated
- **Modules**: `src/python-frontend/python_converter.h` and the thirteen `src/python-frontend/converter/*.cpp` files
- **Summary**: One class with 152 methods, implemented across thirteen translation units and included by 37 files; every converter file must know the whole surface.
- **First seen**: 2026-08-31
- **Reason**: Blast radius 5 — repo-wide for the Python frontend, not one unattended PR. Listed under *Too large to automate*; a human should schedule it.

## cpp-om-library-headers

- **Status**: dropped
- **Score**: 12/25 (leverage 1, locality 3, blast radius 3, heat 5)
- **Files**: ~12 estimated
- **Modules**: `src/cpp/library/{string,vector,type_traits,map,set,list}`
- **Summary**: The hottest directory in the repo by file touches, but these are operational models that deliberately mirror libc++'s shape.
- **First seen**: 2026-08-31
- **Reason**: Leverage 1 — fails the deletion test. These headers exist to reproduce a published interface (the C++ standard library) verbatim; "deepening" them would make ESBMC diverge from the thing it is modelling. Their shape is a requirement, not a defect.
