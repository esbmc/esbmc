# Scope: clang-c frontend to IREP2 (Phase 6)

Opened per `frontends-to-irep2.md` §6, which requires each of Phases 5-9 to
start with its own scope doc: census, phased decomposition, gates, risks.
Phase 5 (jimple) closed at `scope-jimple-irep2.md` §31; §39 of the parent
records what this phase inherits from it.

## 1. Census

### 1.1 Surface

| File | LOC | Legacy construction sites |
|---|---|---|
| `clang_c_convert.cpp` | 5 391 | 245 |
| `nested_func_transform.cpp` | 2 283 | — (a GOTO-level transform, not construction) |
| `clang_c_adjust_expr.cpp` | 1 828 | 209 |
| `clang_c_language.cpp` | 758 | 2 |
| `clang_c_adjust_polymorphic_functions.cpp` | 700 | 120 |
| `clang_c_main.cpp` | 454 | 77 |
| `padding.cpp` | 369 | 10 |
| `clang_c_convert_literals.cpp` | 273 | 10 |
| `clang_c_lexer.cpp` | 250 | **already IREP2** |
| **total** | **13 778** | ~673 |

"Construction sites" counts lines mentioning `exprt`, `typet`, a `code_*t`,
`symbol_expr`, `gen_zero` or `from_integer` — the same rough proxy the parent
used for its 971 figure, not a precise obligation count.

`clang_c_lexer.cpp` is already native: `parse_integer`, `do_parse_expr` and
friends return `expr2tc` directly. It is the frontend's `#pragma`/annotation
expression parser and needs no migration.

### 1.2 Corpus

| Suite | Tests |
|---|---|
| esbmc | 1 694 |
| esbmc-unix | 391 |
| esbmc-unix2 | 186 |
| cstd | 142 |
| k-induction | 122 |
| floats | 102 |
| floats-regression | 62 |
| llvm | 48 |
| clang_builtins | 17 |
| csmith | 7 |
| **sampled total** | **2 771** |

Not exhaustive -- cheri-c, cuda and others also parse through this frontend --
but enough to price the gate.

## 2. The gate scales; measured, not assumed

Phase 5's gate was A/B byte-identity of `--goto-functions-only` plus a mutant.
The obvious worry is that jimple's corpus was 26 tests and this one is 2 771.
Measured: **25 tests dump in 5 s**, so ~0.2 s/test single-threaded, ~9 minutes
for the sampled corpus, and well under two minutes at `-j10`. An A/B is two
passes plus one rebuild.

**The gate transfers.** That is worth stating plainly because it was the main
reason to suspect Phase 6 would need a different method, and it does not.

## 3. The seam does *not* transfer

This is the real structural difference, and it decides the decomposition.

jimple's seam returns by value:

```cpp
virtual exprt to_exprt(contextt &, const std::string &, const std::string &) const;
```

which is why the parallel-method technique worked: add `to_expr2t` with a
migrating default (`migrate_expr(to_exprt(...))`), override it one class at a
time, and every unmigrated class keeps working untouched. 27 overrides migrated
independently over 21 PRs.

clang-c's seams are out-parameter and in-place:

```cpp
virtual bool get_expr(const clang::Stmt &stmt, exprt &new_expr);   // convert
virtual bool get_type(const clang::QualType &type, typet &new_type);
void clang_c_adjust::adjust_expr(exprt &expr);                     // adjust
```

Three consequences:

1. **`get_expr` recurses through its own out-param.** A parallel
   `get_expr(stmt, expr2tc &)` cannot call the legacy one for the sub-statements
   it has not migrated without migrating each result individually -- which is
   the migrating default, but paid per node rather than per class.
2. **`adjust_expr` mutates in place.** There is no "return a new node" seam to
   parallel at all; the IREP2 counterpart is a different function shape, and
   Python already has one (`python_adjust`), which is the precedent to read
   rather than invent.
3. **The dispatch is an if-else chain on `expr.id()`**, not a virtual hierarchy.
   Migration is therefore per-*arm*, and the arms are not independently
   addressable the way 27 subclasses were.

## 4. Proposed decomposition (not yet executed)

- **C.1** Census the corpus by construct, before writing anything
  (parent §39.1). jimple had five expression kinds at zero occurrences and one
  was migrated blind; this frontend's arm list is longer and the same failure is
  available.
- **C.2** Pick the seam. Either a parallel `get_expr2t` carrying a per-node
  migrating default, or start at `adjust` where Python's shape already exists.
  This is the decision the phase turns on and it should be made against §3, not
  by analogy with Phase 5.
- **C.3** Migrate leaf arms first (literals, symbols), as jimple did, since they
  are the ones with no operand recursion.
- **C.4** `get_type` -> `type2tc` last, as in jimple (§7 there), because symbol
  construction still takes a `typet`.

## 5. Risks

- **R1 — the arm list is long and the corpus is wide.** A per-arm A/B over
  2 771 tests is cheap (§2) but a *mutant* per arm is a rebuild each. Phase 5
  spent most of its wall-clock on mutant rebuilds at 26 tests; here the rebuild
  dominates and the dump does not. Batch mutants.
- **R2 — this frontend feeds every other one.** `clang_c_adjust` output is
  consumed by clang-cpp, and CUDA and CHERI-C parse through the same converter.
  A divergence here is not contained to one suite.
- **R3 — §20.1 of `scope-coupled-arith-assign-conversion.md` is live here.**
  Four of the seven structural gaps between the two `implicit_typecast_followed`
  copies are C++-shaped, but `incomplete_array` sources and the const/volatile
  qualifier warnings are plain C and reachable from this frontend. They were
  dormant for jimple (§22.1 there) and are not dormant now.
- **R4 — no opt-out flag exists for this path.** jimple's A/B compared two
  binaries because it had no runtime switch; the same applies here, so every A/B
  costs a rebuild. `--no-irep2-native-body` governs `goto_convert`, not the
  frontend.

## 6. Status

Census complete; nothing migrated. Next action is C.2 -- the seam decision --
which needs `python_adjust`'s shape read properly first, since §3.2 says that is
the existing precedent for an in-place IREP2 adjuster.
