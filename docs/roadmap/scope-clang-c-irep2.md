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

## 6. C.2 decided: start at adjust, not convert

§3.2 said `python_adjust` is the precedent to read rather than invent. Read, it
decides the phase.

### 6.1 What Python actually built

`python_adjust::adjust_expr(expr2tc &)` (python_adjust.cpp:274) is the same
shape as `clang_c_adjust::adjust_expr(exprt &)` -- in-place, recursive,
dispatching on kind -- but over IREP2. It recurses with
`expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); })`, which is the
operand-surgery rule of parent §38.3 in its natural form rather than as advice.

It is wired into `python_language.cpp` behind **two** flags
(`src/esbmc/options.cpp:210,214`):

| Flag | Placement | Effect |
|---|---|---|
| `--python-irep2-adjust` | *after* `clang_cpp_adjust` | shadow; default off; currently behaviour-inert |
| `--python-irep2-adjust-only` | *instead of* `clang_cpp_adjust` | the real flip |

### 6.2 Why this settles the seam

Two reasons to take the adjuster rather than the converter:

1. **The shape exists and has been debugged.** `get_expr`'s out-param recursion
   (§3.1) has no precedent anywhere in the tree; a per-node migrating default
   would be invented here for the first time. The adjuster's does not need
   inventing.
2. **It restores the runtime A/B, which R4 said was lost.** A flag-gated pass
   means flag-off versus flag-on is *the same binary*. Phase 5 paid a rebuild
   per A/B and per mutant, and §5 R1 predicted the rebuild would dominate at
   this scale. A flag removes that cost entirely -- the same economics
   `--no-irep2-native-body` gave the `goto_convert` work.

So R4 is downgraded: it is true that no opt-out exists *today*, but the first
commit of C.3 can add one, and should.

### 6.3 The warning that comes with it

Python's own comment records a negative result worth inheriting verbatim. The
shadow placement is inert **because `clang_cpp_adjust` already did the work** --
the IREP2 pass "currently resolves nothing" and "only writes a symbol back when
it changes the value, so the flag is behaviour-inert."

That is parent §39.1's third failure mode -- *a caller downstream re-does the
work* -- sitting at the centre of the design. A mutant on a shadow-placed pass
will not move a single test, and it will not move it for a reason that has
nothing to do with whether the pass is correct.

Moving the pass earlier does not fix it either: Python prototyped that (B.3,
2026-06-25) and got *no verdict at all* across a 20-test fixture, because
running both adjusters over the same nodes double-resolves them -- the
"two-places-resolve hazard". Their conclusion, which applies unchanged here:
the before-placement is viable only once it **replaces** the legacy adjuster.

**Consequence for this phase.** The shadow flag is worth having as a
crash/regression net, but it cannot be the verification gate. The gate has to
be the `-only` placement over the §1.2 corpus, arm by arm, with the legacy
adjuster's arm disabled in the same run -- otherwise every mutant reports the
false zero §39.1 warns about.

## 7. Revised decomposition

- **C.1** ✅ census (§1).
- **C.2** ✅ seam decided: `clang_c_adjust` -> a parallel IREP2 adjuster, behind
  a flag pair mirroring Python's.
- **C.3** Add the flag pair and an empty IREP2 adjuster that recurses and does
  nothing. Gate: whole-corpus A/B, flag-on versus flag-off, same binary.
- **C.4** Migrate arms one at a time under the `-only` flag, leaf arms first.
  Each arm's mutant must be run with the legacy arm off (§6.3).
- **C.5** `get_type` -> `type2tc`, and only then consider `get_expr` (§3.1),
  which may not be worth migrating at all if the adjuster carries the semantics.

## 8. C.3 executed (#6894): the walk is inert, and two harness facts it exposed

Added `clang_c_adjust_irep2` behind `--clang-c-irep2-adjust`, wired after
`clang_c_adjust` in `clang_c_languaget::typecheck`. The walk is **read-only**:
it reads each code symbol's `get_value2()` and recurses via `Foreach_operand`,
never writing a symbol back. Read-only is not stylistic -- it keeps the pass
inert *by construction* rather than by argument, and side-steps the round-trip
losses `python_adjust` documents (a bitfield's `#bitfield` flag, an explicit
alignment attribute), which is exactly where C headers live.

### 8.1 Result

Flag-off versus flag-on, same binary, over the 1 672 runnable tests of
`regression/esbmc`:

| Measure | Result |
|---|---|
| new failures | **12** -- 73 non-zero exits flag-off, 85 flag-on, over all 1 686 |
| content divergence | 17 tests, **real** (see §8.3) |
| raw divergence before the §9 lookup fix | 1 550 / 1 672 |

*(Both of the first two rows previously read "none". §8.5 records how that
happened and why the errors were the ones this project's own harness rules
predict.)*

### 8.2 Every C symbol trips the migrate warning

```
WARNING: migrate_expr: symbol 'c:@F@printf' missing renaming delimiters,
treating as level0 with base name 'c:@F@printf'
```

`sym_name_to_symbol` looks the name up in `migrate_namespace_lookup` and, on
failure, falls through to the renaming parser, which finds no `?`/`!` and warns
(migrate.cpp:686). The lookup fails because `clang_c_languaget::typecheck`
builds into **`new_context`** and only `c_link`s into the global context
afterwards, so the namespace the migrator consults does not yet hold these
symbols.

The migration is not wrong -- the fallback returns the same level-0 symbol the
lookup would have. But it is a diagnostic-noise defect waiting for the flip:
the moment this pass stops being optional, every C compilation emits one warning
per symbol. C.4 owns fixing the lookup, not silencing the warning.

### 8.3 The 17 are real, and the walk aborts on unions

Stripping the warnings left 17 divergences, clustered on `to_union_*` (5),
`github_994-cast-to-bitfield-*` (3) and `github_709*` (4). Unions and bitfields
are precisely the round-trip losses `python_adjust` names.

The abort is direct:

```
Assertion failed: (is_union_type(type)), function constant_union2t,
irep2_expr.h:495
```

`migrate_expr`'s union arm builds
`constant_union2tc(migrate_type(expr.type()), ...)` (migrate.cpp:923). The
union's type at that point is still a **by-name tag**, so `migrate_type` yields
`symbol_type2t` and the constructor's assertion fires.

This is the transient-by-name-aggregate hazard `python_adjust` documents, and
running the walk *after* `clang_c_adjust` was meant to avoid it -- Python's
comment says a post-`clang_cpp_adjust` walk is safe because the types are
resolved by then. **That inheritance does not hold for C.** `clang_c_adjust`
does not leave every aggregate tag resolved, and 12 of 1 686 tests reach a union
constant that proves it.

**This is the substantive Phase 6 finding**, and C.4 cannot use the adjuster
seam until it is answered: either the walk runs somewhere the types are
resolved, or aggregate resolution has to precede it.

### 8.5 How this section came to say the opposite

The first version of §8.1/§8.3 reported no new failures and attributed the 17 to
baseline nondeterminism. Both were measurement errors, and both are the errors
parent §7 warns about by name:

| Claim | Method | Defect |
|---|---|---|
| "no new failures, 20 = 20" | exit codes over the **first 300 tests alphabetically** | rule 6 -- *sample dense and unbiased*. A prefix contains no `to_union_*`, no `github_9*`; the entire defect class sorts after it. |
| "nine of ten differ against themselves" | flag-off twice, stripping **only** the migrate warnings | the real A/B normalises temp paths and timing too. Re-run with the same normaliser: **1 of 13**. The control was measuring noise the comparison already removed. |

The second is the more instructive. Rule 7 says run the baseline against itself
-- I did, and still got a wrong answer, because the control was not normalised
the same way as the thing it was controlling for. **A control has to be
identical to the measurement in every respect but the variable under test**;
mine differed in two.

Both errors pointed the same way -- towards "the pass is fine" -- which is the
direction that requires no further work.

### 8.4 What C.3 does and does not establish

Establishes: the IREP2 walk reaches every code symbol in the C corpus, migrates
each value, and neither crashes nor changes behaviour. `migrate_expr` handles
every construct 1 672 C tests contain.

Establishes nothing about a pass that *writes*. Read-only is the point of C.3
and the limit of it; the first write-back is C.4's risk, and §6.3's warning
stands.

## 9. §8.2 fixed (#6897): the gate is now readable without filters

The lookup defect §8.2 deferred is fixed. `migrate_namespace_lookup` now points
at the context being built for the duration of the walk and is restored after,
the same save/restore `dereferencet` performs for the same reason
(`dereference.cpp:637`):

```cpp
namespacet ns(context);
const namespacet *old_ns = std::exchange(migrate_namespace_lookup, &ns);
...
migrate_namespace_lookup = old_ns;
```

### 9.1 The measurement that matters

| | raw A/B divergence over 1 672 tests |
|---|---|
| C.3, as merged | 1 550 |
| with §8.2 fixed | **17** |

Seventeen is exactly the set §8.3 showed the baseline cannot reproduce against
itself. So the divergence attributable to the pass is zero, and -- the point of
fixing it now rather than later -- **the comparison no longer needs a filter to
be readable**.

That is worth more than the warning suppression. C.4 runs this A/B once per
migrated arm; a gate whose raw output is 1 550 lines of noise is a gate people
stop reading, and §8.3 already showed how convincing a false positive looks when
it lands on unions and bitfields. A gate whose raw output is 17 known names is
one where a real eighteenth stands out.

### 9.2 The remaining 17 are still owed an explanation

Not this phase's defect, but it is now the only thing between the corpus and a
clean A/B, and it should be characterised before C.4 leans on the gate: are they
one cause or several, and is the nondeterminism in `goto_convert`'s ordering, in
a hash-table iteration order, or in the frontend? A single reproducer would
settle it.

## 10. Status

C.1-C.3 done (#6894), §8.2 fixed (#6897). Next: characterise the 17 (§9.2), then
C.4 -- the `-only` placement, one arm at a time, legacy arm disabled in the same
run.

## 11. The union assert (#6899), and why "read-only" was never true

### 11.1 The abort was an asymmetry, not a rule

`constant_struct2t` permits a `symbol_id` type as *"a transient pre-resolution
state"* -- added deliberately so migrating a constant aggregate before its tag
resolves does not abort. `constant_union2t`, whose comment calls it "almost the
same as constant_struct2t", never got the disjunct. #6899 adds it.

Result: non-zero exits with and without the flag go from 73 / 85 to **68 / 68**,
and content divergence from 17 to 5.

### 11.2 The remaining five, separated properly

Running each candidate three ways -- flag-off twice, then flag-on -- with the
*same* normaliser the A/B uses:

| test | off vs off | off vs on |
|---|---|---|
| `github_746` | 12 | 12 |
| `github_1200` | **0** | **48** |
| `github_1377` | **0** | **66** |
| `github_2618` | **0** | **94** |

Only `github_746` is unstable. The other three are stable against themselves and
change under the flag: **real divergences caused by the pass.**

### 11.3 A read-only walk that is not read-only

The divergence is a GOTO instruction reordering -- a `NONDET` initialisation
swapping position with the next instruction's location comment. A frontend walk
that writes nothing should not be able to do that.

Two hypotheses, both tested:

1. *The `get_value()` guard back-migrates.* `get_value()` on a symbol whose
   IREP2 side is authoritative materialises `migrate_expr_back(value_)` into the
   legacy cache. Removing the call: divergence unchanged at 48 / 66 / 94.
   **Refuted.**
2. *It is the iteration or the namespace swap.* Emptying the walk body entirely,
   leaving only the symbol snapshot and the `migrate_namespace_lookup`
   save/restore: divergence **0**. **Refuted.**

What remains is the reading itself. `symbolt::get_value2()` is a *materialising*
accessor: it populates `value_`, sets `value2_valid_`, and the pipeline is
sensitive to the cache state it leaves behind.

So the C.3 premise -- "read-only keeps the pass inert by construction" -- is
wrong twice over. There is no read-only way to inspect a symbol's IREP2 value:
the accessor is the mutation. §8.4 claimed read-only was "the point of C.3 and
the limit of it"; the limit is tighter than that.

### 11.4 What C.4 has to carry

- The seam is still the right one (§6.2 stands: the shape exists, and the flag
  gives a same-binary A/B).
- But **any** placement that reads `get_value2()` perturbs three tests, before a
  single arm is migrated. C.4's first job is to find out what downstream reads
  the cache state and why it changes instruction order -- not to migrate an arm
  on top of an unexplained perturbation.
- `github_1200` is a 2-second reproducer with a 48-line diff.

## 12. Status

C.1-C.3 done (#6894), §8.2 lookup fixed (#6897), union assert fixed (#6899).
C.4 is blocked on §11.4 -- three tests whose GOTO changes when a symbol's IREP2
value is merely read.

## 13. §11.4 answered: the gate itself is the casualty

Three tests (`github_1200`, `github_1377`, `github_2618`) are stable against
themselves and change under the flag. §11.3 narrowed the cause to *reading*
`get_value2()`. Two further bisects finish the job.

### 13.1 It is global state, not the symbol

Reading the value on a **copy** of the symbol -- so the real symbol's caches are
never touched, but `migrate_expr` still runs -- leaves the divergence at exactly
48 / 66 / 94 lines. So it is not the symbol's lazy cache. Emptying the walk
entirely gives 0 (§11.3), so it is the migration.

### 13.2 What actually changes

The diff is a permutation of nondet initialisations:

```
<  ASSIGN r=NONDET(unsigned long int);        >  ASSIGN s=NONDET(signed char *);
<  ASSIGN ATOI_MAP=NONDET(unsigned char [256]); >  ASSIGN r=NONDET(unsigned long int);
<  ASSIGN s=NONDET(signed char *);           >  ASSIGN ATOI_MAP=NONDET(unsigned char [256]);
```

Same instructions, different order -- the signature of iterating a container
whose order the migration perturbs.

### 13.3 The mechanism: `irep_idt` orders by interning sequence

```cpp
inline bool operator<(const irep_idt &b) const { return no < b.no; }
inline size_t hash() const { return no; }
```

`no` is the index the string received when it was interned in the append-only
pool (`src/util/base/string_pool.h`). So **every `std::map`/`unordered_map`
keyed by `irep_idt` iterates in interning order, not lexicographic order.**

`migrate_expr` interns strings -- type tags, `nondet$`-prefixed names, component
names. Running it before `goto_convert` shifts the `no` of every string interned
afterwards, permuting iteration of whichever `irep_idt`-keyed container decides
nondet-initialisation order.

### 13.4 Why this matters far beyond clang-c

This is a **pre-existing latent order-dependence**, not a defect the walk
introduces; the walk only exposes it. But the consequence lands squarely on the
migration programme:

**Any pass that interns a string before `goto_convert` can permute the GOTO
dump.** No migration pass can avoid interning strings -- that is what building
IREP2 nodes does. So G3, `--goto-functions-only` byte-identity, is not a sound
gate for a frontend-resident pass, and that applies to **Phases 6-9 alike**, not
just this one.

Phase 5 never hit it because jimple's corpus is 26 tests and its overrides run
*inside* the conversion that was already interning those strings, in the same
order.

### 13.5 Options for C.4

1. **Normalise instruction order within a block before diffing.** Cheapest;
   weakens the gate exactly where §13.2 shows real bugs could hide.
2. **Fix the container.** Order the offending map by string rather than by `no`.
   A real fix, wide blast radius, and it would make ESBMC's GOTO output
   deterministic under interning -- valuable independently of this programme.
3. **Change gate.** Use G1 (verdict parity) and G2 (counterexample text) instead
   of G3 for frontend-resident passes. Weaker per-test, but unaffected.

Option 2 is the one worth costing first: it is the only one that leaves the gate
intact, and the resulting determinism is worth having on its own.

## 14. Status

C.1-C.3 done (#6894), lookup fixed (#6897), union assert fixed (#6899). C.4 is
blocked on §13.5 -- a gate decision, not a code task. The three divergent tests
are explained and are not migration defects.

## 15. §13.5 option 2 taken (#6901): the container was one line

§13.5 offered three responses to the interning-order problem and judged option 2
-- fix the container -- worth costing first, on the grounds that it is the only
one that leaves the gate intact. Costed: it is a single container with a single
consumer.

### 15.1 The container

```cpp
typedef std::unordered_set<expr2tc, irep2_hash> loop_varst;   // loopst.h:14
```

consumed by `make_nondet_assign` (goto_k_induction.cpp:159), which emits one
havoc assignment per element in iteration order. `irep2_hash` folds in
`irep_idt::hash()`, which returns `no` -- the interning sequence number -- so the
bucket layout, and with it the iteration order, moves whenever anything earlier
in the run interns a string.

That is the whole mechanism. Not systemic: one set, one loop.

### 15.2 Sorting by `operator<` would not have worked

The obvious fix -- copy to a vector and `std::sort` with the default comparator
-- is wrong here. `expr2tc::operator<` delegates to `expr2t::lt`, which recurses
into fields; a `symbol2t`'s field is its `irep_idt`, whose `operator<` is
`no < b.no`. **The default ordering is interning-ordered too.** #6901 sorts by
`pretty()` instead, which is text.

Worth recording because the same trap waits for anyone who tries to make another
IREP2-keyed container deterministic: in this codebase, sorting `irep_idt` is not
a text sort.

### 15.3 Result

| test | off vs off | off vs on, before | off vs on, after |
|---|---|---|---|
| `github_1200` | 0 | 48 | **0** |
| `github_1377` | 0 | 66 | **0** |
| `github_2618` | 0 | 94 | **0** |
| `github_746` | 12 | 12 | 12 |

`github_746` is unaffected -- it differs against itself, a separate
nondeterminism this fix does not touch and does not claim to.

202 k-induction tests, 668 unit tests and 1 037 C regression tests pass.

### 15.4 What this restores

G3 -- `--goto-functions-only` byte-identity -- is a sound gate again for a
frontend-resident pass, which §13.4 had just declared it was not for Phases 6-9
alike. The gate survived because the order-dependence turned out to be one
container rather than a property of the IR.

That is a better outcome than §13.5 expected from option 2, and it is worth
noting *why* the estimate was pessimistic: §13.4 reasoned from the generality of
the cause (every `irep_idt`-keyed container is interning-ordered) to the
generality of the fix. Only one such container was actually on the path from
frontend to GOTO dump.

## 16. Status

C.1-C.3 done (#6894), lookup fixed (#6897), union assert fixed (#6899), havoc
order fixed (#6901). The flag's divergence over `regression/esbmc` is now one
test, and that test is unstable without the flag.

C.4 is unblocked: migrate arms under the `-only` placement, one at a time, with
the legacy arm disabled in the same run (§6.3).

