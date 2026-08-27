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

## 17. C.4's first arm: `adjust_symbol` is not migratable

With the gate restored (§15.4), C.4 begins. The natural first arm is
`clang_c_adjust::adjust_symbol(exprt &)` -- a true leaf, 20 lines, no operand
recursion. It is not takeable, and the reason generalises.

### 17.1 Two carriers IREP2 cannot hold

```cpp
locationt location = expr.location();          // saved
...
expr = symbol_expr(symbol);
expr.location() = location;                    // restored
if (expr.type().is_code())
{
  address_of_exprt tmp(expr);
  tmp.implicit(true);                          // <-- flag
  ...
}
```

1. **Location.** `symbol2t` has no location slot -- `if2t` is the only
   value-level kind that carries one (parent §38.4). The save/restore is a no-op
   in IREP2, so the migrated arm silently drops it.
2. **`#implicit`.** IREP2 has no representation for it: grepping `irep2_expr.h`
   for `implicit` returns nothing.

The second is the blocking one, because the flag is **read**:

```cpp
// clang_c_adjust_expr.cpp:840
op.is_address_of() && op.implicit() && op.operands().size() == 1 && ...
```

A sibling arm branches on it. Migrating `adjust_symbol` alone would set the flag
nowhere and that check would stop firing -- a behaviour change, not a
representation detail.

### 17.2 The census this prompted

Counting legacy-only flag uses (`.set("#...")`, `implicit()`, `cmt_*`) per arm:

| arm | flag uses |
|---|---|
| `adjust_side_effect_function_call` (142 lines) | 4 |
| `adjust_address_of` (66) | 2 |
| `adjust_base_to_derived`, `adjust_dereference`, `adjust_expr`, `adjust_symbol(exprt&)` | 1 each |
| **all 28 others** | **0** |

So the flag problem is concentrated: six arms carry it, twenty-eight do not.
`adjust_expr_binary_arithmetic` (114 lines), `adjust_index` (59),
`adjust_expr_shifts` (56), `adjust_ptr_mem` (60) and `adjust_side_effect_*` are
all clean, and several are far more substantial than the leaf that blocked.

### 17.3 Revised plan for C.4

- **Do not** start with `adjust_symbol`. Start with a zero-flag arm; the
  candidates are large enough to be worth the harness cost and clean enough to
  be faithful.
- The six flag-carrying arms need a decision that is **not** per-arm: either
  `address_of2t` and friends gain the flags (an IR change with wide blast
  radius), or those arms stay legacy permanently and the `-only` placement is
  never fully reachable.
- That decision should be taken once, with the six named, rather than
  rediscovered per arm.

**Generalisation for Phases 7-9.** Jimple had no analogue of this because it sets
no `#`-flags at all (§22.1 there: the frontend emits no qualifier, no
`#reference`). C, C++ and Solidity all do. The question "which legacy-only irep
flags does this frontend's adjuster read?" belongs in each scope doc's census,
next to the type-constructor census §22.1 established.

## 18. Status

C.1-C.3 done (#6894), lookup (#6897), union assert (#6899), havoc order
(#6901). C.4 re-planned per §17.3; next action is the first zero-flag arm, with
`adjust_expr_binary_arithmetic` and `adjust_index` the leading candidates.

## 19. `adjust_index` is implementable, and per-arm gating has an ordering hazard

`adjust_index` (59 lines, zero flags) is the leading candidate from §17.3.
Checked before executing, per the pattern the parent doc uses.

### 19.1 The kit covers it, and #6873 is load-bearing

Every primitive the arm needs has an IREP2 form:

| legacy | IREP2 |
|---|---|
| `gen_typecast(ns, e, t)` | `c_implicit_typecast(expr2tc &, const type2tc &, ns)` |
| `ns.follow(typet)` | `ns.follow(const type2tc &)` -- namespace.h:21 |
| `index_exprt` -> `dereference` rewrite | build `dereference2tc(elem, add2tc(...))` directly |

`gen_typecast` is a three-line wrapper over `c_typecastt::implicit_typecast`
(typecast.cpp:9), so the IREP2 counterpart is the overload **#6873** fixed. That
matters: before that fix the two copies disagreed on folding a cast of a
constant, and `adjust_index`'s `gen_typecast(ns, index_expr, index_type())` is
exactly a cast of a frequently-constant index. Migrating this arm on top of the
unfixed overload would have diverged on every constant subscript -- the same
failure jimple's assignment hit (`scope-jimple-irep2.md` §22.2).

So Phase 4's claim that the kit already exists (parent §38) holds here, provided
#6873 is in.

### 19.2 The hazard: gating an arm changes traversal, not just the arm

§6.3 requires the legacy arm disabled in the same run, or the mutant is
shadowed. But `clang_c_adjust::adjust_expr` dispatches *and* recurses through
the same arm:

```cpp
void clang_c_adjust::adjust_index(index_exprt &index)
{
  adjust_operands(index);      // <-- the recursion lives here
  ...
}
```

Skipping the arm therefore skips the recursion into the index's operands, not
just the index rewrite. A naive `if (!flag) adjust_index(...)` would leave the
subtree unadjusted and the comparison would measure that, not the migration.

Two workable shapes:

1. **Split the arm**: legacy keeps `adjust_operands`, the flag skips only the
   rewrite below it. Smallest change, but it edits the legacy pass for every arm
   migrated.
2. **Move recursion up**: have the IREP2 pass own the traversal for the whole
   expression once it owns any arm -- which is the `-only` placement in
   miniature, and closer to where the phase must end up anyway.

Shape 2 is the better target and the larger step. Shape 1 is what makes the
*first* arm measurable without restructuring.

### 19.3 Plan

- C.4a: implement `adjust_index` in the IREP2 pass, gate the legacy rewrite
  (shape 1), write back, A/B over `regression/esbmc`, mutant-check with the
  legacy rewrite off.
- C.4b: once two or three arms are in, switch to shape 2 rather than accumulating
  per-arm edits in the legacy pass.

## 20. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901).
C.4a specified above; the primitives are confirmed present and the traversal
hazard is identified before rather than after the first attempt.

## 21. C.4a done (#6907): the first arm, and three ways to get it wrong

`adjust_index` is migrated. A/B over `regression/esbmc`: **2 divergences, both
`github_746`**, which differs against itself without the flag. The dereference
rewrite mutant moves 23 tests, so the arm is live.

Getting there took three corrections, and all three are transferable.

### 21.1 #6873 was a hard prerequisite, exactly as predicted

§19.1 said migrating this arm on top of the unfixed `c_typecast` overload would
diverge on every constant subscript. First A/B: **296** divergences, all of the
form

```
< p[0]                        > p[(signed long int)0]
```

The branch was stacked off master, which does not contain #6873. Merging it:
296 -> 11.

A prediction recorded before the attempt, confirmed by the attempt, at the exact
magnitude implied -- which is worth more than the fix, because it means the
reasoning in §19.1 can be trusted for the next arm.

### 21.2 `index_type2()` is not the IREP2 form of `index_type()`

```cpp
typet   index_type()  { return signed_size_type(); }
type2tc index_type2() { return get_int_type(config.ansi_c.address_width); }
```

The `2` suffix in `c_types.h` marks *an IREP2 type constructor*, not *the IREP2
counterpart of the same-named legacy one*. The arm uses
`migrate_type(index_type())`.

This one did not change the count, so it was not the cause of the residue -- but
it is a live trap for every later arm, and the naming makes it invisible.

### 21.3 The gate reached C++ and the replacement did not

The residue after #6873 was 11, of which 9 were `.cpp` and 2 were the known
`github_746` pair. Cause: `clang_cpp_adjust` **derives from** `clang_c_adjust`,
so gating the rewrite on `config.options.get_bool_option(...)` inside the shared
arm disabled it for C++ as well -- where the IREP2 pass, wired only into
`clang_c_languaget::typecheck`, never runs. The adjustment was simply lost, and
it presented as a missing `int`->`long` widening cast, which looks exactly like a
migration defect.

The hand-over is now `clang_c_adjust::set_irep2_owns_index()`, called by the C
driver alone.

**The general rule for C.4:** the arms live in a base class three frontends
inherit. A per-arm hand-over must be per-*instance*, never a global option read
inside the arm. Any future arm that gets this wrong will present as a C++
regression with a plausible-looking C-shaped explanation.

### 21.4 Method note

Each of the three was found by the same loop -- run the A/B, look at *one* diff,
form a hypothesis, test it -- and two of the three hypotheses were wrong first
(a vacuous-cast theory refuted by `int it_pos;` in the source, and an
index-type theory that changed nothing). The cost of a wrong hypothesis here is
one A/B, about ten minutes. That is the argument for keeping the corpus A/B
cheap, which §9.1 made for a different reason.

## 22. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
first arm (#6907). C.4b next: migrate a second zero-flag arm --
`adjust_expr_binary_arithmetic` (114 lines) or `adjust_ptr_mem` (60) -- then
switch to shape 2 (§19.2) rather than accumulating per-arm edits in the legacy
pass.

## 23. C.4b: `adjust_expr_shifts` is not hand-over-able, and the criterion that says so

The obvious second arm was `adjust_expr_shifts` -- 56 lines, zero flags,
self-contained. Checked before implementing, and it fails, for a reason §17's
flag census does not cover.

### 23.1 The blocker

The arm's own assertion states its input:

```cpp
assert(expr.id() == "shr" || expr.id() == "shl");
```

and its job is to resolve `"shr"` into `"lshr"` or `"ashr"` by the left
operand's signedness. IREP2 has `lshr2t`, `ashr2t` and `shl2t` -- and **no plain
`shr2t`**. `migrate_expr` has forward arms for all three resolved kinds and none
for the unresolved one; `i_shr` does not exist.

Under shape 1 the legacy arm returns early and the node reaches the IREP2 pass
unresolved -- as a `"shr"` that `migrate_expr` cannot consume. The hand-over
does not merely move a transformation; it strands a node kind IREP2 has no
representation for.

### 23.2 The criterion is not "rewrites a kind"

Four arms rewrite `expr.id()`:

| arm | rewrites to | pre-adjust kind | hand-over-able |
|---|---|---|---|
| `adjust_index` | `dereference` | `index` -- `index2t` exists | **yes** (#6907) |
| `adjust_dereference` | `index` | `dereference` -- exists | yes (but 1 flag, §17.2) |
| `adjust_float_arith` | `ieee_add`/`sub`/`mul`/`div` | `+`/`-`/`*`/`/` -- exist | yes |
| `adjust_expr_shifts` | `lshr`/`ashr` | **`shr` -- does not exist** | **no** |

So kind-rewriting is not the problem -- #6907 rewrites a kind and is faithful.
The criterion is:

> **Shape 1 is safe for an arm iff the node *as it arrives* survives
> `migrate_expr`.** An arm that resolves a placeholder kind IREP2 does not model
> cannot be handed over post hoc; it has to migrate where the node is built
> (shape 2, or the converter).

That is a third category alongside §17's two, and it is cheap to test for: run
the C.3 walk with the arm's legacy rewrite disabled and see whether migration
aborts.

### 23.3 Next arm

`adjust_float_arith` (45 lines, zero flags, pre-adjust kinds all representable).
It is also the arm with a working precedent: `python_adjust::promote_to_ieee`
(esbmc/esbmc#6839) performs the same `+`->`ieee_add` promotion over IREP2
already, so the second arm has an implementation to mirror rather than derive.

## 24. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907). C.4b re-aimed at `adjust_float_arith` per §23.3;
`adjust_expr_shifts` is deferred to shape 2 with the reason recorded.

## 25. C.4b attempted and withdrawn: applicability can be path-dependent

`adjust_float_arith` cleared every check §23 knew to make -- 45 lines, zero
flags, pre-adjust kinds (`+`/`-`/`*`/`/` on floats) all representable, and a
working IREP2 precedent in `python_adjust::promote_to_ieee` (#6839). It was
implemented, measured, and **withdrawn**: A/B over `regression/esbmc` gave
**141 divergences**, clustered on `complex_*`.

### 25.1 Why

The legacy arm has exactly one call site:

```cpp
// clang_c_adjust::adjust_expr_binary_arithmetic, end of function
if (expr.id() == "+" || expr.id() == "-" || expr.id() == "*" || expr.id() == "/")
  adjust_float_arith(expr);
```

So it applies only to nodes that reached *that dispatch path*. The IREP2 pass
walks every node and has no notion of how a node was reached, so it promoted
float arithmetic the legacy pass never routes there -- most visibly the adds and
multiplies the complex lowering builds.

Tightening the node test does not fix it. Two nodes can be identical -- same
kind, same operand types -- and differ only in whether the legacy dispatch would
have handed them to this arm. That difference is not recoverable from the node.

### 25.2 The third clause

§17 gave one blocker (the arm sets a flag IREP2 cannot hold), §23 a second (the
node as it arrives is not representable). This is a third, and the most
restrictive:

> **Shape 1 is safe only for an arm whose applicability is a property of the
> node itself, not of the path that reached it.**

`adjust_index` satisfies it -- "is this an `index2t`?" is intrinsic, which is
why #6907 is faithful at 2 divergences. `adjust_float_arith` does not: "did this
node come through binary-arithmetic adjustment?" is extrinsic.

### 25.3 What this costs the plan

The three clauses together are demanding, and they are not independently rare.
The §17.2 census counted 28 zero-flag arms and implied a long runway; §23 and
§25 both fired on the first two candidates drawn from it. The honest reading is
that shape 1 suits a minority of arms, and the count is unknown until each is
checked against all three clauses.

That strengthens the §19.2 argument for **shape 2** -- give the IREP2 pass the
traversal, so it reconstructs the dispatch context rather than inheriting nodes
stripped of it -- and it moves that from "do it after two or three arms" to the
next substantive step. #6907 stands on its own; it does not need more per-arm
siblings to be worth having.

### 25.4 Method

The arm was written, measured, and reverted inside one iteration, because the
corpus A/B costs about ten minutes (§9.1, §21.4). Withdrawal on measurement is
the cheap outcome the harness was built to make possible; the expensive outcome
would have been shipping 141 divergences behind a default-off flag where nothing
would have looked at them again.

## 26. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907). C.4b withdrawn (§25). Next substantive step is **shape 2**:
move traversal ownership into the IREP2 pass, per §19.2 and §25.3.

## 27. Sizing shape 2: the per-arm blocker census

Shape 2 gives the IREP2 pass the traversal, so it reconstructs the dispatch
context §25 showed shape 1 cannot inherit. It is close to all-or-nothing: an
unmigrated arm cannot be called on an `expr2tc` without the
`migrate_expr_back` -> `migrate_expr` round trip parent §38.3 bans. So the
question is not "which arm next" but "how many arms, and which are hard".

Censusing all 31 arms against the three clauses (§17 flags, §23 input-kind
representability, §25 intrinsic applicability -- the last read off the call
graph: an arm invoked from `adjust_expr`'s dispatch is intrinsic, one invoked
from another arm is not):

| | count |
|---|---|
| zero-flag **and** reached from the top dispatch | **18 / 31** |
| carries legacy-only flags | 4 (`adjust_address_of` 2, `adjust_side_effect_function_call` 4, `adjust_dereference` 1, `adjust_symbol` 1) |
| reached only from another arm (path-dependent) | 9 |

The 18:

`adjust_base_to_derived`, `adjust_builtin_va_arg`, `adjust_comma`,
`adjust_expr_binary_arithmetic`, `adjust_expr_binary_boolean`,
`adjust_expr_rel`, `adjust_expr_shifts`, `adjust_expr_unary_boolean`,
`adjust_expr_unary_complex`, `adjust_if`, `adjust_index`, `adjust_member`,
`adjust_ptr_mem`, `adjust_reference`, `adjust_side_effect`, `adjust_sizeof`,
`adjust_struct`, `adjust_type`.

### 27.1 What the census changes

§25.3 read the two failures pessimistically -- "shape 1 suits a minority, the
count is unknown". The count is now known and it is better than that reading:
**18 of 31 arms are clean on both the flag and the applicability clause.**
`adjust_expr_shifts` is in the 18 and still needs §23's treatment (its input
kind is unrepresentable), but that is one arm with a known, local remedy --
resolve `shr` at construction rather than in the adjuster.

Under shape 2 the path-dependence clause largely dissolves: the nine
called-from-another-arm cases become reachable once the IREP2 pass owns the
dispatch, because the caller is then also IREP2 and supplies the context.
`adjust_float_arith` fails under shape 1 for exactly the reason shape 2 fixes.

So the real shape-2 obligation is: **18 arms to write, 9 that follow once their
callers are IREP2, and 4 blocked on flags IREP2 cannot represent.**

### 27.2 The four are the decision

`#implicit` (`adjust_address_of`, `adjust_symbol`), the `cmt_*` uses in
`adjust_side_effect_function_call`, and `adjust_dereference`'s single flag are
not a scheduling problem -- they are the one design question the phase has to
answer, and §17.3 already flagged that it should be answered once rather than
per arm. Either those flags gain IREP2 representation, or the `-only` placement
is permanently partial and the legacy adjuster survives for them.

That decision gates shape 2 and should be taken before the 18 are written, not
after.

## 28. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), C.4b withdrawn (§25). Shape 2 sized (§27): 18 arms clean, 9
unblocked by shape 2 itself, 4 gated on the flag-representation decision, which
is the next thing to settle.

## 29. The flag decision, part 1: `#implicit`'s reader does not need it

§27.2 named the four flag-carrying arms as the one design question gating shape
2. Two of them (`adjust_symbol`, `adjust_address_of`) carry only `#implicit`,
and the flag has exactly one reader:

```cpp
// clang_c_adjust::adjust_address_of -- "address of function designator",
// ANSI-C 99 6.3.2.1 p4
if (op.is_address_of() && op.implicit() && op.operands().size() == 1 &&
    op.op0().id() == "symbol" && op.op0().type().is_code())
```

It collapses `&(&f)` for a function designator, where the inner `&f` is the
sugar `adjust_symbol` inserts. The flag distinguishes that synthesised inner
`address_of` from a user-written one -- but a user cannot write the shape it is
distinguishing from: `&f` is an rvalue in both C and C++, so `&(&f)` is a
constraint violation, not an alternative parse. **The shape alone determines the
case.**

### 29.1 Measured

Dropping `op.implicit()` from the condition, leaving the shape test:

| check | result |
|---|---|
| GOTO A/B over `regression/esbmc` (1 672) | **2 divergences**, both `github_746`, which differs against itself |
| `esbmc-cpp/cpp` suite | **575 passed, none failed** |

The C++ check matters because `adjust_address_of` is inherited by
`clang_cpp_adjust`, and function designators are commoner there.

### 29.2 What it unblocks

If the read does not need the flag, the writes exist only to feed it, and
`#implicit` need not be represented in IREP2 at all. That removes the blocker
from two of §27.2's four arms and reduces the gating decision to
`adjust_side_effect_function_call`'s four `cmt_*` uses and
`adjust_dereference`'s single flag.

**Corrected count for shape 2:** 18 arms clean, 9 unblocked by shape 2 itself,
**2** gated on flags -- not 4.

### 29.3 Scope of the claim

Measured, not proven. The corpus contains whatever function-designator shapes
these tests contain, and the standard argument covers why the distinguished case
is unwritable, but neither rules out a frontend constructing an explicit
`address_of(address_of(code symbol))` some other way -- the CBMC adapter and the
Solidity frontend both build ireps directly. Removing the flag outright is a
separate PR with its own sweep; this section establishes only that it is not a
shape-2 blocker. The probe was reverted rather than shipped.

## 30. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), C.4b withdrawn (§25), shape 2 sized (§27). Flag decision half
settled (§29): `#implicit` is not a blocker. Next is the other half --
`adjust_side_effect_function_call`'s `cmt_*` uses and `adjust_dereference`.

## 31. Correction to §29: `#implicit` has two readers, and the second is needed

§29 stated that `#implicit` "has exactly one reader" and concluded it is not a
shape-2 blocker. The census behind that was incomplete. There are **two**:

| site | what it does |
|---|---|
| `adjust_address_of:840` | collapses `&(&f)` for a function designator |
| `adjust_side_effect_function_call:1159` | "do implicit dereference" -- strips the sugar `address_of` off a call's function operand |

§29's probe dropped the first and measured no divergence. That result stands
for what it tested: reader 1 is redundant. It says nothing about reader 2, which
the probe left in place.

### 31.1 Why the second reader is not redundant

```cpp
if (f_op.is_address_of() && f_op.implicit() && (f_op.operands().size() == 1))
{ /* strip the address_of: call f directly */ }
else if (f_op.type().is_pointer())
{ /* wrap in an implicit dereference */ }
```

Here the flag separates two cases that are **both reachable from valid C**:

- `f(x)` -- `adjust_symbol` rewrote `f` to an implicit `&f`, which this strips;
- `(&f)(x)` -- the user wrote the address-of explicitly, which is legal, and
  which must take the second branch.

Unlike §29's case, the distinguished alternative is writable. The shape is
identical; only the provenance differs, and provenance is exactly what the flag
records.

### 31.2 Corrected conclusion

`#implicit` **is** load-bearing, and shape 2 needs it represented -- or
`adjust_side_effect_function_call` and `adjust_symbol` stay legacy. §29.2's
"corrected count" of 2 gated arms is withdrawn; it returns to **4**, as §27.2
had it.

The `cmt_*` uses at `:1089-1090` are a separate blocker of the same kind:
`cmt_identifier`/`cmt_base_name` attach parameter identity to an *argument
expression*, and IREP2 has no per-expression comment slot at all.

### 31.3 The recurring error

This is the second overclaim in this phase with the same cause -- §11's
"read-only" and now §29's "one reader" -- and both times the fix was a census I
had not run. The pattern is: I checked the site I was looking at and inferred a
property of the whole. For a flag, the property that matters is over **all**
readers, and `grep` gives it in seconds.

Rule for the remaining flag work: **enumerate every reader before reasoning
about a flag, and quote the list.** A claim of redundancy is a claim about a
set, not about a site.

## 32. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), C.4b withdrawn (§25), shape 2 sized (§27), §29 corrected
here. Four arms remain gated on flag representation; the decision §27.2 asked
for is still open, and is now known to be a real IR question rather than a
census artefact.

## 33. The `#implicit` census, run properly

§31.3 set the rule: enumerate every reader before reasoning about a flag, and
quote the list. Done, over all of `src/`.

### 33.1 Reads -- two, both on `address_of`

| site | condition |
|---|---|
| `clang_c_adjust_expr.cpp:840` | `op.is_address_of() && op.implicit() && ...` |
| `clang_c_adjust_expr.cpp:1159` | `f_op.is_address_of() && f_op.implicit() && ...` |

### 33.2 Writes -- nine, by node kind

| node kind written | sites |
|---|---|
| **`address_of`** | `clang_c_adjust_expr.cpp:360` (the function-designator sugar), `:846` (clears it), `:894` |
| `dereference` | `c_typecast.cpp:670`, `clang_c_adjust_expr.cpp:1168`, `clang_cpp_adjust_expr.cpp:420`, `:439`, `clang_cpp_convert.cpp:2907`, `:2946` |

### 33.3 What the census yields

Both readers test `is_address_of()` first. **Every `#implicit` written on a
`dereference` node is therefore never read** -- six of the nine writes are dead
metadata, spanning `util`, the C adjuster, the C++ adjuster and the C++
converter.

So the requirement on IREP2 is far narrower than §31.2 implied. Shape 2 needs
`#implicit` carried **only on `address_of`**, produced by essentially one writer
(`:360`, the `&f` sugar; `:894` builds the same sugar for a code-typed
dereference, `:846` clears it). One bit on `address_of2t` discharges both
readers.

That is a materially cheaper answer than "give the flags IREP2 representation"
sounded in §27.2, and it is only visible from the full list -- from either
reader alone the flag looks like a general property of expressions.

### 33.4 Two follow-ups, both out of scope here

- **The six dead writes** are a simplification candidate in their own right, and
  the `dereference` ones in `clang_cpp_convert` sit on the C++ reference model,
  so removing them needs the C++ suite rather than this phase's corpus.
- **Whether `address_of2t` should gain a bit, or the provenance should be
  recovered another way**, is still the open decision. The census bounds its
  cost; it does not take it.

## 34. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), C.4b withdrawn (§25), shape 2 sized (§27), §29 corrected in
§31, `#implicit` censused here. Remaining for the flag decision:
`adjust_side_effect_function_call`'s `cmt_identifier`/`cmt_base_name`, which
attach parameter identity to argument expressions and have no IREP2 slot at all.

## 35. The flag decision, settled: one bit on `address_of2t`

§27.2 named flag representation as the single design question gating shape 2,
and §31 restored the count to four arms. Censusing both flag families over all
of `src/` -- the §31.3 rule applied properly -- collapses it to one bit.

### 35.1 `cmt_identifier` / `cmt_base_name` on argument expressions are dead

`adjust_side_effect_function_call:1089-1090` writes them onto `arguments[i]`,
an *expression*. Every reader found is on `code_typet::argumentt`, a **type**
component:

| reader | subject |
|---|---|
| `std_types.h:334,339` (`get_identifier`, `get_base_name`) | `code_typet::argumentt` |
| `clang_c_adjust_polymorphic_functions.cpp` (12 sites) | `code_type.arguments()[i]` |
| `assign_params_as_non_det.cpp:77`, `clang_cpp_adjust_code_gen.cpp:82,172`, `clang_cpp_convert.cpp:2247,2256,2266,2291` (raw `get("#identifier")`) | `...arguments()` |

Nothing reads either field off an argument expression. The type-level carrier is
separately represented in IREP2 already (`code_type2t::argument_names`), so it
is not at issue.

### 35.2 What remains live

| use | verdict |
|---|---|
| `#implicit` on `dereference` (6 writes) | dead -- both readers test `is_address_of()` first (§33) |
| `#implicit` on `address_of` (`:360` sugar, `:894`, cleared at `:846`) | **live** -- feeds the reader at `:1159` |
| `#implicit` read at `:840` | redundant, measured (§29) |
| `#implicit` read at `:1159` | **live** -- distinguishes `f(x)` from `(&f)(x)` (§31.1) |
| `cmt_*` on argument expressions | dead (§35.1) |

**One live carrier: `#implicit` on `address_of`.** One bit on `address_of2t`
discharges it, and with it all four arms §27.2 listed.

### 35.3 The decision

Add an `implicit` bit to `address_of2t`. Cost: one field, its `fields` tuple
entry for hashing and comparison, and the two migrate arms. Everything else the
flag families touch is dead metadata that need not migrate at all.

The alternative -- leaving four arms permanently legacy -- was priced against a
requirement that the census has now shown to be six times smaller than it
looked. §27.2 asked for this decision to be taken once rather than rediscovered
per arm; it is taken.

### 35.4 Method note

Three sections of this phase (§29, §31, §33, §35) are one question asked four
times, each time with a wider search. The first answer was wrong, the second
over-corrected, and only the exhaustive census produced a number worth
building on. The rule §31.3 states is cheap -- `grep` over `src/` costs seconds
-- and each time I skipped it the error pointed the same way: toward the
conclusion that required less work.

## 36. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), C.4b withdrawn (§25), shape 2 sized (§27), flag decision
settled (§35). Shape 2 is now unblocked in principle: **18 arms clean, 9
unblocked by shape 2 itself, 4 discharged by one bit on `address_of2t`.**

Next: add the bit, with the two migrate arms and a corpus A/B, then begin the 18.

## 37. The bit is in (#6912)

`address_of2t` now carries `implicit`, defaulted false, in its `fields` tuple
and in both migrate directions. Nothing sets it from the IREP2 side yet -- this
is the representation §35.3 decided on, not a user of it.

### 37.1 The check that mattered

Adding a field to the `fields` tuple changes every `address_of` node's hash and
its comparison. Given §13 -- where a hash-ordered container permuted the GOTO
dump -- that is the risk worth measuring, not the field itself.

A/B over `regression/esbmc`: **2 divergences, both `github_746`**, which differs
against itself. 668/668 unit tests.

### 37.2 A constructor assertion worth noting

```cpp
assert(ptrobj->expr_id != expr2t::address_of_id);
```

IREP2 cannot represent a nested `address_of` at all. That independently
corroborates §29: the reader at `:840`, which exists to collapse `&(&f)`, is
looking for a shape that cannot survive into IREP2 -- so its redundancy is not
merely a corpus observation. The live reader is `:1159`, whose shape
(`address_of(symbol)`) is perfectly representable, which is why the bit is
needed at all.

### 37.3 Three pre-existing failures found on the way

Running the C++ suite further than earlier runs had reached surfaced
`github_2242_1`, `github_2242_2` and `github_3897_collision`. All three fail on
**master**, so they are unrelated to this stack. Also `regression/esbmc/github_2572_2`
fails on master (`--ir-ieee`, `assertion 0*f==0`). Recorded, not fixed: none is
this phase's work, and quietly fixing them would mix concerns.

## 38. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912). Shape 2's blockers are discharged:
**18 arms clean, 9 unblocked by shape 2 itself, 4 unblocked by the bit.**

Next is the first shape-2 step: give `clang_c_adjust_irep2` the dispatch, so the
9 path-dependent arms become addressable, starting with the arms already proven
under shape 1.

## 39. `adjust_member` withdrawn: a fourth clause, about invariants not kinds

`adjust_member` passed every check the phase had: 21 lines, zero flags,
intrinsic applicability (reached from `adjust_expr`'s `is_member()` arm), and a
representable input kind (`member2t` exists). It was implemented and
**withdrawn** at 191 divergences.

### 39.1 The failure is in migration, not in the arm

```
Assertion failed: (source->type->type_id == struct_id || union_id ||
                   complex_id || symbol_id), function member2t,
                   irep2_expr.h:1587
```

The arm's whole job is to make the base of a member access reachable -- wrapping
a pointer base in a dereference, an array base in a zero index. Handing it over
means the legacy pass leaves `member(pointer_base, ...)` in place, and
`member2t`'s constructor **forbids a pointer source**. The abort happens inside
`get_value2()`, before the IREP2 arm is ever called.

### 39.2 The clause

§23 said shape 1 is safe iff the node *as it arrives* survives `migrate_expr`,
and read that as a statement about node **kinds** -- `shr` has no IREP2 kind.
This is the same sentence with a different subject:

> The node as it arrives must satisfy IREP2's **construction invariants**, not
> merely have a representable kind. Legacy IR does not maintain those invariants
> mid-adjustment; that is what the adjuster is for.

`member2t` requires a struct-like source. `constant_union2t` required a union
type (§11, fixed by relaxing it). `address_of2t` forbids a nested operand
(§37.2). Each is an invariant IREP2 enforces at construction and legacy reaches
only after the relevant arm has run.

**So an arm that establishes an IREP2 construction invariant can never be handed
over post hoc** -- by definition, the node before it runs is one IREP2 refuses to
build. That is a sharper and more useful rule than the kind-based reading, and
it predicts §23's `shr` case as a special instance.

### 39.3 Consequence

Three arms attempted under shape 1, one shipped:

| arm | outcome |
|---|---|
| `adjust_index` | shipped (#6907) |
| `adjust_float_arith` | withdrawn -- path-dependent applicability (§25) |
| `adjust_member` | withdrawn -- establishes a construction invariant (§39) |

`adjust_index` succeeded because the invariant it establishes -- turning `p[i]`
into `*(p+i)` -- is not one `index2t` enforces: `index2t` accepts a pointer
source, so the un-adjusted node migrates fine.

The §27 census counted 18 "clean" arms against three clauses. This fourth clause
is not visible in that census, and checking it needs the §23 test run per arm --
disable the legacy rewrite, walk the corpus, see whether migration aborts. That
test is cheap and should now precede any implementation.

## 40. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912). Two arms withdrawn on measurement
(§25, §39). Next action is the §39.3 pre-check across the 18, which will say how
many survive all four clauses -- and if the answer is "few", that is the
argument for going straight to shape 2.

## 41. Correction to §39: the invariant is relaxable, and the pattern is established

§39 concluded that "an arm that establishes an IREP2 construction invariant can
never be handed over post hoc". Too strong. The static census of construction
assertions shows why.

### 41.1 `index2t` was already relaxed for exactly this

```cpp
/* A `symbol_id` or `pointer_id` source is permitted only as a transient
   pre-resolution state (V.1k two-phase source invariant, see member2t
   above); the IREP2-native adjuster resolves a symbol source to an
   array/vector and rewrites a pointer source `p[i]` to `*(p+i)` before
   symex. */
assert(is_array_type(source) || is_vector_type(source) ||
       source->type->type_id == type2t::symbol_id ||
       source->type->type_id == type2t::pointer_id);
```

`adjust_index` did not succeed because its invariant happened not to bite. It
succeeded because someone had **already** relaxed `index2t` to admit the
transient pointer source, and documented the rewrite -- `p[i]` to `*(p+i)` --
that #6907 went on to implement.

`member2t` carries the sibling comment ("see member2t above") and permits
`symbol_id` but **not** `pointer_id`. The relaxation was applied to one of the
pair and not the other.

### 41.2 So §39's wall is a checklist

The remedy is the one #6899 already used for `constant_union2t`: add the
transient disjunct, with the same justification -- the adjuster re-establishes
the strong form before symex. `member2t` needs `pointer_id`, one line, and the
V.1k two-phase invariant is the standing rationale for it.

Restated:

> An arm that establishes an IREP2 construction invariant needs that invariant
> relaxed to admit its *input* state before it can be handed over. The pattern
> is established (`index2t`, `constant_struct2t`, `constant_union2t` after
> #6899); what it costs per arm is one disjunct plus the A/B.

### 41.3 The invariant census, as a work list

Node kinds whose constructor asserts a shape an adjuster arm establishes:

| kind | invariant | status |
|---|---|---|
| `index2t` | source array/vector/symbol/**pointer** | already relaxed |
| `constant_struct2t` | type struct/complex/**symbol** | already relaxed |
| `constant_union2t` | type union/**symbol** | relaxed in #6899 |
| `member2t` | source struct/union/complex/symbol | **needs `pointer_id`** |
| `if2t` | type matches both arms | check `adjust_if` |
| `constant_array2t` | type is array | check `adjust_struct` |

The first three are the precedent; the fourth is `adjust_member`'s one-line
unblock; the last two are unchecked and should be, before those arms are
attempted.

### 41.4 What this changes

§40 offered "if few arms survive all four clauses, go straight to shape 2".
The fourth clause is now a per-arm one-liner with an established pattern, not a
wall, so it does not by itself argue for abandoning shape 1. The argument for
shape 2 rests where §25 put it -- path-dependent applicability -- which no
relaxation fixes.

## 42. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912). §39 corrected here. Next:
relax `member2t` with the `pointer_id` disjunct and re-run #6907's withdrawn
sibling, which is now a one-line change plus the arm already written.

## 43. `adjust_member` shipped (#6921): §41's remedy, applied

§41 predicted the withdrawal in §39 was a one-line relaxation rather than a
wall. It was two lines, and the measurement shows why both were needed:

| | A/B divergences |
|---|---|
| arm handed over, invariant untouched (§39) | 191 |
| + `pointer_id` disjunct | 6 |
| + `array_id` disjunct | **2** (`github_746`, unstable against itself) |

`adjust_member` has two branches -- a pointer base becomes
`member(dereference(base))`, an array base becomes `member(index(base, 0))` --
so both are transient states its own input can be in. §41's work list named the
pointer case from `index2t`'s precedent; the array case only surfaced when the
second A/B still aborted, on the same assertion with a different type id.

**Lesson for the remaining relaxations:** the disjuncts an invariant needs are
one per *branch* of the arm that establishes it, not one per arm. Reading the
arm tells you the list without measuring; §41.3's work list should be re-derived
that way for `if2t` and `constant_array2t` before those arms are attempted.

Dropping the dereference changes 187 tests, so the arm is live -- and 187 is
also the size of the class §39 mistook for a wall.

### 43.1 Where shape 1 now stands

| arm | outcome |
|---|---|
| `adjust_index` | shipped (#6907) |
| `adjust_member` | shipped (#6921) |
| `adjust_float_arith` | withdrawn -- path-dependent (§25), only shape 2 fixes it |
| `adjust_expr_shifts` | blocked -- `shr` has no IREP2 kind (§23) |

Two of four attempted arms now ship, and the two failures have distinct,
understood causes. That is a better rate than §25.3 feared and does not need
shape 2 to continue.

## 44. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). Next: derive the
disjunct list for `if2t` and `constant_array2t` from their arms (§43), then take
`adjust_if` or `adjust_struct`.

## 45. The disjunct lists for `if2t` and `constant_array2t`, derived not measured

§43 said the disjuncts an invariant needs are one per *branch* of the arm that
establishes it, and that reading the arm gives the list. Applied to the two
unchecked entries in §41.3.

### 45.1 `adjust_if` needs a different kind of relaxation

```cpp
gen_typecast(ns, expr.op0(), bool_type());
if (expr.type() != expr.op1().type() || expr.type() != expr.op2().type())
{
  gen_typecast(ns, expr.op1(), expr.type());
  gen_typecast(ns, expr.op2(), expr.type());
}
```

and `if2t`:

```cpp
assert(type->type_id == trueval->type->type_id);
assert(type->type_id == falseval->type->type_id);
```

The arm establishes exactly that invariant, so handing it over hands `if2t` a
node it refuses. But this invariant is **relational** -- an equality between two
fields -- not a whitelist of type ids. There is no disjunct to add. Admitting
the transient state means dropping the equality for everyone, which is a
materially weaker change than `index2t`'s and `member2t`'s: those still
constrain the source to a listed set, whereas this would constrain nothing.

So `adjust_if` is not the next arm. If it is taken later it needs either a
marker distinguishing pre-adjust nodes, or the arm migrating at construction
rather than post hoc.

**The §43 rule needs the qualifier:** it holds for invariants that whitelist a
field's shape. A relational invariant has no transient form that is weaker but
still meaningful, so relaxing it is a different decision.

### 45.2 `adjust_struct` looks clear

```cpp
const typet &t = ns.follow(expr.type());
... insert gen_zero padding operands where components are padding ...
adjust_expr(ops[i]);
```

The arm changes the **operands**, never the type. `constant_struct2t`'s only
assertion is on the type (`struct_id || complex_id || symbol_id`), already
relaxed for the transient symbol case, and it asserts **nothing** about operand
count -- the operand/component agreement `python_adjust` documents is enforced
by that pass, not by the constructor.

So an un-padded literal constructs fine, and the arm should be hand-over-able
with no relaxation at all. It is the next candidate.

Two cautions, both from this phase's own history: the recursion at `ops[i]`
lives inside the arm, so the split must keep it (§19.2); and the arm is reached
from `adjust_expr`'s dispatch, so applicability is intrinsic (§25) -- both
already checked against the §27 census.

## 46. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). §41.3's work list
is now fully derived: `adjust_struct` next, `adjust_if` deferred with its reason
recorded.

## 47. `adjust_struct`: §45.2's prediction was wrong, for a reason worth having

§45.2 predicted `adjust_struct` would need no relaxation, because it changes
operands rather than the type and `constant_struct2t` asserts nothing about
operand count. That reasoning was sound and the conclusion is still wrong: the
arm cannot be written at all in IREP2 as things stand.

### 47.1 The blocker is type metadata, not an invariant

```cpp
if (c.get_is_padding() && !already_padded)
  ops.insert(ops.begin() + i, gen_zero(c.type()));
```

The arm inserts a zero operand for each component that **is padding**.
`struct_type2t` carries `members`, `member_names`, `member_pretty_names`, `name`
and `packed` -- and **no per-member padding flag**. `migrate.cpp` mentions
`is_padding` nowhere, so the legacy `componentt`'s flag is dropped on the way in.

This is a fifth blocker category, and it is not §17's: that was a flag on an
*expression*, this is metadata on a *type component*. §17's census looked for
`.set("#...")` in the arms and would never have found it, because the write
happens in `padding.cpp`, not in the adjuster.

### 47.2 Reconstructible, but on a naming convention

Padding components are named from two prefixes
(`src/util/irep/pad_names.h`):

```cpp
inline constexpr std::string_view pad_prefix           = "anon_pad#";
inline constexpr std::string_view pad_bit_field_prefix = "anon_bit_field_pad#";
```

`member_names` survives migration, so an IREP2 arm could test the prefix --
and there is precedent: `goto_check.cpp:973` already identifies padding that way
(`has_prefix(*it, pad_prefix)`), on the IREP2 side of the pipeline.

So the arm is writable, but on a convention rather than a flag, and it would
need both prefixes. Whether that is acceptable is a judgement about how load-
bearing the convention should become, not a measurement -- and `goto_check`
having already made that call is the strongest argument that it is.

### 47.3 What this says about the census

The §27 census screened arms on flags, input-kind representability and
applicability. §39 added construction invariants; this adds **type metadata the
arm reads**. Four of the five categories were found by attempting an arm and
failing, not by the census -- which is the honest summary of how much a static
screen can tell you here.

A better pre-check, for the arms remaining: *list everything the arm reads that
is not an operand or a type id, and confirm each survives `migrate_type`.*
`is_padding` would have been caught by that; so would `#implicit`.

## 48. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). `adjust_struct`
is writable via the §47.2 convention; `adjust_if` is deferred (§45.1). Next
action is to apply §47.3's pre-check to the remaining 15 clean arms before
attempting any of them.

## 49. §47.3's pre-check, run across the clean arms

*List everything the arm reads that is not an operand or a type id, and confirm
each survives migration.* Applied to the 18:

| arm | non-migrating reads |
|---|---|
| `adjust_comma`, `adjust_expr_binary_boolean`, `adjust_expr_rel`, `adjust_expr_unary_boolean`, `adjust_reference`, `adjust_sizeof` | **none** |
| `adjust_member` | none -- shipped (#6921) |
| `adjust_if` | none on this axis; blocked by §45.1's relational invariant |
| `adjust_base_to_derived`, `adjust_builtin_va_arg`, `adjust_side_effect` | `location()` |
| `adjust_struct` | `is_padding`, `incomplete()` |
| `adjust_type` | `incomplete()` |
| `adjust_index`, `adjust_expr_binary_arithmetic`, `adjust_expr_shifts`, `adjust_expr_unary_complex`, `adjust_ptr_mem` | (`id()` string compare -- see below) |

### 49.1 One category the check over-flagged

The first run flagged `expr.id() == "..."` as a non-migrating read. It is not:
an `id()` comparison is a kind test, and becomes `is_<kind>2t(expr)`.
`adjust_index` does it and shipped fine (#6907). Reclassified as benign, which
moves five arms out of the suspect list.

Worth recording because it is the same failure mode as §29 in miniature -- a
pattern match standing in for the property actually being asked about.

### 49.2 The real finding: `location()` is a third blocker family

Three arms read `expr.location()`. Only `if2t` carries a location among
value-level IREP2 kinds (parent §38.4), so an arm that reads or propagates a
location cannot do so natively. That is neither a flag (§17), an invariant
(§39), nor type metadata (§47) -- it is the location model, and it is the same
gap `scope-jimple-irep2.md` §17.1 hit on `adjust_symbol`.

### 49.3 The list this leaves

Six arms are clear on every known axis: `adjust_comma`,
`adjust_expr_binary_boolean`, `adjust_expr_rel`, `adjust_expr_unary_boolean`,
`adjust_reference`, `adjust_sizeof`.

That is a real list, produced statically, and it is the first time this phase has
had one -- §47.3 was written precisely because four of five blocker categories
had been found by attempting arms rather than screening them. Whether the screen
is now complete will be told by whether the next arm ships without a surprise.

## 50. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). Six screened
candidates (§49.3); next is `adjust_expr_rel`, the smallest of them at 11 lines.

## 51. `adjust_expr_rel` withdrawn: legacy type identity is finer than IREP2's

The first arm drawn from §49.3's screened list. Implemented, measured at **4
divergences** -- the `github_746` pair plus `aligned_attr` and
`aligned_attr_fail` -- and withdrawn.

### 51.1 What differs

```
< ASSERT ... (!((signed int)default_global_var == 42))
> ASSERT ... (!(default_global_var == 42))
```

The arm is `gen_typecast_arithmetic(ns, op0, op1)`, which inserts a cast when
the operand types differ. `default_global_var` carries an alignment attribute,
so its legacy type is not equal to plain `int` and the legacy pass casts.
`migrate_type` drops the attribute, both operands migrate to `signedbv 32`, the
IREP2 helper sees equal types and inserts nothing.

The cast is structurally vacuous -- same width, same signedness -- so the IREP2
output is arguably the better one. It is still a divergence, and byte-identity
is this phase's gate, so the arm does not ship on that argument.

### 51.2 Not a new category -- a second face of §47

§47 found `adjust_struct` blocked because `is_padding` does not survive
`migrate_type`. This is the same gap seen from the other side: the attribute
survives nowhere, so **legacy type equality is strictly finer than IREP2's**,
and any arm whose behaviour is conditioned on type *inequality* can diverge
wherever a dropped attribute was the only difference.

That is a much wider blast radius than §47's, because it does not require the
arm to read the metadata. `adjust_expr_rel` reads nothing unusual -- it passes
the screen in §49 -- and still diverges, because the helper it calls compares
types.

### 51.3 What this does to the screen

§49.3 produced six candidates and asked whether the screen was complete. The
answer is no, and the missing test is not about what an arm *reads* but about
what it *compares*:

> An arm that calls a helper which branches on type equality can diverge
> wherever `migrate_type` drops an attribute, regardless of what the arm itself
> reads.

Of the six, `adjust_expr_rel` and `adjust_expr_binary_boolean` call
`gen_typecast*`; `adjust_comma`, `adjust_expr_unary_boolean`,
`adjust_reference` and `adjust_sizeof` need re-reading against this test before
being attempted.

The alternative to screening each is to decide the question once: either
`migrate_type` carries the attributes, or the phase accepts that vacuous casts
may be dropped and moves the gate from byte-identity to verdict parity for arms
of this shape. That is the same choice §13.5 posed for interning order, and it
was worth taking once there.

## 52. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). Three arms
withdrawn on measurement (§25, §39 -- later shipped, §51). Next action is the
§51.3 decision, not another arm.

## 53. §51.3 decided: do not carry the attributes

The choice was "either `migrate_type` carries the dropped type attributes, or
the gate moves for arms of this shape". Priced:

### 53.1 Nothing on the IREP2 side consumes them

`alignment` is a named sub-irep, and its only consumer is

```cpp
BigInt alignment(const typet &type, const namespacet &ns)   // type_byte_size.cpp:472
{
  const exprt &given_alignment = static_cast<const exprt &>(type.find("alignment"));
```

-- a **legacy** `typet`. `type_byte_size`'s IREP2 overloads take `type2tc` and
never look for it. (`object_descriptor2t::alignment` is a computed field on a
descriptor, not a type attribute.) The wider attribute census across the
frontends -- `#reference` 5, `#bitfield` 5, `#extint` 3, `#rvalue_reference` 2,
and four singletons -- is likewise all legacy-side.

So carrying them into `type2t` would add representation, hashing cost and
comparison semantics for information **no IREP2 consumer reads**. That is
disproportionate, and it is the wrong direction for a migration whose point is
that IREP2 is the destination.

### 53.2 The divergence class is provably vacuous

The `aligned_attr` difference is a cast between two types of identical width and
signedness. It changes no verdict; §51.1 already noted the IREP2 output is
arguably the better one.

### 53.3 Decision

**Do not carry the attributes.** For an arm whose behaviour is conditioned on
type *equality*, normalise structurally-vacuous casts out of both sides before
diffing, and keep byte-identity everywhere else.

One condition, or this becomes a way to hide real divergences: *vacuous* must be
**verified per case** -- same width, same signedness, same kind -- not assumed
because the test name mentions an attribute. A cast that changes width is never
in this class.

### 53.4 The cheaper next step

Four of §49.3's six candidates call no type-comparing helper at all:
`adjust_comma`, `adjust_expr_unary_boolean`, `adjust_reference`,
`adjust_sizeof`. They need neither the relaxation of §43 nor the gate change of
§53.3, and taking one of them makes progress without weakening anything --
which is the better move before spending effort on a normaliser whose first user
would be a single arm.

## 54. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). §51.3 decided
(§53). Next: `adjust_comma` or `adjust_sizeof`, both untouched by the type-
identity gap.

## 55. `adjust_comma` withdrawn: a fourth clause, about consumers not producers

§53.4 picked `adjust_comma` as the cheaper next step -- two statements, no
type-comparing helper, clean on every axis §49 screens. It does not ship, and
the reason is a category none of the four recorded blocker families covers.

### 55.1 The census, run first

Per `frontends-to-irep2.md` §39.1, an `fprintf` in the legacy arm, over the
§1.2 corpus (2 809 tests that parse a `.c`/`.i` through this frontend):

| | |
|---|---|
| calls to `adjust_comma` | 3 924 |
| tests containing at least one | 1 102 |
| calls where `expr.type() != expr.op1().type()` | 86, in 8 tests |

So the arm is live, and the 86 are the only calls that do anything: the
converter already gives a comma its right operand's type (`BO_Comma` at
`clang_c_convert.cpp:4397`), and the assignment only bites once adjusting that
operand has retyped it.

### 55.2 The gate, made sensitive before it was trusted

The A/B is `--goto-functions-only` under `--clang-c-irep2-adjust` against the
same binary without it. Two harness facts cost more than the arm did:

- **The dump goes to stderr.** A sweep that hashes stdout compares empty
  strings and reports byte-identity for anything at all. §18.6 does not say
  this, and neither did this document.
- **§21.3's three normalisations are not enough.** A fourth is needed: clang
  prints AST node addresses in its diagnostics, so `regression/esbmc/github_746`
  and its `_nocolor` twin differ run-to-run under ASLR. Normalise `0x[0-9a-f]{6,}`.

With both applied the self-control (same binary, twice) is clean, and a
reachability mutant -- rebuild the node as `code_comma2tc(t, side_1, side_1)`,
which survives `remove_sideeffects` because that pass keeps the *last* operand
(`goto_sideeffects.cpp:1370`) -- moves **1 092** tests, against 1 102 that
contain a comma. The gate sees this arm.

### 55.3 The measurement

| | divergences |
|---|---|
| master, flag-on vs flag-off | **1** |
| arm handed over, flag-on vs flag-off | **8** |

The pre-existing one is `esbmc-unix/github_2220`, unrelated: a `member` whose
base is still a pointer reaches `c_expr2string`, which cannot print it and dumps
the raw irep instead. It is a member-arm residual, not a comma one, and it is
recorded here only because any future sweep will see it.

The other **7** are new, and they are the census's 86 retypes: `csmith01`-`04`,
`csmith06`, `esbmc-unix/github_2513_6`, `esbmc-unix/github_4435`. (The eighth
retyping test, `esbmc/00_aiob_4_true-unreach-call`, does not diverge -- a retype
no consumer reads.)

### 55.4 The clause

Every divergence has the same shape:

```
off:  IF !(cur == c)          THEN GOTO 2
on:   IF !((_Bool)(cur == c)) THEN GOTO 2
```

The condition is `(..., cur == c)`. `adjust_expr_rel` types the right operand
`bool`; `adjust_comma` copies that onto the comma node; the condition's
`gen_typecast_bool` then finds a `bool` and inserts nothing. Hand the arm over
and step two moves to a pass that runs after the whole legacy walk -- so the
consumer sees the converter's `int` and inserts a cast. The IREP2 arm then
corrects a type nobody will read again.

So the clause is not about what the arm reads (§49's screen) nor about what
migration can represent (§39, §47, §51). It is:

> **§19.2's hand-over shape is sound only for an arm whose rewrite no other arm
> in the same pass consumes.** `adjust_index` and `adjust_member` rewrite
> expression *shape*, which no sibling re-reads. An arm that writes a node's
> *type* is a producer for every sibling that types its operands.

### 55.5 This exhausts §49.3's list

Applied to the remaining five, three fall from source alone:

| arm | verdict |
|---|---|
| `adjust_expr_unary_boolean`, `adjust_expr_binary_boolean` | both write `expr.type() = bool_type()`, and `clang_c_adjust_expr.cpp:148` reads exactly that (`op0().type().id() == "bool"`) from the `unary-`/`bitnot` arm. Same clause, with the consumer named in the tree |
| `adjust_sizeof` | fills the VLA byte-size operand, and `migrate.cpp:783` **aborts** on a `sizeof` carrying anything but two operands. Handing it over aborts the pass before the arm runs |
| `adjust_reference` | `clang_c_adjust::adjust_reference` is empty for C; only `clang_cpp_adjust`'s override does work, and the IREP2 pass is wired into the C driver alone. Vacuous |
| `adjust_expr_rel` | already withdrawn, §51 |

§49.3 asked whether the screen was complete, and said it would be told by
whether the next arm shipped without a surprise. It did not, and the screen was
incomplete in a way that retires the whole list.

## 56. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921). §51.3 decided
(§53). `adjust_comma` withdrawn (§55), and with it the last of §49.3's
candidates.

Next is **not another arm**: the hand-over shape has no candidates left. §55.4
forces what §19.2 and §25.3 already preferred -- **shape 2**, moving traversal
ownership into the IREP2 pass so arm order is preserved and a migrated arm's
output reaches its consumers in the same walk. `esbmc-unix/github_2220`'s
pre-existing divergence is the first thing that shape should be measured
against.

## 57. Shape 2, tested on the arm §55 withdrew

§55.4 blamed the comma arm's divergences on *when* the rewrite ran, not on the
rewrite. That is a falsifiable claim, and it is cheaper to test on one arm than
to discover after restructuring the pass. The experiment: run the **identical**
IREP2 rewrite at the point `clang_c_adjust::adjust_expr` dispatches the arm,
via a per-node round trip -- `migrate_expr`, rewrite, `migrate_expr_back` --
instead of in the trailing whole-program pass.

### 57.1 Result

| configuration | divergences vs flag-off |
|---|---|
| master | 1 |
| arm in the trailing IREP2 pass (§55) | 8 |
| **same rewrite at the dispatch point** | **1** |

The one is `github_2220` in every row: pre-existing, member-arm, unrelated.
So the seven are fully explained by ordering. §55.4's clause holds, and shape 2
-- which fixes ordering by construction, because a migrated arm runs where the
legacy arm ran -- is validated on a real arm before anyone restructures
anything.

A second fact falls out, and shape 2 depends on it: the per-node round trip is
**lossless** for this construct. §3.1 predicted the migrating default would have
to be paid per node rather than per class; this is the first measurement that it
can be paid at all.

### 57.2 What the mutants say, and which one counts

Two were run, and only the second is evidence:

- **`side_2->type` -> `side_1->type`** moves 8 tests -- but at least one
  (`csmith01`) moves by **aborting**. That is `frontends-to-irep2.md` §39.1's
  fifth row: the mutation makes the operation invalid and the crash, not the
  value, moves the output. It proves the arm is reached and nothing more.
- **the valid alternative** -- round trip, no retype -- moves **8** tests and
  aborts on none. They are exactly the 86 retypes' 8 tests from §55.1,
  `esbmc/00_aiob_4_true-unreach-call` included: it did not diverge under §55's
  trailing-pass shape because that pass corrected it after the fact, and here
  nothing does. This is the mutant that isolates the value.

Recorded because the first mutant looked conclusive and was not. An abort is a
*louder* signal than a divergence and a weaker one.

### 57.3 Cost

`csmith01`, the heaviest comma test in the corpus: 2.79 s -> 3.18 s, +14 %.
That is the whole opt-in path, dominated by the trailing pass migrating every
symbol value, not by the round trip. The per-node trip migrates a node's whole
subtree, so nested commas cost O(depth x size); C comma chains are shallow and
the corpus does not exercise a deep one. Flag is default-off.

### 57.4 What this unblocks

§55.5 retired §49.3's list because every remaining arm writes a type a sibling
consumes. Shape 2 removes that clause entirely -- a migrated arm runs in
sequence, so its consumers see its output. `adjust_expr_unary_boolean` and
`adjust_expr_binary_boolean` come back on the list on exactly the evidence that
took them off it. `adjust_sizeof` does not: `migrate.cpp:783`'s two-operand
requirement is independent of ordering.

## 58. The boolean arms, and the bound on §57's round trip

§57.4 put `adjust_expr_unary_boolean` and `adjust_expr_binary_boolean` back on
the list: shape 2 removes the §55.4 clause that took them off. They still do not
ship, and the reason bounds the dispatch-point shape itself.

### 58.1 Pre-check, from source

Cheaper than §57's census, and it answers two things at once:

- **No operand invariant to relax.** `not2t`, `and2t` and `or2t` come from
  `ESBMC_DEFINE_LOGIC_2OP`, which fixes the *node's* type to bool and asserts
  nothing about operands. An `and` whose operands are still `int` migrates
  cleanly -- unlike `member2t`/`index2t`, which needed #6921's relaxation.
- **Half of each arm is dead for C.** `expr.type() = bool_type()` never changes
  anything: the converter already emits bool for `UO_LNot`, `BO_LAnd` and
  `BO_LOr` (`clang_c_convert.cpp:4227`, `:4380`, `:4384`), and `migrate_expr`
  *asserts* on `and`/`or` that it is bool already (`migrate.cpp:1118`). The
  operand conversion is the whole of the live work.

### 58.2 The measurement

| | divergences vs flag-off |
|---|---|
| master | 1 |
| both arms at the dispatch point | **5** |

Four are new, in two unrelated classes:

- `esbmc/deep_binary_chain_{pass,fail}` **time out** -- see §58.3.
- `csmith01`, `csmith02` differ by one cast: legacy emits
  `(_Bool)(*l_60 == (unsigned int *)0)`, the IREP2 path emits the comparison
  bare. Migration normalises a comparison to `equality2t`, which is bool by
  construction, so `c_implicit_typecast` correctly declines a bool-to-bool cast
  the legacy copy inserts. This is §53.3's vacuous class and is verifiable as
  such per case -- same kind, same width -- but it is not why the arms are
  withdrawn.

### 58.3 The round trip is quadratic, and the corpus already proves it

§57.3 noted that a per-node round trip migrates the node's whole subtree, so
nested operators cost O(depth x size), and said the corpus did not exercise a
deep one. It does -- for boolean operators, not commas:

```c
#define A6 A5 && A5 && A5 && A5
#define DEEP A6 && A6 && A6
int deep(int x) { return DEEP; }
```

| `deep_binary_chain_pass`, `--goto-functions-only` | |
|---|---|
| flag-off | 34.5 s |
| both arms at the dispatch point | **> 200 s (killed)** |

Every `&&` node re-migrates the whole chain beneath it. Comma survived §57
because C comma chains are shallow; `&&` chains are not, and this one is a
macro expansion of a kind real code produces.

### 58.4 What this bounds

The dispatch-point round trip is a *probe*, and §57 was right to call it one; it
is viable only where nesting is shallow, which is a property of the operator,
not of the technique. It does not generalise, and #6992 should be read as the
comma arm plus a validated diagnosis -- not as a shape to repeat per arm.

Shape 2 proper is unaffected, and this sharpens what it has to be: **migrate a
symbol's value once, walk it natively, dispatch in sequence**. §3.1's "migrating
default paid per node rather than per class" is affordable only if "per node"
means *dispatch* per node, not *migration* per node. That distinction was
implicit before this measurement and is the design constraint now.

## 59. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921), comma arm at its
dispatch point (#6992). `adjust_comma` in the trailing pass withdrawn (§55);
boolean arms in the dispatch-point shape withdrawn (§58).

Next: shape 2 proper, per §58.4 -- one migration per symbol value, native walk,
in-sequence dispatch. The two arms withdrawn here are its first customers, and
`deep_binary_chain_pass` is its performance gate: it must stay at ~34 s.

## 60. Shape 2 sized: the increment is the coupled component, not the arm

§58.4 left shape 2 as "one migration per symbol value, native walk, in-sequence
dispatch". That says what it must do. This says how large the smallest sound
step is, and the answer is why the phase has stalled twice.

### 60.1 The coupling census

Classify each arm of `clang_c_adjust`'s dispatcher two ways. A **producer**
writes a node's type or inserts a typecast. A **consumer** reads an operand's
type -- directly, or by handing the operand to a typecast helper that reads it,
which is the case the first pass of this census missed.

| | count |
|---|---|
| arms examined | 30 |
| producers | 20 |
| consumers | 19 |
| **both** | **19** |
| neither | 10 |

Nineteen arms both produce and consume. In an expression dispatcher any
expression can be any other's operand, so those nineteen are mutually reachable:
they form one strongly-coupled component.

### 60.2 Why that kills arm-at-a-time migration

§55.4 said an arm cannot move to the trailing pass alone if a sibling consumes
its output. The natural repair is to move a *set* closed under "consumes" --
inside the trailing pass, a native walk recurses children-first, so a
producer and its consumer run in the right relative order if both have moved.

The census prices that closure: for any arm in the component it is the whole
component. There is no small closed set. Nineteen arms, ~690 lines, move
together or not at all (`adjust_index`, `adjust_member` and `adjust_comma`, 99
lines, are already native).

### 60.3 The three routes, two of them closed

| route | status |
|---|---|
| arm at a time into the trailing pass | **closed** -- §55.4, and §60.2 shows the repair does not shrink |
| arm at a time at its dispatch point, via a per-node round trip | **closed** -- §58.3, quadratic on `&&` chains |
| the coupled component at once, one migration per symbol value | open |

The third has neither defect by construction: migrating once is linear, and a
single native walk dispatches every arm of the component in sequence, so no
producer outruns its consumer. It is also the only route whose endpoint deletes
the legacy dispatcher rather than shadowing it.

### 60.4 What is still separable

Ten arms are neither producer nor consumer. Most are helpers (`adjust_operands`,
`adjust_type`, the two `adjust_symbol` overloads, `adjust_argc_argv`) or already
resolved (`adjust_reference`, empty for C -- §55.5). Three are genuine dispatch
arms and are the only remaining single-arm candidates:

- `adjust_struct` -- inserts padding operands; writes no type
- `adjust_expr_unary_complex`
- `adjust_side_effect_function_call`

They are *candidates*, not cleared: §49.3 produced a list that looked clean and
was not, and each still has to face the A/B and a valid-alternative mutant. But
they are the only work in this phase that does not require the big step first.

### 60.5 Gates for the big step

Unchanged from §57, plus one this phase learned the hard way:

- A/B byte-identity over the §1.2 corpus, stderr-hashed, §21.3's three
  normalisations plus §55.2's address normalisation.
- A valid-alternative mutant per migrated arm -- not one that aborts (§57.2).
- **`deep_binary_chain_pass` stays at ~34 s.** §58.3 made this a standing
  performance gate rather than a note.
- The pre-existing `github_2220` divergence is the baseline, not a regression;
  it should be fixed or explained before the component lands, since a
  whole-component A/B cannot afford an unexplained non-zero baseline.

## 61. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921), comma arm at its
dispatch point (#6992). Withdrawn: `adjust_comma` in the trailing pass (§55),
the boolean arms at their dispatch point (§58).

Next, in order: (a) resolve `github_2220` so the baseline is zero (§60.5), then
(b) the coupled component as one step (§60.3), with (c) the three separable arms
of §60.4 available in parallel to anyone who wants a smaller piece.

## 62. `github_2220` diagnosed: array bounds live in types

§60.5 wanted the A/B baseline at zero before the coupled component moves, and
named `esbmc-unix/github_2220` as the one thing in the way. It is not a quirk of
that program, and it does not have a small fix.

### 62.1 The reproducer

Eight lines, reduced from a 90-line test:

```c
struct dirent { char d_name[256]; };
unsigned long strlen(const char *);
void g(struct dirent *entry)
{
  char buf[strlen(entry->d_name) + 2];
  buf[0] = 0;
}
```

Flag-off prints `strlen(&entry->d_name[0])`. Flag-on prints a raw irep dump: a
`member` whose base is still a pointer, which `c_expr2string` cannot render.

### 62.2 The cause

`char buf[...]` is a VLA, so its bound is an expression carried in a **type**.
`clang_c_adjust::adjust_type` walks exactly that -- `/* adjust the size
expression for VLAs */`, calling `adjust_expr` on the size. The IREP2 pass walks
`get_value2()` and nothing else. So once `irep2_owns_arms` hands an arm over,
nothing rewrites a member or index inside an array bound.

Latent since the index arm (#6907) and equally true of `adjust_member` (#6921).
It surfaced as one test only because a VLA whose bound subscripts a struct
member is rare.

### 62.3 Two fixes that do not work, and why the second matters

- **Walking the symbol's type.** Correct, and insufficient. Instrumentation
  confirms the member arm fires exactly once, on `buf`'s symbol type, and the
  output does not change: the function body's `code_decl2t` carries its own copy
  of the array type.
- **Walking each expression node's type.** Not expressible. `expr2t::type` is
  **`const`** (`irep2.h:909`) -- an IREP2 node's type is immutable by design, so
  a type-adjusting walk must *rebuild* every node whose type changed, kind by
  kind, rather than assign through it.

That second point is the finding. There is no generic "same node, new type"
operation in IREP2, so mirroring `adjust_type` is not a patch to the shadow pass
-- it is a structural piece of a walk that constructs nodes anyway.

### 62.4 Consequence for §60.5

The zero-baseline precondition is withdrawn as a *precondition*. It cannot be
met cheaply, and it does not need to be: shape 2 rebuilds nodes natively, so the
type descent comes free with it. The baseline is 1 until the component moves,
and that 1 is now explained rather than outstanding -- which is what §60.5
actually needed.

`regression/esbmc/github_2220_vla_bound` pins it as KNOWNBUG. Its regex matches
flag-off output and not flag-on, so it fails for this defect and will XPASS the
moment the defect goes -- checked both ways rather than assumed.

## 63. What the component actually executes

§60 sized the port at 19 coupled arms; `frontends-to-irep2.md` §39.1 requires a
census before writing, and this phase has twice found an arm whose work was
already dead (`adjust_comma`'s type write, §55.1; both boolean arms', §58.1).
An `fprintf` at each arm's entry, over the §1.2 corpus, flag-off:

| arm | calls | tests |
|---|---:|---:|
| `adjust_side_effect` | 94 596 | 2 559 |
| `adjust_index` | 48 499 | 1 877 |
| `adjust_address_of` | 46 499 | 2 035 |
| `adjust_side_effect_function_call` | 44 017 | 2 465 |
| `adjust_function_call_arguments` | 44 017 | 2 465 |
| `adjust_side_effect_assignment` | 40 018 | 1 415 |
| `adjust_expr_binary_boolean` | 30 045 | 453 |
| `adjust_expr_rel` | 29 956 | 2 038 |
| `adjust_member` | 29 883 | 490 |
| `adjust_expr_binary_arithmetic` | 28 098 | 891 |
| `adjust_expr_shifts` | 7 916 | 109 |
| `adjust_dereference` | 6 709 | 336 |
| `adjust_expr_unary_boolean` | 6 655 | 364 |
| `adjust_sizeof` | 4 999 | 1 346 |
| `adjust_struct` | 4 165 | 350 |
| `adjust_comma` | 3 924 | 1 102 |
| `adjust_side_effect_statement_expression` | 3 120 | 1 094 |
| `adjust_if` | 1 372 | 114 |
| `adjust_builtin_va_arg` | 52 | 9 |
| `adjust_expr_unary_complex` | 22 | 4 |
| **`adjust_base_to_derived`** | **0** | **0** |
| **`adjust_ptr_mem`** | **0** | **0** |

### 63.1 Two arms never fire, and porting them blind is the §28 trap

`adjust_base_to_derived` is guarded by `#base_to_derived` on a typecast, and
`adjust_ptr_mem` by an `id() == "ptr_mem"` node: both are C++ shapes, reachable
only through `clang_cpp_adjust`, which derives from this class. Phase 6 needs no
native counterpart for either, and **must not claim one verified** on a C-only
A/B -- that is exactly `scope-jimple-irep2.md` §28, where `nondet` was migrated
before anyone knew it occurred zero times and the byte-identity claim held for
nine PRs because nothing executed the override. They come back in Phase 7.

That takes the port from 19 arms to **17**.

### 63.2 The thin tail is where the mutants will lie

`adjust_builtin_va_arg` (9 tests) and `adjust_expr_unary_complex` (4 tests) have
corpus support two orders of magnitude below the head. §39.1's first row -- an
unmoved mutant means the corpus is thin -- is a near-certainty for both, so each
needs a written test *before* it is ported, not after its mutant comes back
silent.

This also corrects §60.4: `adjust_expr_unary_complex` was listed as one of three
arms still separable, on the ground that it neither produces nor consumes type
information. That remains true, but at 4 tests it is not a cheap win -- the
measurement it would need costs more than the arm.

### 63.3 One pair moves together

`adjust_side_effect_function_call` and `adjust_function_call_arguments` have
identical counts, 44 017 in the same 2 465 tests: the latter is called only by
the former. They are one unit of work, not two.

## 64. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921), comma arm at its
dispatch point (#6992). Withdrawn: `adjust_comma` in the trailing pass (§55),
the boolean arms at their dispatch point (§58). Baseline explained (§62).

Next: the 17 live arms of §63, ported behind a `--clang-c-irep2-adjust-only`
mode mirroring `--python-irep2-adjust-only`, with the divergence count against
flag-off as the progress metric and the flip when it reaches the §62 baseline of
1. Write tests for the §63.2 tail first.

## 65. §63.2 is wrong: few tests is not the same as undetectable

§63.2 said the thin tail -- `adjust_builtin_va_arg` at 9 tests and
`adjust_expr_unary_complex` at 4 -- would need tests written before porting,
because §39.1's first row makes an unmoved mutant near-certain at that coverage.
That inference does not hold, and the way to find out was to run the mutant
rather than reason about the count.

### 65.1 The measurement

Two valid alternatives, both compiled and swept against the flag-off baseline:

| mutant | tests moved | arm fires in |
|---|---:|---:|
| A -- `adjust_expr_unary_complex` never negates the real part, so `-z` becomes `~z` | 3 | 4 |
| B -- `adjust_builtin_va_arg` lowers to a differently-named intrinsic | 8 | 9 |

A moved `complex_23`, `complex_25`, `complex_26`; B moved the `va_start` /
`va_copy` / `vasprintf` tests and three printf-family ones. Both arms are
mutation-detectable by the corpus as it stands, so **no new tests are needed
before porting either**.

`complex_24` did *not* move under A, and should not have: it uses only `~z`,
which is the branch A leaves alone. A measurement that moved everything would be
the suspicious one.

### 65.2 What the reasoning got wrong

§39.1's row is "the corpus is thin" -- an arm the corpus does not *exercise*.
§63.2 read it as a statement about test *count*. Those differ: a mutant needs to
move one dump, and four tests that genuinely execute the arm supply that as
surely as four hundred. Coverage breadth matters for finding defects the mutant
was not designed to model; it is not the threshold for whether the gate has
teeth.

The operative test is therefore not "how many tests touch this arm" but "does a
valid-alternative mutant move the dump" -- which is one build and one sweep, and
answers the question instead of estimating it. `adjust_base_to_derived` and
`adjust_ptr_mem` remain genuinely undetectable here (§63.1) because they fire
zero times; that is the real form of the concern and the census already found it.

### 65.3 Consequence

§63.2's precondition is withdrawn. §60.4's downgrade of
`adjust_expr_unary_complex` is also withdrawn: at 22 calls in 4 tests, with a
mutant that moves 3 of them, it is exactly the small separable arm §60.4
originally called it.

## 66. `--clang-c-irep2-adjust-only`, and the first number

§60.3 left the coupled component as one step, which is 17 arms (§63.1) and no
way to show progress in between: the trailing-pass shape cannot move an arm
singly (§55.4) and the dispatch-point shape is quadratic (§58.3). The hop-off
flag fixes the measurement problem, exactly as `--python-irep2-adjust-only`
does for V.4: the IREP2 pass *replaces* `clang_c_adjust` instead of shadowing
it, so every arm ported makes strictly more tests match and the divergence
count is monotone. Default off.

### 66.1 The starting number

Against the flag-off baseline over the §1.2 corpus:

| | tests |
|---|---:|
| already identical | **1 001** |
| `migrate expr failed` before any arm runs | **575** |
| migrates, output differs | **1 233** |
| **diverging** | **1 808** of 2 809 |

1 001 already match, which is more than expected for a pass implementing three
of seventeen arms -- most tests never reach the constructs the missing arms
handle.

### 66.2 The 575 are a different workstream, and they come first

`migrate_expr` presumes a **post-adjust** tree. Run without the legacy pass, it
aborts on shapes the converter emits and `clang_c_adjust` lowers: this is the
same class as the union-constant assert (#6899) and the `member2t`/`index2t`
construction invariants (#6907, #6921), each of which was found and relaxed one
at a time. Measured at scale it is 575 tests -- a fifth of the corpus -- and
they cannot be measured *at all* until it is relaxed, because they die before
the first arm.

So the port has two workstreams, not one, and their order is forced:

1. **Migration preconditions.** Enumerate the constructs on which
   `migrate_expr` aborts pre-adjust, and relax or teach each. Until this is
   done, 575 tests contribute nothing to the metric.
2. **The 17 arms**, whose progress the remaining 1 233 measure.

### 66.3 A harness defect worth its own issue

An aborted `esbmc` does not remove its `esbmc-headers-*` temp directory (~7.4 MB
per run). Any sweep over a mode that aborts -- which `-only` does on 575 tests
today -- leaks toward 20 GB and fills the disk, after which every subsequent
measurement is an ENOSPC artifact that reads as "identical" rather than as an
error. The sweep harness now gives each run a private `TMPDIR` and deletes it;
the underlying cleanup-on-abort gap is ESBMC's, not the harness's.

## 67. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921), comma arm at its
dispatch point (#6992), hop-off flag (this section). Withdrawn: `adjust_comma`
in the trailing pass (§55), the boolean arms at their dispatch point (§58).
Baseline explained (§62); tail arms cleared for porting (§65).

Next: §66.2 workstream 1 -- census the constructs behind the 575
`migrate expr failed` tests, which is the same shape of work as #6899/#6907/#6921
and now has a number attached to it.

## 68. §66.2 is wrong: migration has no preconditions of its own

§66.2 read 575 tests as aborting inside `migrate_expr`, called that a separate
workstream, and put it ahead of porting arms. Measured directly -- an `fprintf`
at `migrate_expr`'s failure site, swept under `-only` -- it is **70 tests and
two constructs**:

| construct | tests |
|---|---:|
| `shr` | 64 |
| `builtin_va_arg` | 6 |

### 68.1 Where the 575 came from

A proxy: "output under 2 KB with `-only`, at least 2 KB flag-off". That bucket
holds three unrelated things -- real migrate aborts, a different early error, and
tests whose GOTO dump is simply short (`clang_builtins/nontemporal_load_*`
complete normally and were counted as failures). The lesson is the ordinary one:
a proxy measured because it was cheap, when the direct measurement was one
`fprintf` away.

### 68.2 And they are not preconditions -- they are unported arms

Both constructs are the *input* to an arm that has not moved:
`adjust_expr_shifts` rewrites `shr` into `ashr`/`lshr` by signedness, and
`adjust_builtin_va_arg` lowers `builtin_va_arg` to a call. Without the arm, the
raw form reaches `migrate_expr`, which has no case for it.

The same holds for every other migration failure in the corpus. Sampling 303
tests, 116 error under `-only`:

| message | count |
|---|---:|
| `Function X not found` | 93 |
| `PARSING ERROR` | 10 |
| `shr` | 6 |
| `cannot remove side effect (assign_shr)` | 3 |
| `and takes boolean operands only` | 2 |
| `sizeof node must carry a type operand and a value operand` | 1 |
| `do_function_call: unexpected callee expression (id: member)` | 1 |

`and takes boolean operands only` is `migrate.cpp:1118` firing because the
boolean arms have not run (§58.1 noted the assert from the other direction);
the `sizeof` arity error is `migrate.cpp:783` because `adjust_sizeof` has not
filled the VLA operand (§55.5, likewise). So:

> `migrate_expr`'s preconditions *are* "the legacy arms have run". There is no
> separate relaxation workstream. §66.2's ordering is withdrawn.

### 68.3 What to port first, on evidence

`Function X not found` is 80 % of the sampled errors. That is the function-call
path -- `adjust_side_effect_function_call` with `adjust_function_call_arguments`,
which §63.3 already found move as one unit, and which §63's census puts at
44 017 calls in 2 465 tests. It is both the most-executed arm and the dominant
blocker, so it is the first thing to port rather than the last.

Then `adjust_expr_shifts` (109 tests), which clears the 64 `shr` aborts and the
`assign_shr` side-effect error with them.

The `PARSING ERROR` rows are not attributed: they may fail flag-off too, and
that was not checked.

## 69. Status

C.1-C.3 (#6894), lookup (#6897), union assert (#6899), havoc order (#6901),
index arm (#6907), address_of bit (#6912), member arm (#6921), comma arm at its
dispatch point (#6992), hop-off flag (§66). Withdrawn: `adjust_comma` in the
trailing pass (§55), the boolean arms at their dispatch point (§58), §63.2's
test precondition (§65), §66.2's migration workstream (§68).

Metric: **1 808 of 2 809 diverge** under `--clang-c-irep2-adjust-only`; 649 of
those error, 1 159 differ silently. Next: port the function-call pair (§68.3),
and re-measure.

## 70. The first `-only` blocker: implicitly-declared callees

§68.3 named the function-call pair as the dominant blocker on the strength of
`Function X not found` being 80 % of sampled errors. Reduced, the trigger is
narrower than the arm:

```c
int main(void) { undeclared_fn(1); return 0; }
```

A call to a function with no visible declaration. Ordinary calls are fine under
`-only`; `clang_builtins/atomic_store` trips it only because it calls `assert`
without including `assert.h`, so `assert` is a function rather than a macro.

### 70.1 It is a symbol-table side effect, not a rewrite

`clang_c_adjust::adjust_side_effect_function_call` looks the callee up and, when
it is absent, **creates** the symbol (`context.add(new_symbol)`). That is not an
expression rewrite, so it ports independently of the arm's other ~139 lines --
and it is the general point: some arms do work that is not
representation-bound, and those pieces can move first and cheaply.

`declare_implicit_callee` does the same natively. Result under `-only`:

| | tests erroring |
|---|---:|
| before | 649 |
| after | **304** |

### 70.2 Both spellings, and a gate

A bare `f(x);` is a `sideeffect2t` of kind `function_call`; an assigned call is
a `code_function_call2t`. A handler matching one misses the other, and the
discarded-result form is exactly the failing case.

The declaration is also gated on a new `sole_adjuster` flag, set only under
`-only`. Work that *substitutes* for the legacy pass has no `irep2_owns_arms`
counterpart to disable on the legacy side, so unlike an arm's rewrite it is not
shadow-safe by construction. In shadow mode the legacy pass declares the callee
first and the native code is a no-op, so this is intent rather than a fix -- but
the distinction is real and the rest of the port will meet it again.

### 70.3 The metric did not move, and that is correct

Divergences stayed at 1 808. Clearing an *error* does not make a test
byte-identical; it lets the test run further, moving it from "errors" to
"differs". §66's divergence count is the right *exit* condition and a poor
*progress* signal, since almost all of its movement is concentrated at the end.
Track the staged counts -- errors, then differs, then identical -- which move
throughout.

### 70.4 Harness: a per-run temp path defeated the A/B

Isolating each sweep run's `TMPDIR` (§66.3, to stop aborted runs leaking header
dirs) introduced a fresh random path component per run, which §55.2's
normalisation list does not cover. The A/B then reported **793** divergences in
*flag-off against itself* -- read at first as a shadow-mode regression from this
patch, which it was not. Normalising the temp-dir name restores a clean
self-control and shadow mode to its baseline of 2 (`github_2220` and the §62
KNOWNBUG that pins the same defect).

Third harness defect in this phase, after stdout-vs-stderr and ASLR addresses
(§55.2). The pattern is constant: a per-run artefact enters the output, and the
gate reports divergence everywhere rather than failing loudly. **Run the
self-control after any change to how the sweep invokes ESBMC**, not only after
changes to ESBMC.

## 71. Status

Metric under `--clang-c-irep2-adjust-only`: **1 808 of 2 809 diverge**, of which
**304 error** (was 649). Shadow mode: 2, both the §62 VLA defect.

Next: the remaining 304. `shr` (64 tests) and `builtin_va_arg` (6) are
`adjust_expr_shifts` and `adjust_builtin_va_arg` (§68.2); the rest need the same
reduce-then-classify treatment this section applied.

## 72. `shr` shows the hop-off's ordering is not universally satisfiable

§71 put the shift arm next: `shr` is 64 of the tests still erroring under
`-only`, and 10 of the 14 real errors in the §70 sample. It does not port, and
the reason is about the `-only` architecture rather than the arm.

### 72.1 The 10 `PARSING ERROR`s are not ours

First, §68.3's unattributed row, resolved: all 10 fail flag-off as well. They are
pre-existing parse failures, not `-only` failures, and they should be subtracted
from every error count in §68-§71. The sample's 24 errors are 14.

### 72.2 IREP2 has no untyped shift, and the choice needs the promotion

`clang_c_adjust::adjust_expr_shifts` promotes both operands
(`gen_typecast_arithmetic`) and *then* reads `op0.type()` to pick `lshr` for
unsigned or `ashr` for signed. IREP2 has `lshr2t`, `ashr2t` and `shl2t` and no
signedness-agnostic `shr`, so a raw `shr` is not representable: migration must
make the arm's choice.

It cannot make it correctly. `-only` migrates a symbol's whole value up front,
before any native arm runs, so the only type available is the **unpromoted**
one, and promotion changes it:

```c
unsigned char x = 200;
int y = x >> 1;      // flag-off: ASSIGN y=(signed int)x >> 1
```

`x` is `unsignedbv` before promotion and `signedbv` after, so a migration-time
decision picks `lshr` where the arm picks `ashr`. For a promoted `unsigned char`
the two agree numerically -- the promoted value cannot be negative -- so this is
a byte-identity failure rather than a wrong answer. Byte-identity is the gate.

### 72.3 What this bounds

The hop-off's order is *migrate, then adjust natively*. That is only satisfiable
when every construct's IREP2 form is determined **before** adjustment. `shr` is
the first proof that it is not: its node kind is a *result* of adjustment.

So teaching `migrate_expr` about `shr` is not the fix -- it would have to
duplicate the promotion to be right, which is the arm. The resolutions are:

1. **Decide earlier.** Have the converter emit `ashr`/`lshr` directly; it knows
   the operand types and C11 6.5.7p3's promotion rule. This changes flag-off
   output and needs its own A/B, but it removes the construct from the adjuster
   entirely.
2. **Adjust before migrating**, i.e. keep the legacy pass -- which is what
   shadow mode already does, and what `-only` exists to stop doing.
3. **Construct natively end to end** (C.2), where no migration boundary exists
   and the question does not arise.

Only (1) and (3) make progress. (1) is a small, self-contained change and is the
next step; (3) is the phase's actual goal and this is evidence for taking the
converter, not the adjuster, as its vehicle.

## 73. Status

Metric under `--clang-c-irep2-adjust-only`: 1 808 of 2 809 diverge; **304 error,
of which the pre-existing parse failures (§72.1) are not ours**. Shadow mode: 2,
both the §62 VLA defect.

Next: §72.3 option 1 -- emit `ashr`/`lshr` from the converter -- measured
flag-off first, since it moves output on the default path.

## 74. The shift kind, decided at conversion

§72.3 left two viable routes for `shr`; this takes option 1. `clang_c_convert`
now emits `lshr` or `ashr` directly instead of a signedness-agnostic `shr`.

### 74.1 Why the converter can decide and migration cannot

Clang has already applied the integer promotion by the time the converter sees
the node, and records it as an `ImplicitCastExpr <IntegralCast>`:

```
BinaryOperator 'int' '>>'
|-ImplicitCastExpr 'int' <IntegralCast>
| `-ImplicitCastExpr 'unsigned char' <LValueToRValue>
`-IntegerLiteral 'int' 1
```

C11 6.5.7p3 gives the result the type of the *promoted* left operand, so the
node's own type is exactly the signedness `adjust_expr_shifts` computes. One
ternary, no promotion logic duplicated -- which is the difference from teaching
`migrate_expr` the same trick (§72.2), where only the unpromoted type exists.

### 74.2 The A/B caught a dispatcher bug

`clang_c_adjust::adjust_expr` routes to the shift arm on
`id() == "shl" || id() == "shr"`. Emitting the typed ids moved those nodes out
of its reach, so the arm stopped running -- and the two things it does besides
choosing the kind, `gen_typecast_arithmetic` on both operands and
`expr.type() = op0.type()`, sit *outside* the `shr` branch and apply to every
shift. One default-path divergence (`esbmc/github_323`) and 46 bytes of missing
casts. The fix is both halves: emit the typed id **and** route it.

Generalises: moving a decision earlier can silently detach a node from a
dispatcher keyed on the old spelling. Grep for the id being replaced is part of
the change, not a follow-up.

### 74.3 The gate cannot see this change

`c_expr2string` prints `ashr` and `lshr` identically as `>>`, so flipping the
kind produces a byte-identical dump. The clean A/B (0 of 2 809 on the default
path) says the surrounding structure is unchanged and **nothing** about the
choice -- §39.1's fourth row, met for the first time in this phase.

So the gate is a semantic test: for `a >= 0x80000000`, `a >> 1 < 0x80000000`
holds under a logical shift and fails under an arithmetic one. Nondeterministic
input, so it cannot be constant-folded. `regression/esbmc/shift_kind_unsigned`,
mutation-checked -- flipping the ternary fails it.

The typedef case is covered there too: `t` resolves for integer typedefs, so
`u32 >> 1` picks `lshr`. That was a live risk, since the arm follows the type
(`ns.follow(op0.type())`) and the converter does not.

## 75. Status

`shr` no longer errors under `-only` (64 tests). Default path byte-identical.
Remaining sampled `-only` errors: 4 `assign_shr`, 2 `and takes boolean operands
only`, 1 `sizeof` arity, 1 `do_function_call` member callee -- plus 10
pre-existing parse failures that are not ours (§72.1).

Next: `assign_shr`, the same class one level up. `adjust_side_effect_assignment`
picks `assign_lshr`/`assign_ashr` from the **unpromoted** LHS type, which the
converter also has, so option 1 applies again -- with its own default-path A/B
and its own semantic test, since the printer is blind here too.

## 76. `assign_shr`, the same decision one level up

`E1 >>= E2` carries the same problem as §74's `E1 >> E2` and the same fix.
`clang_c_convert` now emits `assign_lshr`/`assign_ashr`; the kind follows E1's
own type, per C11 6.5.16.2p3's rewrite to `E1 = E1 >> E2`, which is what
`adjust_side_effect_assignment` already used (`ns.follow(op0.type())` -- the
*unpromoted* LHS type, unlike the binary case).

Three details differ from §74 and each was a way to get it wrong:

- **The type arrives after the switch.** A compound assignment's type comes from
  `get_type(compop.getType(), ...)` further down, so the decision cannot sit in
  the opcode switch. It is made where `lhs` exists, using `ns.follow(lhs.type())`
  -- mirroring the arm exactly rather than trusting the node type to be resolved.
- **Falling out of the dispatcher is worse here.** `adjust_side_effect_assignment`
  ends with `gen_typecast_arithmetic(ns, op0, op1)`, which converts *both*
  operands to a common type. For a shift E2 is a bit count, not a value in that
  type (the reason #6924 exists), so an unrouted `assign_lshr` would not merely
  lose a cast -- it would gain a wrong one. The condition admits the typed forms
  and returns early for them.
- **Solidity still emits the untyped form** (`solidity_convert_expr.cpp:3180`),
  so the arm's rewrite stays for it. This is a C-frontend change only.

### 76.1 Gate

Default path byte-identical (0 of 2 809). The printer shows `>>=` whatever the
kind, so byte-identity is again blind (§74.3) and the real gate is
`regression/esbmc/shift_kind_compound_assign`: nondeterministic inputs, and
assertions in **both** directions -- an unsigned value with the high bit set must
end below `0x80000000` (false under an arithmetic shift), and a negative `int`
must stay negative (false under a logical one).

Mutating both arms kills it. That alone does not show both directions are
covered, because ESBMC reports only the first violated property -- so the signed
case was also run standalone against the mutant binary, where it fails on its
own. A two-directional test can otherwise be carried entirely by one half.

## 77. Status

Sampled `-only` errors: **14, of which 10 are the pre-existing parse failures**
(§72.1). Real remainder: 2 `and takes boolean operands only`, 1 `sizeof` arity,
1 `do_function_call` member callee. Both shift classes are gone.

Next: `and takes boolean operands only` -- `migrate.cpp:1118` asserting because
the boolean arms have not run. §58.1 showed those arms' type write is dead for C
(the converter already emits bool), so the assert is firing on operands, not on
the node: worth reducing before assuming which.

## 78. The boolean arms port after all -- in this shape

§58 withdrew `adjust_expr_binary_boolean` and `adjust_expr_unary_boolean`
because the *dispatch-point round trip* was quadratic on `&&` chains. That was
an objection to the shape, not the arm. Under `-only` the walk migrates a symbol
once and never round-trips per node, so the same arm goes in without the cost,
and their live half -- the operand conversion, §58.1 -- is now native.

This is the first evidence that `-only` absorbs work the other two shapes
rejected, which is a point in favour of §60.3's third route beyond its being the
last one standing.

### 78.1 The reduction took three attempts

The error is `goto_convert`'s short-circuit lowering rejecting a non-boolean
operand (`goto_sideeffects.cpp:1267`), so a bare `x && f()` looks like it should
fail. It does not, and neither does the same condition in an `if`. What fails:

```c
int foo(int x) { return 1; }
int main(void) { int il; for (il = 0; foo(il) && il < 2; ++il) {} return 0; }
```

A call on the **left** of `&&`, in a **`for`** condition. Fixing this from the
error message alone would have meant patching against a case that could not be
triggered -- and the message names the check, not the shape that reaches it.

### 78.2 Gate

Default path byte-identical (0 of 2 809); shadow mode unchanged at 2, both the
§62 VLA defect. `regression/esbmc/irep2_only_boolean_operands` pins the fix by
the `(_Bool)` cast on the call result; disabling the conversion makes ESBMC abort
and fails it.

The first `test.desc` regex was written from a guess at the lowering and did not
match: the condition becomes
`IF !((_Bool)return_value$_foo$1 ? il < 2 ? 1 : 0 : 0)`, a ternary. Read the
output before writing the pattern.

## 79. Status

Sampled `-only` errors: **12, of which 10 are the pre-existing parse failures**
(§72.1) -- so **2 real**: one `sizeof` arity, one `do_function_call` member
callee. `llvm/sizeof` and `llvm/struct_method` are the two tests.

Next: the `sizeof` arity error. `migrate.cpp:783` aborts on a one-operand
`sizeof`, which `adjust_sizeof` fills for VLAs -- and §55.5 already found that
arm blocked on exactly this, from the other direction. That makes it a migration
ordering question like §72's `shr`, not a port: check whether the converter can
supply the value operand before assuming the arm must.

## 80. The VLA `sizeof` operand: compute in migration, not in the converter

`migrate.cpp` aborts on a one-operand `sizeof`. The frontend emits that shape
for a VLA -- clang cannot evaluate a non-constant size -- and
`clang_c_adjust::adjust_sizeof` fills the value in. Under `-only` that pass does
not run, so the node dies before anything can fix it.

### 80.1 The converter cannot do it, measured

§79 asked whether the converter could supply the operand, since `adjust_sizeof`
does not need a constant either: it calls `c_sizeof(measured, ns)`, and
`clang_c_convertert` has an `ns`. Tried, and it moves the default path on four
VLA tests (`github_588`, `github_588_1`, `cwe_excessive_alloc_vla{,_pass}`).

The arm calls `adjust_type(measured)` *before* `c_sizeof`, so it measures a
resolved type; at conversion the symbols are not resolved yet and the resulting
expression differs. Reverted.

### 80.2 Migration can, and the distinction is the point

`migrate_expr` computing the size is **default-path-neutral by construction**:
the adjuster fills the operand first, so the normal pipeline never reaches a
one-operand `sizeof`. Confirmed empirically as well -- 0 of 2 809 on the default
path, shadow unchanged at 2.

This draws a line §72 left implicit:

> Migration may **compute** what is a pure function of information the node
> already carries. It may not **decide** something that depends on an adjustment
> having run.

`c_sizeof` of the measured type is the first: derivable wherever the type is.
The `shr` kind is the second: correct only after integer promotion. That is why
§72 refused one and §80 permits the other, and it is a usable test for the
remaining constructs rather than a case-by-case judgement.

Fail-closed behaviour is kept: an empty operand list still aborts, and so does a
one-operand node whose size `c_sizeof` cannot build.

## 81. Status

Sampled `-only` errors: **11, of which 10 are the pre-existing parse failures**
(§72.1). **One real error left**: `do_function_call: unexpected callee
expression (id: member)`, in `llvm/struct_method`.

Next: that one. It is a C++ shape reaching the C driver (a member callee), so
the first question is whether `llvm/struct_method` is a C test at all -- §63.1
found two arms that are C++-only and must not be claimed verified on this
corpus.

## 82. The function-pointer callee, and the sample reaching zero

`llvm/struct_method` is plain C -- a function-pointer struct member called as
`x.update()` -- so §81's scope question is answered: this is Phase 6 work, not
one of §63.1's C++-only arms.

`goto_convert`'s `do_function_call` accepts a symbol or a dereference as callee.
`adjust_side_effect_function_call` ends with an implicit-dereference step that
wraps a pointer-typed callee; without it the `member` node arrives bare. The
rewrite is driven only by the callee's type, depends on no adjustment, and the
failure surfaces after migration -- so by §80's rule it belongs in the native
pass, and that is where it went.

The arm also unwraps an *implicit* `address_of` callee. `address_of2t` carries
no `implicit` flag, so that half is deliberately not mirrored: it would have to
be guessed, and guessing it wrong is silent. If it matters it will appear as a
divergence or an error, not as a bad rewrite.

### 82.1 Where the metric now stands

| | before this series | now |
|---|---:|---:|
| `-only` divergences | 1 808 | **1 623** |
| `-only` errors | 304 | **185** |
| sampled *real* errors | 14 | **0** |

The sample's remaining 10 are §72.1's pre-existing parse failures, which fail
flag-off too. Every error the sample can see is now either fixed or not ours --
which means the sample has stopped being a useful instrument and the full-corpus
185 is the number to work from.

### 82.2 Gate

Default path 0 of 2 809; shadow unchanged at 2.
`regression/esbmc/irep2_only_fnptr_callee` pins the dereferenced callee, and
disabling the rewrite reproduces the original `unexpected callee expression`
error and fails the test.

The test pins *only* the callee, not the surrounding output: `-only` still
mislowers other parts of that function (`OTHER A` where an assignment belongs),
and freezing that would pin the bugs the remaining arms still have.

## 83. Status

`-only`: **1 623 of 2 813 diverge, 185 error**. Shadow: 2, both the §62 VLA
defect. Default path unchanged throughout.

Next: the 185 need re-classifying by message before picking a target -- the
303-test sample is exhausted (§82.1), so the next step is the full-corpus
equivalent of §68's error census rather than another reduction.

## 84. The full-corpus error census, and the ternary arm

§83 retired the 303-test sample. Run over all 2 813 tests, with each erroring
test *also* run flag-off so the census attributes its own rows:

| | tests |
|---|---:|
| erroring under `-only` | 185 |
| **pre-existing** -- fail flag-off too | **167** |
| **ours** | **18** |

That is the number that matters, and it was hidden until the census did the
attribution itself. §68 carried `PARSING ERROR` as an unattributed row for four
sections before §72.1 checked it by hand; measuring it per-row costs one extra
run per erroring test and removes the guesswork.

The 18 were three classes: the ternary condition (10), `builtin_va_arg` (6), and
`Can't generate zero for type complex` (2).

### 84.1 The ternary arm ports

`goto_sideeffects.cpp:1317` rejects a non-boolean `?:` condition, and
`clang_c_adjust::adjust_if` supplies the cast plus reconciles the arms with the
node's type. Both are type-driven, neither depends on an adjustment, and the
failure is downstream of migration -- §80's rule again, so it goes in the native
walk.

Two things `if2t`'s constructor settles before writing the rebuild:

- it takes an optional **location**, and `if2t` is the only value-level kind
  carrying one (§49.2, where §21.2/§26.2/§27 are three defects from forgetting
  it), so the original is passed through;
- it **asserts** each arm's `type_id` matches the node's -- which is exactly what
  the arm's second half establishes, so the arms must be reconciled *before* the
  node is rebuilt, not after.

### 84.2 Result

Ours **18 → 8**, and the class cleared completely: nothing was hiding behind it,
which is not guaranteed when a census reports first-error-per-test.

Default path 0 of 2 809; shadow unchanged at 2.
`regression/esbmc/irep2_only_ternary_cond` pins the `(_Bool)` condition, and
disabling the arm reproduces `first argument of 'if' must be boolean` and fails
it.

## 85. Status

`-only`: 1 623 of 2 813 diverge; **175 error, of which 167 are pre-existing --
8 are ours**: 6 `builtin_va_arg`, 2 `Can't generate zero for type complex`.
Shadow: 2, both the §62 VLA defect. Default path unchanged throughout.

Next: `builtin_va_arg`. §68.2 already identified it as the input to an unported
arm, and §72 showed that class splits -- some constructs can move to the
converter, some cannot. `adjust_builtin_va_arg` lowers the node to a call to
`__ESBMC_va_arg`, which is a rewrite rather than a decision, so the question is
whether it can run natively after migration or whether `migrate_expr` rejects
the node first, as it did for `shr`.

## 86. `builtin_va_arg`: a lowering migration may replay

`migrate_expr` rejects the node first. IREP2 has no `builtin_va_arg` kind, so
under `-only` the node dies exactly where `shr` died (§72) -- and not where
§82's and §84's arms failed, which was downstream of a migration that had
succeeded.

That settles placement before soundness. §82 and §84 could go in the native walk
because migration left something to walk; here it leaves nothing, so the rewrite
has to happen *during* migration, as §80's VLA `sizeof` does.

§80's rule then says it may. The lowering reads the node's type -- which becomes
the call's return type -- and its single operand, cast to `void *`. Both are
already on the node and neither is the result of an earlier adjustment: a
computation, not a decision. The cast needs no layering exception either, since
`gen_typecast` is one line over `c_typecastt` (`clang-c-frontend/typecast.cpp:9`)
and `util` may call that directly.

### 86.1 The symbol is declared, with a different type

`adjust_builtin_va_arg` does a second thing the arm does not: it moves an
`__ESBMC_va_arg` symbol into the context, deliberately typed `void (void *)` to
"avoid collisions of the same symbol with different types". `migrate_expr` holds
a `namespacet`, not a `contextt`, so §70's `declare_implicit_callee` declares it
instead -- and types it from the call site:

| | `__ESBMC_va_arg` |
|---|---|
| default path | `void (void *)` |
| `-only` | `signed int (void *)` -- the first call site |

Inert, and measured to be so rather than argued. `do_function_call` builds the
`va_arg` side-effect from the *assignment target's* type
(`builtin_functions.cpp:1500`), never from the callee symbol, so a function
reading an `int`, a `double` and a `char *` off one `va_list` lowers identically
under both flags -- as does the cross-function case, where the second function's
migration finds the symbol already in the table and takes
`sym_name_to_symbol`'s use-the-table-type path.

Not special-cased. §82 declined to mirror the `address_of` unwrap on a guess,
and forcing a `void` return here would be the same guess; if it matters it will
surface as a divergence.

The call sites do each log `missing renaming delimiters`. That is §70's ordering
-- migration walks the callee before the adjuster declares it -- and reproduces
on any implicitly-declared callee under `-only`, so it is not this arm's.

### 86.2 Result and gate

Ours **8 → 2** -- the whole `builtin_va_arg` class, all 6 of it -- and `-only`
divergences 1 623 of 2 813 → **1 612 of 2 816**. The two denominators differ
because each arm since §82 adds its own test, so treat the drop as bounded by
the 6 rather than read exactly from the difference.

Default path 0 of 2 814 common rows against the pre-arm sweep; shadow unchanged
at 2, both the §62 VLA defect. Disabling the arm reproduces `ERROR:
builtin_va_arg` and fails `regression/esbmc/irep2_only_va_arg`, which pins four
lines: the return-value temporary's type and `=va_arg(ap[0])` for an `int` and
again for a `double`. The second pair is what regression-protects §86.1 -- it is
the call whose type differs from the declared callee's, so a lowering that read
the symbol instead of the node would produce `signed int` there and fail.

`=va_arg(ap[0])` earns its place in the expected output. `make_va_list`
(`builtin_functions.cpp:835`) strips typecasts, so the `void *` cast looks
unobservable; it is observable because the array-to-pointer decay it forces
yields `address_of(index(ap, 0))`, which `make_va_list` unwraps to `ap[0]`.
Drop the cast and the operand prints as a bare `ap`.

## 87. Status

`-only`: 1 612 of 2 816 diverge; **169 error, of which 167 are pre-existing --
2 are ours**, both `Can't generate zero for type complex`. Shadow: 2, both the
§62 VLA defect. Default path unchanged throughout.

Next: that last class. `gen_zero` (`irep2_utils.cpp:63`) has no `complex_id`
arm and falls through to its aborting `default`, so the tempting fix is to add
one. §88 has to establish first *which* caller asks for a complex zero under
`-only` and not on the default path: the abort names the type, not the
adjustment whose absence produced the request, and every arm since §72 has
turned on that distinction.

## 88. Complex arithmetic, and an abort that was doing its job

The caller is `goto_check`'s `div_by_zero_check` (`goto_check.cpp:172`), reached
from `check_rec`'s `div_id` arm. Taken from a backtrace, not inferred -- §87
asked for the caller precisely because the message names only the type.

That settles the question against a `gen_zero` arm, and not on style grounds.
On the default path `adjust_expr_binary_arithmetic` decomposes a complex `/`
into per-component `ieee_div`, and `check_rec` exempts `ieee_div_id` from the
divisor check as defined behaviour (`goto_check.cpp:1265`). The default path
therefore never asks for a complex zero -- so teaching `gen_zero` to build one
would not restore parity, it would *add* a division-by-zero claim the default
path does not emit. The abort is not a missing case. It is the fail-closed
signal that the lowering upstream of it never ran, and the two erroring tests
were the only shape loud enough to say so: all 27 complex tests were diverging,
division was just the one that could not fail quietly.

### 88.1 Native, and already provided for

The complex `div2t` migrates without complaint; the failure is in `goto_check`,
downstream. §80's rule puts the arm in the native walk with §82 and §84, not in
`migrate.cpp` with §80 and §86 -- the placement question §86 had to answer the
other way.

Three things were already in place, which is most of why the arm is short:
`complex_type2t` synthesises a `(real, imag)` member view
(`irep2_type.h:556`), `member2t` and `constant_struct2t` each already name
complex as an accepted source, and `migrate_expr` synthesises the
`c:@__ESBMC_rounding_mode` symbol for a legacy `ieee_*` node carrying none --
which is exactly what `clang_c_adjust` emits here, so the native arm names the
same symbol rather than inventing a rounding mode of its own.

The element type picks the component operator: `ieee_*` for a floatbv, plain
`add`/`div` for an integer complex. The second is not a detail -- an integer
complex division *does* get a divisor check, on `denom`, exactly as the default
path gives it. The lowering does not suppress the check; it moves it onto the
operand the standard actually divides by. Both paths report it identically, on
`b.real * b.real + b.imag * b.imag != 0` -- measured, and pinned by
§88.3's third test.

Reaching that arm at all takes care, and the first draft of the harness did
not. Absent imaginary types `I` expands to `_Complex_I` (C11 7.3.1p6), whose
type is `const float _Complex` (7.3.1p4), so *every* expression written with it
has a floating element type no matter what
the operands or the assigned-to object are: `int complex w = (4 + 0 * I) / (p +
0 * I)` is a float division truncated on assignment, and it lowers to
`ieee_div` -- exempt from the divisor check. The integer arm is reachable only
by building the operands through `__real__`/`__imag__`. Two of the three tests
below asserted the integer element type and got the float one; both now
construct their operands componentwise.

### 88.2 The side-effecting operand is left alone, on purpose

Each operand is read twice, once per component, so an operand that performs a
side effect would be evaluated twice. `clang_c_adjust` binds it to a context
temporary first (`bind_sideeffect_operands`) and wraps the result in a
statement expression. `complex_25` exists to pin exactly that -- it counts calls
through `f() + z`, `f() * f()` and `z * d()`.

That half is unported. Porting it means reproducing the temporary's name
(`<file>:<line>$complex$`, `file_local`, module-tagged so `c_link` can rename it
across TUs), and getting the name wrong buys a divergence rather than a match --
so it is a separate piece of work, not a guess to make here.

What the arm does instead is **return**, leaving the node exactly as this mode
left it before §88 existed. The first draft aborted instead; declining is
measurably identical to that abort, because an unlowered complex reaching the
solver segfaults it -- `complex_25` ends in a core dump under `-only` either
way, so neither choice produces a verdict. Stated plainly because the earlier
wording ("diverges either way") reads as though declining were benign: it is
not, it is the same non-verdict arrived at without an `abort()` in the
frontend. What declining does buy is that it never trades a crash for a *wrong*
answer, which lowering a side-effecting operand would.

### 88.3 Result and gate

**Ours reaches 0.** The census's one remaining row is a `__TIMEOUT__` on
`esbmc/deep_binary_chain_pass`, which another session's parallel build pushed
past the 120 s cap; re-run under normal load it takes 35 s, errors nowhere, and
is byte-identical between the default and shadow paths. The same contention put
a third row in the shadow sweep, and it is the same test for the same reason.
Both sweeps whose numbers are quoted below carry **zero** timeout rows.

| | before | after |
|---|---:|---:|
| `-only` errors that are ours | 2 | **0** |
| `-only` divergences | 1 612 | **1 598** |
| complex tests byte-identical to the default path | 0 of 27 | **14** |

Default path 0 of 2 816 common rows; shadow unchanged at 2, both the §62 VLA
defect.

The 13 complex tests still diverging split cleanly, and neither cause is this
arm's: 12 of them (`complex_01`–`04`, `13`, `14`, `19`, `20`, `23`, `24`, `26`,
`github_268`) only because `assert` stays a `FUNCTION_CALL` where the default
path emits `ASSERT` -- an unported arm with nothing to do with complex -- and
`complex_25` for §88.2.

Two things this arm does *not* finish, both re-measured on the post-patch
binary rather than carried over:

- `complex_25` still core-dumps under `-only` (§88.2). It is the corpus's only
  remaining complex crash, and the binary arm cannot close it -- the operand
  binding is what closes it.
- **Unary complex is unported.** `clang_c_adjust::adjust_expr_unary_complex`
  lowers `-z` (negate both components) and GNU `~z` (conjugation); the IREP2
  adjuster has no counterpart, so both reach the solver unlowered and segfault
  it, exactly as §88.2's operands do. No corpus test covers it, which is why no
  census row ever pointed at it -- found by reading the legacy adjuster's other
  complex entry point, not by sweeping. It is the natural successor to this
  arm: same shape, same helpers, no operand-binding blocker, since neither `-`
  nor `~` reads its operand twice.

So: complex is not finished. What it no longer does is *abort in `gen_zero`*,
which is a narrower claim than "no longer erroring".

Three tests pin the arm, at the verdict rather than the shape, because a wrong
per-component formula still produces a correctly-shaped lowering.
`irep2_only_complex_arith` asserts each of `+`, `-`, `*`, `/` over both element
types; its expected values are *scalar* expressions over `__real__ b` /
`__imag__ b`, so a mutated formula is not restated on both sides of the
comparison. `irep2_only_complex_arith_fail` keeps a genuinely violated property
reportable through the lowering, and `irep2_only_complex_div_zero_fail` pins the
divisor check onto the lowered denominator by regexing the guard text.

Mutation-checked, one rebuild per mutant:

| mutant | killed by |
|---|---|
| arm disabled | all three -- `ERROR: Can't generate zero for type complex` |
| `mul` real: `ar*br - ai*bi` → `+` | `..._arith` |
| `div` imag: `ai*br - ar*bi` → `+` | `..._arith` |
| `div` denom: `br*br + bi*bi` → `-` | `..._arith`, `..._div_zero_fail` |
| `add` imag: `ai + bi` → `ai + br` | `..._arith` |

The denominator mutant is the one the third test earns its place on: it is the
only mutant that leaves a *plausible* divisor check standing, and only the
pinned guard text distinguishes it.

## 89. Status

`-only`: **1 598 of 2 818 diverge; 167 error, none of them introduced by §88.**
Shadow: 2, both the §62 VLA defect. Default path unchanged throughout.

Read "none introduced by §88" strictly: it means no row errors now that did not
error under `-only` before this arm. It does *not* mean the hop-off matches the
default path on those 167 -- `complex_25` errors here and passes there, and
§88.3 now says so. The census counts against the previous `-only` run, not
against the default path, and every "pre-existing" in §§72–88 carries that
sense.

Next, in order:

1. **Unary complex** (§88.3). Sized at one arm and blocked by nothing; the only
   reason it is not already done is that no test covers it, so it never
   surfaced in a census. Add the coverage with the fix.
2. **`assert`**, which is the first target that is not an error at all. The
   error census has run out of signal, so the instrument changes with the
   target, from "what aborts" to "what the divergence set is made of". §88.3
   supplies the first reading: `assert` holds 12 complex tests on its own, and
   nothing counted it because a `FUNCTION_CALL` where an `ASSERT` belongs fails
   quietly. Size it across the whole corpus first; the 12 are only the ones
   §88 happened to look at.

## 90. Unary complex, and a corpus that could not report it

`clang_c_adjust::adjust_expr_unary_complex` lowers `-z` into a negated pair and
GNU `~z` into a conjugated one. It is dispatched under
`(unary- || bitnot) && type is complex` (`clang_c_adjust_expr.cpp:134`), which
`is_complex_unary` mirrors exactly.

§88.3 named it as this arm's successor and said "No corpus test covers it,
which is why no census row ever pointed at it". The first clause is wrong.
`complex_23` applies both operators over both element types, `complex_24`
conjugates, and `complex_25:39` conjugates a *call*. Three tests, all inside
the swept suites, none of them new.

What they cannot do is report. None of the three includes `<assert.h>`, and an
implicitly-declared `assert` stays a `FUNCTION_CALL` under `-only` (§88.3, and
§91 for how much narrower that target is than it looked), so `complex_23`'s
eight assertions are never checked and it returns SUCCESSFUL whether or not
`-z` was lowered:

| test | default path | `-only`, pre-patch | `-only`, post-patch |
|---|---|---|---|
| `complex_23` (`-z`, `~z`, both element types) | SUCCESSFUL | SUCCESSFUL | SUCCESSFUL |
| `complex_24` (`~z`, property violated) | **FAILED** | SUCCESSFUL | SUCCESSFUL |

`complex_24` is the sharp one: `-only` reports a *wrong verdict* on it, and did
so before this arm and after it, because what is broken there is the assertion,
not the operator.

So the census was not blind to unary complex; it was reading a corpus in which
the arm that would have reported it is itself unported. The two claims §88.3
ran together -- "no row pointed at it" and "no test covers it" -- come apart
here, and only the first was ever measured.

### 90.1 The arm, and where the element type stops mattering

Placement follows §80's rule, as §88.1's did: the node migrates cleanly, so the
arm belongs in the native walk and not in `migrate.cpp`.

`member2t` and `constant_struct2t` already accept a complex source (§88.1), so
what is left is choosing which component to negate. Unlike §88's arm, the
element type does not also pick a component *operator*: negation is a sign-bit
flip, exact and independent of the rounding mode, so there is no `ieee_neg` to
select and no `c:@__ESBMC_rounding_mode` symbol to name. The integral element
type reaches the same six lines and -- unlike §88.1's integer division, which
brings a divisor check the float form is exempt from -- brings nothing with it.
That is the whole reason this arm is short where §88's is a switch.

`complex_23`'s five unary sites, pre-patch under `-only`, post-patch, and on
the default path:

| site | `-only`, pre-patch | `-only`, post-patch | default path |
|---|---|---|---|
| `n = -z` | `n=-z` | `n={ .real=-z.real, .imag=-z.imag }` | identical to post-patch |
| `c = ~z` | `c=~z` | `c={ .real=z.real, .imag=-z.imag }` | identical to post-patch |
| `t = -z` (typedef'd) | `t=-z` | `t={ .real=-z.real, .imag=-z.imag }` | identical to post-patch |
| `ni = -zi` (`__complex__ int`) | `ni=-zi` | `ni={ .real=-zi.real, .imag=-zi.imag }` | identical to post-patch |
| `ci = ~zi` (`__complex__ int`) | `ci=~zi` | `ci={ .real=zi.real, .imag=-zi.imag }` | identical to post-patch |

Post-patch and default agree character for character on all five. What still
separates the two dumps for this test is `assert`, and nothing else.

### 90.2 The declined operand, measured rather than argued

The legacy unary path calls `bind_sideeffect_operands` too
(`clang_c_adjust_expr.cpp:666`), so §88.2's decline transfers unchanged: each
operand is read once per component, the temporary-binding half is unported, and
the arm returns rather than lowering an operand that performs a side effect.

§88.2 defended that with two claims it could not measure, because `complex_25`
mixes binary and unary operators and core-dumps on the binary ones first. A
unary-only program -- `-f()` and `~f()` over a call that increments a counter,
the shape `complex_25:35` and `:39` already use -- separates them:

| probe | default path | `-only`, guard present | `-only`, guard removed |
|---|---|---|---|
| components **read** | SUCCESSFUL | *core dump, no verdict* | **FAILED**: `calls == 1` |
| components **unread** | SUCCESSFUL | SUCCESSFUL | **FAILED**: `calls == 1` |

Both halves hold: declining costs a verdict, lowering would cost correctness,
and the violated property is exactly the double evaluation §88.2 predicted.

The second row is the one worth keeping. §88.2 concluded that declining is
"measurably identical to that abort" from `complex_25` core-dumping either way,
and that generalises less far than it looks: the crash comes from *reading* the
declined result, not from declining it. A program that leaves the result unread
never presents a complex-typed node to the encoder, while the call counter
still records the double evaluation. So the guard is pinnable by a *passing*
test, and `irep2_only_complex_unary_sideeffect` is it -- SUCCESSFUL with the
guard, FAILED on `calls == 1` without it, both measured.

`complex_25` remains blocked: it asserts `calls == 2` immediately after `-f()`,
but it reads its results and its `f() + z` reaches the binary arm's decline, so
it still core-dumps under `-only`. It is no longer the only candidate, which is
the correction -- "the operand binding blocks all coverage of the decline" was
inferred from the one test that happened to be in front of us, and it is wrong
in the same shape §90's opening catches §88.3 in.

### 90.3 Result and gate

The arm moves no row into byte-identity, and the divergence total does not
move:

| | before | after |
|---|---:|---:|
| `-only` divergences | 1 598 of 2 818 | 1 598 of 2 821 |
| complex tests byte-identical to the default path | 14 of 27 | 14 of 27 |
| `-only` rows whose dump changed | -- | 2 |
| default-path rows whose dump changed | -- | 0 of 2 821 |

Both sweeps carry zero timeout rows. The two changed rows are `complex_23` and
`complex_24`; both still diverge, on `assert`. The denominator moves by three,
not five: §88's three tests entered the corpus after its own sweep was taken,
and two draft names from that section (`irep2_only_complex_div`,
`..._div_fail`) left with it.

Reported this way on purpose. The instrument is GOTO-level byte identity
against the default path, and on this corpus it cannot see this arm, because
every existing test that exercises unary complex is quietened by `assert`
before the difference can reach a verdict. What the arm does buy is visible
one level down: five sites in `complex_23` that now match character for
character, and two of the three new tests, which core-dump under `-only`
without it.

Three tests pin the arm at the verdict, the first two following §88.3's shape
-- nondet operands assumed into NaN-free ranges, expected values written as
scalar expressions over the components, so a mutated formula is not restated on
both sides of the comparison. `irep2_only_complex_unary` asserts `-z` and `~z`
over both element types, over a typedef'd complex, and over `-(z * z)`, whose
operand is the binary arm's own output and so is read out of a struct literal
rather than a symbol; `irep2_only_complex_unary_fail` keeps a genuinely
violated property reportable through the lowering;
`irep2_only_complex_unary_sideeffect` pins the decline of §90.2.

Mutation-checked, one rebuild per mutant:

| mutant | killed by |
|---|---|
| arm disabled | `..._unary`, `..._unary_fail` -- core dump, no verdict |
| real component not negated for `-z` | `..._unary` |
| imaginary component not negated | `..._unary`, `..._unary_fail` |
| real component negated for `~z` too | `..._unary` |
| `is_neg2t` swapped for `is_bitnot2t` | `..._unary` |
| side-effect guard removed | `..._sideeffect` |

`..._unary_fail` earns its place on the second mutant: dropping the imaginary
negation makes `~z` the identity, which turns the violated property true and
the test SUCCESSFUL.

`..._sideeffect` is the only one of the three that the first mutant does *not*
kill, and that is the point of it: disabling the arm and declining inside it
leave an unread result in the same place, so the test separates the guard from
the arm rather than restating it. It is also a Phase-2 contract test rather
than a scaffold -- single evaluation of a side-effecting operand stays true
once the binding lands and the decline goes away, so nothing here has to be
deleted to make progress.

The same shape pins the binary arm's identical guard, which shipped untested in
§88: `f() + z` with the result unread is SUCCESSFUL under `-only` today.
Left for the commit that ports the binding, since that is what gives the two
guards a common fix.

## 91. Status

`-only`: **1 598 of 2 821 diverge; 167 error, none of them ours.** Shadow: 2,
both the §62 VLA defect. Default path unchanged: 0 of 2 821 rows moved.

All three sweeps carry zero timeout rows. The shadow sweep first reported a
third row -- this section's own `irep2_only_complex_unary` -- which was an
artefact of editing that test between sweeps, not a shadow-mode divergence; on
one source revision the three modes agree byte for byte on it. Recorded because
a manifest is only comparable against another taken over the same inputs, and
nothing in the harness checks that.

`assert` is still the next target, but §88.3 mis-scoped it and §90's opening
only found half of that. Of the 13 diverging complex rows:

- **10** call `assert` with no `<assert.h>` in the file. That is the shape that
  stays a `FUNCTION_CALL`: with the header, assertions lower and report
  normally under `-only` -- `complex_24` plus one `#include` reports FAILED,
  the same verdict the default path gives it. So the target is the
  *implicitly-declared* callee path (§70's neighbourhood), not `assert`
  lowering, and the fix is narrower than "port `assert`".
- **`complex_25`, `complex_26`** diverge on §88.2's unported operand binding,
  visible as the `main.c:13$complex$` temporary the default path declares and
  this mode does not.
- **`github_268`** has nothing to do with either: 10 882 diff lines of missing
  `(_Bool)` casts on conditions and array-to-pointer decay in call arguments.

Only `complex_23` has been measured down to a single remaining cause. For the
other nine the implicit-declaration path is established as *a* cause, not as
the only one -- which is the distinction §88.3 lost, and worth holding onto
before sizing the target.

Next, in order:

1. **The implicitly-declared `assert`**, sized across the whole corpus rather
   than across the complex tests, and sized as "how many rows have this as
   their *last* difference" rather than "how many contain it".
2. **The complex operand binding** (§88.2, §90.2). It is what `complex_25` and
   `complex_26` diverge on, it closes the corpus's last complex crash, and it
   retires both arms' declines together. The work is reproducing the
   temporary's name -- `<file>:<line>$complex$`, `file_local`, module-tagged --
   closely enough that `c_link` renames it the same way across TUs.


## 104. All four C suites censused — and no unowned cause remains

`esbmc-unix` is the last of §101's list. **53 of 60 sampled tests differ**, and
after correcting three tag rules (see below) every cause is owned. That closes
the census:

| suite | differing | dominant cause |
|---|---|---|
| `regression/esbmc` | 78 of 120 (65 %) | `__builtin_expect` — 37 (#7086) |
| `cstd` | 134 of 142 (94 %) | warning 34 (#7093), `assert` 34 (#7087) |
| `floats` | 97 of 102 (95 %) | name-matched builtins — 25 (#7088) |
| `esbmc-unix` | 53 of 60 (88 %) | **padding — 46 (#7100)** |

**Across all four, every measured divergence is owned by an open PR.** There is
no adjuster arm left to write for the censused C corpus. §101 said that of one
suite; it now holds for the corpus §1.2 prices.

### 104.1 The dominant cause differs per suite, which changes the priorities

This is the useful result, and it is not visible from any single suite.
`#7100` (struct/union padding) accounts for 9 tests in `regression/esbmc` and
**46 of 53** in `esbmc-unix`: the unix headers are dense in padded unions
(`pthread_attr_t` is a union of a `char[36]` and a `long`, whose
`union_pad#` is missing under the flag). A reviewer sizing these PRs from
`regression/esbmc` alone would rank #7100 sixth; on the corpus it is first or
second.

Likewise #7088 barely registers in `regression/esbmc` and is the top cause in
`floats`, where `fabs`/`inf` are everywhere.

### 104.2 Three tag rules were wrong

Recorded because the same rules will be reused:

- `union_pad#` is a padding token that `anon_pad` does not match — unions pad
  under a different name.
- `&f` for a function designator (#7092) needs its own rule; it is not a
  by-name pattern like the others.
- `&"lit"[0]` versus `"lit"` is the string-literal half of the decay class
  (#7098), and reads as a quoted-string difference rather than an index.

Each initially produced an UNTAGGED test that looked like a new cause. §98.2 and
§104.3 already record that symptom-tagging misattributes; this adds that it also
*over*-reports, and the fix is to read every untagged residue rather than trust
the tally.

### 104.3 What is left, and it is not an arm

- **Landing the PRs.** §99 gives the order and the two non-mechanical conflicts.
  Master additionally does not build at 2284cf241d (#7111).
- **W3/W4**, coupled per §102.2 and needing a design decision, not a port.
- **`goto_convert`**, for §98's residual `DEAD` questions.
- **`adjust_type` beyond padding** and **`adjust_float_arith`'s vector branch**,
  both witnessless so far; §103.2 argues the latter's scalar path is dead
  legacy code rather than unported work.
## 103. The `floats` suite censused — and the `ieee_*` gap does not exist

`floats` is the second of §101's uncensused suites: **97 of 102 differ**. By
cause, over the first 50:

| cause | tests | owner |
|---|---:|---|
| name-matched builtins (`fabs`, `inf`, ...) | 25 | #7088 |
| `migrate_expr` renaming warning | 18 | #7093 |
| `assert` base name | 18 | #7087 |
| boolean cast on a condition | 4 | #7099 |
| array-to-pointer decay | 3 | #7098 |
| `for`-init hoist | 1 | #7105 |

**Every cause is owned.** Like `cstd` (§102), the suite is denser in the same
things rather than differently affected.

### 103.1 A cause that was not there

A first pass tagged 2 tests as an `ieee_*` promotion gap — the
`adjust_float_arith` arm §104.2 deferred — on the strength of `IEEE_ADD` and
friends appearing in the default symbol table and not under `-only`. The arm was
written to close it. It fires **zero** times, and the tag was a false positive.

Measured directly:

```c
double a = nondet_double(), b = nondet_double();
double c = a + b;
```

`double c=IEEE_ADD(a, b);` on the default path **and** under `-only` on an
unmodified baseline. The promotion is already there without the legacy arm.

The two tagged tests are `Float-no-simp8`/`9`, and their `IEEE_*` lines sit
inside **libm model functions absent from the `-only` symbol table altogether** —
the same "1066 vs 1067 symbols" seen in §102's `cstd` samples. An unlowered
builtin call (#7088) changes which operational-model functions get linked, so
whole model bodies appear or disappear. The tag matched a consequence of a cause
already owned, in a function that is not the program's.

The arm was deleted rather than shipped. It would have been an arm with no
witness, which is what §94.1 and §102.2 both declined to do.

### 103.2 A dead-code candidate

That measurement says something about the legacy pass, not just the port: if a
float `+` is already an `ieee_add` before `clang_c_adjust` runs — which is what
`-only` on an unmodified baseline demonstrates, since it never calls
`adjust_float_arith` — then **`adjust_float_arith`'s scalar path is dead on the C
frontend**. Its vector branch may not be; the arm explicitly handles a
vector-of-float and returns before attaching a rounding mode.

That makes it a candidate for the dead-code process rather than for porting:
`CLAUDE.md`'s C-Dead sub-mode, with the vector case checked separately. Recorded
here rather than acted on, because deleting a legacy arm needs its own proof and
is not this scope's business.

### 103.3 Suites remaining

`esbmc-unix` (438) is the last of §101's list. On the evidence of `cstd` and
`floats`, expect the same four owned causes and no new ones.

## 99. The fifteen open PRs do not batch-merge — an integration attempt

Fifteen Phase 6 PRs were open with none merged, so rather than add a sixteenth
arm this section reports an attempt to merge them all onto master at once and
measure the combined divergence. **The combined number was not obtained**: the
integration does not build, for two reasons that matter to whoever lands them.

### 99.1 The matcher move collides with a function master added

PRs #7086/#7088 move `compare_float_suffix`, `compare_unscore_builtin`,
`is_abs_builtin_name`, `is_name_matched_builtin` and `shadows_user_definition`
out of `clang_c_adjust_expr.cpp` into `builtin_names.{h,cpp}`, deleting that
region. Since those branches were cut, master gained **#7028**, which added
`float_lowering_id` *inside the same region* and calls it from
`do_special_functions`.

Git flags this as a conflict, so it is not silent — but the conflict looks like
"branch deleted a block, master edited it", and resolving it the obvious way
(take the deletion) removes `float_lowering_id` and the build fails with
`use of undeclared identifier 'float_lowering_id'`. The resolution has to keep
master's new function and move only the five matchers.

### 99.2 Adjacent helpers lose their shared closing brace

Eight of the branches add a `static` helper and a dispatch line to
`clang_c_adjust_irep2.cpp`, all in the same two places. When two of them are
merged, git's conflict region covers each side's body but **not** the closing
brace, which is shared context. Resolving with "keep both sides" therefore
produces two function headers and one brace — `function definition is not
allowed here` — and where the arms are longer it splices two bodies together
(`redefinition of 'before0'`).

Again: git does flag it. The hazard is that these conflicts *look* like the
textbook "both added something, keep both" case and are not.

### 99.3 What follows for the merge

- **Merge one PR at a time, in number order, building after each.** The
  dependency chain (#7086 → #7087 → #7088 → #7090 → #7091 → #7092) is only the
  declared order; the master-based arms (#7093-#7102) touch the same two
  places and will each need a trivial-but-manual resolution once the earlier
  ones land.
- **#7088 needs its own master merge before anything else**, because of §99.1.
  That resolution is a judgement, not a mechanical one.
- The doc-section collisions already seen (§92/§93 renumbering) are the benign
  half of the same phenomenon and can be resolved by keeping both sides; the
  source ones cannot.

### 99.4 What this section does not claim

No combined divergence figure. Every per-arm number in §§90-98 was measured
against master with only that arm applied, and those stand; how far the fifteen
together take the corpus is unmeasured, and will stay unmeasured until they can
be built together. Stated plainly because the obvious summary — "201 down to N"
— is one this scope has not earned.

## 98. The `DEAD` class: a live variable marked dead, and a hoist that does not fix it

§97 left 13 untagged tests, most showing a `DEAD` instruction in a different
place. Read properly, that class is two things, and only one of them is an
unported arm.

### 98.1 The defect

For a `for` loop declaring its own variable:

```c
for (int i = 0; i < 10; ++i)
  buf[i] = (char)nondet_int();
```

| path | placement |
|---|---|
| default | `DECL i`, the loop, `GOTO 1`, then `2: DEAD ...@i` |
| `-only` | `DECL i`, **`DEAD ...@i`**, then the loop guard |

Under the flag the loop variable is marked dead *before the body that reads and
writes it*. That is a scope error, not a spelling one.

No verdict impact is demonstrated: an assertion over a value accumulated in such
a loop verifies identically on both paths (`--unwind 12`). Stated rather than
implied, because "marks a live variable dead" sounds worse than what has been
shown, and because symex's own treatment of `DEAD` for a scalar is what absorbs
it. It remains a divergence that must close before the round-trip can be deleted.

Pinned as `regression/esbmc/irep2_only_for_scope_knownbug`, KNOWNBUG, whose
regex is confirmed to match the **default** path — so the test flips to CORE when
the placement is fixed rather than being vacuously green.

### 98.2 The arm this is not

`clang_c_adjust::adjust_for` hoists the loop's init into an enclosing block —
`for (a; b; c) d;` becomes `{ a; for (; b; c) d; }` — and its comment records
that it is *"the only structurally-mutating adjust_* method"*, fenced on a
non-nil init because re-running it would move a now-nil init into a fresh block
and break goto_convert (#5298). That arm is unported, so it was the obvious
candidate.

It was written and measured, and **it is not the fix**. Ported, it:

- fires on 4 of the sample's tests (a plain `for (int i = 0; ...)` does not reach
  it: clang has already hoisted, so `init` is nil);
- converges those tests — `github_4978` from 14 differing lines to 8,
  `github_1890_2` 28 to 24, 33 767 to 33 743 corpus-wide;
- **leaves the misplaced `DEAD` exactly where it was.**

So it is a faithful port that moves location comments and nothing a test can
meaningfully pin, while the defect that dominates its own class survives. Not
shipped on its own for that reason; it belongs in the change that fixes §98.1,
where the two can be gated together. The measurements are recorded here so the
next attempt starts from them rather than repeating the dead end.

### 98.3 Where to start

The placement is `convert_block`'s destructor unwinding, not the adjuster: the
`-only` and default goto programs differ in where `i`'s scope ends, and the
adjuster only decides what block structure `goto_convert` is handed. The next
step is to find which block the `-only` path puts the declaration in — the hoist
above changes that structure and does *not* move the `DEAD`, which is the useful
negative result.
## 93. The four tests that diverged against themselves

PR #7086 recorded that `gcc_nested_func_02`, `gcc_nested_func_collision`,
`gcc_nested_func_sibling_calls_uncaptured` and `github_746` differ between two
runs of the *same* binary on the same input, and recommended teaching
`irep2_canon` to strip the noise. Three of the four are fixed at the source
instead.

`transform_nested_functions` wrote its rewritten source to
`create_tmp_file("esbmc-nested.%%%%-%%%%.c")`. The helpers it lifts have
internal linkage, so clang's USR for each embeds the **basename of the file it
was parsed from** -- and `generateUSRForDecl` is what
`clang_c_convertert::get_decl_name` uses. The random basename therefore reached
the symbol table: same input, same flags, a different goto program every run.
The `#line 1 "<source>"` directive already at the top of the rewritten file
fixes locations, not USRs.

The fix keeps the uniqueness and moves it somewhere the USR cannot see: a
per-run temp **directory** with a deterministic file name inside it, derived
from the source's basename (`esbmc-nested.main.c`). Two translation units whose
sources share a basename collide, which is the hazard two real file-static
functions in same-named files already carry.

This is worth more than the gate it unblocks. A goto program that is not a
function of its input undermines counterexample reproducibility and any caching
keyed on it, and it silently defeats *any* differential harness -- this one
spent two sessions attributing those tests to whichever patch was in hand before
the self-comparison caught that they diverge against themselves.

`github_746` is untouched: its difference is clang AST-dump node addresses in an
error message, which is diagnostic text rather than program content, and belongs
in the canonicaliser, as that note said.

## 102. The migrate warning, and a correction to §92.2

§96's census put the `migrate_expr` "missing renaming delimiters" warning at the
top of the remaining causes, 31 tests. Taken on its own terms rather than as a
divergence row:

`sym_name_to_symbol` returns level0 immediately for a symbol it finds in the
namespace. Reaching the warning means the symbol was *not* found, and the name
carries no `?`/`!`; the function then treats it as level0 -- which is what it
is. A level0 symbol carries no renaming delimiters by definition, so their
absence is not an anomaly, and the message names no action. It fires once per
occurrence.

Measured over the sample: under `-only` every instance is an implicitly-declared
callee (`assert` 23, `perror` 3, `strlen` 1, `signbit` 1) -- library functions
used without their headers. On the **default path** it fires too, on
`sizeof(int[n])`, a VLA type whose extent symbol is reached before it is in the
context. Both are ordinary construction order, not defects.

Demoted to `log_debug("migrate", ...)`. The information is unchanged at
`--verbosity 9`.

### 102.1 What was considered and dropped

A first version kept `log_warning` for the level2 case -- a name carrying `#`
but no delimiters is genuinely malformed. It was dropped because that is a
**new branch whose reachability cannot be shown**: the names come from ESBMC's
own renaming, which always emits delimiters, and no C input reaches it.
`CLAUDE.md`'s dead-code rule is that an added branch must be proven reachable or
removed, and an unprovable guard is worth less than the simpler code. If a
malformed level2 name is ever produced, the guard can come back with the input
that produces it.

### 102.2 §92.2 overstated the masking

§92.2 said the warning makes the divergence count "not a sufficient statistic"
because an arm can be entirely correct and score zero, and put 17 tests behind
it. That was measured on the base-name branch, where the `assert` tests had lost
their goto difference and had only the warning left. At the current stack tip --
which does **not** include that branch -- the warning is the sole difference in
**2** of 105 tests, and on master in **1** of 201. The general point stands; the
number attached to it was specific to one branch's state and should not be
carried forward.

On master this change takes the sample from 201 to 200. Its value is the output
it stops printing, and that it lets the base-name arm's effect be seen.

## 92. The base-name defect in `declare_implicit_callee`

The fix is one line: `declare_implicit_callee` gives the symbol
`get_pretty_name(id)` -- the existing helper in `util/symtab/pretty.h`, which is
the same `rfind('@')` split every other consumer of a mangled C identifier uses
-- instead of the identifier itself.

`complex_01`'s goto program is byte-identical to the default path afterwards.

### 92.1 What it actually fixes is a vacuous pass

Before: `assert(x == 2)` with `x == 1` and no `#include <assert.h>` reports
**VERIFICATION SUCCESSFUL** under `-only`, because no `ASSERT` is emitted at
all. That is the dangerous direction -- the `__builtin_expect` defect (PR #7086)
reported a spurious failure, which is loud; this one silently drops the property
and reports success. Both are wrong verdicts and only one of them complains.

`irep2_only_implicit_assert_fail` pins it at the verdict (FAILED, SUCCESSFUL on
master) and `irep2_only_implicit_assert` pins the emitted `ASSERT` in the goto
dump (`FUNCTION_CALL: c:@F@assert(x == 1)` on master). Both discriminate against
a control binary.

### 92.2 The divergence count does not move, and the reason is instructive

| | control | patched |
|---|---:|---:|
| full-output divergence, 297-test sample | 201 | **201** |
| of those, goto program identical (warning-only) | 1 | **17** |

The arm closes 16 tests at the goto level and the headline metric registers
nothing, because what replaces the missing `ASSERT` is a *different* divergence
on the same tests: `migrate_expr` warns

```
WARNING: migrate_expr: symbol 'c:@F@assert' missing renaming delimiters,
treating as level0 with base name 'c:@F@assert'
```

once per occurrence, on stderr, which the A/B captures. The symbol genuinely is
not in the context when the enclosing body is migrated -- `get_value2()` runs
before `declare_implicit_callee` adds it -- so this is inherent to the ordering,
not to the fix.

Two consequences worth carrying forward. First, **the divergence count is not a
sufficient statistic for this phase any more**: an arm can be entirely correct
and score zero. §89 already moved the instrument once, from "what aborts" to
"what the divergence set is made of"; this moves it again, to "goto program
versus diagnostics", and the sweep should report both columns from here on.
Second, the warning is itself a candidate: it is emitted for every
implicitly-declared callee under `-only` and says nothing a user can act on.

### 92.3 A local-only failure class, explained

`irep2_only_complex_arith` and `irep2_only_complex_arith_int` fail on master on
this host and pass in CI. The cause is PR #7086's finding: they call `assert`, Darwin's
`assert.h` routes it through `__builtin_expect`, and the nondet result violates
the assertion. glibc's `assert.h` does not, so CI never saw it. PR #7086's arm makes
both pass locally. Anyone baselining this suite on macOS should expect that
class to disappear with it rather than treat it as noise.

## 93. Status

`-only` on the 297-test sample: **201 of 297 by full output** (unchanged by this
arm, §92.2), **184 by goto program** (200 before). PR #7086's arm, measured
separately from master, takes the full-output count to 131.

Gates: 42 of 44 in the
`irep2_only|complex_|gcc_popcount|gcc_bswap|github_223` slice green; the two
failures are the §92.3 pair and fail on master identically. The whole-suite gate
is still owed.

Next:

1. The name-matched builtin family with `shadows_user_definition` (PR #7088).
2. The `missing renaming delimiters` warning (§92.2) -- worth closing on its own
   terms, and it unblocks the divergence metric.
## 92. `assert` is two mechanisms, and neither of them is `assert`

*(Renumbered from §90: master took that number for the unary-complex arm.)*

§89 named `assert` as the next target on the strength of §88.3's twelve complex
tests. Sized across the corpus first, as §89 asked, it splits into two unrelated
causes, and the dominant one is not an assertion arm at all.

### 92.1 The dominant cause is `__builtin_expect`, and it is a wrong verdict

Darwin's `assert.h` expands `assert(e)` to
`__builtin_expect(!(e), 0) ? __assert_rtn(...) : (void)0` under `__DARWIN_UNIX03`.
`do_special_functions` folds `__builtin_expect(v, hint)` to `v`
(`clang_c_adjust_expr.cpp:1587`); the IREP2 pass did not, so the call survives
into the goto program. It has no body, so its result is nondet:

```c
#include <assert.h>
int main(void) { int x = 1; assert(x == 1); return 0; }
```

`VERIFICATION SUCCESSFUL` on the default path, `VERIFICATION FAILED` under
`-only`. Every `assert` on this host was nondet in this mode, which is why the
class was large. That is a different kind of finding from §82/§84/§88, all of
which were shape divergences: this one is a wrong answer, and the divergence
count was measuring it only incidentally.

**Host-dependence, stated rather than discovered later.** glibc's `assert.h`
does not use `__builtin_expect`, so the share this arm carries here (36 of 70
diverging tests in a first sample) will be smaller on Linux CI. The arm is worth
landing regardless: `__builtin_expect` occurs directly in four corpus files and
in the libm operational model's `predict_true`/`predict_false`
(`src/c2goto/library/libm/musl/libm.h:92`), and a nondet assertion is a wrong
verdict wherever it occurs, not a formatting difference.

### 92.2 The second cause is a base-name defect in §70's arm

A test that calls `assert` *without* including the header -- `complex_01` and
its neighbours -- gets an implicit declaration, and §70's `declare_implicit_callee`
declares it. `clang_c_adjust` sets `new_symbol.id = identifier` and
`new_symbol.name = f_op.name()`, the base name; the IREP2 arm sets **both** to
the identifier. `do_function_call_symbol` matches on the base name
(`builtin_functions.cpp:933`), so under `-only` the symbol is named
`c:@F@assert`, nothing matches, and the call stays a `FUNCTION_CALL` where the
default path emits `ASSERT` -- exactly the shape §88.3 reported. `migrate_expr`
also warns that `c:@F@assert` is "missing renaming delimiters", which is the
same defect visible from the other side.

This is a one-line defect in an already-merged arm rather than a port, so it is
the next slice and not this one.

### 92.3 Scope: the reserved spellings only

Ported here: `__builtin_expect`, `__builtin_popcount{,l,ll}` and the `__popcnt*`
aliases, `__builtin_parity{,l,ll}`, `__builtin_bswap{16,32,64}`. All are
`__builtin_`-prefixed, which `is_name_matched_builtin`'s comment
(`clang_c_adjust_expr.cpp:1381-1385`) calls out as reserved: a program
cannot supply its own definition, so unlike `is_name_matched_builtin`'s family
(`abs`, `isnan`, `isinf`, `inf`, `huge_val`, ...) they need no
`shadows_user_definition` query. That family, and the composite lowerings
(`__builtin_isinf_sign`, `__builtin_fpclassify`), are the natural successor.

**Not ported: the relational family.** `__builtin_isgreater` and its siblings
occur **zero** times in the C corpus's sources. §39.1 of the parent roadmap
records what porting an unexercised construct costs -- jimple's `nondet` override
held a byte-identity claim for nine PRs because nothing executed it -- so they
wait for a slice that brings its own tests.

One representational note: `popcount2t`'s type is hard-fixed to `int32`
(`irep2_expr.h:1074`) where the legacy node uses `int_type()`. Identical on every
supported target; a 16-bit-`int` target would diverge, and the fix would be to
give `popcount2t` a type rather than to work around it here.

### 92.4 Result

Measured over a 297-test stride sample of `regression/esbmc`, control and
patched binaries both saved before the sweep:

| | control | patched |
|---|---:|---:|
| diverging under `-only` | 201 | **131** |
| identical | 96 | 166 |
| tests that regressed (SAME → DIFF) | -- | **0** |

Default path unchanged: of the 131 tests diverging under both binaries, 127
produce byte-identical default-path dumps. The other four
(`gcc_nested_func_02`, `gcc_nested_func_collision`,
`gcc_nested_func_sibling_calls_uncaptured`, `github_746`) differ **against
themselves** -- re-running one binary twice on the same input reproduces the
difference. The nested-function transform names its synthetic file
`esbmc-nested.<rand>.c` and the clang AST dump prints node addresses, and
`irep2_canon` strips neither. Any A/B over this corpus will show those four
forever; they should be added to the canonicaliser rather than re-investigated.

Three tests pin the arm, at the verdict: the fold is invisible in shape terms
once it has happened, so only a value distinguishes it. The hint operand is
`0` in the passing test and `1` in the failing one, which makes returning the
hint instead of the value kill both.

| mutant | killed by |
|---|---|
| arm disabled | `..._expect`, `..._bit_ops` |
| `__builtin_expect` yields the hint, not the value | `..._expect`, `..._expect_fail` |
| parity `popcount & 1` → `& 2` | `..._bit_ops` |
| `bswap` → identity | `..._bit_ops` |

## 93. Status

`-only`: **131 of 297 diverge** on the stride sample (201 before this arm), 0
regressions, default path unchanged. Unary complex (§88.3) is in flight
separately.

Gates run: 694 unit tests green; the 85-test
`gcc_popcount|gcc_bswap|builtin_|complex_|irep2_only|csmith|github_223` slice
green. **The whole-suite gate did not run** -- the machine's 1-minute load
average was above 10 and `ctest -L esbmc` did not complete inside the 5-minute
cap at any stride tried. Stated rather than omitted: this arm is gated on
`sole_adjuster` and so cannot reach the default path by construction, but the
suite number is owed and not paid.

Next, in order:

1. **§92.2's base-name defect.** One line, and it closes the `assert` class
   §88.3 actually named.
2. The name-matched builtin family (§92.3), which needs `shadows_user_definition`
   ported alongside it -- a symbol-table query, so the same shape of work as §70.


## 105. `adjust_float_arith` probed: unreached by the corpus, and its scalar half
## unreachable by construction

§103.2 argued from one measurement that `adjust_float_arith`'s scalar path is
dead. Probed properly, with an `fprintf` inside its `need_float_adjust` block and
a run over 90 tests sampled across all four C suites:

| | hits |
|---|---:|
| scalar float `+ - * /` | **0** |
| vector-of-float `+ - * /` | **0** |

The block is not reached by the corpus **at all**. It is reachable: a
hand-written GCC vector-of-float program hits it twice. Nothing in
`regression/esbmc`, `cstd`, `floats` or `esbmc-unix` does.

### 105.1 Why the scalar half cannot be reached

`adjust_expr` dispatches to `adjust_expr_binary_arithmetic` on the ids
`+ - * / mod bitand bitxor bitor`. A float-typed node with one of those ids would
have to come from the converter, and the converter does not produce one:
`double c = a + b;` is already `IEEE_ADD(a, b)` in the symbol table **under
`--clang-c-irep2-adjust-only`**, which never calls `adjust_float_arith` at all.
So the id-rewrite is a no-op for scalars, and the rounding-mode attachment below
it — guarded by an early `return` for vectors — is unreachable outright.

That is a construction argument, not just a probe result, and it is the half of
this that does not depend on corpus coverage.

### 105.2 What was shipped instead of a deletion

Not a deletion. `CLAUDE.md`'s C-Dead sub-mode wants the removed branch shown
unreachable, and §29.4 is explicit that "no corpus input reaches it" is an honest
negative rather than a proof — the vector half *is* reachable, so the arm cannot
go as a unit.

What the probe did expose is a **live path with no test**: vector float
arithmetic was lowered by an arm nothing in the corpus executed.
`regression/esbmc/gcc_vector_float_arith` pins all four operators, and a mutant
that drops the vector lowering fails it — so the path is now protected before
anyone tries to remove the arm around it.

### 105.4 Extended to C++, and one reason not to delete after all

§105.3 left the deletion to its own PR. Two further measurements, and it should
stay left.

`adjust_float_arith` is `clang_c_adjust`'s, which `clang_cpp_adjust` inherits, so
CUDA, CHERI-C and C++ all reach it. The probe extended:

| frontend | corpus | PROBE hits |
|---|---|---:|
| C | 90 tests, four suites | 0 |
| C++ | 25 tests of `esbmc-cpp` | 0 |
| C, C++ | a two-line `double c = a + b;` in each | 0 |

So the block is unreached across both frontends, not just C.

**And yet the rounding-mode `set` is not value-neutral to remove.** The arm sets
`rounding_mode` to `symbol_exprt(CPROVER_PREFIX "rounding_mode")`, i.e.
`__ESBMC_rounding_mode`; `migrate_rounding_mode` (`migrate.cpp:857`) defaults to
`c:@__ESBMC_rounding_mode` when the attribute is absent. **Two different symbol
names for the same thing**, and the unprefixed one is not the global the symbol
table holds.

That makes the deletion safe only on unreachability, not on equivalence — the
"harmless even if reached" argument does not hold, because if it were ever
reached the two spellings would differ. Worth recording on its own: an `ieee_*`
node built by this arm carries a rounding-mode operand naming a symbol that does
not exist, which would be a free variable at the solver. It never bites because
nothing reaches it, and it is one more reason the arm reads as vestigial rather
than as load-bearing.

The deletion therefore needs the C-Dead gates on a shared arm reached by four
frontends, of which this host can meaningfully exercise two. That is a Linux-CI
job, and it is recorded here rather than attempted.

### 105.3 For whoever takes the deletion

- The scalar id-rewrite and the rounding-mode `set` can go on §105.1's argument,
  leaving the vector branch.
- That is a legacy-side simplification, not a port, and it needs its own PR with
  the C-Dead gates; it does not block anything in Phase 6.
## 106. The `cstd` suite censused — and W4 has a witness

§101 said the unowned work would come from the suites never censused. `cstd`
is the first of them, measured with `symtab_sweep.sh`:

**134 of 142 differ** (94 %), against 78 of 120 (65 %) for `regression/esbmc`.
Over the first 60, by cause:

| cause | tests | owner |
|---|---:|---|
| `migrate_expr` renaming warning | 34 | #7093 |
| `assert` base name | 34 | #7087 |
| boolean cast on a condition | 21 | #7099 |
| array-to-pointer decay | 17 | #7098 |
| **`#cformat` char hint lost** | **14** | **— none** |

The four owned causes carry most of it, which is the useful half of the answer:
the suite is not differently broken, it is more densely affected by the same
things. `cstd` is libc-facing, so nearly every test calls `assert` and indexes a
buffer.

### 106.1 The new cause, and why it is W4

```
default:  signed char [14] str={ 'T', 'e', 's', 't', ' ', ... };
-only:    signed char [14] str={ 84, 101, 115, 116, 32, ... };
```

Same fourteen values; only the rendering differs. `string2array`
(`util/expr/string2array.cpp:25`) sets `#cformat` to `'T'` on each element as it
converts a string literal to a char array, and `c_expr2stringt::convert_constant`
(`util/lang/c_expr2string.cpp:1120`) prints `cformat` verbatim when present.
`scope-coupled-arith-assign-conversion.md` §20.1 item 7 already records that the
**IREP2 `c_typecastt` copy does not do `string2array`** — this is that gap, seen
from the printer.

That makes it **W4**, the wall §4 lists as "untouched, deferred": the
counterexample printer consuming the attributes. Until now W4 had no witness
outside the C++ printer. It has fourteen in `cstd` alone, reachable from C with a
single flag.

### 106.2 Why the obvious fix is not available, and what that says about B-4

`convert_constant` falls through to integer rendering only when `cformat` is
absent, so teaching it to render a char-typed constant as `'T'` would be
additive — the default path, where the hint is present, could not change.

It is still not available. A legacy `typet` cannot distinguish `char` from
`int8_t`: both are `signedbv` of width 8. What distinguishes them is
**`#cpp_type`** — which is one of the three W3 attributes. So inferring the
rendering from the type requires reading a W3 attribute to decide whether to stop
reading a W3 attribute.

**W3 and W4 are therefore coupled**, and §37's conclusion that B-4 "has no
viable executable content left" needs this qualification: the semantics half
(§33's four scalar spellings) and the presentation half are the same problem seen
twice, and neither can be closed while the other holds the type information. That
is a stronger statement than §37 makes, and it is the reason this section stops
at a finding rather than an arm.

### 106.3 What follows

- `esbmc-unix` (438 tests) and `floats` (102) are still uncensused; `cstd`
  suggests they will be dense in the same four owned causes.
- The `#cformat` class needs the W3 semantics/presentation split (§33) decided
  first. It is not an adjuster arm.
## 101. The symbol-table census, and what it says is left

§100.1 established that the adjuster's output is the symbol table, not the goto
program. That instrument is now in the harness — `irep2_symtab_dump` in
`scripts/irep2-migration/lib.sh` and `symtab_sweep.sh` beside it — so the
question "what does this pass still do differently" can be asked directly.

Over the first 120 tests of the §1.2 sample, `--clang-c-irep2-adjust-only`:
**78 differing symbol tables, 42 identical.** Blank-line-only differences are
ignored (`diff -B`): the printer varies its blank lines with block nesting,
which four tests differ by and nothing was adjusted differently in them.

Every remaining cause is owned by an open PR:

| cause | tests | owner |
|---|---:|---|
| `__builtin_expect` left as a call | 37 | #7086 |
| `migrate_expr` renaming warning | 22 | #7093 |
| array-to-pointer decay | 15 | #7098 |
| struct/union padding | 14 | #7100 |
| `for`-init hoist | 13 | #7105 |
| boolean cast on a condition | 9 | #7099 |
| conversion at a call argument | (in the residue) | #7091 |
| nested-function file name | (in the residue) | #7094 |

**There is no unowned adjuster work left in this sample.** That is the honest
answer to "what is the next arm": there isn't one here. Sixteen PRs carry the
whole of the measured gap, and the next material step is landing them, not
writing another arm (§99 gives the order and the two conflicts to expect).

What the census does *not* cover, and where the next unowned work will come from
when it is needed:

- **The other suites.** This sample is `regression/esbmc`; `esbmc-unix`,
  `cstd`, `floats` and the rest are in §1.2's corpus and have never been
  symbol-table censused.
- **`adjust_type` beyond padding** — symbol-type resolution and VLA size
  expressions (§96.2), unported and witnessless so far.
- **`goto_convert`**, which is where §98's remaining `DEAD` questions live, and
  which is not this scope's subject.
## 97. The baseline was two tests too high

§96's residue read left `intrinsic_unroll_misplaced_warning` and `github_746`
untagged. Neither is a divergence: their whole diff is run-to-run noise the
canonicaliser did not strip.

| test | the entire difference |
|---|---|
| `intrinsic_unroll_misplaced_warning` | `operational-model library (clib): ... deserialise 0.197s ...` vs `0.198s` |
| `github_746` | clang AST-dump node addresses in an error message (`0x8e529b0a8`) |

Both differ **against themselves** — the same binary, twice, on the same input.
§90.4 flagged the second and PR #7094 fixed three of that group at the source
(the nested-function transform's random file name, which was a real defect); this
is the remainder, which is diagnostic text and belongs in `irep2_canon` exactly
as §90.4 said.

`irep2_canon` now drops the clib summary line and rewrites hex addresses to
`0xADDR`.

**Every divergence count in §§90-96 is therefore two too high.** Master's
baseline is **200 of 297**, not 202. The per-arm deltas are unaffected — both
tests were noise on both sides of every A/B — but the absolute numbers should be
read with this correction, and re-measured counts from here use the fixed
canonicaliser.

The lesson is the one §90.4 already stated and this scope keeps re-learning: run
the same binary twice before believing a diff. It cost three sessions of
mis-attribution for the nested-function group, and two units of a headline
number here.
## 94. The name-matched builtin family, and the guard it needs

§90.3 deferred these because they are the spellings a program may reuse:
`is_name_matched_builtin`'s list, plus `sqrt` and the ordered-comparison
builtins. Ported here.

### 94.1 The matchers are shared, not copied

`compare_float_suffix`, `compare_unscore_builtin`, `is_abs_builtin_name`,
`is_name_matched_builtin` and `shadows_user_definition` moved from
`clang_c_adjust_expr.cpp` (where four of the five were `static inline`) into
`clang-c-frontend/builtin_names.{h,cpp}`, and the legacy member now delegates.

That is not tidying. §39.2 and `scope-coupled-arith-assign-conversion.md` §20
record two defects found in independently-written copies of
`c_typecastt` -- a dropped `floatbv` case and an unfolded constant cast -- each
of which produced a silent divergence for years. A second copy of "which
spellings are `isnan`" would be the same shape of bug, and the two passes must
agree by construction rather than by review.

### 94.2 The shadow guard is the load-bearing part

`abs`, `isinf`, `fabs` are names a program is free to define (#6904). The arm
runs behind `builtin_shadows_user_definition` for exactly that reason, and
`irep2_only_builtin_shadowed` -- a program with its own `fabs` returning 42 --
is the only test that detects the guard's removal. It is worth noting that this
test does *not* discriminate against the pre-arm control: with no lowering at
all the user's body is called too, and the verdict is the same. It is a mutation
test by nature, which §39.1's table anticipates.

### 94.3 What is declined, with the reason

- **`inf`/`huge_val`/`nan` under `--fixedbv`.** The legacy arm builds a bit
  pattern off `bv_width` rather than an `ieee_floatt`, and `constant_floatbv2t`
  takes an `ieee_floatt`. Declining leaves the call where this mode already had
  it, as §88.2's operand rule does.
- **`__builtin_sqrt`.** Neither pass lowers it: the legacy arm is
  `compare_float_suffix(identifier, "sqrt")`, which matches `sqrt`/`sqrtf`/
  `sqrtl` and *not* the `__builtin_` spelling. Reproduced on the default path
  before writing the arm's test to it; the test uses plain `sqrt`. Whether that
  asymmetry is intended is a question for the legacy pass, not for this port.
- **`sqrt`'s `py:` guard.** This pass is constructed only from
  `clang_c_languaget::typecheck`, so no Python symbol reaches it.

### 94.4 Result

| | before | after |
|---|---:|---:|
| `-only` divergence, 297-test sample | 131 | **129** |
| tests carrying this family's residue | **7** | **1** |
| regressions | -- | **0** |

The divergence column is again the wrong instrument (§92.2): the family's
residue is gone from six of seven tests, but most of those tests also diverge
for reasons this arm does not touch -- `math_exp02` is now down to the
unported boolean-condition cast alone. The one test still carrying family
residue is `github_2757`, whose `signbit` is *implicitly declared*, so it needs
the base-name fix of §92 as well. Two tests reach byte-identity outright:
`15_qurt_new` (`sqrt`) and `github_1226-2` (`__builtin_isgreaterequal`).

Five mutants, one rebuild each:

| mutant | killed by |
|---|---|
| shadow guard removed | `..._shadowed` only |
| `isnan` → `isinf` | `..._float_class` |
| `inf` ↔ `nan` constants swapped | `..._float_class`, `..._inf_abs` |
| `__builtin_isgreater` → `lessthan` | `..._ordered` |
| `signbit` → `popcount` | `..._signbit` |

The second of those is why the tests look the way they do. A first draft
asserted the predicates over `1.0` alone, and `isnan` → `isinf` **survived** it:
`!isnan(1.0)` and `!isinf(1.0)` are both true, so the test distinguished neither
node. The values are now chosen so the predicates disagree -- an infinity
separates `isinf` from `isnan` and `isfinite`, a zero separates `isnormal` from
`isfinite`.

### 94.5 An unrelated abort the tests surfaced

Asserting all four classification predicates *and* `signbit` over the same
function aborts in the solver, on the **default path**, under Bitwuzla:

```
Assertion failed: (a->sort->id == SMT_SORT_BOOL), function mk_not,
file bitwuzla_conv.cpp, line 346.
```

Each assertion passes alone and in pairs; removing the `signbit(-d)` line clears
it. `signbit2t` is `int32`-typed, so a `not` over it is the suspect, but the
combination is what triggers it and that is not explained yet. Nothing to do
with this arm -- recorded because it cost a test-writing iteration, and because
`signbit` now lives in its own test file for this reason rather than by design.

## 95. Status

`-only` on the 297-test sample: **129 of 297**. §90 + §92 + this arm together
take it from 201.

Gates: 66 of 67 in the
`irep2_only|complex_|gcc_popcount|gcc_bswap|math_|github_*|15_qurt` slice green;
`github_2572_2` fails identically on master (`--z3 --ir-ieee`). Whole-suite gate
still owed (§91).

Next:

1. The **CPROVER intrinsic family** (`same_object`, `POINTER_OFFSET`,
   `POINTER_OBJECT`, ...), which is the last block of `do_special_functions`
   with corpus traffic.
2. The `missing renaming delimiters` warning (§92.2). Checked while picking this
   slice: the warning comes from `sym_name_to_symbol`, shared by every frontend,
   and firing it is inherent to migrating a body before its implicit callee is
   declared. Not the one-liner §93 implied.

## 96. The divergence set, censused -- and §95's next target was the wrong one

§95 named the CPROVER intrinsic family next, on the strength of its appearing in
the leftover call-position census. Sized properly before starting it, it is
**three tests**. Classifying all 129 remaining divergences by cause first, as
§89 asked and §92.2 insisted on:

| cause | tests |
|---|---:|
| `migrate_expr` "missing renaming delimiters" warning | 31 |
| array-to-pointer decay | 29 |
| `assert` left as a FUNCTION_CALL (§92, PR pending) | 25 |
| usual arithmetic conversions | 17 |
| function-to-pointer decay | 12 |
| CPROVER intrinsics (`same_object`, `POINTER_OFFSET`, ...) | ~3 |

`DEAD` placement appears in 43, but in **no** test is it the whole diff: it is
cascade, an extra declaration's shadow rather than a cause. The census asks
whether a tag ever appears alone before it is worth a slice, and that check is
what kept a 43-row entry off this list.

### 96.1 The arm

The three conversion rows above are one mechanism -- `gen_typecast_arithmetic`,
which `clang_c_adjust` calls from `adjust_expr_rel` and
`adjust_expr_binary_arithmetic` -- and its IREP2 counterpart is the shared
`c_implicit_typecast_arithmetic(expr2tc &, expr2tc &, ns)` that §38.1 named as
Phase 4's already-extracted helper. This ports the **relational** call site
only. `adjust_expr_rel`'s other half, `expr.type() = bool_type()`, has nothing to
do: IREP2's comparison kinds are bool-typed by construction.

Binary arithmetic is deliberately not in this slice. That call site also carries
`adjust_float_arith`'s `ieee_*` promotion, which is the defect
`scope-coupled-arith-assign-conversion.md` §17 spent three sections on for
Python (#6839); it deserves its own gates rather than a shared A/B with this.

### 96.2 Result

| | before | after |
|---|---:|---:|
| `-only` divergence, 297-test sample | 129 | **122** |
| regressions | -- | **0** |

Seven tests reach byte-identity: `fam_true_2`, `github_263`,
`simplifier-equality-fail`, `simplifier{17,19,21}_no`, `simplifier4`.

The per-cause counts barely move (array decay 29 → 26, promotion 17 → 17), and
that is the honest reading: the same conversion is owed at assignment, at binary
arithmetic and at call arguments, and this arm covers one operator position of
several. What it does clear, it clears completely.

### 96.3 The tests were written twice

The first pair asserted `-1 < 1u` and a local `int a[4]` compared against a
pointer. Both passed **against the control binary**, which is to say they proved
nothing: clang already inserts the arithmetic conversions for `i < u`, so the
adjuster has no work there, and the local-array shape does not reach this arm at
all (it aborts under `-only` in `irep2_utils`' width assertion, before and
after).

Rewriting them at the shapes the arm demonstrably fires on -- taken from the
diffs of the seven tests it cleared, not from what the C standard suggests it
ought to do -- gives an array-typed **struct member** compared against a
pointer (`fam_true_2`'s shape) and a comparison whose operands are both
**boolean** (`simplifier17_no`'s). Both now differ from the control and match
the default path exactly.

| mutant | killed by |
|---|---|
| arm absent (the control binary itself) | both |
| only `op0` written back, `op1` left alone | `..._bool_operands` |

The second mutant survived the first draft of `..._bool_operands`, whose regex
pinned only the left operand's cast. A comparison arm has two operands and a
test for it must say so.

### 96.4 A soundness note from `github_263`

Worth recording separately because it is not a formatting difference. Comparing
two rows of a 2-D array, `a[0] < a[MAX-1]`, the default path emits one assertion
on the decayed pointers. Under `-only` before this arm it emitted **array-bounds
claims instead** -- different properties, not a differently-spelled one. A
divergence census that only counted tests would have scored that the same as a
missing cast.

## 97. Status

`-only` on the 297-test sample: **122 of 297** (201 at the start of this
sequence; §90 → 131, §94 → 129, this arm → 122).

Gates: 694 unit tests green; 130 of 130 in the
`irep2_only|simplifier|fam_|github_263|complex_` slice green. Whole-suite gate
still owed (§91) -- retried at three strides this session and it does not
complete inside the 5-minute cap under this machine's load.

Next, by the §96 census rather than by guess:

1. **Function-to-pointer decay** (12 tests) -- the same conversion mechanism at
   argument position, `adjust_function_call_arguments`' `gen_typecast`.
2. **The conversion at assignment and binary arithmetic** (17), taking
   `adjust_float_arith` with it.
3. The CPROVER intrinsics (~3), which are cheap but small.


## 95. A statement's controlling expression, and the census re-run on master

Re-censusing the 202 tests that diverge on current master, tagged by cause:

| cause | tests | owner |
|---|---:|---|
| `__builtin_expect` | 112 | PR #7086 |
| `migrate_expr` renaming warning | 31 | PR #7093 |
| array-to-pointer decay | 29 | PR #7098 |
| `assert` as a FUNCTION_CALL | 25 | PR #7087 |
| function-to-pointer decay | 21 | PR #7092 |
| **boolean cast on a condition** | **19** | **— none** |
| usual arithmetic conversions | 15 | PR #7097 |

Every large cause had an open PR except one, which is what this arm takes.
Tagging by owner rather than by symptom is the reading §98.2 and §104.3 both
argued for; done this way it selects the next task without a guess.

### 95.1 The arm

`adjust_ifthenelse`, `adjust_while` (which also serves `dowhile`) and
`adjust_for` each apply `gen_typecast_bool` to the statement's controlling
expression, because goto_convert's branch lowering wants a boolean guard and
clang leaves `if (a)` with `a` an `int`. `switch` is deliberately not in the
list: its selector is an integer.

This is the statement-level counterpart of §84's `adjust_if_expr`, which does
the same for the *ternary* operator's condition. The two are separate arms
because they are separate legacy functions over separate node kinds.

`code_for2t`'s condition is optional (`for (;;)`), which the nil check covers;
the other three always have one.

### 95.2 Result

| | master | with the arm |
|---|---:|---:|
| `-only` divergence, 297-test sample | 202 | **201** |
| tests with a boolean-cast difference | 19 | **16** |
| regressions | -- | **0** |

The headline moves by one and the class it targets by three, which is the now
familiar gap: a test carrying this difference usually carries others too, and
only clears when the last of them goes. The class count is the honest measure of
this arm; the divergence count measures the backlog.

| mutant | killed by |
|---|---|
| arm absent (master) | `..._statement_conditions` |
| `code_for2t` dropped from the statement list | `..._statement_conditions` |

The second mutant is there because `for` reaches its condition through a
different field than the other three, so a list that omits it still compiles and
still passes an `if`-only test.


## 108. Conversions at call arguments -- and where the function decay actually lives

§97 named function-to-pointer decay next, on the strength of §96's 12-test row.
Porting `adjust_function_call_arguments`' conversion half clears ten tests and
**does not clear that row**, which is the interesting part.

### 108.1 The arm

For each argument, convert to the parameter type; where the parameter list is
exhausted -- a variadic argument -- only the array decay is owed, to `void *`.
That is `gen_typecast(ns, op, argument_type)` and its `is_array_like` fallback,
which is `c_implicit_typecast(expr2tc &, type2tc, ns)` on this side.

`adjust_function_call_arguments`' other half, the `__ESBMC_assigns_impl`
guard that keeps a pointer-to-array `&a` intact (#7010), is **not** ported and
does not need to be yet: it exists to undo `adjust_address_of`'s `&a` → `&a[0]`
rewrite, and that arm is unported, so there is nothing to undo. Checked rather
than assumed -- the eight `__ESBMC_assigns` tests were run both ways, and the
one that differs (`github_4219_..._knownbug`) produces the identical 14-line
diff on the two preceding binaries as well. **When `adjust_address_of` is
ported, this guard must go with it in the same commit.**

### 108.2 The 12-test row was mis-attributed

The census tagged those tests by their symptom -- `&f` in the default dump, bare
`f` under `-only`. The cause is not an argument conversion. Neither copy of
`c_typecastt` decays a bare `code`-typed operand; the sugar is applied at the
symbol, in `clang_c_adjust::adjust_symbol`
(`clang_c_adjust_expr.cpp:366-372`): a symbol whose type is `code` is wrapped in
an implicit `address_of`. So the row belongs to a symbol-level arm and is
untouched by this one, which a probe showed directly -- `apply(callee, 1)` under
`-only` before and after.

Worth stating as a method point, because §96 built its work list on these tags:
a symptom-tagged census names *where a difference shows up*, not what produced
it. Both readings were needed here, and the second only came from probing.

### 108.3 Result

| | before | after |
|---|---:|---:|
| `-only` divergence, 297-test sample | 122 | **112** |
| array-to-pointer decay | 26 | **16** |
| usual arithmetic conversions | 17 | **12** |
| function-to-pointer decay | 20 | 11 |
| regressions | -- | **0** |

The two shapes it fixes, taken from the cleared tests rather than from the
standard: an array passed to a **declared-but-undefined** function with a
pointer parameter (`wchar_model`'s `wcscpy(&dst[0], ...)` -- clang inserts the
decay itself when the callee is defined in the same file, which is why the first
probe found nothing), and the scalar conversion on an `__ESBMC_assume` argument
(`github_1620`'s `ASSUME (_Bool)((signed int)(x != 0))`).

Both tests are goto-shape rather than verdict tests, deliberately: the callees
that exhibit this are bodiless, so there is no verdict to move.

| mutant | killed by |
|---|---|
| arm absent (the control binary) | both |
| declared-parameter branch disabled | `..._call_arg_decay` |
| variadic branch disabled | `..._call_arg_variadic` |

## 109. Status

`-only` on the 297-test sample: **112 of 297** (201 → 131 → 129 → 122 → 112
across §90, §94, §96 and this arm).

Gates: 694 unit tests green; 108 of 108 in the affected slice green. Whole-suite
gate still owed (§91); it has not completed inside the 5-minute cap at any
stride tried across three sessions.

Next:

1. **`adjust_symbol`'s function-designator sugar** (§108.2) -- now located
   exactly, and it owns the 11-test decay row.
2. The conversion at assignment and binary arithmetic (12), with
   `adjust_float_arith`.

3. `adjust_address_of`, which must bring #7010's assigns guard with it (§98.1).

3. `adjust_address_of`, which must bring #7010's assigns guard with it (§108.1).

## 100. The function-designator sugar, and the cast that was not a conversion

§108.2 located this arm: `clang_c_adjust::adjust_symbol` wraps a symbol whose
type is `code` in an implicit `address_of`, and `adjust_side_effect_function_call`
strips that sugar back off when the call is direct. Both halves port; the
`implicit` bit `address_of2t` already carries (#6912, added for exactly this)
is what tells `f(x)` from a user-written `(&f)(x)`.

### 100.1 A spurious cast, and a predicate that already existed

With the sugar in place and nothing else, the argument came out as
`apply((signed int (*)(signed int))(&callee), 1)` where the default path has
`apply(&callee, 1)`. The cast comes from §108's argument conversion:
`implicit_typecast_followed` reaches its "very generous: between any two
function pointers it's ok" branch, and then casts anyway, because the decision
after that branch is a bare `src_type == dest_type` -- and IREP2's
`code_type2t::fields` includes `argument_names`, which hold *symbol ids*, not
source spellings. Naming both parameters `x` does not make them equal; only
being the same declaration does.

This is #6749's defect in a second place, and the predicate written for it --
`same_function_pointer_ignoring_argument_names`, with C11 6.7.6.3p15 and C++
[dcl.fct]p5 in its comment -- was `static` in `dereference.cpp`. Moved verbatim
to `irep2/irep2_utils.h` and used from both, on §94.1's reasoning: a second copy
of "when are two function types the same" is the shape of bug this file keeps
finding.

Not done here, and worth its own decision: the same `src_type == dest_type`
sits in `implicit_typecast_followed` itself, so *every* consumer of the IREP2
`c_typecast` -- `python_adjust` included -- can still be handed this cast. Fixing
it there would be one condition and would need its own A/B over the Python
corpus, which is why it is named rather than done.

### 100.2 Applied from the parent, not at the symbol

`address_of2t`'s constructor asserts its operand is not another `address_of`
(`irep2_expr.h:1417`). The legacy pass builds `&(&f)` for a user-written `&f`
and collapses it in `adjust_address_of`; IREP2 cannot build it at all. So the
sugar runs over a node's *operands*, skipping the case where the node is itself
an `address_of` -- the nesting is never constructed rather than constructed and
undone.

Order matters and cost an iteration: the arm has to run **before**
`adjust_call_callee`, because that is what reads the sugar to decide the call is
direct. Placed after it, every direct call in the corpus kept its `&` and the
probes went from 0 differing lines to ~300.

### 100.3 Result

| | before | after |
|---|---:|---:|
| `-only` divergence, 297-test sample | 112 | **105** |
| function-to-pointer decay divergences | 11 | **3** |
| regressions | -- | **0** |

Default path unchanged: of the 105 tests diverging under both binaries, all but
the four §90.4 self-nondeterministic ones produce byte-identical default-path
dumps. That check matters more than usual here, because this is the first arm in
the sequence to touch a file outside the frontend.

| mutant | killed by |
|---|---|
| arm absent (the control binary) | `..._fn_designator` |
| the `&(&f)` guard removed | `..._fn_designator_call` |
| the implicit-`address_of` strip removed | `..._fn_designator` |

## 101. Status

`-only` on the 297-test sample: **105 of 297** (201 at the start of this
sequence).

Remaining causes, re-censused: the `migrate_expr` renaming warning (31, and
§92.2 explains why it masks arms rather than being one), array-to-pointer decay
(16), usual arithmetic conversions (12), function-to-pointer decay (3).

Next:

1. The conversion at **assignment and binary arithmetic** (12), which brings
   `adjust_float_arith`'s `ieee_*` promotion with it.
2. `adjust_address_of`, which owns most of the remaining array decay and **must
   carry #7010's assigns guard** (§108.1).
3. The `src_type == dest_type` decision in `implicit_typecast_followed` (§100.1),
   which is shared and needs a Python A/B.
## 107. The conversions at binary operators and at assignment

§96's census attributed 17 tests to "usual arithmetic conversions". §97 named
them next. Both arms are ported here, **master-based rather than stacked** —
the six-deep hop-off chain is unmerged, and neither arm needs it.

### 107.1 Clang does most of this already

The binary-arithmetic arm — `gen_typecast_arithmetic` over the operands, then
the node's own type — fires on **one** test in the 297-test sample. That is not
a bug in the port: clang inserts the usual arithmetic conversions into its own
AST for the ordinary integer and floating types, so `gen_typecast_arithmetic`
has nothing left to do. The shape where it does is a **bit-precise** operand:

```c
_ExtInt(10) x, y;
_ExtInt(10) z = x + y;      // -only gave `x + y`, the default path
                            // `(signed int)x + (signed int)y`
```

Proven live before shipping, per §90.4's rule: the arm changes
`bitvector_04`'s dump and nothing else in the sample.

The assignment arm is wider, and its most visible effect is not arithmetic at
all — it is **array-to-pointer decay at an assignment statement**:

```c
int a[3]; int *p;
p = a;                      // -only gave `ASSIGN p=a`
```

Clang inserts the decay for the *initialiser* `int *p = a;`, which is why the
first draft of the test passed against the control and proved nothing. The
separate assignment statement is the shape that reaches the adjuster —
`r_ok18`'s, found by reading what the arm changed rather than by reasoning about
C.

### 107.2 What is left out

- **`adjust_float_arith`'s `ieee_*` promotion**, which the same legacy function
  calls for `+ - * /`. That is the defect
  `scope-coupled-arith-assign-conversion.md` §17 spent three sections on for
  Python (#6839) and it deserves its own gates rather than a shared A/B.
- **Compound assignment** (`assign+`, ...). `adjust_side_effect_assignment`
  gives those a complex lowering of their own; only the plain `assign` case is
  here.

### 107.3 Result

| | master | with both arms |
|---|---:|---:|
| `-only` divergence, 297-test sample | 201 | **200** |
| tests whose `-only` dump changes | -- | **4** |
| regressions | -- | **0** |

One test reaches byte-identity (`github_286_8`); three more move toward it
(`bitvector_04`, `github_65`, `r_ok18`). A one-test gain is a fair return for
this arm and is reported as such: the census row it was drawn from counted
*symptoms at the stack tip*, and §98.2 already recorded that a symptom-tagged
census names where a difference shows up rather than what produced it. This is
the second time that distinction has cost a prediction.

| mutant | killed by |
|---|---|
| arm absent (master, for both) | both tests |
| assignment arm disabled | `..._assign_array_decay` only |
| binary-arithmetic arm disabled | `..._bitint_arith` only |


## 94. `adjust_address_of`'s array decay, and a guard that had no witness

`&a` on an array is `&a[0]`: the pointer designates the first element, not the
array object. Ported here; the pointer's subtype follows the element.

Master-based rather than stacked, like the arms before it. Baseline re-measured
against current master first, since #7028 and the contracts series landed since
§96: **202 of 297** diverge, not 201.

### 94.1 The #7010 guard, written and then removed

`adjust_function_call_arguments` undoes this very rewrite for
`__ESBMC_assigns_impl` arguments, because an assigns clause names an lvalue and
the decay makes `&a` indistinguishable from a clause naming the first element —
the frame silently shrinks (#7010). §98.1 flagged that the guard must travel
with this arm.

It was written: a flag set while walking a clause's subtree, suppressing the
decay inside it. Then measured, and **removed**, because it never fires:

- With the guard disabled, all 14 `function_contract` assigns tests still agree
  between the default path and `-only`.
- On a purpose-built array-typed clause (`__ESBMC_assigns(a)` for `int a[4]`),
  this arm changes **nothing**: the `-only` goto dump is byte-identical to
  master's. The `&a` the macro expands to never reaches the arm as an
  `address_of` over an array.

An arm no input executes is the trap §90.4 records, and the same reasoning that
removed the level2 warning branch in §102.1 applies: a guard whose reachability
cannot be shown is worth less than the simpler code. If the shape is ever
produced, the guard comes back with the input that produces it.

### 94.2 A pre-existing divergence found while checking

The purpose-built clause above **already diverges on master** under `-only`:

```c
int a[4];
__ESBMC_contract void bump(void) { __ESBMC_assigns(a); a[3] = 7; }
```

`--enforce-contract bump` is SUCCESSFUL on the default path and FAILED under
`-only`, on master, before this arm. So `-only` mishandles an array-typed
assigns clause for a reason that is not the decay and is not yet identified.
Recorded rather than chased: it is the first contracts-specific `-only`
divergence this scope has seen, and the `function_contract` suite is not
registered on macOS (`gotcha`: run its `test.desc` by hand, as here).

### 94.3 Result

## 100. §98 was wrong: the hoist is the fix, and the bug was in my port

§98.2 reported that porting `adjust_for`'s block hoist "leaves the misplaced
`DEAD` exactly where it was" and concluded it was not the fix. That conclusion
was wrong. The hoist *is* the fix; the port had a bug that made it look
otherwise.

### 100.1 The instrument §98 should have used

§98 compared goto programs. The adjuster's output is the **symbol table**, and
`--symbol-table-only` shows it directly:

```
default:  {  signed int i=0;    for(; i < 3; i++;) s += i;  }
-only:       for(signed int i=0; ; i < 3; i++;) s += i;
```

One command, and the hoist is visibly the difference. Three iterations of this
scope inferred adjuster behaviour from goto programs — two stages downstream —
when the pass's own output was one flag away. That is the reusable lesson.

### 100.2 The bug

`f.init` is itself block-shaped, and the first port made it a single operand of
the new wrapper:

```
default:  {  signed int i=0;   for(...) ... }
first port: { { signed int i=0; } for(...) ... }
```

The inner block ends the declaration's scope at its own closing brace, so `i`
is DEAD before the loop that reads it — the very symptom the arm was meant to
fix, reproduced by the arm. `clang_c_adjust` moves the init *operand* into the
wrapper, so its declaration sits directly there; the port must splice a
block-shaped init rather than nest it.

### 100.3 Result

| | master | with the arm |
|---|---:|---:|
| `-only` divergence, 297-test sample | 202 | **200** |

| regressions | -- | **0** |

Cleared: `github_159_postdecrement_fail`, `github_159_preincrement_fail`, whose
shape is `&Q` on a global array in an initialiser — which is what the test pins.

The conditional distribution the legacy arm also does — `&(c ? a : b)` into
`c ? &a : &b`, which #6291 needs for the pointer analysis to resolve either arm
— is **not** ported: no corpus input reaches it under this flag, and porting it
would be the same unwitnessed instrumentation §94.1 just removed.

| mutant | killed by |
|---|---|
| arm absent (master) | `..._address_of_array` |


| differing lines corpus-wide | 33 767 | **25 341** |
| regressions | -- | **0** |

The line count is the number that matters here: **−8 426, a 25 % reduction**,
the largest of any arm in this sequence, and all of it was hidden behind the
nesting bug. Two tests clear outright (`github_1067`, `github_286_2`); the rest
converge substantially because a wrong scope perturbs every location and
destructor placement after it.

| mutant | killed by |
|---|---|
| arm absent (master) | `..._for_scope` |
| splice replaced by nesting (the original bug) | `..._for_scope` |

The second mutant is the one worth having: it is not a hypothetical, it is the
code that shipped in the §98 measurement.

### 100.4 Consequences for #7102

PR #7102 records §98's conclusion and adds
`irep2_only_for_scope_knownbug` as KNOWNBUG. Both are now wrong: the KNOWNBUG
passes with this arm, so `testing_tool.py` would exit 77 (unexpected pass). When
these two land, #7102's test must become CORE or be dropped in favour of
`irep2_only_for_scope` here, and §98.2's "not the fix" must be read together
with this section.

## 96. The type symbols were never walked

§95's census left 29 tests with no cause any open PR owned. Reading five of
them found three showing the same thing: struct and union types in the `-only`
symbol table have **no padding**.

```
default:  struct s { signed int a; signed char c; unsigned _ExtInt(24) anon_pad#2; }
-only:    struct s { signed int a; signed char c; }
```

### 96.1 Why, and why it is not cosmetic

`clang_c_adjust::adjust()` walks the symbol list twice over: once for every
**type** symbol, through `adjust_type`, which pads a complete struct or union
(`clang_c_adjust_expr.cpp:1006`); and once for values. `clang_c_adjust_irep2`
only ever walked values — `if (!s->is_type && s->get_value().is_not_nil())`.
Type symbols were skipped entirely, so nothing padded them.

That is a layout difference, not a spelling one. The symbol table's type is what
ESBMC sizes objects and computes member offsets from, so every hole in a
`-only` layout shifts the members after it. It is the most consequential
divergence this scope has found, and it was invisible in the census because the
tests carrying it were tagged by whatever *else* they also diverged on.

### 96.2 Reuse, not reimplementation

`add_padding` is shared (`clang-c-frontend/padding.h`), operates on `typet`, and
`adjust_type`'s own `#ifndef NDEBUG` block asserts it is idempotent — it re-pads
a copy and requires the result to be equal. So the arm calls it rather than
growing a second layout algorithm over `type2tc`; §94.1 and §100.1 are two
records of what a second copy of a shared rule costs here.

Only the padding half of `adjust_type` is ported. The rest — resolving a
`symbol` type through the symbol table, and adjusting a VLA's size expression —
has no witness in the corpus under this flag, and §94.1 is the standing reason
not to ship an arm without one.

### 96.3 Result

| | master | with the arm |
|---|---:|---:|
| `-only` divergence, 297-test sample | 202 | **193** |
| regressions | -- | **0** |

Nine tests reach byte-identity: `github_{133_flex_array,170,345_false,357,6950_fail}`,
`github_732-1-align_check`, `github_963-no-union`, `overflow_24`,
`time_h_localtime_r_null_fail`. The best single-arm result on master since the
sequence began, and the reason is that padding is a *precondition* for the rest
rather than one more spelling: several of those tests had no other cause left.

The union half is exercised by the corpus rather than by the test —
`github_345_false` is a bitfield union and clears here — while the test pins the
struct cases directly, including interior padding (`char` then `int`) as well as
tail padding.

| mutant | killed by |
|---|---|
| arm absent (master) | `..._struct_padding` |

`sizeof(struct s)` is **not** a usable probe: clang folds it in its own AST, so
it reads 8 on both paths and a verdict test built on it passes against the
control. The first draft of this test did exactly that and proved nothing.

## 111. The two aborts §110.4 named, and what fixed one of them (2026-08-22)

§110.4 put the two tests where the hop-off emits no symbol table at all ahead of
the remaining spelling differences: a hard stop is not a divergence you can
measure. Neither is the by-name union tag the class comment on
`clang_c_adjust_irep2` documents — that attribution in §110.4 is wrong, and the
two tests fail for two unrelated reasons.

### 111.1 `builtin_memcpy`: an array operand of pointer arithmetic

```
Assertion failed: (p2 || (is_bv_type(t) == is_bv_type(v1->type) &&
  t->get_width() == v1->type->get_width())), assert_arith_2ops_consistency
```

Reduced to `char a[9]; char *p = a + 1;` — nothing to do with `memcpy`.
`clang_c_convert` drops the decay cast on purpose
(`case clang::CK_ArrayToPointerDecay: break;`) and leaves `clang_c_adjust` to
insert the `&a[0]`. Under the flag that pass does not run, so `migrate_expr`
builds `add2t` with a pointer result and an *array* operand, and `add2t`'s
invariant is a post-adjust one.

The adjuster cannot fix it: the node has to be constructed before any pass can
walk it. So the decay goes where the node is built —
`decay_array_operand` in `migrate.cpp`, applied to both operands of the `plus`
and `minus` arms. On the legacy path the operands are already `&a[0]`, so it is
a no-op there; the C++, C and 400-test Python slices are unchanged.

Only `+` and `-` assert on an array operand. `a > q`, `a == q` and `a[1]` all
migrate today — measured, not assumed.

The `-` case needed the guard widened. Keying the decay on a pointer *result*
type fixes `a + 1` and not `a - q`, whose result is `ptrdiff_t`: C11 6.3.2.1p3
decays an array operand in either position regardless of what the operator
returns, and C has no array arithmetic for the unconditional form to catch.

`--goto-functions-only` on the new test is byte-identical between the two paths,
which is the §1.3 gate. The *symbol table* still prints `a + 1` where legacy
prints `&a[0] + 1`, and that is not this defect: the pass writes a value back
only when it changed it (`value != before`), and `before` is now already
decayed, so nothing is written and the unadjusted legacy value survives in the
table. Closing that means comparing the write-back against the legacy value
rather than the migrated one — a change to the pass's write-back policy, not
another arm.

| mutant | killed by |
|---|---|
| decay absent (master) | `irep2_only_array_arith_decay`, `..._memcpy` |

### 111.2 `cwe_uninit_array_vla` is still open, and it is a segfault

`int a[n];` with a runtime `n` segfaults under the flag with no diagnostic —
`int main(int argc, char **argv){ int n = argc; int a[n]; return 0; }` is
enough, and the array need not be read. It survives this patch, so it is a
second cause and not a second symptom.

**§112 corrects this: the VLA is not the trigger.** The reduction kept the VLA
because it stopped as soon as the crash reproduced, and the crash reproduces on
`int main(int argc, char **argv){ return 0; }` with no array at all. Reduce past
the construct you came in for.

## 112. `argc'`/`argv'` — a symbol-table side effect the sole adjuster owed
## (2026-08-22)

§111.2 named the VLA as the second abort's cause. It is not. Reduced past the
construct the test was named for:

```c
int main(int argc, char **argv) { return 0; }
```

segfaults under `--clang-c-irep2-adjust-only`. No array, no VLA, no body. The
same program on the default path is fine.

`clang_c_main` looks the symbols up without a null check
(`const symbolt &argc_symbol = *ns.lookup("argc'");`, clang_c_main.cpp:157), and
they are created by `clang_c_adjust::adjust_argc_argv`. Under the flag the
legacy pass does not run, so the lookup dereferences null.

That is the same class as `declare_implicit_callee` (§70): a **symbol-table side
effect** rather than an expression rewrite, so it belongs to whichever pass is
in charge rather than to the dispatcher. Extracted as a free
`declare_argc_argv(contextt &, const symbolt &)` and called from both, so there
is one definition rather than a port to keep in step.

### 112.1 What this says about the census

The two "aborts" §110.4 ranked ahead of the spelling causes were:

| test | actual cause | closed by |
|---|---|---|
| `builtin_memcpy` | undecayed array operand of `+` | §111.1 |
| `cwe_uninit_array_vla` | missing `argc'`/`argv'` | this section |

Neither was the by-name union tag §110.4 attributed them to, and neither had
anything to do with the construct its test is named for. Both attributions came
from reading the test name and the class comment instead of reducing. The rule
that follows is the one §104.2 already states for tags and applies equally to
crashes: read the residue, do not infer it.

Stride-8 sample over `regression/esbmc`, symbol tables, blank lines ignored:

| branch | same | differing |
|---|---:|---:|
| master `595f52b025` | 90 | 138 |
| §112 alone, on master | 97 | 132 |
| §111.1 + §112 stacked | 100 | 129 |

**These two figures are inflated and §113.2 corrects them to 94/134 for the
stacked pair.** The sample is a stride over the sorted test list, so adding
regression test directories -- which these patches do, and which are
byte-identical by construction -- changes *which* tests are sampled. Compare
across branches only on a test list pinned to one commit.
| §111.1 + §112 stacked | **100** | **129** |

`cwe_uninit_array_vla`'s symbol table is byte-identical once it runs, and so is
the reduction's; the three-argument `envp` form is byte-identical too.

| mutant | killed by |
|---|---|
| side effect absent (master) | `irep2_only_argc_argv`, `..._argc_argv_envp` |

### 112.2 A pre-existing abort found alongside, and not fixed here

`int main(int argc) { return 0; }` -- one argument -- aborts on
`assert(false)` at clang_c_main.cpp:399 on **both** paths. It is not a hop-off
defect and it is not in this scope; recorded so the next reduction does not
mistake it for one.

### 112.3 Next

The remaining causes are all spelling differences again, and §110.1's table
still ranks them. The two that are not printer artefacts are the
function-pointer cast at a call argument and the coupled arith-assign
(`scope-coupled-arith-assign-conversion.md`).

## 113. The compound assignment, and a census that was measuring itself
## (2026-08-22)

§112.3 left two non-printer causes. This closes the second and disqualifies the
first.

### 113.1 The compound assignment was a third abort, not a spelling difference

`compound_assign_narrow_overflow` appears in §110.1's untagged residue as
`(signed int)b += a;` versus `b += a;` — a text difference. It is not: with
`--goto-functions-only` the hop-off *aborts*, on the same
`assert_arith_2ops_consistency` §111.1 met. The symbol-table census could not
see it because the abort happens in `goto_convert`, two stages after the pass
whose output that census reads. **A symbol-table census under-reports by
construction; a differing text there may be a crash further on.**

`adjust_plain_assignment` ports only the `"assign"` case of
`clang_c_adjust::adjust_side_effect_assignment`, and its comment says the
compound spellings were "left where this mode already had them". Where they
were was: unconverted. C11 6.5.16.2p3 makes `b op= a` equivalent to
`b = b op (a)`, so a `char` target promotes to `int` before the operation, and
without that promotion `goto_convert`'s lowering builds `add2t` on a `char` and
an `int`.

Measured across all ten spellings on `char b; int a; b op= a;`:

| spelling | before |
|---|---|
| `+= -= *= /= %=` | **abort** |
| `&= \|= ^=` | diverge, no abort |
| `<<= >>=` | already byte-identical |

So the port is the tail of the legacy arm — the arithmetic conversion on *both*
operands — and not its shift branch, which returns early after promoting only
the right operand and which the corpus shows is already the migrated shape.
All ten are byte-identical afterwards, in the symbol table and in the goto
program.

### 113.2 The census had started measuring its own tests

§112's table reported the stacked pair at 100 same / 129 differing. On a test
list pinned to `595f52b025` it is **94 / 134**. The difference is not drift: the
sample is `awk 'NR%8==0'` over the sorted `test.desc` list, and every patch in
this sequence adds regression directories which are byte-identical by
construction. Adding them both inserts guaranteed-same entries and shifts which
other tests land on a stride position.

Corrected, on the pinned list (230 tests):

| branch | same | differing |
|---|---:|---:|
| master `595f52b025` | 90 | 138 |
| + §111.1 + §112 | 94 | 134 |
| + §113.1 | **95** | **133** |

§113.1 gains one test on this sample, which is the honest number: the sample
holds few narrow-target compound assignments. Its value is the three abort
classes it removes, not the sample delta.

### 113.3 The function-pointer cast is a "do not mirror", not a gap

§110.4's other non-printer cause: `atexit((void (*)())(&free_g2))` versus
`atexit(&free_g2)`, and unlike §110.2's `(void)0` this one *does* reach the
goto program. It is still not work.

Instrumented at the conversion site, `arg->type == params[i]` is **true**: the
argument is `void (*)(void)` and the parameter `void (*)()`, and `migrate_type`
maps both to the same `code_type2t` — same empty argument vector, same return
type, same ellipsis flag. The legacy pass emits a cast because the *legacy*
types differ; in IREP2 the cast is the identity, and no pass reading IREP2 can
know it is owed.

Confirmed twice over: disabling `same_function_pointer_ignoring_argument_names`
at that site does not restore the cast, so the §100.1 guard is not suppressing
it; and `atexit-1` returns the same verdict on both paths under its own flags.

Emitting an identity typecast to match the legacy printer is §110.2's mistake
with a different node. Closing it for real means `code_type2t` carrying the
prototyped/unprototyped distinction, which is a representation change and needs
its own justification.

### 113.4 Next

Every cause §110.1 names is now either closed or argued not to be work, except
the printer-only set (`+1` vs `1`, the float literal suffix, block indentation)
and the `migrate_expr` renaming warning. The measured 133 residue on the pinned
sample wants a fresh cause census before another arm is written — the old one
is stale, and §113.1 shows it was reading the wrong stage.

## 120. The verdict census reaches zero — and the last live row was a comma
## (2026-08-23)

(§120: PRs #7266, #7271, #7274, #7275 and #7278 are in flight against this file
and claim §115-§119.)

With those five merged locally, both censuses were re-run on the pinned
stride-8 list (226 C sources).

| | before the series | after |
|---|---:|---:|
| goto program differing | 24 | **6** |
| **verdict differing** | 3 | **0** |

Every test in the sample now returns the same verdict on both paths. That is
the first time this scope has measured zero on the instrument that matters.

### 120.1 §119.4's row was already closed

`github_3487`'s `bad_optional_access` is `member2t::do_simplify` calling
`.value()` on an unresolvable component number
(`expr_simplifier.cpp:1327`). The member it fails to find is **`anon_pad#1`**:
the source is a `constant_struct2t` still carrying the pre-padding operand list,
so the padding member the read names is not in its type. §119's arm removes the
producer, and the test verifies.

**Do not "harden" the `.value()`.** `member2t`'s constructor already asserts
that the member resolves (`irep2_expr.h:1607-1610`), exempting only the
transient symbol/pointer/array source types — which a `constant_struct2t` is
not. So an assert-enabled build (CI's DebugOpt) would have caught this at
construction, and the uncaught exception is only how a `-DNDEBUG` build notices
the same violated invariant. Returning `expr2tc()` there would convert an
asserted invariant into a silent decline and hide the next producer. The three
unguarded `.value()` calls in that file (1226, 1243, 1327) are correct as they
stand.

### 120.2 The six remaining goto rows, classified

| rows | cause | verdict |
|---:|---|---|
| 2 | identity cast from an alignment attribute (`aligned_attr`, `github_2337_6`) | do not mirror, §115.1 |
| 3 | function-pointer identity cast (`atexit-1`, `github_5138_fail`, `github_5296`) | do not mirror, §113.3 |
| 1 | `00_aiob_4_true-unreach-call` | **work**, §120.3 |

Five of six were already-recorded decisions. Only one was live.

### 120.3 The comma expression's type

Reduced from `00_aiob_4`:

```c
unsigned g[42][3];
unsigned i;
int main(void) { i = 3; if ((i = i, g[i])[0] != 0) return 1; return 0; }
```

```
<         ASSERT (signed long int)i >= 0 // array bounds violated: array `g' lower bound
<         ASSERT (signed long int)i < 42 // array bounds violated: array `g' upper bound
<         IF !(g[(signed long int)i][0] != 0) THEN GOTO 1
---
>         IF !((&g[(signed long int)i][0])[0] != 0) THEN GOTO 1
```

C11 6.5.17p2 gives a comma expression its right operand's type. Clang hands the
node the *decayed* type when that operand is an array, and
`clang_c_adjust::adjust_comma` overwrites it (`expr.type() = expr.op1().type()`,
`clang_c_adjust_expr.cpp:1884`). This pass had no such arm — `adjust_sole_arms`
never matched `code_comma2t` at all — so the pointer type survived,
`adjust_index` took its `p[i]` sugar path, and the row was indexed as a pointer.

The visible cost is the two named array-bounds ASSERTs. It is **not** a
soundness hole: an out-of-range subscript is still caught, as
`dereference failure: array bounds violated` rather than
``array bounds violated: array `g' upper bound``. What is lost is the check's
attribution, and with it the ability of a `test.desc` to pin *which* check
fired — which is why the `_fail` test regexes the named form.

`adjust_comma_at_dispatch` (`clang_c_adjust_irep2.cpp:875-885`) already performs
exactly this rewrite for the `--clang-c-irep2-adjust` probe; the sole-adjuster
path simply never called it. The arm added here is the same three lines,
natively.

### 120.4 Result

Census 24 → 23 for this patch alone, with `00_aiob_4_true-unreach-call`
converging and nothing new; default path byte-identical on all 226 C sources.
Suites: `esbmc` 1857/1857, `cstd` 142/142, `floats` 106/106,
`function_contract` 414/414, `goto-coverage` 144/144.

Applied on top of the other five, the goto residue falls to **5**, and all five
are recorded do-not-mirror decisions.

### 120.5 Next

The stride-8 sample is exhausted: zero verdict divergences, and every goto
difference is a recorded decision. The sample was pinned at 226 of 1 863 tests,
so the honest next step is **not another slice but a wider census** — run the
verdict comparison over the whole `regression/esbmc` suite rather than 1-in-8,
which is where any remaining live divergence now has to come from. §118.5's
artefact warning applies at that scale: the run must clean up after itself.
## 125. The unary promotion the complex arm displaced (2026-08-23)

(§125: PRs #7266-#7285 are in flight against this file and claim §115-§124.)

§124.4 guessed that `github_4078_unary_bool` was an unported arm and said it
was cheap to check first. It was, and it was.

### 125.1 One `if`, two obligations

`clang_c_adjust` handles `unary-` and `bitnot` in a single arm that does two
things: recurse into the operands, and promote a boolean operand to the node's
type (`clang_c_adjust_expr.cpp:150-160`, added for #4078). This pass ported the
*complex* half of the unary story — `adjust_complex_unary`, guarded by
`is_complex_unary`, which requires `is_complex_type(expr->type)` — and nothing
covers the ordinary case. A `~(a || b)` therefore kept a boolean operand, and
the solver was handed a boolean where it wanted a bitvector:

```
$ esbmc main.c --unwind 2 --clang-c-irep2-adjust-only
ERROR: Bitwuzla error encountered
```

The port is the same shape as the legacy arm, expressed with
`c_implicit_typecast`, and hangs off the existing `is_complex_unary` branch as
its `else`.

### 125.2 Result

Closes `github_4078_unary_bool` and `github_4078_unary_bool_fail`. The
`gcc_vector_float_{arith,scalar_mul}` pair that §124.4 grouped with them is a
different cause — both still abort in bitwuzla after this patch — so the
"SIGABRT on vectors / unary bool" row was two causes, as its name half-admitted.
That is the second time a cluster named by *symptom* has split on inspection
(§123.2 was the first); the census groups by signal, and a signal is not a
cause.

Whole-suite verdict residue **11 → 9**. Both new tests abort in bitwuzla on the
pre-patch binary and return SUCCESSFUL / FAILED after; a mutant swapping `neg`
for `bitnot` moves both, the `_fail` one inverting.

Default path byte-identical on 226 C sources. Suites: `esbmc` 1857/1857,
`cstd` 142/142, `floats` 106/106, `cbmc` 307/307, `goto-coverage` 144/144.

### 125.3 Next
## 119. §118.6's unsound row closed — the literal's type is a pre-padding copy
## (2026-08-23)

(§119: PRs #7266, #7271, #7274 and #7275 are in flight against this file and
claim §115-§118.)

§118.3 named `github_2335_4` — FAILED by default, SUCCESSFUL under the flag —
as the highest-value row left, being the only one unsound rather than merely
wrong. It is `clang_c_adjust::adjust_struct`, which this pass never had.

### 119.1 The missing arm, and why the obvious port does not work

The legacy arm inserts a zero operand at each synthetic padding member so a
literal's operand list matches its padded type
(`clang_c_adjust_expr.cpp:202-226`). Ported directly it does nothing: the
literal's own type reports two members where the tag reports three.

The reason is that the value's type is an **inline copy the converter recorded
before `add_padding` ran**. `pad_type_symbol` pads the *type symbols*, and the
legacy arm needs no more than that because there the value's type is a
`symbol_typet` — `ns.follow` resolves it to the padded one. Under this flag the
value has come through `migrate_expr`, which resolved that symbol type to a
concrete `struct_type2t` snapshot, and `ns.follow` on a concrete type is the
identity. The padded layout has to be read off the tag symbol by name.

`pad_struct_operands` already existed for exactly this job, file-local in
`python_adjust.cpp` (§V.3's Optional/union literals). It moves to
`irep2_utils`, so the two frontends share one definition rather than the second
copy §39.2 of `frontends-to-irep2.md` warns about.

### 119.2 The first pair of tests did not reproduce, and why

The obvious test — read a trailing member out of a padded literal — passes on
the *pre-patch* binary. Trailing padding shifts nothing before it, so every
declared member still resolves at its own index. §39.1's first failure mode:
the corpus was thin, and a green mutant meant the test, not the code.

What `github_2335_4` actually exercises is a dispatch through a function-pointer
member of an array element, where the short literal changes the flow rather
than a single read. Reduced to 21 lines:

```c
struct command { char *name; void (*function)(void); char state_needed; };
const struct command commands[] = {{"c1", c1, 0}, {"c2", c2, 1}};
/* parse() dispatches through commands[i].function; c1() leaks on the second call */
```

```
$ esbmc v2.c --memory-leak-check ...                              VERIFICATION FAILED
$ esbmc v2.c --memory-leak-check ... --clang-c-irep2-adjust-only  VERIFICATION SUCCESSFUL
```

That is the shipped `_fail` test, and it is the strongest kind available here:
the pre-patch binary does not merely print differently, it **misses a real
leak**. Corrupting the arm (skip the insert) returns both tests to SUCCESSFUL.

### 119.3 Result

Census on the pinned stride-8 list, same base: **24 → 22**, with
`github_2335_4` and `github_578_success3` converging and nothing new. The two
were one cause with two symptoms, as §118.3 predicted. The verdict census's
three-row residue is now one (`github_3487`, §118.4).

`irep2_utils` is shared, so the default path was measured: byte-identical on all
226 C sources. Suites: `esbmc` 1857/1857, `cstd` 142/142, `function_contract`
414/414, `python/list` 294/294, `python` class/struct/optional/union 110/110,
`esbmc-cpp/cpp` 931/933 — `ch9_7` and `ch13_10` exceed the harness's 120 s cap
locally and return their expected verdicts on master and here alike.

### 119.4 Next

`github_3487` — `ERROR: uncaught exception [St19bad_optional_access]` under the
flag, SUCCESSFUL without. The last row of §118.1's three, and the only one that
is a crash in ESBMC's own code rather than a modelling gap.
## 116. §114.2's two three-test causes: one is a crash, one is not work
## (2026-08-23)

§114.2 deferred "the promotion at a comparison, and the decay rendered as a
cast", three tests each, on the reading that both were spelling-level. Reduced,
neither is what its census tag said.

### 116.1 The comparison cast is §113.3's class, not a missing promotion
## 129. The false alarm was not in the builtin's lowering (2026-08-24)

(§129: PRs #7278-#7290 are in flight against this file and claim §119-§128;
#7284 has since landed as §123.)

§128.4 picked `builtin_arith_overflow` on the reasoning that it "names a builtin
family (`__builtin_*_overflow`) whose lowering is a specific, findable arm — the
same shape as §117 and §125, both of which were unported name-matched builtins."

The family is the right place to look and the wrong place to patch. The
builtins are lowered identically on both paths; what this pass corrupts is the
type of the *argument* they are handed, in `expr2t::with_type` — a generic irep2
utility with no connection to builtins. The family matters only because its
lowering is the one consumer that reads that type.

### 129.1 A trait that is structurally right and semantically wrong

`c_typecastt::implicit_typecast_followed` re-attaches a pointer argument's own
type when source and destination compare equal without being identical
(`c_typecast.cpp:838-842` — qualifier differences). It does so through
`expr2t::with_type`, which rebuilds a node from a new type plus its remaining
fields, gated on `supports_with_type_v`:

1. the kind's first `fields` entry is `&expr2t::type`, and
2. the kind is constructible from `(const type2tc &, rest...)`.

`address_of2t` passes both — and means something else by the type it accepts.
Its primary constructor takes the **pointee** type and builds
`pointer_type2tc(subtype)` itself, a signature its own comment already calls
"slightly unintuitive". Handed a pointer type it wraps it a second time, so
`&y` on an `int` acquires type `int **`.

Gate 2 is what makes this the only such kind. Forty-one kinds synthesise their
own type in the constructor and still list `&expr2t::type` first; a compiled
fold of both gates over `expr_kinds.inc` counts 106 passing gate 1, 66 passing
both, so 40 are excluded by gate 2 alone. Representative rows:

| kind | constructor's first parameter | gate 2 |
|---|---|---|
| `constant_bool2t` | `bool value` | rejects |
| `constant_fixedbv2t` | `const fixedbvt &` | rejects |
| `constant_floatbv2t` | `const ieee_floatt &` | rejects |
| `same_object2t` | `const expr2tc &v1` | rejects |
| `overflow2t`, `overflow_cast2t`, `overflow_neg2t` | `const expr2tc &operand` | rejects |
| **`address_of2t`** | **`const type2tc &subtype`** | **accepts** |

All forty are excluded because their constructor will not take a leading
`type2tc` at all — the case the trait's comment was written for. `address_of2t`
takes one and means the subtype by it, which no structural test can see. That
it is the only kind to do so is exhaustive, not sampled: every `expr2t(...)`
base initializer lives in `irep2_expr.h`, and tallying them by first argument
gives `type` x43, `get_empty_type()` x20, `get_bool_type()` x11, `t` x5,
`value.spec.get_type()` x2, `size_type2()` x1, `get_int32_type()` x1, and
`pointer_type2tc(subtype)` x1.

The trait is left alone: the callers do want the rebuild, they want it built
from `to_pointer_type(new_type).subtype`. The override is an explicit
specialization of `rebuild_with_type<address_of2t>` rather than an early return
in `with_type`, because the trait still admits the kind — an early return leaves
the dispatcher expanding a `case address_of_id:` arm that performs the very
double-wrap being fixed, dead only by virtue of being shadowed. Specialising
puts the exception on the generic mechanism it overrides, and any future
reordering that would have resurrected the bug now has nothing to resurrect.

One field does not survive the rebuild. `pointer_type2t` carries
`carry_provenance` beside `subtype`, and `address_of2t`'s constructor rebuilds
the pointer with that parameter's `false` default, so `with_type(T)` returns a
node whose type is not `T` when `T` carries provenance. The drop is inherited,
not introduced — `migrate.cpp:1609` builds address-of nodes through the same
constructor — and it is unobservable today: `can_carry_provenance` is set only
under `ESBMC_CHERI_CLANG`, and every `with_type` caller derives its new type
from `expr->type`, which for an address-of is already `false`. Closing it
properly means an `address_of2t` constructor overload taking the full pointer
type, which is a separate change with `migrate.cpp` in its blast radius.

### 129.2 Why only a builtin reproduces it

An ordinary call masks the widening completely. Measured on the pre-patch
binary, under the flag:

| shape | verdict |
|---|---|
| `void store(int *p) { *p = 4; } store(&y);` | SUCCESSFUL |
| the same with no body | SUCCESSFUL |
| `memcpy(&y, &s, sizeof(int))` | SUCCESSFUL |
| `*(&y) = 4;` | SUCCESSFUL |
| `__builtin_sadd_overflow(2, x, &y)` | **FAILED** — spurious out-of-bounds |

Binding an argument to a parameter discards the argument's type: the callee
dereferences `p`, whose type is its own and correct. The overflow builtins have
no body, so nothing binds. `goto_symext::run_builtin` synthesises the store
itself, and sizes it from the argument:

```cpp
symex_assign(code_assign2tc(
  dereference2tc(
    to_pointer_type(func_call.operands[2]->type).subtype,
    func_call.operands[2]),
  op));
```

`operands[2]->type`, not `func_type.arguments[2]` — which the assert at
`run_builtin.cpp:113`, thirty lines up, has already established is a pointer. With the argument widened, the subtype
is `int *`, so the store writes eight bytes into a four-byte object and
`dereferencet` reports the out-of-bounds. That asymmetry is why the four probes
above are silent: this is the only consumer in the corpus that reads an
argument's own type where a parameter type exists.

The lesson for the census is narrower than "read the verdict". A defect can be
one node deep and still surface in exactly one test, because only one consumer
looks at the field it corrupted. The reduced test is not the minimal program
exhibiting the wrong type — every probe above carries the wrong type too — it is
the minimal program with a consumer that reads it.

### 129.3 The default path never reaches the arm

`with_type` is shared — `base_type`, `goto_symex_state`, `symex_main`,
`smt_solver` and `python_adjust` all call it — so the patch was measured for
reach rather than argued safe. A counter in the new arm, over the pinned
226-source corpus:

| path | tests firing | firings |
|---|---:|---:|
| default | **0** | **0** |
| `--clang-c-irep2-adjust-only` | 138 | 937 |

Zero on the default path over the whole corpus, and zero on a 40-test Python
and a 24-test C++ sample. The default path is unchanged by construction, which
is a stronger statement than the byte-identity diff the earlier sections take.

### 129.4 Result

Whole-suite verdict residue **5 → 4**; `builtin_arith_overflow` is `SUCCESSFUL`
on both paths. Both new tests reproduce the false alarm on the pre-patch binary.
The pinned 233-test sample carries three divergences — `github_2335_4`,
`github_3487`, `memset-const-2` — and all three were re-measured on the same
tree with the patch reverted and are unchanged by it. Suites, rebased onto
master: `esbmc` 1891/1891, unit 713/713.

Four tests pin the fix, and all four fail when the specialization is removed:
the two regression directories, plus an `address_of2t` row in
`unit/irep2/with_type.test.cpp`'s supported-kinds table and a case asserting one
level of indirection, an unchanged `ptr_obj`, a preserved `implicit`, and
`irep2_cast_error` on a non-pointer `new_type`. The unit case is what pins
`implicit`: it is `false` on every call site that reaches the arm, so a mutant
dropping it survives both regression tests.

The `_fail` test pins the violated property (`^  assertion uy == 5$`) rather
than the verdict alone: the pre-patch binary also fails that program, on the
spurious out-of-bounds, so a bare `VERIFICATION FAILED` regex would have
measured nothing. A negative test against a false *alarm* has to name which
property fails, not that one does.

### 129.5 Next

| test | signature |
|---|---|
| `github_2174` | false alarm SUCCESSFUL → FAILED |
| `github_301` | `ERROR: Bitwuzla error encountered` |
| `32_floppy` | SIGSEGV, no verdict |
| `complex_25` | §88.2 binding |

`github_2174` is the last false alarm and was expected to share this cause. It
does not: with the address-of correctly typed, `atomic_init(&a, 10)` followed by
`atomic_load(&a)` still returns something other than 10 under the flag, so the
`__c11_atomic_*` lowering is a separate arm. It is the one to take next — and
being body-less builtins reading their arguments' types, they are the same
family of consumer §129.2 describes.

## 122. The largest cluster was one line, and it is not a frontend bug
## (2026-08-23)

(§122: PRs #7266, #7271, #7274, #7275, #7278, #7280 and #7282 are in flight
against this file and claim §115-§121.)

§121.4 named the seven-test `to_struct_type() called on type whose type_id is
union` cluster as the next target, on the reasoning that its signature named the
defect precisely and that a `union_bitfield` and a `struct_bitfields` test
sitting together pointed at the bitfield lowering. The signature was right; the
guess about *where* was wrong.

### 122.1 The site is `value_sett::assign`, not the frontend

```
#3  to_struct_type (t=...) at type_kinds.inc:23
#4  is_subclass_of (subclass=..., superclass=..., ns=...) at base_type.cpp:445
#5  value_sett::assign (...) at pointer-analysis/value_set.cpp:1299
```

`value_sett::assign` opens its aggregate branch with
`if (is_struct_type(lhs_type) || is_union_type(lhs_type))` — unions included —
and then, for a concrete rhs whose type is not `base_type_eq` to the lhs, asks
`is_subclass_of(lhs_type, rhs->type, ns)`. That helper is struct-only: it opens
by casting *both* operands with `to_struct_type`. Handed a union it throws, and
nothing catches it.

Inheritance has no union analogue, so a union pair that is not `base_type_eq` is
simply incompatible — exactly the case the branch already drops two lines above
for a mismatched `type_id`. Guarding the `is_subclass_of` call on both types
being structs closes all seven:

| test | before | after |
|---|---|---|
| `github_162`, `github_162_fail` | abort | SUCCESSFUL, both paths |
| `github_571_{1,2,3}` | abort | SUCCESSFUL, both paths |
| `struct_bitfields_16` | abort | SUCCESSFUL, both paths |
| `union_bitfield_0` | abort | SUCCESSFUL, both paths |

Reduced, the trigger is two lines:

```c
union a { int : 5; };
int main(void) { union a x = {}; return 0; }
```

### 122.2 Why the printers could not see it

The symbol table **and** the goto program are byte-identical between the two
paths for that reducer, and the flag still aborts while the default path
verifies. Whatever makes the two union types structurally unequal is a property
`--show-symbol-table` and `--goto-functions-only` both elide. This is the
sharpest instance yet of §121's lesson: neither printer is an oracle for
behaviour, and a census that reads one is measuring the printer.

### 122.3 A note on scope

The defect is in `pointer-analysis`, shared by every frontend, and the fix is
not flag-gated. Several probes for a default-path reproducer — mismatched union
tags through a cast, a union returned from an uninterpreted function, a
self-assignment through a `char *` round trip — did **not** find one, so the only
known trigger remains `--clang-c-irep2-adjust-only`. Recorded as such rather
than claimed as a user-facing fix: the call is wrong on its own terms
(a struct-only helper reached with a union), and the guard is the same
incompatibility test the branch already applies.

`value_set.cpp` is shared, so the wider suites were run: `esbmc` 1857/1857,
`cstd` 142/142, `floats` 106/106, `function_contract` 414/414, `esbmc-unix`
435/438 — the three are `03_boundedBuffer`, `github_595` and
`github_6480_deepening`, all exceeding the harness's 120 s cap locally and all
returning identical verdicts on master and here. Default path byte-identical on
226 C sources.

### 122.4 Next

Whole-suite verdict residue **24 → 17**. The clusters left, largest first:

| tests | signature |
|---:|---|
| 6 | SIGSEGV in complex arithmetic (`complex_25`, `github_382_6`, `github_6713_complex_*`) |
| 4 | false alarm SUCCESSFUL → FAILED (`builtin_arith_overflow`, `github_2174`, two `pragma_unroll`) |
| 4 | SIGABRT on vectors / unary bool |
| 3 | unclustered (`32_floppy`, `github_301`, `github_1934-1`) |

The complex-arithmetic six is next by size, and `github_6713` names an issue
whose own fix (#6713, the compound-assignment lowering) is already in the tree —
so the reproducers are pinned and the divergence is in how this pass carries
that lowering.
## 121. The whole-suite verdict census — the sample's zero was a sampling
## artefact (2026-08-23)

(§121: PRs #7266, #7271, #7274, #7275, #7278 and #7280 are in flight against
this file and claim §115-§120.)

§120.5 asked for the verdict comparison to be widened from the pinned stride-8
list to all of `regression/esbmc`, on the argument that any remaining live
divergence had to come from the other seven-eighths. It does — emphatically.

| | sample (226) | whole suite (1 742) |
|---|---:|---:|
| same verdict | 226 | 1 742 |
| **differing verdict** | **0** | **25** |
| skipped (`test.desc` already carries the flag) | 6 | 43 |

**Zero on the sample meant nothing.** The stride-8 list is 1-in-8 of an
alphabetical listing, and it contained not one of the 25. Every "exit criterion
met" claim this scope has made against a stride sample should be read with that
in mind: the sample was sized for a *goto-dump* census, where divergences were
dense, and it was never re-sized when the instrument changed to verdicts, where
they are rare and clustered.

### 121.1 The 25, by failure mode

| mode | tests |
|---|---:|
| SIGABRT (mostly a solver sort mismatch) | 8 |
| uncaught `irep2_cast_error` | 7 |
| SIGSEGV | 6 |
| verdict flip, SUCCESSFUL → FAILED (false alarm) | 4 |

Twenty-one of the 25 are crashes. Only four produce a wrong answer rather than
no answer, and all four are false alarms rather than missed bugs — worth noting,
though 21 aborts is not a comfortable position either.

### 121.2 What this patch closes

`bitvector_04`:

```c
_ExtInt(10) x = nondet_float();
_ExtInt(10) y = nondet_int();
_ExtInt(10) z = x + y;
```

```
<         ASSIGN z=(signed _ExtInt(10))((signed int)x + (signed int)y);
>         ASSIGN z=(signed int)x + (signed int)y;
```

The operands promote to `int` for the addition, and
`clang_c_adjust::adjust_decl` ends with a `gen_typecast` of the initialiser back
to the declared type (`clang_c_adjust_code.cpp:104`). This pass had no
`code_decl2t` arm at all, so the 10-bit object was initialised from an `int` and
bitwuzla aborted on the mismatched sorts.

Note this is *not* `adjust_assign`: the first port targeted `code_assign2t` and
did nothing, because a declaration's initialiser rides in `code_decl2t::init`
rather than lowering to a separate assignment.

The tests need **nondet** operands. With constants the initialiser folds before
the mismatch can reach the encoder, and a constant-initialised pair passes on
the unfixed binary — §39.1's first failure mode again, and the second time in
three sittings that the first test written was the thin one.

Census 24 → 24 on the pinned sample (which does not contain `bitvector_04`) with
nothing new, and 25 → 24 on the whole-suite verdict census. Default path
byte-identical on all 226 C sources of the sample. Suites: `esbmc` 1857/1857,
`cstd` 142/142, `floats` 106/106, `function_contract` 414/414, `goto-coverage`
144/144.

### 121.3 The residue of 24, clustered

Ordered by cluster size, since these are a handful of causes rather than 24:

| tests | signature | members |
|---:|---|---|
| 7 | `to_struct_type() called on type whose type_id is union` | `github_162{,_fail}`, `github_571_{1,2,3}`, `struct_bitfields_16`, `union_bitfield_0` |
| 6 | SIGSEGV in complex arithmetic | `complex_25`, `github_382_6`, `github_6713_complex_{compound,div_nondet}{,_fail}` |
| 4 | SIGABRT on vectors / unary bool | `gcc_vector_float_{arith,scalar_mul}`, `github_4078_unary_bool{,_fail}` |
| 4 | false alarm, SUCCESSFUL → FAILED | `builtin_arith_overflow`, `github_2174`, `github_4715_irep2_bodies_pragma_unroll_01`, `pragma_unroll_nested_dowhile_true` |
| 3 | unclustered | `32_floppy`, `github_301`, `github_1934-1` |

All reproduce on **master** as well as on the six-PR branch — checked directly
for the `to_struct_type` cluster — so none is a regression from the series.

### 121.4 Next

The `to_struct_type()`-on-a-union cluster, at seven tests the largest. Its
signature names the defect precisely: something in the pass assumes a struct
where the type is a union, and both a `union_bitfield` and a `struct_bitfields`
test are in it, so the bitfield lowering is where to look.

The methodological point stands on its own: **census on the whole suite, not a
stride sample**. The sample was calibrated for a denser instrument and silently
under-reported by a factor of infinity once the instrument changed.

## 128. The array-typed expression statement (2026-08-23)

(§128: PRs #7267-#7288 are in flight against this file and claim §116-§127.)

§127.4 picked `github_1934-1` as the most precisely signposted of the six
remaining rows. Its message —

```
ERROR: Can't construct rvalue reference to array type during dereference
```

— names the consumer, and the producer is a nine-line program:

```c
struct Base { int ss[128]; };
int main() { struct Base x, *y = &x; y->ss; }
```

### 128.1 A statement whose value is an array

`clang_c_adjust::adjust_code` rewrites an array-typed expression statement to
`&y->ss[0]` (`clang_c_adjust_code.cpp:57-74`), with its own comment explaining
why: the dereference code does not assume such an object exists, and the
statement's value is unused, so taking the first element's address is free.
This pass had no `code_expression2t` arm — the only place it touches that kind
is `declare_implicit_callee`, which reads the operand and does not rewrite it —
so the bare array reached `dereferencet`.

Ported with the same shape, extended to `vector` as well as `array` because
`is_array_like` (the legacy predicate) covers both.

### 128.2 The assignment exemption, which nothing reaches

The legacy arm exempts an assignment operand (`op.statement() != "assign"`):
there the array is the assignment target, not a discarded value. Mirrored here,
and **no input found reaches it**. The probes:

| probe | result |
|---|---|
| `struct` assignment (`b = a` with an array member) | struct-typed, so the array guard already excludes it |
| array-to-pointer assignment (`q = p`) | pointer-typed, excluded |
| **vector** assignment (`b = a`, `v4f`) — legal in C, and vector-typed | reaches the arm as a `code_assign2t` statement, not wrapped in `code_expression2t` |

Removing the exemption leaves all three byte-identical to the default path.

This is **not** §94.1's case, where a ported guard *undid* a rewrite the legacy
pass also performed and so added behaviour that had to be justified. Here the
guard makes this pass's condition identical to legacy's; dropping it would be
the deviation, and could only be justified by an input showing legacy's own
guard is dead. Kept, and recorded as untested rather than left to look tested.

### 128.3 Result

Whole-suite verdict residue **6 → 5**. Both tests abort on the pre-patch binary
and return SUCCESSFUL / FAILED after. Default path byte-identical on 226 C
sources. Suites: `esbmc` 1854/1857, `cstd` 142/142, `cbmc` 307/307,
`extensions` 201/201 — the three are `github_302`, `github_2335_1` and
`github_4634`, all exceeding the harness's 120 s cap locally and all returning
their expected verdict on master and here alike.

### 128.4 Next

| test | signature |
|---|---|
| `github_301` | `ERROR: Bitwuzla error encountered` |
| `32_floppy` | SIGSEGV, no verdict |
| `builtin_arith_overflow`, `github_2174` | false alarm SUCCESSFUL → FAILED |
| `complex_25` | §88.2 binding |

The two false alarms are next: they are the only rows producing a wrong answer
rather than no answer, and `builtin_arith_overflow` names a builtin family
(`__builtin_*_overflow`) whose lowering is a specific, findable arm — the same
shape as §117 and §125, both of which were unported name-matched builtins.
## 127. The "unclustered four" were four causes (2026-08-23)

(§127: PRs #7266-#7287 are in flight against this file and claim §115-§126.)

§126.4 said "unclustered" meant only that nobody had read them. Read, the four
are four distinct causes — no two share a signature:

| test | `-only` outcome |
|---|---|
| `github_382_6` | `ERROR: Unexpected type in int/ptr typecast` — fixed here |
| `github_301` | `ERROR: Bitwuzla error encountered` |
| `github_1934-1` | `ERROR: Can't construct rvalue reference to array type during dereference` |
| `32_floppy` | no verdict (SIGSEGV) |

That closes the question §123.2 opened: grouping by signal number produced one
four-test "cluster" containing four unrelated defects, and the two earlier
splits were not bad luck.

### 127.1 `*main`, and a missing arm rather than a wrong one
## 124. A field excluded from equality, dropped by four rebuilds (2026-08-23)

(§124: PRs #7266, #7271, #7274, #7275, #7278, #7280, #7282, #7283 and #7284 are
in flight against this file and claim §115-§123.)

§123.5 named the four false-alarm rows as next, on the grounds that they were
the only ones left producing a wrong *answer* rather than no answer, and that
two naming `pragma_unroll` were probably one cause. They were.

### 124.1 The mechanism, which the test's own comment predicted

`github_4715_irep2_bodies_pragma_unroll_01` documents its failure mode in
advance:

> If the count were dropped on the round-trip the loop would run to its natural
> bound of 8, writing a[3..7] out of the 3-element array: a spurious
> array-bounds violation seen only under the flag.

Which is exactly what happens — though not on the round-trip. `migrate_expr`
carries `#pragma_unroll` onto the IREP2 loop and `migrate_expr_back` writes it
out again; both halves are correct. What drops it is this pass. Four sites
rebuild a loop node, and every one of them omitted the count:

| site | node |
|---|---|
| `adjust_statement_condition` | `code_while2tc(cond, body, loc)` |
| `adjust_statement_condition` | `code_dowhile2tc(cond, body, loc)` |
| `adjust_statement_condition` | `code_for2tc(init, cond, iter, body, loc)` |
| `hoist_for_init` | `code_for2tc(nil, cond, iter, body, loc)` |

The constructor's last parameter defaults to `0`, and `0` means "no pragma".

### 124.2 Why nothing caught it

`pragma_unroll_count` is deliberately **excluded** from the loop kinds' `fields`
tuple (`irep2_expr.h:2258-2264`, alongside `location`), so it takes no part in
`operator==`. This pass writes a symbol's value back only when it changed —

```cpp
if (value != before)
  s->set_value(value);
```

— and a rebuilt loop that dropped the count compares **equal** to the original
that had it. The guard cannot see the loss, the A/B census cannot see it
(`--goto-functions-only` prints the unrolled program, not the annotation), and
only a verdict differs.

That is a general hazard, not a one-off: any excluded-from-`fields` member is
invisible to both the change guard and structural equality, so every rebuild has
to carry it by hand. `location` is the other one, and §115 was the same bug in
that field.

### 124.3 Result

Closes `github_4715_irep2_bodies_pragma_unroll_01` and
`pragma_unroll_nested_dowhile_true`; the other two false alarms
(`builtin_arith_overflow`, `github_2174`) are unaffected and are a different
cause. Whole-suite verdict residue **13 → 11**.

The `_fail` test earns its place with an under-unroll mutant rather than the
absent patch: forcing the carried count to `1` truncates the loop before the
out-of-bounds write and turns FAILED into SUCCESSFUL. The positive test moves
against the unfixed binary directly (FAILED → SUCCESSFUL). Both loop shapes are
covered — a `while` for the condition-rebuild sites and a `for` for the hoist.

Default path byte-identical on 226 C sources. Suites: `esbmc` 1857/1857,
`cstd` 142/142, `goto-coverage` 144/144, `k-induction` 122/122,
`loop-invariants` 81/81.

### 124.4 Next

| tests | signature |
|---:|---|
| 4 | SIGABRT on vectors / unary bool (`gcc_vector_float_{arith,scalar_mul}`, `github_4078_unary_bool{,_fail}`) |
## 123. The complex cluster was four, not six — and a decline that crashes
## (2026-08-23)

(§123: PRs #7266, #7271, #7274, #7275, #7278, #7280, #7282 and #7283 are in
flight against this file and claim §115-§122.)

§122.4 grouped six SIGSEGVs as "complex arithmetic". Read individually, that
grouping was wrong in one place and incomplete in another.

### 123.1 The four that share a cause

`adjust_compound_assignment` *declines* a complex operand, with a comment
deferring to `clang_c_adjust::lower_complex_compound_assignment` — an arm that
does not run under `--clang-c-irep2-adjust-only`, because the flag replaces the
legacy pass rather than shadowing it. Nobody performs the lowering, and #6713's
own comment says what happens next: `goto_convert`'s `remove_assignment`
rebuilds `a op b` long after adjustment, so the SMT layer is handed a raw
complex operator. Bitwuzla faults inside `mk_bvadd`:

```
#0  bitwuzla_mk_term2 ()
#1  bitwuzla_convt::mk_bvadd (...) at bitwuzla_conv.cpp:107
#2  smt_solver_baset::convert_ast_node (...) at smt_solver.cpp:745
```

The port is small because the decomposition already exists here: rewrite
`a op= b` to `a = a op b` and hand the binary node to `adjust_complex_arith`.
Closes `github_6713_complex_{compound,div_nondet}{,_fail}` — all four agree on
both paths after.

**A decline is not free.** §88.2 justified leaving these nodes alone with
"declining only leaves the node where this mode already had it", which was true
of the *shape* and false of the *outcome*: what this mode already had was a
segfault. A decline that hands the backend a node it cannot encode is a crash
with extra steps, and the other declines in this file should be re-read with
that in mind rather than assumed safe.

### 123.2 `github_382_6` is not a complex test at all

```c
int main(void) { global_var3 = *main; assert(global_var3 == *main); return 0; }
```

```
<         ASSIGN global_var3=(unsigned int)(&(*(&main)));
>         ASSIGN global_var3=*(&main);
```

C11 6.3.2.1p4: dereferencing a pointer to a function yields a function
designator, which converts straight back to a pointer — `*f` is `f`, and
`******f` too. `clang_c_adjust::adjust_dereference` re-takes the address for
exactly this case (`clang_c_adjust_expr.cpp:918-927`, its comment says
"allowing ******...*p"). This pass had **no dereference arm at all**, so the
code-typed dereference reached a consumer wanting a pointer.

Only that arm is ported. The array (`*a` → `a[0]`) and pointer-subtype arms
above it retype a node migration already builds with the right type, and no
corpus input distinguishes them — porting them would be §94.1's guard again, an
arm nothing executes.

### 127.2 A mutant that was an alternative implementation

Replacing `address_of2tc(type, expr, true)` with `to_dereference2t(expr).value`
— strip the dereference instead of re-addressing it — left both tests passing.
That is not §39.1's "unreachable by construction": `*(&f)` and `&f` denote the
same pointer, so the mutation is a *semantically equivalent rewrite*, and no
test can distinguish them because there is nothing to distinguish. It is a
sixth way for a mutant to sit still, and the useful response is to note the
arm has an equally valid alternative form rather than to hunt for a test.

The discriminating mutant is the absent patch, which the base binary supplies:
both tests abort there and return SUCCESSFUL / FAILED here. The `_fail` test
needed `***f` rather than `*f` to reach that state — with a single dereference
it failed identically on both binaries and measured nothing.

### 127.3 Result

Whole-suite verdict residue **7 → 6**. Default path byte-identical on 226 C
sources. Suites: `esbmc` 1857/1857, `cstd` 142/142, `cbmc` 307/307, `floats`
106/106, `extensions` 201/201.

### 127.4 Next

| test | signature |
|---|---|
| `github_301` | bitwuzla error |
| `github_1934-1` | rvalue reference to array during dereference |
| `32_floppy` | SIGSEGV |
| `builtin_arith_overflow`, `github_2174` | false alarm SUCCESSFUL → FAILED |
| `complex_25` | §88.2 binding |

Six rows, six causes, and no grouping left to exploit — each is now its own
investigation. `github_1934-1`'s message names a specific construction site
(`dereference` on an array-typed rvalue reference) and is the most precisely
signposted, so it is next.
Dereferencing a function. It was grouped with the others only because the
census records a signal number, and `SIGSEGV` is not a cause. Moved to the
unclustered rows.

### 123.3 `complex_25` is the §88.2 decline, and it is the real one

```c
_Complex double f(void) { calls++; return 1.0 + 2.0i; }
```

`adjust_complex_arith` reads each operand twice, once per component, so it
declines any operand carrying a side effect rather than evaluating it twice —
and `complex_25` is built entirely from side-effecting complex calls. The
decline is correct as far as it goes; the consequence is §123.1's, a node the
backend faults on.

Closing it needs the binding `clang_c_adjust` does first — a context temporary
plus a statement expression — which §88.2 records as separate work and this
patch does not attempt. It is now the *only* known input where the decline is
reachable, which makes it the concrete justification for doing that port.

### 123.4 Result

Whole-suite verdict residue **17 → 13**. Default path byte-identical on 226 C
sources. Suites: `esbmc` 1857/1857, `floats` 106/106, `floats-regression`
65/65, `cstd` 142/142.

Both new tests produce **no verdict at all** on the pre-patch binary — it
segfaults — and SUCCESSFUL / FAILED after. A mutant lowering `*=` as `+=` moves
both, the `_fail` one inverting.

### 123.5 Next

| tests | signature |
|---:|---|
| 4 | false alarm SUCCESSFUL → FAILED (`builtin_arith_overflow`, `github_2174`, `github_4715_irep2_bodies_pragma_unroll_01`, `pragma_unroll_nested_dowhile_true`) |
| 4 | SIGABRT on vectors / unary bool (`gcc_vector_float_{arith,scalar_mul}`, `github_4078_unary_bool{,_fail}`) |
| 4 | unclustered (`32_floppy`, `github_301`, `github_1934-1`, `github_382_6`) |
| 1 | `complex_25`, the §88.2 binding |

The false-alarm four are next. They are the only rows left that produce a wrong
*answer* rather than no answer, and two of them name `pragma_unroll`, so that
pair is likely one cause.

## 126. Vector float arithmetic — the half clang does not lower itself
## (2026-08-23)

(§126: PRs #7266-#7286 are in flight against this file and claim §115-§125.)

§125.3 named the `gcc_vector_float_{arith,scalar_mul}` pair and predicted the
cause would be §123.1's again — a deliberate decline that hands the backend an
unencodable node. It is a *missing* arm rather than a decline, but the shape of
the consequence is identical.

### 126.1 Why the scalar case never showed this

Under the flag a scalar `float a + b` is byte-identical between the two paths:

```
ASSIGN s=IEEE_ADD(a, b);
```

on both. Nothing in this pass promotes it — **clang emits `ieee_add` itself**
for scalar float arithmetic, and `migrate_ieee_arith_2op` carries it across. So
the pass never needed a float-promotion arm and the gap was invisible.

For a vector of float clang hands over the plain operator, and
`clang_c_adjust::adjust_float_arith` promotes it
(`clang_c_adjust_expr.cpp:796-817`, the `t.is_vector()` widening). That pass
does not run under this flag:

```
<         ASSIGN s=IEEE_ADD(a, b);
>         ASSIGN s=a + b;
```

and the backend aborts on a bitvector operator over a floating-point vector.

### 126.2 The rounding mode the legacy arm does not attach

`adjust_float_arith` returns *before* setting `rounding_mode` when the type is a
vector, with the comment "BUG: setting rounding_mode breaks migration". The
attribute-less legacy node then reaches `migrate_rounding_mode`, which
synthesises the default `c:@__ESBMC_rounding_mode` symbol for it. So the node
the default path actually produces carries that symbol, and the arm here builds
the same one — the goto dumps are byte-identical after the patch, which is what
confirms the reasoning rather than an argument from the comment.

### 126.3 Result

Closes both. Whole-suite verdict residue **9 → 7**. Default path byte-identical
on 226 C sources. Suites: `esbmc` 1857/1857, `floats` 106/106,
`floats-regression` 65/65, `cstd` 142/142, `cbmc` 307/307, `extensions`
201/201.

A note on the test rather than the code: the first version of the positive test
carried `+/-/*//` in its block comment, and the `*/` inside it closed the
comment early — `PARSING ERROR` on *all three* binaries, including the default
path. A test that fails identically everywhere is not measuring anything, and
the three-way comparison is what caught it.

### 126.4 Next

| tests | signature |
|---:|---|
| 4 | unclustered (`32_floppy`, `github_301`, `github_1934-1`, `github_382_6`) |
| 2 | vector float arithmetic (`gcc_vector_float_{arith,scalar_mul}`) |
| 2 | false alarm (`builtin_arith_overflow`, `github_2174`) |
| 1 | `complex_25`, the §88.2 binding |

The vector pair is the next coherent cause: `adjust_complex_arith` declines a
vector operand deliberately (`§88` records that the legacy pass "returns before
attaching a rounding mode for them"), and §123.1's finding applies — a decline
that hands the backend an unencodable node is a crash, not a no-op. Check
whether the same reasoning that closed the complex compound assignment closes
these.
| 2 | false alarm (`builtin_arith_overflow`, `github_2174`) |
| 1 | `complex_25`, the §88.2 binding |

The vector/unary-bool four are next by size. `github_4078_unary_bool` names an
issue whose fix is the integer promotion of a boolean operand under `unary-`
(`clang_c_adjust_expr.cpp:150-160`) — an arm this pass may not have ported, and
a cheap thing to check first.
No cluster larger than the unclustered four, and "unclustered" now means only
that nobody has read them — they were grouped by signal number and §123.2/§125.2
both show that is not a grouping. The next step is to read those four
individually, starting with `github_382_6` (`global_var3 = *main`, a function
dereference), which is the smallest input of the seven.
## 118. The census re-run through verdicts, not goto dumps — and what it saw
## (2026-08-23)

(§118: PRs #7266, #7271 and #7274 are in flight against this file and claim
§115-§117.)

§117.4 asked for this: two of the previous three causes had been mis-tagged as
spellings when they were programs the flag cannot verify, because
`--goto-functions-only` stops before the encoder. Re-run reading each test's
**verdict** under its own `test.desc` flags, default path against
`--clang-c-irep2-adjust-only`, on the same pinned stride-8 list, at a build with
those three PRs merged locally.

For reference the goto census at that same build is **9 differing, 217 same** —
down from 24, better than any of the three alone, because several tests carried
more than one cause.

### 118.1 What the verdict census found

| | tests |
|---|---:|
| same verdict | **217** |
| differing verdict | **3** |
| skipped (`test.desc` already carries the flag) | 6 |

Three, and none of them is a spelling:

| test | default | `-only` | |
|---|---|---|---|
| `github_2572_2` | SUCCESSFUL | **FAILED** | §118.2, fixed here |
| `github_2335_4` | FAILED | **SUCCESSFUL** | §118.3, the unsound direction |
| `github_3487` | SUCCESSFUL | **uncaught `bad_optional_access`** | §118.4 |

The goto census ranked the second of these as "struct padding in an aggregate
initialiser" and did not see the third at all. That is the whole argument for
this instrument: a `diff` row says the printers disagree, and says nothing about
whether the verifier still works.

### 118.2 `__builtin_isinf_sign` — the one fixed here

`do_special_functions` spells it exactly, and deliberately: the neighbouring
`isinf` arm matches a *base* name a program may reuse (`is_name_matched_builtin`,
#6904), whereas `__builtin_isinf_sign` is reserved. This pass mirrored the base-
name arm and not the exact one, so the call survived — and the symbol is
bodyless, which makes the result nondet rather than differently shaped:

```c
assert(__builtin_isinf_sign(1.0) == 0);   /* SUCCESSFUL by default, FAILED under the flag */
```

Ported as the same nested conditional the legacy arm builds,
`isinf ? (signbit ? -1 : 1) : 0`. `github_2572_2` agrees on both paths after it,
and both new tests move under a sign-swap mutant — the `_fail` one inverts,
which is the stronger signal of the pair.

### 118.3 `github_2335_4` is the unsound direction, and it is next

A test that FAILS by default SUCCEEDS under the flag. The goto diff is a missing
`anon_pad#3` in an aggregate initialiser for an array of structs, so the
initialiser is being built without the padding member the layout carries. A
frame that verifies because a member vanished is exactly the shape §110.2 warns
about read in the opposite direction, and it is the highest-value row left.

`github_578_success3` shows the same missing-padding spelling
(`anon_bit_field_pad#1`, `anon_pad#2`) without a verdict change, so the two are
one cause with two symptoms and should be taken together.

### 118.4 `github_3487` aborts in an optional

`ERROR: uncaught exception [St19bad_optional_access]: bad optional access` under
the flag, SUCCESSFUL without. Not diagnosed here beyond the reproduction; an
unhandled `std::optional` access is a defect wherever it is, and it is the only
row of the three that is a crash in ESBMC's own code rather than a modelling
gap.

### 118.5 A harness note worth keeping

Running a test with its own `test.desc` flags from its source directory writes
that test's output artefacts into the *source tree* —
`cwe_dead_code_dead_store_sarif` takes `--sarif-output out.sarif`, and the stale
file a census run left behind then failed the real `ctest` run of that test on a
later invocation. The census must either run in a copy or clean up after itself;
a suite failure immediately following a census run should be checked against
`git status` before it is believed.

### 118.6 Next

`github_2335_4` / `github_578_success3` — the missing padding member in an
aggregate initialiser, the one row in the residue that is unsound rather than
merely wrong.
## 117. The `POINTER_OFFSET` group is a third abort, and `offsetof` was fatal
## (2026-08-23)

(§117 rather than §115: PRs #7266 and #7271 are in flight against this file and
claim §115 and §116.)

The `POINTER_OFFSET` spelling was the largest remaining group in §114's table,
at three tests. It is five, and it is not a spelling — the second time in two
sittings that a row tagged `diff` turned out to be a program the flag cannot
verify at all.

### 117.1 The three intrinsics matched by name, not by prefix

`do_special_functions` selects most of its lowerings by a reserved
`__builtin_` prefix, and this pass mirrors those. Three it selects by the
`__ESBMC_` name instead — `POINTER_OFFSET`, `POINTER_OBJECT`, `same_object` —
and none had been ported. Each lowers to a node the backend evaluates in place
(`pointer_offset2t`, `pointer_object2t`, `same_object2t`, all long-standing).

Left as calls the symbols are bodyless, and `goto_check` refuses them:

```c
#include <stddef.h>
struct s { int x; int y; };
int main(void) { assert(offsetof(struct s, y) == 4); return 0; }
```

```
$ esbmc pv.c                              # VERIFICATION SUCCESSFUL
$ esbmc pv.c --clang-c-irep2-adjust-only
ERROR: Function call to non-intrinsic prefixed with __ESBMC (fatal)
```

`offsetof` is the reachable one: `clang_c_language.cpp:705` defines the macro
as `((size_t)__ESBMC_POINTER_OFFSET(&((type*)0)->member))`, so *every* use of
`<stddef.h>`'s `offsetof` was fatal under this flag. The census saw `&0->y` in
an `ASSIGN` on one side and a `FUNCTION_CALL` to a temporary on the other, and
recorded a spelling difference.

### 117.2 Result

Census on the stride-8 list pinned to a file before the A/B (233 entries, 226
with a `.c` source), same base:

| goto program | before | after |
|---|---:|---:|
| same | 202 | **207** |
| diff | **24** | **19** |

Five converge — `github_2512_8`, `github_2512_12`, `github_426_2`,
`github_1064-3-32`, `pointer-offset2` — against the three §114 tagged; two
carried the cause under another tag. None diverges that did not before, and the
default path is byte-identical on all 226 C sources (the arm is reached only
from `adjust_special_functions`, which the flag gates, but it was measured
rather than argued). Suites: `esbmc` 1857/1857, `cstd` 142/142, `floats`
106/106, `function_contract` 414/414, `goto-coverage` 144/144.

### 117.3 Mutants, and the one that did not move

Per §39.1 of `frontends-to-irep2.md`, each arm was corrupted rather than
deleted:

| mutation | ok-test |
|---|---|
| `pointer_offset` → `pointer_object` | FAILED ✓ |
| `same_object(a, b)` → `same_object(a, a)` | FAILED ✓ |
| `pointer_object` → `pointer_offset` | **SUCCESSFUL ✗** |

The third is §39.1's *first* failure mode, not its second: the test asserted
`POINTER_OBJECT(&a) == POINTER_OBJECT(&a)`, which holds whatever the intrinsic
lowers to. Rewritten against two distinct globals — where every intrinsic here
yields offset 0, so only the object id separates them — the mutant moves. The
arm was fine; the test was not.

### 117.4 Next

The residue is 19: temporary numbering (`tmp$3` vs `tmp$4`, 2 tests), struct
padding in an aggregate initialiser (`github_578_success3`), and the untagged
remainder. The identity-cast class stays closed as non-work.

A re-census is now worth more than another slice. Two of the last three causes
were mis-tagged because `--goto-functions-only` stops before the encoder, and
the residue is small enough to read every row through a full verification run
instead of a goto dump.
## 115. §114.2's dominant cause closed — the hoist's wrapper block had no close
## (2026-08-23)

§114.1 diagnosed the 14-test `DEAD` divergence as `hoist_for_init`'s
provenance and named `convert_block`'s `unwind_destructor_stack` as the site
that reads it. The mechanism is one step earlier than that, and it is not a
provenance choice at all.

### 115.1 A default-constructed `locationt` is not nil

`hoist_for_init` builds the wrapper block's close location as

```cpp
locationt end_location;
if (!is_nil_expr(f.body) && is_code_block2t(f.body))
  end_location = to_code_block2t(f.body).end_location;
```

and hands it to `code_block2tc`. When the loop body is not a block the variable
is left default-constructed — and `irept::is_nil()` is `id() == "nil"`, so a
default-constructed `locationt` (id `""`) is *empty but not nil*.
`migrate_expr_back` guards only on nil:

```cpp
if (ref2.end_location.is_not_nil())
  block.end_location(ref2.end_location);
```

so it wrote an empty `#end_location` onto the legacy block. `convert_block`
then stamped that empty location on every destructor it unwound, and
`goto_programt::output` renders an empty location as blank where it renders a
nil one as `no location`. The legacy hoist never calls `end_location(...)` at
all on this shape, which is why the two disagreed.

`goto_convert_functions.cpp:1834-1839` already spells the correct idiom for the
same question about the same field, with the same `else end_location.make_nil()`
arm. Mirroring it is the whole patch.

### 115.2 Result

Reproducer (§114.1's, unchanged), `--no-irep2-native-body` on both sides:

```
<         // 48 no location
>         // 48
```

is byte-identical after the patch, and a block-bodied control
(`for (...) { s = s + i; }`) was byte-identical before and after — the shape
the legacy hoist does set the close location on.

Two-stage census re-run on a stride-8 list of `regression/esbmc` (233 entries,
226 with a `.c` source), `--clang-c-irep2-adjust-only` against the legacy pass,
blank lines and timing lines ignored:

| goto program | before | after |
|---|---:|---:|
| same | 202 | **210** |
| diff | **24** | **16** |
| crash | 0 | 0 |

Eight tests converge; none diverges that did not before. The 24 matches §114's
count on the same suite, so the samples are comparable. Ten of those 24 carried
a `no location` line in their diff; the eight that converged are the ones where
it was the *only* cause, and the two that remain
(`00_aiob_4_true-unreach-call`, `github_2572_2`) show it only as instruction
renumbering downstream of a different divergence, not as a location mismatch of
their own.

The residue of 16 is the §114 tail, unmoved and untouched by this patch:
integer promotion missing at a comparison, array decay rendered as a cast
rather than `&a[0]`, the function-pointer identity cast §113.3 rules a
"do not mirror", struct padding in an aggregate initialiser, and
`POINTER_OFFSET` spelling.

### 115.3 The gate, and what pins it

`regression/esbmc/irep2_only_for_hoist_dead_location` asserts the `no location`
line adjacent to the loop variable's `DEAD` under
`--goto-functions-only --clang-c-irep2-adjust-only`. It is a positive regex, so
it is mutation-checked the only way that means anything here: run it against
the pre-patch binary, where the line reads `// 48` and the regex does not
match. §39.1's fifth failure mode does not apply — the mutation is the absence
of the patch, and the printer is exactly what the test reads.

### 115.4 Next

The two three-test causes §114.2 deferred. The first reduces to two lines, and
the trigger is not the comparison:

```c
__attribute__((aligned)) int g = 42;
int main(void) { int p = 1; if (g == 42) p = 2; return p; }
```

Legacy emits `(signed int)g == 42`; the hop-off emits `g == 42`. Dropping the
attribute makes the pair byte-identical, so the operand needs no promotion —
`g` is already `int`. What legacy emits is an identity cast, and it emits it
because the *legacy* types differ on an alignment attribute.

`signedbv_type2t` has exactly one field, `width`
(`src/irep2/irep2_type.h:209-221`). `__attribute__((aligned)) int` and plain
`int` are therefore not merely observed-equal after `migrate_type`, they are the
same node by construction, and the symbol tables are byte-identical on both
paths. No pass reading IREP2 can know the cast is owed — §113.3's argument
verbatim, reached from a different node.

**Do not mirror.** Three of the 24 close as non-work, on the same footing as
`atexit`'s function-pointer cast.

### 116.2 The array decay is not a spelling difference at all

The other cause reduces to six lines:

```c
char a[4];
char b[4];
int main(int argc, char **argv) { char *c = argc == 1 ? a : b; return c[0]; }
```

```
<         ASSIGN c=argc == 1 ? &a[0] : &b[0];
>         ASSIGN c=argc == 1 ? (signed char *)a : (signed char *)b;
```

The same conversion through an `if` statement (`if (argc == 1) c = a; else c =
b;`) and through a plain initialiser is byte-identical, so it is the ternary
that is special — and the site is not in the adjuster.

`migrate_expr`'s `if` arm coerces any branch whose `type_id` differs from the
node's, by construction a typecast (`migrate.cpp:1186`). It was added for the C
`assert` idiom `cond ? 0 : __assert_fail()`, whose branches diverge from a void
result, and its comment claims "well-typed ternaries already have matching
branch types". That premise is false: a well-typed conditional yielding a
pointer from array operands has branch `type_id` `array` against a node
`pointer`, so the coercion fires on the common path and wins the race against
the adjuster, which never sees an array to decay.

The consequence is worse than a spelling. `typecast(array, pointer)` is not a
form the SMT backend accepts:

```
$ esbmc v.c --clang-c-irep2-adjust-only
ERROR: Unexpected type in int/ptr typecast
```

on a nine-line program that verifies on the default path. The census could not
see it: `--goto-functions-only` stops before the encoder, so the row read
`diff`, not `crash`.

The fix gives that pair its C conversion (C11 6.3.2.1p3) rather than a cast —
`&a[0]`, exactly what `c_typecastt::do_typecast` already spells for the same
pair on both of its copies — and falls through to the typecast for every other
divergent pair, so the `assert` idiom the arm was written for is untouched.

### 116.3 Result

`regression/esbmc/irep2_only_ternary_array_decay{,_fail}` pin the verdict rather
than the printer: both abort with `Unexpected type in int/ptr typecast` before
the patch, and return SUCCESSFUL / FAILED-with-`array bounds violated` after.
That is the strongest mutant this scope has had — the pre-patch binary does not
merely print differently, it produces no verdict at all.

Census on a stride-8 list of `regression/esbmc` pinned to a file before the A/B
(233 entries, 226 with a `.c` source), against the same base:

| goto program | before | after |
|---|---:|---:|
| same | 202 | **204** |
| diff | **24** | **22** |

`github_6966_fail` and `memset-const-2` converge; none diverges that did not
before. §114 tagged this cause at three tests — the third carries a second
cause and stays.

`migrate_expr` is shared, so the default path was measured separately: over the
same 226 C sources, `--goto-functions-only` with no `-only` flag is
**byte-identical on all 226** between master and the patch. Suites:
`esbmc` 1857/1857, `cstd` 142/142, `floats` 106/106, `function_contract`
414/414, `goto-coverage` 144/144, `python/list` 294/294, `esbmc-cpp/cpp`
931/933 — `ch9_7` and `ch13_10` exceed the harness's 120 s cap locally and
`ch9_7` takes 2 m 04 s on master against 2 m 01 s here, so neither is this
patch.

### 116.4 Next

The residue is 22, and the named causes left are the temporary numbering
(`tmp$3` vs `tmp$4`, 2 tests), struct padding in an aggregate initialiser
(`github_578_success3`), and `POINTER_OFFSET` spelling in an `offsetof`
lowering (`github_2512_8`, `github_426_2`, `github_2512_12`) — the largest
remaining group and the one to reduce first. Three tests are untagged.
```
<         IF !((signed int)g == 42) THEN GOTO 1
>         IF !(g == 42) THEN GOTO 1
```

Dropping the attribute makes the pair byte-identical, so what legacy emits is
an *identity* cast: the alignment attribute leaves `g`'s type unequal to plain
`signed int` in ESBMC's type model, and `gen_typecast_arithmetic` casts on that
inequality. That puts it in §113.3's class rather than the promotion class it
was tagged as — decide whether to mirror it before writing a slice.

The array decay rendered as a cast rather than `&a[0]` (`github_6966_fail`,
`memset-const-2`) has not been reduced yet.


## 114. The two-stage census §113.4 asked for — the residue is 24, not 133
## (2026-08-22)

Every census in this scope so far has read one stage. §113.1 showed why that is
not enough: a symbol-table difference can be a `goto_convert` crash, and a
symbol-table difference can equally be nothing at all. Re-run reading both, on
the test list pinned to `595f52b025` (230 tests, stride 8), at the tip of
§111.1 + §112 + §113.1:

| symbol table | goto program | tests |
|---|---|---:|
| same | same | 95 |
| **diff** | same | **109** |
| diff | diff | **24** |
| — | crash | **0** |

Three things follow.

**The abort classes are gone.** Zero crashes in the sample, against three
distinct ones at the start of the day (§111.1, §112, §113.1). That is the whole
value of those three patches; the same-count moved by 5.

**109 of the 133 differences do not reach the goto program.** They are the class
§110.2 established with `(void)0`: the adjuster writes a value back only when it
changed it, and the un-written-back legacy value is what the symbol-table
printer shows, while `goto_convert` re-migrates from the same legacy value and
lands in the same place. Chasing them is chasing a printer.

**The residue that matters is 24.** Causes, read rather than tallied:

| cause | tests | note |
|---|---:|---|
| `DEAD` location: `no location` vs blank | **14** | §114.1 |
| integer promotion missing at a comparison | 3 | |
| array decay rendered as a cast, not `&a[0]` | 3 | |
| temporary numbering (`tmp$3` vs `tmp$4`) | 2 | |
| function-pointer identity cast | 1 | not work, §113.3 |
| untagged | 3 | |

(Tags overlap; four tests carry two.)

### 114.1 The dominant cause is the for-init hoist, and it is one line of provenance

```c
int main(void) { int s = 0; for (int i = 0; i < 3; i++) s = s + i; return s; }
```

```
<         // 48 no location
>         // 48
```

The `DEAD` for the loop-scoped `i`. Legacy leaves its location nil, which
`goto_programt::output` renders `no location`; the hop-off gives it an
empty-but-not-nil one, which renders blank. `goto_convert_functions.cpp`'s
`emitted_location` already documents this exact asymmetry — in the other
direction, where reproducing *blank* was the correct choice.

The mechanism is §105's `hoist_for_init`. Rewriting `code_for2t` into a block
moves the loop from `convert_for` to `convert_block`, and `convert_block` stamps
the block's `end_location` on every destructor it unwinds
(`unwind_destructor_stack`, goto_convert.cpp:2215) whereas `convert_for` leaves
it nil. `migrate_expr_back` is not the culprit — it already guards
`if (ref2.end_location.is_not_nil())`.

Reproduced with `--no-irep2-native-body` on both sides, so this is the legacy
converter's own asymmetry and not the W1 dispatcher's.

### 114.2 Next

`hoist_for_init`'s destructor-location provenance, which is 14 of the 24. The
other live causes — the promotion at a comparison, and the decay rendered as a
cast — are three tests each and worth a census of their own once the dominant
one is out of the way.
second cause and not a second symptom. §80 records that the VLA `sizeof`
operand is computed in migration; that is the place to look first.

That is the next target: it is the only remaining input in the censused C
corpus on which the pass produces nothing at all.
## 110. The census re-run after the sixteen PRs landed (2026-08-22)

§104 closed the census with "every measured divergence is owned by an open PR".
Those PRs are merged, so the question is what the symbol-table gap looks like
now. Re-measured on master at `595f52b025` over `regression/esbmc`,
`--clang-c-irep2-adjust-only` versus the legacy pass, blank-line differences
ignored:

| | tests |
|---|---:|
| whole suite | **693 same, 1147 differing, 1 skipped** |
| stride-8 sample | 90 same, 138 differing |

The suite figure is the first one taken; §101's 78-of-120 was a prefix of the
same suite before any of the sixteen landed, so the two are not comparable and
neither is offered as a delta. The stride sample is what the causes below are
counted on.

### 110.1 The dominant cause does not reach the goto program

| cause | tests | note |
|---|---:|---|
| `(void)0` vs `0` in a conditional arm | 89 | §110.2 |
| untagged residue | 22 | §110.4 |
| implicit callee has no location | 13 | §110.3, fixed here |
| indentation only | 11 | printer, from the §105 for-init hoist |
| `migrate_expr` renaming warning | 4 | |
| `volatile` dropped from a DECL statement | 1 | symbol keeps it; the statement does not |
| padding | 1 | |

### 110.2 `(void)0` is a legacy artefact, and mirroring it would be wrong

Reduced:

```c
void f(void);
int main() { int x = 1; x ? f() : (void)0; return 0; }
```

Legacy prints `(_Bool)x ? f() : 0;`, the hop-off `(_Bool)x ? f() : (void)0;`.
The source says `(void)0`, so the hop-off is the faithful one. `adjust_if`
compares whole `typet` ireps and casts *both* arms when *either* differs, so an
attribute-only difference between the conditional's type and an arm's is enough
to fire it; `do_typecast` then folds the cast into the constant. `adjust_if_expr`
compares interned `type2tc`, which are equal, and leaves the arm alone.

`--goto-functions-only` on the reduction is byte-identical between the two
paths: `goto_convert` drops the void arm either way. So this is §39.1's "a
caller downstream re-does the work" row of the parent document — 64 % of the
remaining symbol-table gap is a difference that no consumer sees, and the arm
that would close it does not get written.

### 110.3 The implicit callee's location, which is a real loss

`declare_implicit_callee` synthesises the symbol for a callee with no visible
declaration. It read the location off `code_function_call2t::location`, and
took none at all in the other branch: `sideeffect2t` has no location field, so
a bare `assert(x == 1);` — which is a `sideeffect2t` of kind `function_call`
under a `code_expression2t`, not a `code_function_call2t` — produced a symbol
with an empty `Location`. 13 of the 138 differing tests are only this.

The statement's location is the call's **only when the call is the whole
statement**, so that is the one position it is taken from: `adjust_expr`
declares the callee from `code_expression2t` before recursing, passing the
statement's location, and the generic arm keeps declaring the rest unlocated.
The narrower shapes — `a = f();`, `if (f())`, `return f();` — keep the call's
own column in the legacy pass and are left as they were rather than given the
statement's column, which would be the right line and the wrong one. Closing
those needs a `locationt` on `sideeffect2t`, on the pattern the `code_*2t`
kinds already use; that is a separate change and is not made here.

Result on the stride sample: **106 same, 123 differing**, and no test acquires a
divergence it did not have.

| mutant | killed by |
|---|---|
| location not passed (master) | `..._implicit_callee_location` |
| location not passed, nested statement | `..._implicit_callee_location_stmt` |

Both tests were run against the unpatched arm and fail there. The second one
exists because the first would also pass if the location were taken from the
enclosing function rather than the statement.

### 110.4 The untagged residue names four more causes

Read rather than tallied, per §104.2:

- **A cast lost at a call argument** — `atexit((void (*)())(&free_g2))` legacy
  versus `atexit(&free_g2)`. The conversion #7091 ported does not cover a
  function-pointer parameter.
- **The hop-off aborts outright** on `builtin_memcpy` and `cwe_uninit_array_vla`
  — the whole symbol table is missing. This is the by-name union tag the header
  comment on `clang_c_adjust_irep2` already documents.
- **`(signed int)b += a` versus `b += a`** — the coupled arith-assign
  conversion, `scope-coupled-arith-assign-conversion.md`.
- **Printer-only**: `+1` versus `1`, and the float literal suffix
  (`1.175494e-38f` versus `1.175494e-38`).

The abort is the one worth taking next: it is not a spelling difference but a
hard stop, and it puts two tests beyond measurement rather than merely differing.
