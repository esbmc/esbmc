# Scope — Phase 7: clang-cpp → IREP2-native construction

Parent: `frontends-to-irep2.md` §6 (Phases 5-9) and §39.3. Sibling scopes:
`scope-clang-c-irep2.md` (Phase 6), `scope-jimple-irep2.md` (Phase 5),
`scope-coupled-arith-assign-conversion.md` (the typecast pre-flight, §20.1).

Opened 2026-09-10 at master `35db62c320`. This document is the census and the
design questions it forces. No slice is written yet, per §39.1's "census before
writing".

## 1. Census

Re-measured at `35db62c320` with §1's own commands, alongside the 2026-08-03
baseline in `frontends-to-irep2.md` §2:

| frontend | legacy | IREP2 | LOC | baseline (legacy / IREP2 / LOC) |
|---|---:|---:|---:|---|
| jimple | 202 | 120 | 3 918 | 176 / 0 / 3 259 |
| clang-c | 1 142 | 364 | 17 096 | 971 / 49 / 13 783 |
| **clang-cpp** | **643** | **0** | **7 559** | 626 / 0 / 7 394 |
| solidity | 1 420 | 0 | 23 599 | 1 420 / 0 / 23 589 |
| python | 6 395 | 981 | 91 563 | 5 547 / 806 / 79 282 |

Two things this says that the baseline table did not:

- Phases 5 and 6 moved their frontends off zero (jimple 0 → 120, clang-c
  49 → 364). The programme's method works.
- **Every legacy count rose**, clang-cpp's by 17. The frontends are under
  active development, so B-1 is a bar against a moving denominator. Quote the
  measurement commit or the number means nothing.

clang-cpp remains at **0** IREP2 nodes: it constructs none, and relies entirely
on `migrate_expr`/`migrate_type` at the symbol-table seam.

Corpus: `regression/esbmc-cpp` holds 2 842 `test.desc` files, roughly 1.5x
`regression/esbmc`. A whole-suite A/B is therefore a multi-hour run, not the
40 minutes `scope-clang-c-irep2.md` records for Phase 6 — budget for it, and
expect the stride-sample warning from that scope doc (§ "the stride-8 sample is
USELESS for verdict censuses") to bite harder here, not less.

## 2. The blocker, measured: the IREP2 typecast copy has no C++ arms

`frontends-to-irep2.md` §39.2 told Phase 7 to treat
`scope-coupled-arith-assign-conversion.md` §20.1 as its pre-flight list. All
seven gaps are still open at `35db62c320`. Measured, not recalled — the two
copies of `c_typecastt::implicit_typecast_followed` in
`src/util/lang/c_typecast.cpp`:

| copy | lines | length |
|---|---|---:|
| irept | 602-765 | 163 |
| expr2tc | 766-832 | 67 |

Every C++-shaped arm lives in the irept copy alone:

| §20.1 item | arm | line (irept copy) | in expr2tc copy |
|---|---|---:|---|
| 1 | lvalue/rvalue references | 628 (`take_reference_address`) | no |
| 2 | pointer-to-member | 669, 673 (`to-member`) | no |
| 3 | `incomplete_array` source | 682 | no |
| 4 | qualifier warnings | 713, 718 (`disregarding`) | no |
| 5 | `#reference` propagation | 723 | no |
| 6 | derived-object-to-base-pointer | 732, 739 (`address_of_exprt`) | no |
| 7 | string-constant to array | 753 (`string2array`) | no |

Items 1, 2, 6 and 7 are the ones §20.1 marks C++-shaped and dormant for jimple
and Python. They are **live for every assignment clang-cpp converts**, which is
most of them: C++ models `T&` as a pointer, so item 1 alone is on the path of
every reference bind.

### 2.1 Two of the arms are not portable — `pointer_type2t` cannot say "reference"

Items 1 and 5 are not "not yet ported". They are **not representable**.

`is_lvalue_or_rvalue_reference` (`std_types.cpp`) is

```cpp
type.id() == "pointer" && (type.reference() || type.get_bool("#rvalue_reference"))
```

— two irept *attributes*. `pointer_type2t` has exactly two fields, `subtype` and
`carry_provenance`, and `migrate_type`'s pointer arm builds
`pointer_type2tc(subtype, type.can_carry_provenance())` (`migrate.cpp:205`).
**A C++ reference and a plain pointer are the same IREP2 node by
construction.**

This is §113.3's shape — the attributed and plain types being one node — but
the conclusion is the opposite. There the attribute changed only what a printer
emitted, so "do not mirror" was right. Here the arm changes the *expression
built*: `T& F(T& a) { return a; }` must return `&a`, not `(int *)a`. An IREP2
pass cannot decide that, because by the time it sees the type the distinction is
gone.

So item 1 needs `pointer_type2t` to carry the reference kind before any of it
can be written. That is a W2-class representation change and it gates the
phase, exactly as W3's carriage gated Part V.

### 2.2 How much of the corpus this touches — measured

Instrumented each arm with an `fprintf` and ran a stride-10 sample of
`regression/esbmc-cpp` (285 descriptors, 283 runnable) under
`--goto-functions-only`:

| arm | firings | tests (of 283) |
|---|---:|---:|
| `take_reference_address` (item 1) | 20 338 | **199** (70 %) |
| derived-to-base (item 6) | 13 952 | **198** (70 %) |
| source-reference dereference (item 1) | 495 | 87 (31 %) |
| string-constant to array (item 7) | 11 | 6 (2 %) |
| pointer-to-member (item 2) | 0 | **0** |

**Every test in the sample fires at least one of these arms.** There is no
subset of the C++ corpus that avoids them, so there is no first slice that can
be verified while they are missing — which settles the sequencing question §2
raised.

Two readings worth keeping:

- The reference arms are the hot path, not a corner: 70 % of tests, and the
  most-fired arm in the file. The representation gap in §2.1 is therefore the
  phase's critical path, not a detail to schedule late.
- **Pointer-to-member fires zero times.** That is §39.1's "census before
  writing" earning its place: jimple migrated `nondet` before learning it never
  executed, and the byte-identity claim held for nine PRs because nothing ran
  it. Do not port item 2 on the strength of it being in §20.1's list; price it
  against a corpus that contains it first, or leave it declined and recorded.

**Consequence for sequencing.** Phase 7 cannot begin with an adjuster slice.
Porting the four C++ arms into the `expr2tc` overload is the first work item,
and `unit/util/c_typecast.test.cpp` — the differential harness #6873 added — is
where it is pinned. §20.3's lesson is the standing warning: a second
independently-written copy of a conversion is not a translation of the first,
and byte-identity on another frontend's corpus does not establish that it is.

### 2.3 Items 6 and 7 ported

The two representable arms are in, each pinned by a section of
`unit/util/c_typecast.test.cpp`'s `require_overloads_agree` — which runs both
overloads on the same input and requires the migrated results to be equal, so
the mutation check is the harness itself: each section fails on the unported
copy and passes on the ported one.

- **Item 6, derived-to-base.** `address_of2t` takes the *pointee*, not the
  pointer, so the arm passes `dest_ptr_type.subtype` where the irept copy
  assigns `dest_type` whole. Passing `dest_type` would build a pointer to a
  pointer, and the harness catches it.
- **Item 7, string-constant to array.** `migrate_expr` maps a `string-constant`
  to `constant_string2t` (`migrate.cpp:1195`), not to an array, so the arm was
  genuinely missing rather than performed by migration on the way in.
  `constant_string2t::to_array()` is the IREP2 counterpart of `string2array`,
  and the constant is retyped to the destination first, as the irept copy does.

Item 2 (pointer-to-member) stays declined on §2.2's zero. Items 3 and 4 are
warnings and an `incomplete_array` source; neither changes a built expression,
and both are left for the same census to price.

Neither arm changes anything end to end today: a `--goto-functions-only` A/B of
every `regression/esbmc/irep2_only_*` test against master is **95 same, 0
differing**, because goto-convert already normalises the one shape the string
arm touches. That is the expected result for a pre-flight port — it closes a
latent divergence between two copies, and the differential harness is what pins
it. **PR #7701.**

Items 1 and 5 remain blocked on §2.1 and are the phase's critical path.

### 2.4 §2.1 priced: `carry_provenance` is the precedent, and it cost no call sites

The reference-kind field is not a novel change to a core type. `pointer_type2t`
already carries a second, discriminating field of exactly this shape:

| property | `carry_provenance` |
|---|---|
| origin | PR #2464 (CHERI capability bounds) |
| in the `fields` tuple | **yes** — participates in equality, `crc`, `hash` |
| forward migration | `migrate.cpp:205`, `type.can_carry_provenance()` |
| back migration | `migrate.cpp:3117-3118`, `thetype.can_carry_provenance(true)` |

So the mechanism a reference kind needs — a discriminator on the pointer type,
reflected in value identity, carried both ways across the seam — is merged, in
tree, and has been for some time.

Cost, measured:

- **Producer side: zero.** The constructor is
  `pointer_type2t(const type2tc &st, const bool &p = false)`; the second
  parameter is already defaulted, so a third defaulted parameter leaves all
  **41** `pointer_type2tc(` construction sites across 22 files untouched.
- **Consumer side: 20 call sites in 7 files** — 14 calls to
  `is_lvalue_or_rvalue_reference` and 6 direct reads of `.reference()` /
  `#rvalue_reference`.

The one difference from the precedent is arity: `carry_provenance` is a bool,
whereas a reference kind has three states (not a reference, lvalue, rvalue), so
it wants an enum. That changes the field's type, not the mechanics.

This is what §2.1's decision costs. It is a smaller change than the phase it
unblocks, and it is the phase's critical path (§2.2: 199 of 283 sampled tests).

### 2.5 The reference kind is in, and `fields` was the right side of the choice

`pointer_type2t` now carries `pointer_ref_kindt { NONE, LVALUE, RVALUE }`,
following §2.4's precedent: a third constructor parameter, defaulted, so all 41
construction sites are untouched; `migrate_type` reads `#rvalue_reference` then
`#reference`, and `migrate_type_back` sets them again.

An enum rather than `carry_provenance`'s bool, because a pointer has three
possible spellings, not two.

**The one real decision was whether it belongs in the `fields` tuple**, and it
was settled by measurement rather than by argument.

| | in `fields` | excluded via `excluded_field_bytes` |
|---|---|---|
| value identity | a reference-derived pointer stops comparing equal to a plain one | preserved exactly |
| risk | whatever consults type equality — `base_type_eq`, value-set, dereference, SMT sort caching | rebuilds silently drop the field, and no A/B can see it because it takes no part in `operator==` |
| precedent | `carry_provenance` (§2.4) | `sideeffect2t::location` (§136) |

The excluded-field route looks safer and is not: PRs #7266 and #7285 were both
bugs of exactly that shape, and the second cost a corpus-wide wrong answer
before anyone noticed, because the defeated guard was `if (value != before)`.

So the question is whether the equality change actually costs anything. It does
not:

| suite | result |
|---|---|
| unit | 854 / 854 |
| C (`-L esbmc`, stride 8) | 804, 0 failures |
| python (stride 12) | 435, 0 failures |
| C++ (stride 3) | 339, 2 failures, both pre-existing on master |

A default-path goto A/B was **not** run and is not quoted: master moved to
`698c6fe353` (`[interval] … dump line breaks`, #7666) after the control binary
for the earlier ticks was built, and a commit that changes dump line breaks is
exactly what such an A/B cannot tolerate. Rebuilding a control costs ~50 minutes
at current machine load. The suites are the oracle here; say so rather than
quote a stale comparison.

**It is not consumer-free, and calling it that was wrong.** Restoring the
attributes fixes a *lossy* round trip: `symbolt::get_type()` derives the legacy
type through `migrate_type_back`, and `goto_convert.cpp:788` reads the predicate
to decide whether a temporary's destructor fires at end-of-full-expression or is
deferred to block scope ([class.temporary]/6). Measured against an independent
older build, nothing observable moves today — counterexamples and `--show-vcc`
are identical across lvalue-ref, rvalue-ref, base-ref and `const int &` programs
— so the fix is real and currently latent. Both halves belong in the record.

Two things the gates caught that are worth carrying forward:

- **`fields_cover_class` does not protect this field.** Dropping `ref_kind` from
  the tuple leaves a 7-byte shortfall against an 8-byte alignment tolerance, so
  it compiles silently. The two `REQUIRE_FALSE` equality assertions in
  `unit/util/migrate.test.cpp` are the only thing pinning that it participates in
  `cmp`/`crc`/`hash`.
- **`rebuild_with_type<address_of2t>` re-defaults it**, as it already does
  `carry_provenance`, because the constructor takes the pointee and builds the
  pointer itself. The item 1 arms must build the pointer type directly rather
  than route an address-of through `with_type`. PR **#7703**.

## 3. The design question Phase 6 leaves open: the pass is not extensible

The legacy frontends are one class specialising another:

```
class clang_cpp_adjust : public clang_c_adjust      (clang_cpp_adjust.h:17)
```

`clang_c_adjust` declares **15 virtual** members and `clang_cpp_adjust`
overrides **13** of them. The C++ frontend is built as a set of deltas on the C
one; it is not a separate adjuster.

The IREP2 side does not reproduce that extension point.
`clang_c_adjust_irep2` (Phase 6, 32 arms) declares **0 virtual** members. As it
stands, `clang_cpp_adjust_irep2` can only duplicate it.

Worse, the two do not decompose the same way, so "add `virtual` and override 13"
does not map. The overrides and the IREP2 arms line up like this:

| `clang_cpp_adjust` override | IREP2 counterpart |
|---|---|
| `adjust_member` | `adjust_member` |
| `adjust_function_call_arguments` | `adjust_call_arguments` |
| `adjust_ifthenelse`, `adjust_while`, `adjust_for` | **one** arm, `adjust_statement_condition` |
| `adjust_side_effect_function_call` | split across `adjust_call_callee` / `declare_implicit_callee` |
| `adjust_switch` | no counterpart (`code_switch2t` appears once, as a location accessor) |
| `adjust_code`, `adjust_decl_block`, `adjust_symbol`, `adjust_reference`, `adjust_side_effect` | none |

`adjust_statement_condition` reads the condition out of whichever of
`code_ifthenelse2t` / `code_while2t` / `code_dowhile2t` / `code_for2t` it is
given, so the IREP2 pass deliberately unified four legacy arms into one. That
is the better factoring, and it is exactly why the C++ override points have
nowhere to attach: `clang_cpp_adjust::adjust_while` exists to change what
happens for a `while`, and the IREP2 pass no longer has a `while` seam.

**This is the first decision Phase 7 must make, and it is not a slice.** Three
options, none yet costed:

- **A — retrofit virtuals onto `clang_c_adjust_irep2`.** Cheapest to write,
  but it re-imports the per-statement-kind seams the unified factoring removed,
  and only where C++ needs them.
- **B — a shared arm table.** #7455 already made the pass's arm order data
  rather than control flow (`[clang-c] Make the IREP2 adjust pass's arm order
  data, not control flow`). A second frontend supplying its own table entries
  is the natural extension of that change and needs no virtual dispatch.
- **C — duplicate.** Rejected on sight; 32 arms is the whole of Phase 6.

Option B is the one that follows from work already merged, and it should be
priced first. Deciding this wrong means re-doing Phase 6 inside Phase 7.

## 4. What does not exist yet

- **No hop-off flag.** `--clang-c-irep2-adjust-only` has no C++ counterpart
  (`grep -n 'cpp-irep2' src/esbmc/options.cpp` is empty). Phase 6's entire
  instrument — A/B one binary against itself with and without the flag — is
  unavailable until one is added. That is the second work item, and it is a
  prerequisite for any census by verdict.
- **No scope-doc census by construct.** §39.1's "census before writing" prices
  every construct once, at the start. For clang-cpp that census cannot be run
  until the flag exists, so §1's counts are the static census only.

## 5. Risks, carried forward from Phases 5 and 6

| # | Risk | Source |
|---|---|---|
| R1 | A second copy of a conversion is not a translation of the first | §20.3, #6873 |
| R2 | An unmoved mutant has five causes and only one is a fact about the code | §39.1's table |
| R3 | A stride sample is useless for a verdict census | `scope-clang-c-irep2.md` |
| R4 | `--goto-functions-only` stops before the encoder, so a `diff` row can be an abort | `scope-clang-c-irep2.md` |
| R5 | Excluded fields (`location`, `pragma_unroll_count`) take no part in `operator==`, defeating both the write-back guard and any dump A/B | PR #7285 / #7266 |
| R6 | A seam loses what IREP2 has no field for, and it surfaces as a printer difference | §137 |

R6 is the newest and the least obvious: §137 found `is_padding` dropped by
`migrate_type_back` for want of a per-member flag. clang-cpp has more such
attributes than any other frontend — `#cpp_type`, `#member_name`, catch-match
spellings (§33) — so W3's carriage problem lands here first.

### 3.1 Option B taken: one arm table, shared

Priced and chosen. `clang_c_adjust_irep2`'s private section became `protected`,
`adjust_sole_arms` became virtual, and the arm row became a template over the
pass (`adjust_arm<Pass>` in `clang_c_adjust_irep2.h`), so `clang_cpp_adjust_irep2`
orders the arms it inherits alongside its own in one table. No per-statement-kind
virtual was reintroduced, which is what option A would have cost.

One compiler constraint shaped the row. A pointer to a base member stored in a
derived-typed table is legal, but GCC 13.3 mis-reads the call once the runner
inlines it and rejects it under `-Werror=array-bounds` at `-O2`; clang 18 accepts
it. The row therefore holds a function pointer produced by a captureless lambda
trampoline, which is an address constant, so the table stays
constant-initialised.

`unit/clang-c-frontend/adjust_arms.test.cpp` reads `arm_order()` as a drift
guard. It checks a **subsequence**, not equality: a sibling frontend adding rows
must not fail the C pass's guard.

### 3.2 The hop-off flag, and the census by verdict

`--clang-cpp-irep2-adjust-only` mirrors `--clang-c-irep2-adjust-only`: the IREP2
pass *replaces* the legacy one, so a divergence under the flag is a missing arm
rather than a shadow-mode artefact. With the table carrying only the inherited C
arms, the C++ corpus named the missing work rather than guesswork doing it.

Everything from §3.3 on is one row of that census.

### 3.3–3.11 The arms the census named

Landed, in order: the reference arms of `implicit_typecast_followed`
(`c_typecast.cpp`), the C++ member-call arm, exception ids
(`clang_cpp_exception_id.{h,cpp}`, freed from `clang_cpp_adjust` so both passes
compute them from one place), and vtable-pointer generation
(`clang_cpp_code_gen.h`, reached through a `gen_symbol_code` hook so a bodyless
symbol still gets its vptr writes).

### 3.12 Base-conversion displacement: the soundness row

The row that mattered. Three tests proved `SUCCESSFUL` under the flag against
the legacy pass's `FAILED` — silently false proofs, not crashes:
`destructors/github_6263_nonvirtual_base_delete`,
`inheritance/github_7025_vbase_nonfirst_member_fail`,
`inheritance/mi_base_subobject_layout_fail`.

Four separate defects, each found by fixing the one before it:

1. **The marker reaches IREP2 on the wrong node kind.** A `#derived_to_base`
   marker names whichever expression is being converted. Counted over
   `regression/esbmc-cpp`: **symbol 32474, dereference 4502, sideeffect 30,
   address_of 19, typecast 58**. Carrying it on `typecast2t` alone therefore
   reached 0.2 % of them. IREP2 has nowhere to hang a flag on an arbitrary node,
   so `migrate_before_dispatch` wraps a marked non-cast node in a same-type
   `typecast2t` — the identity — and `back_typecast` unwraps it, leaving the
   legacy tree unchanged. `#base_to_derived`, by contrast, is on a typecast
   **14991 times out of 14991**, so the field carries it outright.
2. **The displacement must come from ESBMC's own layout.** `base_displacement`
   and its three helpers moved out of `clang_c_adjust_expr.cpp` into
   `clang_c_base_layout.{h,cpp}` so both passes ask one oracle. Recomputing it
   from clang's `ASTRecordLayout` is the mistake #3894 records.
3. **A dereference the converter leaves typed `empty`.**
   `clang_c_adjust::adjust_dereference` takes the node's type from the pointer's
   subtype; the IREP2 port had only the array and function-pointer cases. The
   C++ converter builds `*this` typed `empty` and relies on that assignment, so
   without it every member offset resolved below the dereference is taken
   against the wrong struct.
4. **One cast can carry both markers.** `clang_cpp_convert_vft.cpp` marks a
   `dynamic_cast`'s typecast `#base_to_derived` and `#derived_to_base` at once.
   `clang_c_adjust` resolves them by *re-entering* `adjust_expr` on the
   marker-stripped node, so base-to-derived runs first and the derived-to-base
   displacement applies to its result — the whole node, not the cast's operand.
   An arm that rebuilds a `typecast2t` must forward the marker it is not
   consuming, or the `-16` re-base is dropped while the `+16` is applied.

A fifth defect surfaced only because the pass now rewrites more symbols:
`back_sideeffect` default-constructed the size `exprt` and wrote it into
`#size` unconditionally. An empty irep is a third state — `is_not_nil()` reports
it as *present* — so the forward arm preferred it over the real `size` field and
threw `migrate expr failed`. See `irept-find-location-nil-ambiguity`: absent,
present-nil and present-empty are three states, and `is_nil()` separates only
two of them.

**Measured.** Over the 402 tests in `regression/esbmc-cpp/{inheritance,
destructors,polymorphism_bringup,polymorphism_bringup_overload,try_catch,
inheritance_bringup}`, verdict agreement between the legacy pass and
`--clang-cpp-irep2-adjust-only`:

| | agree | false proofs |
|---|---|---|
| before | 160 / 402 | 3 |
| after | **293 / 402** | **0** |

No test moved away from the legacy verdict. The first attempt did regress one
(`inheritance/mi_dynamic_cast_fail`), which is how defect 4 was found; the
before/after census is what caught it, not the suite, because the three
displacements cancelled in the goto dump and the symbol table — both were
byte-identical while the verdicts differed.

### 3.13 The catch handler's type does not cross the seam

36 of the 172 `try_catch` tests fail under the flag with `exception lowering:
cannot lower an unsupported handler shape` (`remove_exceptions.cpp:928`). The
cause is one carriage loss, measured rather than guessed:

`clang_cpp_adjust::adjust_catch` reads each handler's catch type off the
**handler block's own type**, computes the id from it, writes it to
`exception_id`, and only then resets the block to `code_typet()`. So the catch
type lives on the block's type between conversion and adjust, and nowhere else.

`code_block2t`'s constructor hardcodes `get_empty_type()`
(`irep2_expr.h`), so `migrate_expr` drops it. Instrumented on
`try_catch/lower-exceptions_empty_catchall`, the legacy arm sees
`ty=code ellipsis=1` and produces the id `ellipsis`; the IREP2 arm sees
`tyid=empty ellipsis=0` and falls through `convert_exception_id`'s last-resort
branch to the id `empty`, which matches no throw.

A second, separate defect hides behind it: `is_unresolved_cpp_catch` tests
`exception_list.empty()`, but migrate's source-form arm pushes each handler's
`exception_id` attribute into that list whether or not it is set, so an
unadjusted catch arrives with one **empty id per handler**, never an empty list.
The arm is therefore dead. Fixing the guard alone changes no verdict — the ids
it then computes are `empty` for want of the type — so the two have to be fixed
together.

Three options for the carriage, none yet costed:

- **A — give `code_block2t` a type.** Smallest conceptually, largest blast
  radius: the type participates in `cmp`/`crc`/`hash` for every block in every
  frontend.
- **B — put the handler types in `code_cpp_catch2t`,** parallel to
  `exception_list`. Contained, but stores what the block already knew.
- **C — compute the ids in the converter,** so `exception_id` is set before
  either pass runs and no type needs to cross the seam. Architecturally the
  cleanest, and it deletes work from the legacy pass rather than adding a field.

**C's one assumption holds, measured.** The doubt was whether the class's type
symbol is complete early enough for `convert_exception_id` at the
`CXXTryStmtClass` site. A probe calling it there over the whole `try_catch`
suite saw **266 handlers and 0 fall through to the last-resort id** — every one
resolved to a real name (`ellipsis` 51, `signed_int` 41, a class tag 31, …).
The converter already knows everything the adjust pass reads off the block type.

One trap the probe surfaced: `is_catch` is what suppresses the `tag-` strip, and
neither legacy call site sets it. Whatever computes a handler id must leave it
`false`, or the id matches no throw.

**Taken, and measured.** The id is now read at the `CXXTryStmtClass` site and
written to `exception_id`, so nothing crosses the seam and `adjust_catch` keeps
only the block-type reset. Handler-shape rejections went 36 to 0, `try_catch`
tests producing a verdict 118 to 162, and agreement over the 402-test census
293 to 361 -- no test changing away from the legacy verdict, and the default
path unchanged at `try_catch` 172/172.

### 3.14 What the census names next

**Reproducing the census.** `scripts/irep2-migration/parity_sweep.sh` is the
harness; `PARITY_FLAG` selects what it sweeps:

```sh
for d in inheritance destructors polymorphism_bringup \
         polymorphism_bringup_overload try_catch inheritance_bringup; do
  PARITY_FLAG=--clang-cpp-irep2-adjust-only PARITY_TIMEOUT=40 \
    scripts/irep2-migration/parity_sweep.sh build/src/esbmc/esbmc \
    regression/esbmc-cpp/$d
done
```

Pass the binary by a path, not a bare name, and read the per-suite totals: a
run that measured nothing still prints `0 divergence(s)`.

At this point in the series that reports **33 divergences over 394 tests** --
`destructors` 4 of 14, `try_catch` 29 of 168, and **zero** in `inheritance`
(102), `polymorphism_bringup` (46), `polymorphism_bringup_overload` (49) and
`inheritance_bringup` (15). None is a false proof. Two clusters account for the
four in `destructors` and the arm that closed seven more:

- **`cpp_delete` (7 rows)** -- `destructors/github_6198*` (5) and
  `3_SI_virtual_ntvalDtor` (2). `clang_cpp_adjust::adjust_cpp_delete` attaches a
  `destructor` call to the side effect, which goto_convert emits as
  `~T(&(*p))`; the IREP2 table has no such arm, so `delete p` through a virtual
  destructor runs no destructor and `assert(n == 3)` fails. No seam work: the
  call already travels in `sideeffect2t::arguments[0]` and back
  (`migrate.cpp`, `back_sideeffect_cpp_delete`), so this is a plain arm port.
- **`cpp-pseudo-destructor` (4 rows)** -- `destructors/pseudo-destructor*`
  aborted with `migrate expr failed: cpp-pseudo-destructor`. The node has no
  migration arm at all, because the legacy pass deletes it before anything
  migrates: `adjust_cpp_pseudo_destructor_call` replaces it with its base
  expression. It therefore cannot be an IREP2 arm -- the elimination has to move
  to conversion time, as §3.13's did. **Done**, §3.15: `destructors` now sweeps
  0 divergences.

The remaining 29 `try_catch` rows are false alarms clustered on
`exception_spec_*`, which is `finalize_exception_specification`'s territory.

### 3.15 The pseudo-destructor call, reduced where it is built

`b.~a()` on a scalar has no run-time semantics beyond evaluating the base
([expr.pseudo]/1). The converter built a `cpp-pseudo-destructor` node for it and
`clang_cpp_adjust` replaced the whole call with that base; IREP2 has no kind for
the node, so under the flag it reached `migrate_expr` and aborted.

Reducing it at `clang_cpp_convertert::get_expr`'s exit -- not in one call case
-- keeps it out of the goto program and covers every call spelling, which is the
coverage the legacy arm had. Note the shape is a plain `CallExpr`, not a
`CXXMemberCallExpr`: a pseudo-destructor applies to scalars, so clang never
builds a member call for it. Reducing it in the member-call case, which is where
it looks like it belongs, fires never.

`clang_cpp_adjust::adjust_cpp_pseudo_destructor_call` is now unreachable --
instrumented, it fires 0 times across all five tests that exercise the construct
-- and is left in place. Deleting it is a branch removal, which owes a C-Dead
proof this slice has not run; it is a cleanup candidate, not a loose end in the
migration.

### 3.16 The reference arm, and why the six-suite number was not the whole story

`regression/esbmc-cpp/cpp` had never been swept under the flag. Doing so found
more divergences than the six tracked suites together, and -- unlike them -- it
contains **false proofs**:

| test | legacy | flag |
|---|---|---|
| `github_4183_fail` | FAILED | **SUCCESSFUL** |
| `github_4243_mem_init_fail` | FAILED | **SUCCESSFUL** |
| `github_4243_mem_init` | SUCCESSFUL | FAILED |

The pair inverts, which is the tell: the value is consistently wrong rather than
the analysis being imprecise. `--goto-functions-only` on `github_4183_fail`
differs in one instruction:

```
legacy  ASSIGN *return_value$_operator[]$1 = 7;
flag    ASSIGN  return_value$_operator[]$1 = &7;
```

`std::array::operator[]` returns a reference, so the assignment must go
*through* it. Under the flag the dereference on the left becomes an address-of
on the right: the write lands on the reference variable, the element keeps its
zero-initialised value, and `assert(a[0] == 0)` proves.

The cause is a missing arm, not a subtle one. `clang_cpp_adjust::adjust_reference`
dereferences a reference-typed operand, and `clang_c_adjust` calls it from five
sites -- the relational arm, binary arithmetic, complex unary, and twice in
`adjust_side_effect_assignment`. The IREP2 pass has no counterpart: `grep
adjust_reference` over `clang_c_adjust_irep2.*` and `clang_cpp_adjust_irep2.*`
returns nothing. Every reference-typed operand in those positions is therefore
left as a bare pointer.

This is the next arm, and it should come before the `exception_spec_*` cluster:
it is the only known live soundness defect.

**Written, measured, and not yet shippable.** Ported as a virtual hook (empty
for C, as the legacy one is), it closes all three false proofs and takes 10 rows
overall, 89 to 81. But it **introduces two**, `github_6319_mutex` and
`github_6319_mutex_api`, both `SUCCESSFUL -> FAILED`. Causation is established,
not assumed: gating the hook behind an environment switch gives the legacy
verdict with it off and the regression with it on.

Three things are already ruled out:

- *Ordering.* `clang_c_adjust` calls `adjust_reference` **after** the conversion
  for relational and arithmetic, and **before** it for assignment -- which it
  documents, because otherwise the source is cast to the reference type.
  Matching that ordering exactly does not fix the mutex rows.
- *The predicate.* `pointer_type2t::ref_kind` and legacy's
  `is_lvalue_or_rvalue_reference` agree on both rewrites the mutex test takes.
- *The referent type.* Both rewrites produce a pointee that is a bare
  `symbol_type2t`, which has no width, and symex reports that as a spurious
  alignment failure. Resolving it through `ns.follow` is right on its own terms
  and does not fix the mutex rows either.

What the reproducer actually says: on `github_6319_mutex_api`, instrumenting
both passes shows **legacy dereferences nothing at all** while the IREP2 hook
dereferences two operands -- and both belong to *one* `sideeffect_assign` whose
LHS and RHS are **both** references. That is reference *binding*, not a write
through a reference; dereferencing both copies the mutex instead of binding the
pointer.

So the arm fires where the legacy pass provably does not. Narrowing that
further, on the same test:

- Legacy's `adjust_reference` *is* called, four times, but only on assigns to
  `__owns` -- a `bool`, no reference in sight. It is **never** called on the
  `__m` assignment, which is the one the IREP2 hook rewrites.
- The seam is not conflating anything: `migrate_expr` maps a legacy
  `code`/`assign` to `code_assign2tc` and a `sideeffect`/`assign` to
  `sideeffect_assign2tc`, and the hook's guard is the latter. So the `__m` node
  genuinely is a side-effect assign in both representations.

Which leaves one explanation: the legacy pass does not **visit** that node,
while the IREP2 walk does. The difference is in walk coverage, not in the
predicate, the ordering, the referent type, or the node kind -- all four of
which have been measured and ruled out. That is where the next attempt starts;
inventing a condition to suppress the rewrite without knowing why legacy skips
the node is how an unsoundness ships wearing a fix's clothes.

Note also that a test for this must use a **function returning a reference**
(`b.at() = 7`, which is what `std::array::operator[]` is). A local `int &r`
binding is lowered without going through this path, so a pair built on one
passes with the arm on or off and pins nothing.

**Lesson for the census.** "Zero false proofs" held only over the six suites
§3.14 sweeps. The suites were chosen because early divergences clustered there,
and that selection quietly became the measurement. A number is scoped by what
was swept, and the scope has to be stated with it.

## 6. Next

1. ~~Add the reference kind to `pointer_type2t`~~ — **done**, §2.5, PR #7703.
2. ~~Port items 6 and 7~~ — **done**, see §2.3.
3. ~~Price option B in §3 against option A~~ — **done**, §3.1: option B.
4. ~~Add the C++ hop-off flag, then run the census by verdict~~ — **done**, §3.2.
5. ~~The 109 remaining divergences in §3.12's census~~ -- §3.13, the
   `cpp_delete` arm and §3.15 closed 79 of them. §3.14 has the command that
   reproduces what is left: 29 `try_catch` rows, all on `exception_spec_*`,
   and nothing anywhere else.
6. `finalize_exception_specification` is legacy-only, and those 29 rows are
   its territory. It is the last cluster in these six suites.
7. `scope-clang-c-irep2.md` §134.4's ternary decay, which is inert on C but
   reaches the goto program on C++.
8. ~~`regression/esbmc-cpp/cpp` has never been swept~~ -- swept, and it is the
   larger half: see §3.16. Take it before item 6.
