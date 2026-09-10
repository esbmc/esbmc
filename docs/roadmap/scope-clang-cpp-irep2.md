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

## 6. Next

1. ~~Add the reference kind to `pointer_type2t`~~ — **done**, §2.5, PR #7703.
   Item 1 is now writable, and it is the next slice: 70 % of the corpus needs
   it, and unlike §2.3's arms it will move verdicts, so it owes a
   `SUCCESSFUL`/`FAILED` pair.
2. ~~Port items 6 and 7~~ — **done**, see §2.3.
3. Price option B in §3 against option A.
4. Add the C++ hop-off flag, then run the census by verdict.

Only then does a slice make sense.
