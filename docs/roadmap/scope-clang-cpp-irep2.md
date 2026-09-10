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

### 2.6 Item 1 ported, and what it took (PR #7705)

Both reference arms are in. Three things worth carrying forward, none of which
was visible before the differential harness rejected a first attempt.

**The address-of must carry the destination's own kind.** irept hardcodes
`#reference` in `take_reference_address` and gets away with it because
`operator==` skips comment attributes (`irep.cpp`, literally "comments are NOT
checked") — so `T&` and `T&&` are the same type there. `ref_kind` is a real
field, so a hardcoded kind leaves `do_typecast`'s `dest_type != type` guard true
and appends a cast irept never produces. Two further carriage sites had to
follow: `migrate_expr`'s address-of arm built its pointer from the pointee and
dropped the spelling, and `rebuild_with_type<address_of2t>` re-defaulted it —
that one silently affected every `with_type` caller, not just this one.

**One shape the two copies cannot agree on, and irept is the wrong side.** For a
`T&&` destination irept produces an address-of spelled `#reference`, i.e. an
lvalue reference. Measured, neither side adds a cast and only the spelling
differs:

```
legacy_migrated:  address_of  ref_kind : lvalue_reference
native:           address_of  ref_kind : rvalue_reference
```

The IREP2 side is the faithful one, so that section pins the shape rather than
byte equality. A port is not obliged to reproduce a defect it can see.

**The complexity gate was already failing before any of this.**
`implicit_typecast_followed` sits at 19 against a `core` threshold of 15 on
master, so *any* edit to it fails the gate — which is what #7701 and #7703 were
failing on, not their own additions. Split into `convert_reference` and
`convert_to_pointer`, with the pointer-compatibility disjunction factored out;
the gate then reports no function over threshold. Anything else touching this
function inherits the same obligation.

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

### 3.1 §3 answered: neither recorded option, and the working one is cheap (2026-09-10)

Measured against the table #7455 actually built, not against §3's sketch.

```cpp
struct arm
{
  const char *name;
  void (clang_c_adjust_irep2::*run)(expr2tc &);
  bool (*when)(const expr2tc &);
};
static const arm arms[];          // 24 rows, constant-initialised
```

**Option B as §3 states it does not compile.** "A second frontend supplying its
own table entries" cannot work while `run` is a pointer-to-member of
`clang_c_adjust_irep2`: a derived arm's type does not convert to the base's, and
the table is `static const` on the base, so a C++ row has nowhere to go.

**Option A is still the wrong shape**, for §3's original reason: retrofitting
virtuals re-imports the per-statement seams the unified
`adjust_statement_condition` removed, and only where C++ needs them.

**What works: each frontend owns a table typed on its own class, and the runner
becomes a template.** Five properties, each compiled rather than recalled — and
the fifth changed the design:

| property | result |
|---|---|
| `decltype(&Derived::shared)` | `void (Base::*)(…)` — the *naming* class decides, so the C rows keep their type |
| a table of `void (Derived::*)(…)` | holds **both** `&Derived::own` and `&Base::shared`, via the implicit base→derived member-pointer conversion |
| that table declared `constexpr` | **0 dynamic initialisers** — constant-initialisation survives, which is the property the table's own comment protects |
| `template <class T, size_t N> run_all(T &, const Row (&)[N], …)` | drives either frontend's table |
| the same table compiled with `-Werror` | clang 18 clean at `-O0` and `-O2`; **GCC 13.3 fails at `-O2`** |

**And that last row is why the member pointer is the wrong carrier.** Legal is
not enough: once the runner inlines, GCC's array-bounds analysis reads the
member-pointer discriminator as a vtable index and rejects the call —

```
error: array subscript 'int (**)(...)[0]' is partly outside array bounds
       of 'Derived [1]' [-Werror=array-bounds=]
```

clean at `-O0`, failing at `-O2`, and clang accepts both. ESBMC builds locally
with GCC and has an `ENABLE_WERROR` option, so B′ as first written would break a
developer build while passing CI — the worst way round.

**B″, which is what landed (PR #7714).** Each row carries a trampoline —
`void (*)(Pass &, expr2tc &)`, a captureless lambda the `ARM` macro generates —
instead of a pointer-to-member. There is then no base→derived conversion for the
analysis to mis-read, the row is still an address constant (0 dynamic
initialisers, re-measured), and both compilers accept it at both levels. The 24
C rows keep their order and the goto program is byte-identical on all 95
`irep2_only` tests.

So a `clang_cpp_adjust_irep2` table can list the 24 inherited C arms **by name**
alongside its own, in one ordered table, with no virtual dispatch, no
`std::function`, and no start-up cost. The C table does not change.

The cost is confined to making `arm` and `adjust_sole_arms` generic over the
concrete pass type — the only two places in the dispatch machinery that name
`clang_c_adjust_irep2` — plus `arm_order()`, which
`unit/clang-c-frontend/adjust_arms.test.cpp` reads and which stays per-class.

**What this does not answer.** B′ settles how a C++ pass *dispatches*; it says
nothing about which arms C++ needs. §3's mapping table stands: only
`adjust_member` and `adjust_function_call_arguments` line up by name with an
IREP2 arm, and `adjust_code`, `adjust_decl_block`, `adjust_symbol`,
`adjust_reference` and `adjust_side_effect` have no counterpart at all. Those are
the real Phase 7 work, and §4.1 says representation will not obstruct them.

### 3.2 The pass stands up, and one arm is 95 % of the gap (2026-09-10)

`clang_cpp_adjust_irep2` derives from the C pass and substitutes its own table
(PR #7717). One virtual selects the table; the C arms become `protected`. Its
table lists only the inherited C arms, so running it as the sole adjuster under
`--clang-cpp-irep2-adjust-only` *measures* what C++ needs.

Over 80 `regression/esbmc-cpp` tests, verdict against the default path:

| | count |
|---|---:|
| agree | 17 |
| diverge | 63 |
| …of which one signature | **54** |

```
ERROR: do_function_call: unexpected callee expression (id: member)
```

A C++ method call. Six of the other nine are multi-file tests the sweep fed only
their first source, one expects a parse error. **So one arm — member-function
call lowering — accounts for essentially the whole gap**, and it is the next
slice. §3's mapping table guessed five missing arms from names; the corpus says
start with the one the legacy `adjust_side_effect_function_call` override covers.

Worth noting what already works with inherited arms alone: a reference bind
through a method-free struct verifies identically on both paths. The C arms are
not merely inert on C++ input.

The nine table guards moved to a shared header (one definition, two tables), and
`adjust_arms.test.cpp` pins that the two tables stay in the same order — a row
added to one and not the other should be a decision, not an accident.

### 3.3 The crashes are one cause, and it is not one §3 predicted (2026-09-10)

§3.2 left 24 of 80 tests crashing under `--clang-cpp-irep2-adjust-only` once the
member-call arm landed. Symbolised and clustered by top frame — SIGSEGV is not a
cause, and this file has split symptom-named clusters before — **all 23 that
produce a backtrace share one site**:

```
remove_exceptions(goto_functionst &, contextt &, namespacet const &)
    src/goto-programs/remove_exceptions.cpp:1857
```

Reduced to two lines, with the control beside it:

| program | default | hop-off |
|---|---|---|
| a method call, no exceptions | SUCCESSFUL | SUCCESSFUL |
| `int main(){try{throw 1;}catch(int e){return e-1;}return 1;}` | SUCCESSFUL | **SIGSEGV** |

**Cause.** `clang_cpp_adjust::adjust_catch`
(`clang_cpp_adjust_code.cpp:319-336`) sets `exception_id` on each catch block,
and the throw arm (`clang_cpp_adjust_expr.cpp:483`) does the same, both via
`convert_exception_id`. The IREP2 pass *replaces* `clang_cpp_adjust`, so none of
that runs, and `remove_exceptions` reaches catch/throw nodes with no catchable-type
id.

**It is not a representation gap.** `migrate_expr`'s cpp-catch arm carries
`exception_id` across the seam in both directions, so §4.1's conclusion stands —
what is missing is the arm that *computes* the ids, not a field to hold them.
Worth stating because the migrate census could not have found this: the census
runs after the legacy adjuster, so the attributes were already present when it
looked.

### 3.4 The ids are computed, and the residue is a spelling the seam drops

PR #7719 ports both arms: `convert_exception_id` becomes a free function taking a
namespace (it read no other instance state), and the IREP2 arms populate
`code_cpp_catch2t`'s and `code_cpp_throw2t`'s `exception_list` fields.

| over 80 tests | before | after |
|---|---:|---:|
| agree | 31 | **51** |
| diverge | 49 | 29 |
| crash | 23 | **0** |

**The residue on the reproducer is a W3 instance that reaches a verdict, not a
printer.** `throw 1` yields the id `signedbv` on the hop-off where the legacy path
yields `signed_int`: the id is computed from `migrate_type_back(...)`, and the C
spelling does not survive `migrate_type`. The throw and the handler then disagree
and the exception escapes, so the two-line reproducer still reports FAILED where
the default path succeeds.

Class-typed exceptions are unaffected — their id comes from the tag name, which
does survive — which is why the corpus improves from 31 to 51 regardless.

This is worth separating from §5's R6 as stated. R6 anticipated a dropped
attribute surfacing as a *printer* difference; here it changes a verdict. The
spelling has to be carried, or the ids computed before migration. Reconstructing
`signed_int` from a 32-bit `signedbv` is available and is the wrong answer, for
§137's reason: do not rebuild what the representation dropped, either carry it or
do not claim it.

**What it says about §3's mapping.** Neither of the two arms the corpus has now
demanded — member-call lowering and exception-id assignment — is among the five
§3 predicted from name comparison (`adjust_code`, `adjust_decl_block`,
`adjust_symbol`, `adjust_reference`, `adjust_side_effect`). Two for two, the
corpus named a different arm than the names did. Treat §3's table as an inventory,
not a work order.

### 3.5 The residue is 23 false alarms with one cause, and no missed bugs (2026-09-10)

With the crashes gone, the 29 remaining divergences over the 80-test sample split
by *direction* first, because that is the question that matters for a verifier:

| direction | count |
|---|---:|
| false alarm — default SUCCESSFUL, hop-off FAILED | **23** |
| **missed bug — default FAILED, hop-off SUCCESSFUL** | **0** |
| no verdict on one side (multi-file tests the sweep fed one source) | 6 |

**Zero missed bugs.** Every divergence the C++ hop-off produces today is in the
loud direction. That is worth stating explicitly: a pass under construction that
over-reports is a nuisance, one that under-reports is a soundness hole, and this
one has not produced a single instance of the latter across the sample.

All 23 false alarms share one property:

```
file /esbmc-vfs/cpp/ostream line 138 column 3 function operator<<
dereference failure: NULL pointer
```

which is `o._put_field(val, strlen(val))` in
`operator<<(ostream &, const char *)`. Reduced:

```cpp
#include <iostream>
int main() { const char *p = "hi"; std::cout << p; return 0; }
```

default SUCCESSFUL, hop-off FAILED. The literal form (`std::cout << "hi"`) fails
identically, so it is not the array-to-pointer decay of a string literal — an
already-decayed pointer argument reproduces it.

The trace puts the violated property at **State 1** with no assignment before it,
so the call is not being set up rather than an argument holding a wrong value.
§3.6 instruments it; the answer was not parameter binding.

### 3.6 The ostream false alarm is a dropped code-generation step, not an arm

Instrumented rather than inferred, and the inference in §3.5 would have been
wrong. `adjust_call_arguments` converts `operator<<`'s arguments correctly; the
reference parameter takes the `binds_by_reference` path and the pointer argument
converts pointer→pointer. Carrying the parameter's `ref_kind` into that
address-of changes nothing, which rules the reference machinery out.

Diffing the whole canonicalised goto program for the two-line reproducer — 897
differing lines — finds it:

| | default | hop-off |
|---|---:|---:|
| `@vtable_pointer=&virtual_table…` assignments | **55** | **0** |
| …for `ostream` alone | 7 | 0 |

`clang_cpp_adjust::gen_vptr_initializations`
(`clang_cpp_adjust_code_gen.cpp:5`) writes each class's vtable pointer at
constructor entry. The IREP2 pass replaces `clang_cpp_adjust`, so none of it
runs, every vptr stays uninitialised, and `ostream`'s `_discard()` — dispatched
through `*o->std::ostream@vtable_pointer->…` — dereferences it. The NULL
dereference at `ostream:138` is that, one call later.

**It does not fit the arm table, and that is the finding.** `gen_vptr_initializations`
takes a `symbolt &` and emits statements at the front of a constructor body: it is
per-symbol code *generation*, not a per-node rewrite. §3.1 settled how a C++ pass
dispatches arms; it said nothing about a pass that also has to synthesise code.
`adjust()` walks values and applies the table, and has no hook for this.

So the next step is not a fourth arm but a second kind of work in the pass, and
the order matters — the vptr writes must precede the constructor's own body, as
the legacy pass arranges. `clang_cpp_adjust` has one further step of this shape,
`finalize_exception_specification`, so the hook wants to take both rather than be
built for one.

**A second divergence the same diff shows**, unrelated and smaller: a ternary over
string literals decays per arm on the default path (`val ? &"1"[0] : &"0"[0]`) and
as a whole on the hop-off (`&(val ? "1" : "0")[0]`). That is §134.4's ternary decay
in `scope-clang-c-irep2.md`, recorded there as not reaching symex on C; on C++ it
reaches the goto program. Worth its own row rather than being folded into the vptr
work.

### 3.7 The hook exists, and the corpus is at 91 % (2026-09-10)

PR #7724 gives the pass a `gen_symbol_code` hook beside the node walk and ports
`gen_vptr_initializations` into it. The C pass generates nothing, so the hook is
a no-op there.

| over 80 tests | §3.4 | after |
|---|---:|---:|
| agree | 51 | **73** |
| false alarms | 23 | **3** |
| **missed bugs** | 0 | **0** |
| no verdict one side | 6 | 4 |

Vtable-pointer writes go 0 → **55**, matching the default path exactly.

**Generation runs before the walk, not after.** The legacy pass generates after
adjusting the body; doing that here would mean migrating the value back to legacy
form, mutating it and migrating forward again — the round trip §137 shows is
lossy. Generating first lets the emitted assignments go through the arms like any
other statement. Worth knowing the two passes differ in that order.

**Freeing a member exposes dead code that private status hid.** Making
`gen_vptr_initializations` a free function left `gen_vptr_init_code` and
`gen_vptr_init_lhs` with no callers — `-Werror=unused-function` says so, and
nothing in `src/` or `unit/` referenced either. Both were already dead on master;
as private members nothing could observe that. Removed with the compiler as the
proof.

**What is left.** Three false alarms and four harness artefacts. One of the three
is §3.4's builtin exception spelling (`throw 1` yields `signedbv` where legacy
yields `signed_int`). The others are unexamined.

### 3.8 The last three false alarms are three causes, not one (2026-09-10)

Named, so the next tick does not re-derive them:

| test | violated property |
|---|---|
| `bitset/bitset6` | `cpp/bitset:88` — incorrect alignment accessing a data object |
| `cpp/ch21_31` | the test's own `assert`, via `std::queue` |
| `cpp/github_2284_4` | `cpp/vector:325` — invalid pointer |

Three different OMs and three different properties. Nothing links them yet, and
after §3.3's single-cause cluster it would be easy to assume another; they are
listed separately until measurement says otherwise.

**`ch21_31` reduced**, with the controls that bound it:

```cpp
std::queue<int> q; q.push(12); q.push(75);
q.back() -= q.front();          // default SUCCESSFUL, hop-off FAILED
```

- Reading only — `assert(q.back() == 75 && q.front() == 12)` — agrees on both
  paths, so `back()`/`front()` themselves are fine.
- The same shape on a hand-written struct returning `int &` agrees on both paths,
  both for `b.back() = 63` and `b.back() -= b.front()`.

So it is **not** compound assignment through a reference in general, which was
the obvious guess and is wrong. It is specific to the reference `std::queue`'s
`back()` returns, which reaches through the underlying container. The narrower
`q.back() = 63` and single-push variants both exceed 90 s under the queue OM, so
the next step wants a longer budget or a hand-rolled stand-in for the OM's
indirection rather than a further reduction of the OM itself.

`bitset6` and `github_2284_4` are unexamined.

### 3.9 The stride-10 sample under-covers try/catch, and one port was refuted

**The sample is not representative of the exception paths.** Over
`regression/esbmc-cpp/try_catch` (first 14 directories) the hop-off agrees on 8
and diverges on 6 — 43 % — against 7 of 80 on the stride-10 sample the earlier
sections use. A stride over a directory listing weights by directory size, and
`try_catch` is small next to `algorithm` and `cpp`. Quote the sample when quoting
the 73/80, and census the exception directories separately.

One of the six produces no verdict at all
(`exception_spec_dynamic_violation_fail`, default FAILED), so a crash survives
somewhere the sample never looked.

**`finalize_exception_specification` is not the cause, and porting it was
refuted.** It was the obvious candidate: `clang_cpp_adjust` calls it per code
symbol, the IREP2 pass replaces that adjuster, and 30 tests in the corpus use a
dynamic exception specification. Ported into the §3.7 hook it changes **nothing**
— `try_catch` stays at 8 agree / 6 diverge, and neither
`exception_spec_dynamic_allowed` nor the violation test moves.

Two things the attempt did establish, both worth keeping:

- **The hook must be offered bodyless symbols.** As first written,
  `gen_symbol_code` sat inside `adjust()`'s `get_value().is_not_nil()` guard, so
  a function *declaration* — which is where an exception specification lives —
  never reached it. Instrumenting showed the resolution firing once under the
  hop-off against four times on the default path. Whoever ports this next needs
  the hook moved, not just the function.
- **The port is unpinnable today.** With no verdict moving and the attribute
  invisible in `--symbol-table-only`, no test in this repo can distinguish the
  ported pass from the unported one. It was therefore reverted rather than
  merged: a change that cannot be pinned is the dead instrumentation the gates
  exist to catch, however plausible its motivation.

### 3.10 The hop-off DOES produce false proofs — the sample hid them (2026-09-10)

§3.9 said the stride-10 sample under-covers the exception paths. Censusing four
directories in full says something worse, and it corrects a claim §3.5, §3.7 and
§3.8 all repeated:

| suite | agree | false alarm | **missed bug** | no verdict |
|---|---:|---:|---:|---:|
| `try_catch` | 68 | 52 | 0 | 52 |
| `destructors` | 2 | 0 | **1** | 11 |
| `inheritance` | 56 | 25 | **2** | 23 |
| `polymorphism_bringup` | 9 | 1 | 0 | 36 |
| **total** | 135 | 78 | **3** | 122 |

**"Zero missed bugs" was false.** It held on the 80-test stride sample and I
restated it four times as though it were a property of the pass. It is a property
of the sample. Three tests verify SUCCESSFUL under the hop-off where the default
path finds the bug:

- `destructors/github_6263_nonvirtual_base_delete`
- `inheritance/github_7025_vbase_nonfirst_member_fail`
- `inheritance/mi_base_subobject_layout_fail`

All three are base-subobject layout — the area `scope-clang-c-irep2.md` §3894 and
#7025 already record as having two competing layout oracles. A pass that silently
proves those is unsound in exactly the way that matters, and the `_fail` suffix on
two of them means the corpus was built to catch this.

**And 122 tests produce no verdict at all**, against 4 on the sample. The crash
class §3.3 closed was not the only one.

**What this says about the method.** Sampling by stride over a directory listing
weights by directory size, so the suites that concentrate a *semantic* area —
inheritance, destructors, exceptions — are the ones a stride under-samples, and
they are exactly where a frontend migration breaks. Every number in §3.2 through
§3.8 is a stride-sample number and should be read as such. Full-suite figures for
the four directories above supersede them.

Until the three false proofs are closed, the flag is not merely incomplete; it is
unsound on inheritance, and no verdict it produces there can be trusted.

### 3.11 The false proofs are two unported arms, and they are C arms

`inheritance/mi_base_subobject_layout_fail` is three lines:

```cpp
struct A { int a; A() : a(1) {} };
struct P { virtual ~P() {} int p; P() : p(9) {} };
struct AP : A, P {};
int main() { AP ap; A &as = ap; assert(as.a == 9); }
```

`as.a` is 1, so the assertion is false and the test is a `_fail`. `main`'s goto
body is **byte-identical** on both paths — both emit
`as = &ap.@base@tag-A` and `ASSERT as->a == 9` — so the reference bind is not
where it goes wrong. The symbol table is:

```
~AP(this == 0 ? 0 : (struct AP *)((signed char *)this - 8))   default
~AP((struct AP *)this)                                        hop-off
```

The hop-off drops the base displacement, so `A`'s subobject aliases `P`'s and
`a` reads 9.

**The two arms that compute it are unported**:
`clang_c_adjust::adjust_base_to_derived` and `adjust_derived_to_base`
(`clang_c_adjust_expr.cpp:424,480`) — `grep -c` in
`clang_c_adjust_irep2.cpp` is **0** for both.

Three things follow:

- **They are C arms, missing from the C pass.** `adjust_base_to_derived` runs
  from `clang_c_adjust`'s default `else` branch (`clang_c_adjust_expr.cpp:205`),
  i.e. on every expression the C path adjusts. The C hop-off has the same gap
  and never shows it, because C has no base classes. §3.2's method — give the
  C++ pass the inherited C arms and measure — cannot find a hole in the arms it
  inherits.
- **This is why the corpus, not the inventory, keeps being right.** §3's mapping
  compared `clang_cpp_adjust`'s overrides against the IREP2 arms. These two are
  not overrides; they are base-class arms the IREP2 pass never had, so no
  comparison of the C++ subclass could have named them.
- **It plausibly accounts for all three false proofs**, which are all
  base-subobject layout. Stated as located, not proven: the fix has not been
  written, and §3.9 is a recent reminder that the obvious cause can be refuted.

Their displacement uses ESBMC's own layout rather than clang's
(`clang_cpp_convert.cpp:1914`), which `scope-clang-c-irep2.md`'s #3894 note also
warns about — the port must take the offset from the same oracle the legacy arm
does, not recompute it.

### 3.12 The soundness fix is blocked on carriage, and the precedent is §2.5

Porting §3.11's two arms is not a port. Both are **marker-driven**:

| arm | fires on | set by |
|---|---|---|
| `adjust_derived_to_base` | `#derived_to_base` (`clang_c_adjust_expr.cpp:92`) | the converter, for a conversion it could not route through a `@base@` component |
| `adjust_base_to_derived` | `#base_to_derived` (`:482`) | the converter, for the downcast |

Both markers are irept **attributes**, and:

- `grep -c` for either in `migrate.cpp` is **0** — nothing carries them across
  the seam;
- `typecast2t` has two fields, `from` and `rounding_mode`. There is nowhere to
  put one.

So an IREP2 pass cannot see that a cast needs displacement. It is not that the
arm is unwritten; the information it dispatches on does not survive migration.

**The displacement cannot simply be applied earlier.** The converter's own
comment says why: *"the displacement is only computable once the layout is
padded, which is here"* (`clang_c_adjust_expr.cpp:90`, #7025). Computing it in
`clang_cpp_convertert` is the option the legacy design already rejected.

**This is a W3 instance that blocks a soundness fix**, which is a different
weight class from the two this scope has recorded so far — §137's was a printer
difference and §3.4's changes one exception id. Here the missing carriage is why
three `_fail` tests are silently proved.

**The fix has a precedent in this same document.** §2.4 and §2.5 added
`pointer_ref_kindt` to `pointer_type2t`: a defaulted field, in the `fields`
tuple, carried both ways by `migrate`, pinned by a round-trip unit test and
costing no construction site. A base-conversion marker on `typecast2t` is the
same shape — most of `pointer_ref_kindt`'s cost was discovering the pattern, and
that is now paid. Two cautions from doing it once:

- `fields_cover_class` will **not** catch the field being dropped from the tuple
  if the shortfall sits under the alignment tolerance (§2.5). Pin equality
  explicitly.
- Whatever rebuilds a `typecast2t` must forward it, as
  `rebuild_with_type<address_of2t>` had to (§2.6).

Sequencing: the field first, on its own, with the round-trip test — then the two
arms become an ordinary port against a marker they can read.

## 4. What does not exist yet

- **No hop-off flag** — though a census instrument now exists, §4.1.
  The reason there is no hop-off:
  `clang_cpp_languaget::typecheck` (`clang_cpp_language.cpp`'s `typecheck`) runs
  `clang_cpp_adjust` unconditionally: no option is read, and no IREP2 pass is
  constructed. Compare `clang_c_languaget` (`clang_c_language.cpp:460-490`),
  which reads `clang-c-irep2-adjust-only` and either replaces or shadows the
  legacy pass.

  Measured consequence: instrumenting the IREP2 `implicit_typecast_followed` at
  entry, a C source under `--clang-c-irep2-adjust-only` reaches it (2 entries)
  and a C++ source reaches it **zero** times. So every arm Phase 7 ports is
  dormant until this is wired — §2.3 and §2.6 both had to say "no regression
  pair is possible", and this is why.

  **The cheap first move is the shadow mode, not the replacement.** Phase 6's
  `--clang-c-irep2-adjust` runs the IREP2 walk *in addition* to the legacy pass:
  read-only, byte-identical by construction, and what it buys is migrating every
  value in the corpus through `get_value2()`, which aborts on any construct
  `migrate_expr` cannot represent. Wiring that on the C++ path needs no
  `clang_cpp_adjust_irep2` and no answer to §3 — it is a census instrument, and
  it would price the whole C++ corpus in one run. That is the next work item.
- **No scope-doc census by construct.** §39.1's "census before writing" prices
  every construct once, at the start. For clang-cpp that census cannot be run
  until the flag exists, so §1's counts are the static census only.

### 4.1 The census exists, and it inverts the expected risk (2026-09-10)

`--clang-cpp-irep2-migrate-census` migrates every adjusted symbol's type and
value through IREP2 and discards the result. It is not a shadow of
`clang_c_adjust_irep2`: that pass writes back whatever it changes and its arms
are C-shaped, so on C++ it would re-adjust bodies `clang_cpp_adjust` has already
handled. A census has to leave the program alone.

Read-only, measured with `irep2_canon` over a stride-10 `regression/esbmc-cpp`
sample: **282 of 282** canonicalised goto programs identical, the one flagged
difference being an extra `migrate_expr` diagnostic rather than a program change.

Result over 273 tests:

| migrated | count |
|---|---:|
| symbol types | **367 738** |
| symbol values | **86 917** |
| `migrate_*` diagnostics | **1** |
| aborts | **0** |

**IREP2 represents everything this corpus's C++ frontend output contains.** That
was the open question §4 existed to price, and it reframes the phase: the blocker
is not representation, it is adjuster coverage — §3's arm-table question. W1 was
already dissolved for structured control flow (`frontends-to-irep2.md` §3); this
says the same for C++ *values*, over this corpus.

**Where the census runs is load-bearing, and the first version had it wrong.**
Placed before `c_link` it produced the same counts and looked equally clean — but
`migrate_namespace_lookup` is the *global* context, so pre-link every symbol of
the TU under census is absent from it. `sym_name_to_symbol` does not fail on a
miss: it falls through to building `symbol2tc` from the expression's own type
instead of `migrate_symbol_type`'s, which `migrate.cpp:670-679` warns "screws up
future hash tables". So the pre-link census established only that migration ran
*with every symbol reference on the fallback path*, and any defect reachable only
through `migrate_symbol_type` — incomplete struct, prototype-versus-definition
mismatch — was invisible to it.

Nor was the single pre-link diagnostic a measure of the problem: that message is
`log_debug("migrate", ...)`, so it needs module-level verbosity to appear at all,
and misses whose names carry `?`/`!` or the k-induction `cs$`/`s$` prefixes
return silently. Counting one warning said nothing about how many symbols took
the fallback.

Moved to run after `c_link`, the same test emits **zero** namespace misses under
`--verbosity migrate:9`, where pre-link it emitted one. The counts are unchanged
and the failure count is still zero, so the conclusion survives — but it now
rests on resolved symbol types rather than on substituted ones.

Two consequences worth keeping:

- **The read-only property was true for a reason I had not identified.**
  `c_link`'s `fix_symbol` (`fix_symbol.cpp:9-14`) round-trips every symbol
  through `set_type`/`set_value` unconditionally, clearing both IREP2 valid
  flags — so a pre-link census's migrated values were discarded at the link
  boundary. That, not the cache-flag argument, is why the goto A/B came out
  282/282. Post-link that leg is gone, and the 282/282 sweep becomes the actual
  evidence rather than a construction.
- **Migration signals failure by throwing a `std::string` on some arms**
  (`migrate.cpp:395`) and by aborting on others (`:795`, `:837`, `:859`), and
  the diagnostic names the expression, never the symbol. Each symbol is wrapped,
  so a *throwing* construct names itself and the walk continues — the difference
  between a census and a bisection. An aborting arm still stops the run; the
  wrap does not and cannot cover those.
- **The counts alone cannot show the census ran.** A symbol or value count is
  identical on either representation, so replacing `get_type2()` with
  `get_type()` migrates nothing and prints the same line — a mutant the first
  version of the test survived, and the natural drift path once the C++
  frontend starts writing the IREP2 side directly. The line therefore also
  reports the number of distinct IREP2 `type_id`s, which only the migrated form
  can produce, and the tests pin it at two or more.

**A measurement trap this cost, recorded because it invalidated a first answer.**
The C++ goto dump carries `GOTO program creation time:`, which varies run to run.
A first A/B of this change filtered `time:` and not that prefix, and reported 270
of 282 tests "differing"; the control — same binary, same flags, twice — reported
54 of 60. Nothing was diverging. Use `scripts/irep2-migration/lib.sh`'s
`irep2_canon`, which strips timings, addresses and temp paths, and run the
same-flags control before believing any A/B on this corpus.

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
2. ~~Port items 6 and 7~~ — **done**, §2.3, PR #7701.
   ~~Port item 1~~ — **done**, §2.6, PR #7705.
3. ~~Wire a census instrument on the C++ path~~ — **done**, §4.1. It says
   representation is not the blocker, so §3 is now the critical path.
4. ~~Price option B against option A~~ — **done**, §3.1: neither, but B′
   (per-frontend typed table, template runner) is measured cheap.
5. ~~Make the table generic and stand up `clang_cpp_adjust_irep2`~~ — **done**,
   §3.1 and §3.2 (PRs #7714, #7717). The corpus, not the name mapping, says what
   is missing: write the member-call arm first (54 of 57 real divergences).
6. ~~Port the `exception_id` assignment~~ — **done**, §3.4, PR #7719. Crashes are
   gone; the residue is the builtin spelling the seam drops.
7. ~~Per-symbol code-generation hook and `gen_vptr_initializations`~~ — **done**,
   §3.7, PR #7724. `finalize_exception_specification` has the same shape and is
   not yet ported.
8. The three remaining false alarms (§3.8), which are three causes: a queue
   reference, a bitset alignment and a vector pointer.
9. **Carry the base-conversion markers on `typecast2t` (§3.12)** — a defaulted
   field in the `fields` tuple, following `pointer_ref_kindt`. Without it §3.11's
   arms have nothing to dispatch on.
10. Then port `adjust_base_to_derived` and `adjust_derived_to_base` (§3.11),
    taking the offset from `base_displacement`, not a fresh computation.
11. Then the 122 no-verdict cases.
10. Then §134.4's ternary decay, which reaches the goto program on C++ (§3.6).
    `finalize_exception_specification` is *not* on this list: §3.9 refutes it.

Only then does a slice make sense.
