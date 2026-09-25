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
`adjust_reference` and `adjust_side_effect` have no counterpart at all. Those
are the real Phase 7 work, and §4.1 says representation will not obstruct them.

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

A C++ method call. Six of the other nine are multi-file tests the sweep fed
only their first source, one expects a parse error. **So one arm —
member-function call lowering — accounts for essentially the whole gap**, and
it is the next slice. §3's mapping table guessed five missing arms from names;
the corpus says start with the one the legacy
`adjust_side_effect_function_call` override covers.

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
`convert_exception_id`. The IREP2 pass *replaces* `clang_cpp_adjust`, so none
of that runs, and `remove_exceptions` reaches catch/throw nodes with no
catchable-type id.

**It is not a representation gap.** `migrate_expr`'s cpp-catch arm carries
`exception_id` across the seam in both directions, so §4.1's conclusion stands —
what is missing is the arm that *computes* the ids, not a field to hold them.
Worth stating because the migrate census could not have found this: the census
runs after the legacy adjuster, so the attributes were already present when it
looked.

### 3.4 The ids are computed, and the residue is a spelling the seam drops

PR #7719 ports both arms: `convert_exception_id` becomes a free function taking
a namespace (it read no other instance state), and the IREP2 arms populate
`code_cpp_catch2t`'s and `code_cpp_throw2t`'s `exception_list` fields.

| over 80 tests | before | after |
|---|---:|---:|
| agree | 31 | **51** |
| diverge | 49 | 29 |
| crash | 23 | **0** |

**The residue on the reproducer is a W3 instance that reaches a verdict, not a
printer.** `throw 1` yields the id `signedbv` on the hop-off where the legacy
path yields `signed_int`: the id is computed from `migrate_type_back(...)`, and
the C spelling does not survive `migrate_type`. The throw and the handler then
disagree and the exception escapes, so the two-line reproducer still reports
FAILED where the default path succeeds.

Class-typed exceptions are unaffected — their id comes from the tag name, which
does survive — which is why the corpus improves from 31 to 51 regardless.

This is worth separating from §5's R6 as stated. R6 anticipated a dropped
attribute surfacing as a *printer* difference; here it changes a verdict. The
spelling has to be carried, or the ids computed before migration.
Reconstructing `signed_int` from a 32-bit `signedbv` is available and is the
wrong answer, for §137's reason: do not rebuild what the representation
dropped, either carry it or do not claim it.

**What it says about §3's mapping.** Neither of the two arms the corpus has now
demanded — member-call lowering and exception-id assignment — is among the five
§3 predicted from name comparison (`adjust_code`, `adjust_decl_block`,
`adjust_symbol`, `adjust_reference`, `adjust_side_effect`). Two for two, the
corpus named a different arm than the names did. Treat §3's table as an
inventory, not a work order.

### 3.5 The residue is 23 false alarms with one cause, and no missed bugs (2026-09-10)

With the crashes gone, the 29 remaining divergences over the 80-test sample
split by *direction* first, because that is the question that matters for a
verifier:

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

The trace puts the violated property at **State 1** with no assignment before
it, so the call is not being set up rather than an argument holding a wrong
value. §3.6 instruments it; the answer was not parameter binding.

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

**It does not fit the arm table, and that is the finding.**
`gen_vptr_initializations` takes a `symbolt &` and emits statements at the
front of a constructor body: it is per-symbol code *generation*, not a per-node
rewrite. §3.1 settled how a C++ pass dispatches arms; it said nothing about a
pass that also has to synthesise code. `adjust()` walks values and applies the
table, and has no hook for this.

So the next step is not a fourth arm but a second kind of work in the pass, and
the order matters — the vptr writes must precede the constructor's own body, as
the legacy pass arranges. `clang_cpp_adjust` has one further step of this
shape, `finalize_exception_specification`, so the hook wants to take both
rather than be built for one.

**A second divergence the same diff shows**, unrelated and smaller: a ternary
over string literals decays per arm on the default path (`val ? &"1"[0] :
&"0"[0]`) and as a whole on the hop-off (`&(val ? "1" : "0")[0]`). That is
§134.4's ternary decay in `scope-clang-c-irep2.md`, recorded there as not
reaching symex on C; on C++ it reaches the goto program. Worth its own row
rather than being folded into the vptr work.

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
adjusting the body; doing that here would mean migrating the value back to
legacy form, mutating it and migrating forward again — the round trip §137
shows is lossy. Generating first lets the emitted assignments go through the
arms like any other statement. Worth knowing the two passes differ in that
order.

**Freeing a member exposes dead code that private status hid.** Making
`gen_vptr_initializations` a free function left `gen_vptr_init_code` and
`gen_vptr_init_lhs` with no callers — `-Werror=unused-function` says so, and
nothing in `src/` or `unit/` referenced either. Both were already dead on
master; as private members nothing could observe that. Removed with the
compiler as the proof.

**What is left.** Three false alarms and four harness artefacts. One of the
three is §3.4's builtin exception spelling (`throw 1` yields `signedbv` where
legacy yields `signed_int`). The others are unexamined.

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
`try_catch` is small next to `algorithm` and `cpp`. Quote the sample when
quoting the 73/80, and census the exception directories separately.

One of the six produces no verdict at all
(`exception_spec_dynamic_violation_fail`, default FAILED), so a crash survives
somewhere the sample never looked.

**`finalize_exception_specification` is not the cause, and porting it was
refuted.** It was the obvious candidate: `clang_cpp_adjust` calls it per code
symbol, the IREP2 pass replaces that adjuster, and 30 tests in the corpus use a
dynamic exception specification. Ported into the §3.7 hook it changes
**nothing** — `try_catch` stays at 8 agree / 6 diverge, and neither
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
restated it four times as though it were a property of the pass. It is a
property of the sample. Three tests verify SUCCESSFUL under the hop-off where
the default path finds the bug:

- `destructors/github_6263_nonvirtual_base_delete`
- `inheritance/github_7025_vbase_nonfirst_member_fail`
- `inheritance/mi_base_subobject_layout_fail`

All three are base-subobject layout — the area `scope-clang-c-irep2.md` §3894
and
#7025 already record as having two competing layout oracles. A pass that silently
proves those is unsound in exactly the way that matters, and the `_fail` suffix
on two of them means the corpus was built to catch this.

**And 122 tests produce no verdict at all**, against 4 on the sample. The crash
class §3.3 closed was not the only one.

**What this says about the method.** Sampling by stride over a directory
listing weights by directory size, so the suites that concentrate a *semantic*
area — inheritance, destructors, exceptions — are the ones a stride
under-samples, and they are exactly where a frontend migration breaks. Every
number in §3.2 through §3.8 is a stride-sample number and should be read as
such. Full-suite figures for the four directories above supersede them.

Until the three false proofs are closed, the flag is not merely incomplete; it
is unsound on inheritance, and no verdict it produces there can be trusted.

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
difference being an extra `migrate_expr` diagnostic rather than a program
change.

Result over 273 tests:

| migrated | count |
|---|---:|
| symbol types | **367 738** |
| symbol values | **86 917** |
| `migrate_*` diagnostics | **1** |
| aborts | **0** |

**IREP2 represents everything this corpus's C++ frontend output contains.**
That was the open question §4 existed to price, and it reframes the phase: the
blocker is not representation, it is adjuster coverage — §3's arm-table
question. W1 was already dissolved for structured control flow
(`frontends-to-irep2.md` §3); this says the same for C++ *values*, over this
corpus.

**Where the census runs is load-bearing, and the first version had it wrong.**
Placed before `c_link` it produced the same counts and looked equally clean —
but `migrate_namespace_lookup` is the *global* context, so pre-link every
symbol of the TU under census is absent from it. `sym_name_to_symbol` does not
fail on a miss: it falls through to building `symbol2tc` from the expression's
own type instead of `migrate_symbol_type`'s, which `migrate.cpp:670-679` warns
"screws up future hash tables". So the pre-link census established only that
migration ran *with every symbol reference on the fallback path*, and any
defect reachable only through `migrate_symbol_type` — incomplete struct,
prototype-versus-definition mismatch — was invisible to it.

Nor was the single pre-link diagnostic a measure of the problem: that message
is `log_debug("migrate", ...)`, so it needs module-level verbosity to appear at
all, and misses whose names carry `?`/`!` or the k-induction `cs$`/`s$`
prefixes return silently. Counting one warning said nothing about how many
symbols took the fallback.

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

**A measurement trap this cost, recorded because it invalidated a first
answer.** The C++ goto dump carries `GOTO program creation time:`, which varies
run to run. A first A/B of this change filtered `time:` and not that prefix,
and reported 270 of 282 tests "differing"; the control — same binary, same
flags, twice — reported 54 of 60. Nothing was diverging. Use
`scripts/irep2-migration/lib.sh`'s `irep2_canon`, which strips timings,
addresses and temp paths, and run the same-flags control before believing any
A/B on this corpus.

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
pass (`adjust_arm<Pass>` in `clang_c_adjust_irep2.h`), so
`clang_cpp_adjust_irep2` orders the arms it inherits alongside its own in one
table. No per-statement-kind virtual was reintroduced, which is what option A
would have cost.

One compiler constraint shaped the row. A pointer to a base member stored in a
derived-typed table is legal, but GCC 13.3 mis-reads the call once the runner
inlines it and rejects it under `-Werror=array-bounds` at `-O2`; clang 18
accepts it. The row therefore holds a function pointer produced by a
captureless lambda trampoline, which is an address constant, so the table stays
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

The cause is a missing arm, not a subtle one.
`clang_cpp_adjust::adjust_reference` dereferences a reference-typed operand,
and `clang_c_adjust` calls it from five sites -- the relational arm, binary
arithmetic, complex unary, and twice in `adjust_side_effect_assignment`. The
IREP2 pass has no counterpart: `grep adjust_reference` over
`clang_c_adjust_irep2.*` and `clang_cpp_adjust_irep2.*` returns nothing. Every
reference-typed operand in those positions is therefore left as a bare pointer.

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

Walk coverage is ruled out too. Tracing the symbol being adjusted at the moment
of each rewrite puts both of them inside
`std::lock_guard<std::mutex>::lock_guard(mutex &)` -- the constructor binding
its reference member -- and instrumenting the legacy pass's own symbol loop
shows it adjusts that same constructor. Both passes visit it.

What differs is what is *in* it. Across the whole program legacy's
`adjust_reference` is called four times and never once from that constructor,
so in the legacy tree the member-initialiser binding is not a side-effect
assign; in the IREP2 tree it is, which is why the hook's
`is_sideeffect_assign2t` guard claims it. Since `migrate_expr` keeps
`code`/`assign` and `sideeffect`/`assign` apart, the node is not being
reclassified at the seam -- an **earlier IREP2 arm is producing a
sideeffect_assign where the legacy pass has something else**, and the reference
hook is merely the first arm to notice.

Dumping that constructor shows the damage exactly:

```
legacy  ASSIGN this->__m  =  m::0;          FUNCTION_CALL: lock(this->__m)
flag    ASSIGN *this->__m = *m::0;          FUNCTION_CALL: lock((std::mutex *)this->__m)
```

The hook dereferenced **both sides of a reference binding**, so the constructor
copies the mutex instead of binding a pointer to it. That is the whole
regression, and it says what the missing distinction is: a reference *used as a
value* must be read through, a reference *being bound* must not.

Six causes are now measured and eliminated -- ordering, predicate, referent
type, node kind, visitation, and now the damage itself is understood. The one
open question left is narrow: legacy never routes this binding through
`adjust_side_effect` (its `adjust_reference` is called four times program-wide
and never from this constructor), while in IREP2 the node is a
`sideeffect_assign2t`. Since `migrate_expr` maps `code`/`assign` to
`code_assign2tc`, the binding is not arriving from the seam in that shape --
`adjust_decl_init` is the first arm to check, since a member-initialiser can
reach it as a `code_decl` with an initialiser.

**Answered: `adjust_reference` was the wrong arm.** Legacy does not reach this
binding through `adjust_side_effect` because `clang_cpp_adjust` overrides that
and routes an `assign` to its own `adjust_side_effect_assign` -- a C++-only arm
with a five-way branch on the assignment's shape:

| the assignment | what legacy does |
|---|---|
| rhs is a constructor call | fold `x = T()` into `T(&x)` |
| lhs is a reference symbol | `r = 1` becomes `*r = 1` |
| lhs is a call returning a reference | `X(a) = 5` becomes `*X(a) = 5` |
| **lhs carries `#member_init`** | **adjust the rhs only -- leave the lhs alone** |
| otherwise | fall through to the C arm |

The fourth row is the mutex case, and it is why legacy never dereferences a
member-initialiser's left side. **None of these five branches is ported**, so
the reference hook was a fragment of this arm applied without the branch that
excludes a binding -- which is exactly why it broke a binding.

And the branch that matters cannot be written yet: `#member_init` is a
converter-set irept flag (`clang_cpp_convert.cpp` sets it in three places and
reads it in three, including `should_dereference`), and `grep member_init
src/util/irep/migrate.cpp` returns **nothing**. It is dropped at the seam, the
same class as §3.12's `#derived_to_base` and with the same shape of fix.

**What shipped, and in what shape.** `#member_init` is carried as a field on
`sideeffect_assign2t` (declared next to `op`; after `location` the compiler
packs it into the location's padding and `fields_cover_class` underflows). The
reference handling is a virtual hook -- empty for C, as the legacy one is --
reached from the plain-assignment arm, the relational arm, and the
increment/decrement family, with the member-initialiser case as a branch inside
it. That is *not* the "port `adjust_side_effect_assign` whole" this section
originally called for: only one of that arm's five branches is addressed, and
the constructor-call fold and call-returning-reference branches remain
unported.

Measured over `regression/esbmc-cpp/cpp` with the §3.14 command: 89 divergences
before, 80 after, no regressions and no false proofs.

**What testing this taught, three times over.** A reference *variable* is
dereferenced at conversion time by `get_decl_ref`'s `should_dereference`, so
`int &r; r++;` and `r == w` and `(long)r` are already correct and pin nothing --
a pair built on one passes with the hook in or out. Every test here has to go
through a reference-**returning call** (`b.at()`). Three pairs were written on
local references and had to be rewritten after mutation-checking showed they bit
nothing.

**Still open on this row:**

- Increment/decrement of a reference-returning expression was unhooked entirely
  until review caught it: `b.at()++` emitted
  `ASSIGN return_value$_at$1 = return_value$_at$1 + 1`, arithmetic on the
  reference with the referent untouched. Now hooked, with a pair that bites.
- The relational hook and `convert_reference`'s typecast branch are not pinned by
  any test that bites, and may be unreachable -- both were added while chasing
  the mutex regression. Their reachability is being measured; whichever does not
  fire over the corpus should come out rather than ship unpinned.
- The binding case's correctness is not self-contained: after this hook
  dereferences the rhs, `c_typecastt::convert_reference` (a same-named function
  in `util/lang/c_typecast.cpp`) re-wraps it in an `address_of2tc` because the
  lhs is reference-typed. The round trip is a genuine no-op -- `this->__m =
  &(*m)` -- but it spans two translation units.

Note also that a test for this must use a **function returning a reference**
(`b.at() = 7`, which is what `std::array::operator[]` is). A local `int &r`
binding is lowered without going through this path, so a pair built on one
passes with the arm on or off and pins nothing.

**Lesson for the census.** "Zero false proofs" held only over the six suites
§3.14 sweeps. The suites were chosen because early divergences clustered there,
and that selection quietly became the measurement. A number is scoped by what
was swept, and the scope has to be stated with it.

### 3.17 What is left, bucketed

The 80 divergences `regression/esbmc-cpp/cpp` reports after §3.16 are not one
cause. Grouped by test family:

| family | rows | shape |
|---|---|---|
| `github_5868_*` | 18 | all `SUCCESSFUL -> FAILED`, all STL surface |
| `github_2284_*` | 4 | |
| `ptr_to_member_*` | 4 | pointer-to-member |
| `github_6291_*` | 3 | a reference *parameter* bound to a conditional lvalue |
| `switch_declaration*` | 3 | a declaration in a switch |
| `member_array_*`, `static_array_ctor` | 3 | array member construction |
| ~20 singletons | | map / list / tuple / stream |

`github_5868_*` is the largest and the most likely to share a cause: all 18 are
`SUCCESSFUL -> FAILED`, and the three sampled so far fail *inside the
operational models* rather than in the test --

```
github_5868_container_relational      /esbmc-vfs/cpp/set     line 858  Incorrect alignment
github_5868_is_scalar_const_lookup    /esbmc-vfs/cpp/utility line 131  invalid pointer
github_5868_reverse_iterator_base     /esbmc-vfs/cpp/list    line  29  invalid pointer
```

"Incorrect alignment when accessing data object" is the signature §3.16 met
when a `dereference2t` was built over a bare `symbol_type2t`, so a
type-resolution gap looked like the first hypothesis. **Diagnosing one refuted
it.**

A goto-diff of `github_5868_is_scalar_const_lookup` shows `std::pair`'s
constructor byte-identical between the passes apart from instruction numbering
(+4 under the flag), so the failure inside it comes from different state, not
different code. Normalising the numbering and diffing the whole program gives
179 hunks, and the first is unambiguous:

```
legacy  RETURN: ieee_fma(a, b, c)
flag    RETURN: return_value$_fmal$1      # an actual call to fmal() instead
```

Likewise `nearbyint`. The IREP2 pass folds only the **`__builtin_`-prefixed**
half of `clang_c_adjust::do_special_functions`; the **name-matched** family
(`fma`, `nearbyint`, …) is unported, so those calls reach the operational model
rather than becoming IREP2 nodes. `adjust_special_functions`'s own doc comment
says as much and names the reason it was deferred: unlike a reserved
`__builtin_` spelling, a program may define `fma` itself, so the fold needs a
`shadows_user_definition` query first. That helper is already shared
(`builtin_names.h`'s `builtin_shadows_user_definition`), so no hoist is needed.

That divergence is confirmed. Whether it *causes* the 18 failures is not: they
are reported inside `<set>`, `<utility>` and `<list>`, not in cmath. Port the
name-matched half, re-measure, and see how many of the 18 move -- rather than
assuming, which is the §3.16 mistake.

`github_6291_*` is already known to be something else: `bump((c < 1) ? a : b)`
binds a reference *parameter* to a conditional lvalue, which needs an address-of
of a ternary rather than a dereference of a reference.

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

### 3.13 Phase 7 re-measured on merged master, and a name match that was not evidence
### (2026-09-13)

#7717 and #7742 landed, carrying most of the stack's content with them, so §3.2's
census predates all of it. Stride-12 sweep of `esbmc-cpp/cpp`, verdicts under
`--clang-cpp-irep2-adjust-only` against the default path, skipping
`KNOWNBUG`/`FUTURE` rows and rows that pin an irep2 flag themselves:

| | #7718's census | now |
|---|---:|---:|
| verdicts agree | 31 of 80 | **77 of 84** |
| diverge | 49 | 7 |
| crash | 23 | **0** |

No crashes at all, and all seven residual rows are false alarms — `off`
SUCCESSFUL, `on` FAILED: `array_element_destructors_leak`, `ch21_4`,
`github_3978`, `github_6291_conditional_ref_shapes`,
`github_6717_throw_conditional_ok`, `member_array_ctor_dtor_symmetry`,
`tuple_tie`.

**One attempted fix, reverted, and worth recording as a method failure.**
`adjust_address_of` in the IREP2 pass documents the conditional distribution
`&(c ? a : b)` -> `c ? &a : &b` as deliberately unported, because "no corpus
input reaches it under this flag, and an arm no test executes is the trap §90.4
records". `github_6291_conditional_ref_shapes` is named for the issue that
distribution serves and binds references to conditionals throughout — `bump(i <
1 ? a : (i < 3 ? b : c))`, `bump(n < 1 ? arr[0] : arr[1])` — so the comment
looked falsified by the corpus having grown a row that reaches it.

It is not. The distribution was ported faithfully from legacy, and the row
still diverges identically, so binding a reference to a conditional does not
produce `address_of(if)` on this path — or reaching it is not sufficient. The
arm was therefore exercised by nothing, which is precisely the trap the comment
names, so it is reverted rather than kept.

The inference that failed was: a test named for the issue, containing the source
construct, must reach the arm that serves it. Two steps of plausibility, neither
measured. The comment's claim stands unrefuted, and refuting it needs the arm
instrumented to show it fires — not a name and a source shape.
### 3.14 The seven rows are pre-existing, and the shape is a typed constructor call

§3.13 left seven false alarms. Two name array construction, so
`array_element_destructors_leak` was dumped both ways. Legacy:

```
FUNCTION_CALL: R(&a[0])        FUNCTION_CALL: R(&a[1])
FUNCTION_CALL: ~R(&a[1])       FUNCTION_CALL: ~R(&a[0])
```

under the flag:

```
DECL struct [2] return_value$_R$1;
FUNCTION_CALL: return_value$_R$1=R(&a[0])
FUNCTION_CALL: ~R(&a[1])                    ~R(&a[0])
FUNCTION_CALL: ~R(&return_value$_R$1[1])    ~R(&return_value$_R$1[0])
```

So the constructor call is **value-returning** on the flag path:
`remove_sideeffects` gives it a temporary, and that temporary then acquires
destructors of its own, whose `p` was never allocated — hence a leak reported
on a program that leaks nothing. Note the flag path has *more* destructor
calls, not fewer; the test's own header comment frames the defect as "skipped
array-element destructors", which is the opposite of what the dump shows.

**The obvious suspect was mine, and it is cleared.** Phase 8 added
`align_call_return_type`, whose whole job is to give a call its callee's return
type, guarded by `if (ret.id() == "constructor") return;`. If an array
constructor's return type were not spelled `constructor` after the round trip,
that guard would miss and the hook would create exactly this temporary.
Disabled behind an env gate, all seven rows diverge **identically**:

| row | hook on | hook off |
|---|---|---|
| all seven | FAILED | FAILED |

So the typed constructor call predates Phase 8's arms. Worth stating that the
question had to be asked: the 77-of-84 measurement was taken *after* that arm
landed, and #7718's census predates the merges, so neither isolates it — a gap
in how the work was sequenced, since a shared-arm change should be measured
against the same base before and after.

What is established: the call carries a non-void type under the flag, and
something other than the alignment hook gives it one. That is the next
diagnosis, and the two array rows are the cheapest instance of it.
### 3.15 The constructor-call fold: two rows of the seven, and a split cause

§3.14 established that under the flag a constructor call carries a non-void
type and `remove_sideeffects` gives it a temporary. The unported branch that
explains the shape is the first row of §3.16's table:
`clang_cpp_adjust::adjust_side_effect_assign` folds `x = T(args)` into
`T(&x, args)`, so on the legacy path there is no value-returning call to give a
temporary to.

Reading the callee's `return_type().id() == "constructor"` from the symbol table
is what identifies a constructor: IREP2 has no id for one, which is the same
detour `align_call_return_type` takes.

**It cannot be an arm, and finding out why is the useful part of this row.**
Registered in the table ahead of `adjust_plain_assignment` it produces a
*silently wrong* argument list. The arms run after the operand walk, so by the
time the fold sees the assignment, the walker has already visited the rhs call
as a node in its own right — and `is_call_site` matches any `sideeffect2t`, so
`adjust_call_arguments` has already converted each argument against the
*matching parameter*, with the object argument still absent. Every actual is
therefore converted against its predecessor's formal. Folding then inserts
`&lhs` and converts again against the correct slots, and the two compose:

```cpp
struct T { int a; double b; T(int x, double y) : a(x), b(y) {} };
struct L { T m;   L(int x, double y) : m(x, y) {} };
```

| | the call in `L` |
|---|---|
| legacy | `T(&this->m, x::0, y::1)` |
| fold as an arm | `T(&this->m, (signed int)((T *)x::0), (double)((signed int)y::1))` |

The double is round-tripped through `int`, so `assert(l.m.b == 2.5)` fails on a
correct program. Note the single-`int`-parameter case survives by accident
(`int -> T* -> int` round-trips), which is why none of the 34 pre-existing flag
rows catches it. A by-value **struct** parameter is worse than a wrong value:
paired against `this`, `binds_by_reference` matches on the type id alone
(`struct_id == struct_id`), takes the argument's address, and the call is built
with a pointer where a struct is expected —

```
ERROR: function call: argument "…@S@T@F@T#$@S@A#@a::0" type mismatch:
       got pointer, expected struct
```

— a hard abort on a program legacy verifies. All three shapes are pinned by
pairs in this change.

The legacy passes never meet this because they dispatch **top-down**:
`adjust_side_effect_assign` folds before anything descends into the rhs. The
IREP2 walk is bottom-up, so the fold runs from a new pre-recursion hook,
`adjust_before_operands` (empty for C), called before
`expr->Foreach_operand(...)`. The walker then descends into the *folded* call
and the call-site arms see the complete argument list, which reproduces legacy's
output byte for byte — and no re-entry into those arms is needed.

**It closes three of the seven, and not the two it was diagnosed from:**

| row | as an arm | pre-recursion |
|---|---|---|
| `ch21_4` | **AGREE** | **AGREE** |
| `github_3978` | **AGREE** | **AGREE** |
| `tuple_tie` | DIVERGE | **AGREE** |
| `array_element_destructors_leak` | DIVERGE | DIVERGE |
| `member_array_ctor_dtor_symmetry` | DIVERGE | DIVERGE |
| `github_6291_conditional_ref_shapes` | DIVERGE | DIVERGE |
| `github_6717_throw_conditional_ok` | DIVERGE | DIVERGE |

`tuple_tie` moved only once the argument list was right, which is the ordering
defect above showing up as a verdict. Phase 7 is therefore 80 of 84 agreeing,
from 77.

The two array rows are §3.14's own reproducers, and they still fail. So the
temporary-creating shape is *not* one cause across the seven: the fold covers
the scalar-member spelling and something else produces a value-returning
constructor call for an array element. Stating that rather than stretching one
fix over seven rows is the point of the table.

**What the pair pins.** The two rows that moved are both `std::list` programs,
and the biting construct reduces to a member whose type has a *user-declared*
default constructor:

```cpp
struct Pool { int buf[4]; Pool() { buf[0] = 1; } };
struct L    { Pool p; L() { p.buf[1] = 7; } };
```

Unfolded, `Pool()` runs with no object argument and symex reports an alignment
failure inside `Pool`. Two more pairs cover the argument list: it pins
`l.m.b == 2.5` through a two-parameter constructor with a mixed-width parameter
list, and its failing half asserts `l.m.b != 2.5`, which **proves** under the
as-an-arm placement — a false proof, which is the outcome worth pinning; and a
by-value struct parameter, which aborts under that placement.

**A stride sample is not a fixed set of rows.** The C++ hard-failure sample went
0 -> 1 across this change, naming `ptr_to_member_2` ("tuple field out of range",
the release-active bound from #7758). It is *not* a regression: with the fold
disabled behind an env switch the same row still aborts, so it fails at the
branch's own HEAD. The six new test directories shifted the stride-12 window,
and `ptr_to_member_2` was never in the earlier sample — `ptr_to_member_*` is one
of §3.17's known families. Adding tests to a suite invalidates comparison with
an earlier stride sample of it; compare named rows, or re-run the old sample on
the old tree. An implicitly-declared or `= default` constructor does
**not** reach the fold (probed: `Pool {}`, `Pool() = default`, and a template
specialisation of the `__om_list_pool` shape all agree with or without it), so a
pair built on one of those pins nothing — the same vacuity trap §3.16 hit three
times on local references. The failing half pins the violated property's own
line, not just the verdict: unfolded, the run still reports FAILED, but inside
`Pool` rather than at main's assertion.

**And the array rows have a name now.** Dumped again with the fold in place,
`array_element_destructors_leak`'s `main` still reads

```
DECL struct [2] return_value$_R$1;
FUNCTION_CALL: return_value$_R$1=R(&a[0])
OTHER &return_value$_R$1[0];
```

so the node is not a `sideeffect_assign` at all — the declaration's initialiser
is a single constructor call whose *type is the whole array*, and only element 0
is constructed. That is `clang_cpp_adjust::adjust_decl_block`
(`clang_cpp_adjust_code.cpp:222`), a C++-only arm over the `decl-block` node
which fans one whole-array constructor call out into one call per element,
recursing into nested arrays, and which deliberately excludes static-storage
locals and aggregate initialisation. The IREP2 table has `adjust_decl_init` and
no decl-block arm, so the fan-out never happens. That is the next arm, and it
covers both array rows.

One qualification before relying on it for both: the *member*-array spelling
reaches the same residual through the converter's own per-element fan-out
(`clang_cpp_convert.cpp:2514`), not through `adjust_decl_block`, so
`member_array_ctor_dtor_symmetry` may also need the `&ctor(...)[0]` unwrap
(`clang_cpp_adjust_code.cpp:196`). Measure each row rather than assuming one arm
covers both -- the mistake §3.14 made and this section nearly repeated.

### 3.16 The array shapes, and two things §3.15's fold had wrong (2026-09-13)

§3.15 named `clang_cpp_adjust::adjust_decl_block`'s per-element fan-out as the
next arm and the two array rows as its coverage. Ported, it closes one of them,
and the second exposed two defects in the fold itself. Phase 7 now reads **82 of
84**; only `github_6291_conditional_ref_shapes` and
`github_6717_throw_conditional_ok` remain.

**The fan-out is an arm on the block, not on the declaration.** One declaration
becomes a bare declaration plus one call per element, and a node cannot expand
into several statements in place. Rewriting the declaration into a block of its
own would work syntactically and end the object's scope at that block's brace,
which is the defect #4715 records and which `hoist_for_init` already documents
for the same reason. So the guard is `is_code_block2t` and the arm splices,
exactly as legacy does over `decl-block` — a distinction that does not survive
the seam anyway (`migrate.cpp` maps `code("decl-block")` to `code_block2t`).
The `temporary_object` wrapper needs no special case: its initialiser travels as
`arguments[0]`, so the recursive search for the constructor call reaches it
through `foreach_operand`.

`array_element_destructors_leak` then produces a GOTO program byte-identical to
the legacy one.

**A guard that was not a defect, and the cost of guessing.** The first
hypothesis for the member-array row was that §3.15's fold over-fires: legacy
keys it on `rhs.get_bool("constructor")`, a converter marker, while the port
promoted legacy's *assert* on the `constructor` return type to the guard, and the
two need not be the same set. Narrowing it to calls still missing their object
argument (`arguments.size() + 1 == parameters.size()`) was written, measured
against the row, and **refuted**:

- Legacy does not leave those calls alone. `clang_cpp_convert.cpp:2544` emits
  `assign(array_init$.buf[i], rhs)` with the shared whole-array call as `rhs`,
  and `copy_to_operands` carries the marker onto every element, so legacy's fold
  fires on each. The two GOTO programs are byte-identical there.
- Those calls have zero arguments against one parameter, so the arity guard
  passes them anyway. It was inert on the shape it was written for, which is why
  no test in the branch changed verdict when it was removed.
- Its only measured effect was a regression: a **variadic** constructor supplies
  more arguments than its parameter list declares, so the guard declined the
  fold and the first argument was then converted against the `this` parameter —
  `this->m=T((T *)4, 5, 6)`, a false alarm on a correct program.

Dropped, and `irep2_constructor_assign_variadic{,_fail}` pins it so it cannot
come back. Default arguments and virtual bases were checked and do not have the
shape: clang materialises a `CXXDefaultArgExpr` at the call site, and the hidden
`__is_complete` flag is appended to both lists. Carrying the marker as a field on
`sideeffect2t`, the `#member_init` precedent, remains the faithful fix if a shape
ever does need the distinction.

**Defect 2, the real one: the folded call took the initialiser's type.** Legacy's
`expr.swap(rhs)` keeps the right-hand side's type, which for a member array's
per-element call is *the whole array's*. Under the flag that made the statement
array-valued, so `adjust_expression_statement` wrapped it in `&stmt[0]` (C11
6.5.3.2p4's decay arm), the call acquired a temporary of array type, and the
temporary's elements were destroyed without ever having been constructed — a
destructor assertion firing on a correct program. The call's value is the object
constructed, so it now takes **the object's** type, `a.lhs->type`. That is also
what legacy has, the two spellings coinciding for every scalar shape.

That defect is pinned by `irep2_member_array_construction{,_fail}`, a
three-element class-typed member array with a destructor assertion. Its passing
half turns SUCCESSFUL -> FAILED under the initialiser-type mutation and its
failing half stops matching its own violated-property line, since the first
property to fail becomes the destructor's. A pair without the destructor pins
neither: the elements *are* constructed in place either way, and only the
spurious temporaries differ.

**What the fan-out declines, and why the declines are staged.** A dimension
whose size is not a constant — a class-typed VLA, which is a GNU extension, not
C++ — makes legacy `abort()` with "cannot determine array size for local ctor
init". The arm declines instead, and the decline is *staged*: the per-element
calls are built into a local vector and the declaration is replaced only if all
of them were built, so a size the arm cannot read never costs the object its
initialiser. Aggregate initialisation (`B a[2] = {B(1), B(2)}`) arrives as a
`constant_array` and is declined by the single-constructor-call gate; fanning it
out would construct every element with element 0's arguments. Both declines now
have pairs, `irep2_array_aggregate_init{,_fail}` and the nested `R a[2][3]` in
`irep2_array_element_construction{,_fail}`.

**Open row: a static or global class-typed array is unconstructed under the
flag.** The arm excludes static-storage locals because
`clang_cpp_maint::adjust_init` (`clang_cpp_main.cpp:59`) constructs them — but
that keys on the `#constructor` marker, and this pass's write-back destroys it.
So `static R a[2];` and a file-scope `R g[2];` both reach `__ESBMC_main` with
element 0 constructed on a temporary, the rest nondeterministic, and destructors
on never-constructed memory. It is **pre-existing**: a file-scope global has no
block and no assignment, so nothing in this arm can reach it. It is also
invisible to the 84-row census, which sweeps `regression/esbmc-cpp/cpp` only:
`regression/esbmc-cpp11/constructors/local_array_of_class_ctor` and
`Constructor9-1` — the two tests legacy's own `adjust_decl_block` comment names —
diverge under the flag, and in the first the whole remaining divergence is its
`static B s[2];`.

**Scope of "82 of 84".** The census adds `--clang-cpp-irep2-adjust-only` to each
row by hand; only 42 `test.desc` rows in the tree pass it themselves, and
`member_array_ctor_dtor_symmetry` is not one of them, so CI runs that row under
the legacy adjuster. A census row is evidence about the pass, not a gate on it —
the gate is the flag-pinned pair.

**And a VLA row, undocumented until now.** With the decline staged, a class-typed
variable-length array keeps its initialiser and element 0 is constructed —
elements 1..n-1 are not, silently, the same false-alarm family as the static row
above. `irep2_array_vla_construction{,_fail}` pins element 0 only: a test over
the rest would cement the false alarm, and pinning element 0 stays correct when
the VLA path is completed. The pair is flag-only by necessity — legacy aborts on
this input.

### 3.17 Next: the conditional distribution, this time with a row that shows it

Of the two rows left, `github_6717_throw_conditional_ok` differs from legacy in
exactly one instruction:

```
legacy  FUNCTION_CALL: S(&tmp$1, c::0 ? &a : &b)
flag    FUNCTION_CALL: S(&tmp$1, &(c::0 ? a : b))
```

That is `clang_c_adjust::adjust_address_of`'s distribution of `&(c ? a : b)` into
`c ? &a : &b` ([expr.cond] makes a conditional over same-typed lvalues an
lvalue), which the IREP2 arm does not do — it handles only the array decay.

§3.13 ported that distribution and reverted it, because
`github_6291_conditional_ref_shapes` diverged identically with it. The revert was
right on the evidence then: the arm was exercised by nothing measurable. It is
not evidence now — this row's IR differs at precisely that rewrite, so the port
has a reproducer whose GOTO is expected to change. Re-port it, diff this row's
GOTO, and expect `github_6291` to need something else: a reference *parameter*
bound to a conditional lvalue is a different shape from taking a conditional's
address.

**§3.17's inference was wrong, and the measurement says where the node comes
from.** Ported again, the distribution fires on **nothing**: instrumented, the
arm sees five `address_of(new_object)` and two `address_of(sideeffect)` on
`github_6717_throw_conditional_ok` and no `address_of(if)` at all, and the row's
GOTO is unchanged. So the `&(c ? a : b)` in the GOTO is not built by the pass.
Two further sites are ruled out by the same method: `c_typecastt`'s IREP2
`take_reference_address` (`c_typecast.cpp:781`) already distributes per arm, and
neither it nor `convert_reference` is called with a conditional on this input.
The node is therefore created downstream of the adjuster, and the reverted
distribution is reverted again — with the reason measured this time rather than
inferred from a row that names the issue.

**What the write-back does lose, and what actually consumes it.** Diffing the
back-migrated body of `pick` against the legacy one (`--symbol-table-only`, the
pass's own tree under `--clang-cpp-irep2-adjust-writeback-all`) shows 274 lines
against 170. Most is known round-trip loss — `#location`, `#base_name`,
`#cpp_type`, `#cformat` — but two entries are not cosmetic: the callee's
`return_type: constructor` becomes `empty` (`migrate.cpp:387` maps
`typet("constructor")` to the empty type, and nothing can restore it), and the
side effect's `constructor: 1` marker is dropped. Legacy's tree has five
`return_type: constructor` spellings in that function; the pass's has none.

Carrying the return-type spelling as a field on `code_type2t` was written and
**reverted**: it closes neither row, and instrumented, neither `migrate_type`'s
nor `migrate_type_back`'s code arm is reached on this input — symbol types are
read from `symbolt::get_type2()` (`migrate.cpp:434`), not migrated on read. An
unexercised field on a core IREP2 type is not shippable evidence of anything.

**So the next task is the marker, not the spelling, and it has a named
consumer.** `#constructor` on the *side effect* is read after the frontend by
`clang_cpp_maint::adjust_init` (`clang_cpp_main.cpp:23`, `:61`), which is the
static-initialisation half — §3.16's open row. Carry it on `sideeffect2t`, the
`#member_init` precedent, and measure that row: it has two reproducers (a
file-scope `R g[2]` and a function-local `static R s[2]`) and two regression rows
that diverge for it, `esbmc-cpp11/constructors/Constructor9-1` and
`local_array_of_class_ctor`. `goto-programs/builtin_functions.cpp:679` reads the
same marker but only on the `cpp_new` path, so it is not what these two
conditional rows turn on; those stay open, and the next measurement on them
should be a whole-body diff of the pass's output for `pick`, not another arm
guessed from an instruction.

### 3.18 The marker that does have a consumer: measured, and blocked on with_type

§3.17 named `#constructor` on the side effect as the loss worth carrying, because
`clang_cpp_maint::adjust_init` (`clang_cpp_main.cpp:23`, `:61`) reads it in
`final()`, *after* the pass and its write-back. Carried as a field on
`sideeffect2t` — set from `expr.get_bool("constructor")` in `migrate_expr`,
restored in `back_sideeffect` — it **closes the static/global row §3.16 opened**:

| row | before | after |
|---|---|---|
| a file-scope `R g[2]` | SUCCESSFUL -> FAILED | **AGREE** |
| a function-local `static R s[2]` | SUCCESSFUL -> FAILED | **AGREE** |
| `esbmc-cpp11/constructors/Constructor9-1` | SUCCESSFUL -> FAILED | **AGREE** |
| `esbmc-cpp11/constructors/local_array_of_class_ctor` | SUCCESSFUL -> FAILED | **AGREE** |

`irep2_global_array_construction{,_fail}` is written and mutation-checked against
the carry: both halves change outcome with the restore suppressed, and both agree
with the legacy path.

**It is not shippable in that shape.** Listing the field in `sideeffect2t::fields`
makes the field order stop matching the primary constructor's parameter order —
`location` sits between `kind` and the new field — and that is exactly what
`supports_with_type_v` tests (`irep2_expr.cpp:500`). The trait goes false, so
every `with_type` on a side effect takes the "no substitutable type" error path
and aborts: 14 of 374 `irep2` rows and 12 of 1062 `esbmc-cpp/cpp` rows, including
four that this branch had just brought to parity. §3.16's `#member_init` on
`sideeffect_assign2t` has the same shape and is already with_type-unsupported;
nothing calls with_type on that kind, which is why it went unnoticed there.

Two ways out: leave the field **unreflected** (like `location`, carried but not
compared — cheap, but a `with_type` rebuild between the fold and the write-back
would drop it), or move the parameter ahead of `loc` in the constructor so field
and parameter order agree, which means touching every positional
`sideeffect2tc(..., location)` call site.

**Shipped unreflected.** `ctest -R irep2` 374 of 374, unit 871 of 871, the
Solidity corpus still 507 of 507 agreeing with zero crashes (it shares
`migrate_expr_back`, so it had to be re-swept), the four rows above agreeing, and both halves of the pair still changing outcome with the
restore suppressed. The reflected version's 14 failures are gone. `esbmc-cpp/cpp`
reports 6 failures — `ch8_5` and the five `github_7433*` rows — but they fail
identically with this change stashed and reverted, so they are not its doing:
they expect an *elaborated* type name (`uncaught exception: struct my_error`,
`class std::out_of_range`) and this build's bundled clang 21 prints the
unelaborated one. Those descriptors have a blank flags line, so they pin whatever
LLVM the build used, which is the trap CLAUDE.md's *Pin the mode in every test*
note describes.

Note also what this says about the *other* seam loss, the callee's `constructor`
return type (§3.17): it is real, but symbol types are read from
`symbolt::get_type2()` rather than migrated, so a `code_type2t` field is not
where it would have to be carried.

### 3.19 Next on the last two rows: a present-empty `#type`, not another arm

The whole-body diff §3.17 called for is in hand, and after the marker is carried
the behavioural residue in `pick` is one entry: the write-back gives the side
effect `#type: empty` and `#size: nil` where legacy has neither. `back_sideeffect`
writes `theexpr.cmt_type(cmttype)` unconditionally, and `cmttype` is a
default-constructed `typet` when the alloctype is nil — an *empty* irep, not a
nil one. That is the irept tri-state trap the same function already documents one
line above for `#size`: "an empty irep is a third state that `is_not_nil()`
reports as present", which is why `size` is initialised to `nil_exprt()` there
and not left default-constructed.

So the next measurement is a one-line change — set `cmt_type` only when the
alloctype is not nil — followed by the two rows and a full sweep, rather than
another arm chosen from an instruction. If it moves neither row, the next thing
to diff is what `remove_sideeffects` does with a class-typed conditional whose
arms are lvalues, since that is where `&(c ? a : b)` is actually built.

**The `#type` hypothesis is refuted.** Setting `cmt_type` only when the alloctype
is not nil changes neither row, so the present-empty `#type` is not what they turn
on either. Reverted, unshipped: a fidelity change that moves nothing measurable
does not belong in the tree.

What is left to try is not a guess about which rewrite is missing but a direct
answer to "who builds this node". The `address_of` over the conditional exists in
the final GOTO and does not exist in the pass's tree, and the three sites that
could have built it are eliminated by instrumentation (`adjust_address_of`,
`take_reference_address`, `convert_reference`). The next step is therefore to
instrument the *construction*: a temporary `fprintf` plus `::backtrace_symbols`
in `address_of2t`'s constructor, firing when the operand is an `if2t`, on
`github_6717_throw_conditional_ok` under the flag. There is no backtrace helper
in `src/util`, so that probe brings its own. Note gdb is the wrong tool here:
these constructors are inlined statics and breakpoints slide.

### 3.20 The last two rows: the write-back drops `#reference` inside the body

Counting the reference spelling in the two symbol tables answers it. On
`github_6717_throw_conditional_ok`, legacy's tree carries **four** `#reference`
markers and the pass's written-back tree carries **one**:

| where | legacy | flag |
|---|---:|---:|
| the copy constructor's parameter type | 1 | 1 |
| `pointer` types inside the function body | 3 | **0** |

The parameter survives because §2.5's `pointer_type2t::ref_kind` carries it on
the symbol's type, and `migrate_type_back` restores it (`migrate.cpp:3215`). The
three inside the body do not, and they are what the legacy pipeline reads *after*
the write-back: `c_typecastt::implicit_typecast_followed` tests
`is_lvalue_or_rvalue_reference(dest_type)` before calling
`take_reference_address`, the irept helper that distributes the address-of over a
conditional's arms (`c_typecast.cpp:584`). With the spelling gone the argument
takes the plain pointer conversion instead, which is exactly the
`&(c ? a : b)` the GOTO shows.

So the missing rewrite was never missing: it is legacy's own, and it declines
because the tree it is handed no longer says "reference".

**Why the body loses it.** Every `address_of2tc(subtype, obj)` in the pass
defaults `ref_kind` to `none` — `irep2_expr.cpp:553` documents that default for
`carry_provenance` in the same breath — so any reference binding the pass rebuilds
comes back as a plain pointer. That is the fix's shape: an address-of the pass
builds over a reference-typed object has to carry
`pointer_ref_kindt::LVALUE`, the way #7703 carries it on types. Note this also
touches the fold in this PR, whose `address_of2tc(a.lhs->type, a.lhs)` defaults
the same way; no census row moves on it today, so it is fidelity rather than a
defect, and the two should be fixed together.

### 3.21 Phase 7 closes: 84 of 84

§3.20 named the site. `adjust_call_arguments` took the address of an argument
bound to a reference itself —

```cpp
if (binds_by_reference(callee, arg, params[i], i, context, ns))
  arg = address_of2tc(arg->type, arg);
```

— where the legacy path routes the same binding through
`c_typecastt::implicit_typecast_followed` to `take_reference_address`, which does
two things this did not: it distributes over a conditional's arms, and it records
the reference kind on the pointer it builds. Both matter, and the second is why
the first could not be recovered later: `ref_kind` back-migrates to
`#reference`, and that is the bit the legacy pipeline tests *after* the
write-back before it would have distributed for itself.

So the arm was never missing, twice over: the IREP2 helper already existed in
`c_typecast.cpp` and already did both jobs. It was `static`, and this call site
had grown its own one-line substitute. Exported and called here, with
`binds_by_reference` now reporting the kind the legacy declaration spells (the
IREP2 parameter type need not carry it, which is why that predicate reads the
declaration at all).

Both remaining rows close, so the `regression/esbmc-cpp/cpp` census under
`--clang-cpp-irep2-adjust-only` reads **84 of 84**, from 82 — and from 31 of 80 at
#7718's census. The change is in the *C* pass, shared by every frontend that runs
it, so the nets are wider than usual: `ctest -R irep2` 374 of 374, unit 872 of
872, the Solidity corpus 507 of 507 agreeing with zero crashes, and the C suite's
2292 rows with two THOROUGH k-induction rows timing out under parallel load
(`github_302` passes standalone). `irep2_conditional_reference_bind{,_fail}` pins it: the passing
half turns SUCCESSFUL -> FAILED with the plain address-of restored. The failing
half is a conventional twin — it asserts that neither object moved, which is
false whichever arm is selected, and stays FAILED under the mutation, so the
passing half is the gate.

**What the whole series says about method.** Three ports were written and reverted
before this one: the `address_of` distribution as an arm (twice, §3.13 and
§3.17), and the `constructor` return type as a `code_type2t` field (§3.18). Each
was a guess at *which rewrite is missing*, and each was refuted by instrumenting
the thing it claimed to fix. What worked was counting a marker in the two symbol
tables — four against one — and reading the consumer that tests it. A divergence
census says which rows disagree; only the tree says why.

## 7. The whole C++ corpus, swept once (2026-09-13)

"84 of 84" was a stride-12 sample of **one** subdirectory. The corpus is
`regression/esbmc-cpp` (2891 rows) plus `esbmc-cpp11` (166) and
`esbmc-cpp14/17/20/23` (125): **3182 rows**. Swept end to end under
`--clang-cpp-irep2-adjust-only` against the default path, skipping
`KNOWNBUG`/`FUTURE`, rows that pin an irep2 flag, and rows whose source is
absent (87 in total), the measurable set is **3095**:

| | rows |
|---|---:|
| verdicts agree | **3033** |
| diverge | 55 |
| hard failure under the flag | 7 classified as crashes, plus 18 rows that produce no verdict |

So 98.0% of the corpus agrees, and the residue is 62 rows rather than the two the
sample suggested. Reproduce with `scripts/irep2-migration/cpp_full_census.sh <out.tsv> [jobs]`
(resumable, one TSV row per test: status, directory, both verdicts).

### 7.1 The residue by cause, not by directory

Every hard failure names itself, and they group into five causes:

| cause | rows |
|---|---|
| `Unexpected type: ptrmem` | `ch22_11`, `github_2672{,_fail}`, `ptrmem18`, `ptr_to_member_3{,_fail}`, `github_6293_member_fn_ptr{,_fail}` |
| `uncaught exception [bad_optional_access]` | `alignas_empty_struct{,_fail}`, `stack_class{,_bug}`, `github_3522_2` |
| `with_type called on kind sideeffect_assign` | `ch17_3` — **fixed**, §7.3 |
| `ERROR: compute_pointer_offset` | `ostringstream_str`, `sstream_str_bool` |
| `caught SIGSEGV` | `ptr_to_member_2{,_fail}` |

plus `github_2040` and `github_6368_insert`, which produce no output at all
within 45s — **not defects**: both agree given 90s, so the census's per-run cap
is what they hit. A cap is part of the measurement, and two of the 62 residual
rows were the cap rather than the pass.

The 55 verdict divergences are 36 `SUCCESSFUL -> FAILED`, 14
`SUCCESSFUL -> none`, 4 `FAILED -> none`, and one row where the **flag is
better**: `ch9_7` answers where the default path does not. By family: 27 are
`try_catch` (the `exception_spec_*` cluster §3.14 named, which is
`finalize_exception_specification`'s territory), 17 in `esbmc-cpp/cpp`, 6 in
`bug_fixes` (two POD-initialisation rows, two member-function-pointer rows), 2
union constructors in `esbmc-cpp11/constructors`, and three singletons.

### 7.2 Next, in order of what the causes cost

1. **`with_type` on `sideeffect_assign2t`** — one row today, and §3.18 predicted
   it: that kind carries `#member_init` outside `fields` exactly as
   `sideeffect2t` carried `constructor`, so its field order does not match its
   constructor's parameter order and the generic rebuild refuses it. The fix is
   the specialization already written for the sibling kind.
2. **`bad_optional_access`** (5 rows) — one unchecked `std::optional` somewhere in
   the flag path; the exception name is the whole lead.
3. **`ptrmem`** (8 rows) — the seam has no IREP2 type for a pointer to member.
   The largest family and the only one that needs a new type kind rather than a
   repair.
4. **`try_catch`** (27 rows) — the known exception-specification cluster.

### 7.3 `with_type` on an assignment, and the fix that was already written

`ch17_3` aborted under the flag with `with_type called on kind sideeffect_assign
which has no substitutable type`. §3.18 predicted it and mis-scoped it: it said
nothing calls `with_type` on that kind, which is why carrying `#member_init`
outside `fields` had gone unnoticed. Over the whole corpus something does.

The cause is the one §3.18 measured for the sibling kind. `supports_with_type_v`
requires the `fields` order to match the primary constructor's parameter order;
`sideeffect_assign2t` listed `member_init` after `rhs` while its constructor
takes `location` there, so the trait was false and every `with_type` on an
assignment took the abort. Unreflecting the field makes the trait true, and a
`rebuild_with_type` specialization carries the flag through — the same pair of
moves the `constructor` marker needed, and the specialization for it was already
in the tree to copy.

`ch17_3` now agrees. The gate is a unit test rather than a regression row:
`unit/irep2/with_type.test.cpp` asserts that a `with_type` on an assignment
carrying `member_init` neither aborts nor loses the flag, and the same for a
constructor-marked side effect. That test bites on both halves of the change —
re-reflecting the field aborts it, dropping the specialization fails the flag
assertion — where a regression row could only show the abort. The construct in
`ch17_3` that reaches the rebuild is not reduced: it fires four times there with
`op=assign`, and none of nine hand-written compound- and plain-assignment probes
reaches it, so the reproducer is the corpus row until someone bisects it.

### 7.4 `bad_optional_access`: one cause, three crash faces

The five rows resolve to a single defect, and the exception name was a symptom of
it rather than the fault. `catch throw bad_optional_access` in gdb put the first
one at `member2t::do_simplify` (`expr_simplifier.cpp`), reading
`struct_union_get_component_number(...).value()` on a component the source's type
does not describe. Guarding that moved the crash to
`tuple_node_smt_ast::project` reading past its element vector — #7758's bug,
whose guard this branch reverted — and the third face was the same `.value()` in
`value_sett::make_member`.

Behind all three: a **struct literal shorter than its own type**. Diffing
`alignas_empty_struct`'s GOTO shows it exactly:

```
legacy  ASSIGN test={ .anon_pad#0=0 };
flag    ASSIGN test={  };
```

`adjust_struct` runs and declines. Instrumented, it reports
`ops=0 unpadded=0 padded=0 align=` — the literal has no operands, its type has no
members, `add_padding` adds no pad, and the **alignment is empty**: an explicit
`alignas` travels on the legacy type as an `alignment` sub-irep, `struct_type2t`
has no field for it, so the back-migrated type add_padding sees is not
over-aligned. An over-aligned empty struct occupies its alignment, so its padded
layout has a trailing pad; without the alignment there is nothing to pad to.

Carried as an unreflected `BigInt alignment` on `struct_type2t` — the
`constructor`/`member_init` pattern, and unreflected for the same reason plus one
more: two records differing only in alignment would otherwise stop comparing
equal, which is a wider change than this repair. `fields_cover_class` needs the
matching `excluded_field_bytes` declaration, as it did for the other two.

Both optional guards stay. They are not redundant once the literal is padded:
each was a release-build crash on an invariant only a debug build checks
(`member2t`'s constructor asserts it), and the `make_member` one widens the
may-points-to set to `unknown` where it previously threw — the conservative
direction. The same call also read `datatype_members[no]` unguarded, which is an
out-of-bounds vector read whenever a literal is short, so the guards are a
memory-safety repair independent of the alignment.

All five rows now agree, and `irep2_overaligned_empty_struct{,_fail}` pins the
carry: both halves SIGSEGV with the restore suppressed.

### 7.5 The `ptrmem` family: a type with no IREP2 form, and the arm behind it

Eleven rows — the eight `Unexpected type: ptrmem` ones, the two
`ptr_to_member_2` SIGSEGVs and `github_6717_inline_ptr_to_member`'s verdict
divergence — are one construct: `obj.*pmf` and `obj->*pmf`, a *bound member*
selection. The expression kind was never the problem; `ptr_mem2t` exists and
`migrate.cpp` carries it both ways. Its **type** was: clang types the selection
`BuiltinType::BoundMember`, the converter records `ptrmem_typet`, and
`migrate_type` had no arm for it, so the body failed to migrate at all.

The type is a placeholder rather than storage — `clang_c_adjust::adjust_ptr_mem`
replaces the whole node with the member function before anything computes a
width — so it maps to the empty type, the round-trip-stable form a constructor's
return type already uses. That alone moved the failure to
`ERROR: do_function_call: unexpected callee`, which is the elimination itself
missing: the node survived to goto_convert.

Ported as an arm in both tables (the C++ table substitutes its own list, so a row
added to the C one never dispatches for C++ or Solidity). It dereferences a
pointer base, and for the placeholder type rebuilds the *function* with `this`
prepended to the pointed-to code type's parameters. Two details are legacy's and
kept deliberately: only the parameter **type** is prepended, the argument being
the call site's business; and a pointer-to-*data*-member selection is left alone,
since it carries the member's own type rather than the placeholder.

All eleven rows now agree. `irep2_bound_member_call{,_fail}` pins it over both
spellings; both halves fail with `do_function_call: unexpected callee` when the
arm is gated off.

### 7.6 The 27 `try_catch` rows: `#cpp_type` does not cross the seam

They are one cause, and it is not `finalize_exception_specification` as §3.14
guessed. `try-catch_simple_01` is the whole family in five lines: `throw 1`
caught by `catch (int)`, and under the flag the GOTO reads

```
ASSERT !(exc_thrown && exc_typeid == 92)   // uncaught exception: signedbv
```

where the legacy path reads `uncaught exception: signed_int`. An exception id for
a primitive is its `#cpp_type` **spelling** (`clang_cpp_exception_id.cpp`'s
`append_cpp_spelling_and_fallback`, falling back to `type.id()`), the adjust pass
computes the throw's ids from `migrate_type_back(operand->type)`, and the seam
carries no `#cpp_type` — so the throw reads `signedbv` while the handler, whose
ids never cross the seam, still reads `signed_int`. Nothing matches, and every
throw of a primitive escapes as uncaught: 36 of the census's 55 divergences are
`SUCCESSFUL -> FAILED`, and this is most of them.

**Conversion time is where it belongs, and a bad probe nearly hid that.** §3.15's
precedent is to move a computation to where the legacy spelling still exists.
Instrumented at the converter's `CXXThrowExpr` arm the operand's type id printed
**empty**, which read as "the type is not settled yet" — and was wrong: the probe
read `tmp` *after* `move_to_operands` had moved it. Reading `new_expr.op0()`
instead prints `id=signedbv cpp_type=signed_int`. The spelling is in hand at
conversion time; the reading of the first probe was the defect.

So the throw's ids are recorded there, for any type whose ids follow from the
type alone — pointer and array layers stripped, since `convert_exception_id`
recurses through them. A class type is left to the adjust pass: its id is the
type symbol's name, which crosses the seam intact, and expanding its bases needs
a symbol-table lookup this early in conversion cannot rely on. The legacy pass
recomputes the same strings from a type that still carries the spelling, so the
default path is unchanged.

**16 of the 27 rows now agree**, and `ctest -R try_catch` is 174 of 174. The
remaining 11 all fail with `exception specification violated`, which *is*
`finalize_exception_specification`'s territory as §3.14 said — a legacy-only
step, and the next task.

An array operand is excluded from the recording: it decays between conversion and
the legacy pass, so an id taken from the pre-decay type is not the one the handler
is matched against. Measured, not assumed — and the measurement was worth having
for a second reason. The first version of this change also replaced
`new_expr.type() = tmp.type()` with the operand's type, on the reading that the
original assigned from a moved-from `tmp` and was therefore a bug. It is not: a
cpp-throw's own type is set by the adjust pass, and giving it the operand's type
at conversion made three `try_catch` rows stop failing as they should **on the
default path**. Bisected by stashing the change, and the line is now commented so
the next reader does not repeat the correction. The
alternative fix, carrying `#cpp_type` on the primitive type kinds, is no longer
needed for this family; reconstruction from the width would not have worked
anyway, since on this target a 32-bit signed type is `signed_int` or `wchar_t`, a
64-bit one `signed_long` or `signed_long_long`, and picking wrong turns a valid
`catch` into a false "uncaught exception" with the same symptom.

### 7.7 The other 11: a dynamic exception specification is never resolved

`exception specification violated` on 11 rows, and §3.14 named the right function
for them: `clang_cpp_adjust::finalize_exception_specification`. The converter
stashes a `throw(T...)` specification's declared types under
`exception_spec_decl`; the legacy pass resolves them to exception ids in
`adjust_symbol`, once the namespace is populated. The IREP2 pass *replaces*
`adjust_symbol`, and had no counterpart — so the specification reached symex with
its declared list unresolved, which permits nothing, and every throw through such
a function was a violation.

Two moves, both small. The finaliser leaves `clang_cpp_adjust` for
`clang_cpp_exception_id.{h,cpp}`, next to `convert_exception_id` and shared for
the same stated reason: it reads nothing but the namespace. And the C pass gains
a per-symbol *type* hook, `adjust_symbol_type`, called for **every** symbol rather
than only those with a value — `gen_symbol_code` is gated on a non-nil value, and
a declaration with a specification and no body still needs resolving. C overrides
nothing; C++ resolves the specification there.

**All 27 `try_catch` rows now agree**, from 0 at the start of §7.6, and
`ctest -R try_catch` is 174 of 174. `irep2_dynamic_exception_spec{,_fail}` pins
it — both halves change outcome with the hook gated off, and the mode is pinned
`c++11` because a dynamic specification is ill-formed in C++17.

### 7.8 The two stream rows: a conditional over arrays decays per arm

`ostringstream_str` and `sstream_str_bool` aborted with `ERROR:
compute_pointer_offset, unexpected irep: if`. The dump names the node, and the
GOTO diff names the difference, inside the bool stream operator:

```
legacy  _put_field(&(*o::0), val::1 ? &"1"[0] : &"0"[0], 1)
flag    _put_field(&(*o::0), &(val::1 ? "1" : "0")[0], 1)
```

`c_typecastt::do_typecast`'s irept copy has an explicit arm for this
(`c_typecast.cpp:985`): an array-typed destination that *is* an `if` typecasts
each arm instead of indexing the conditional. The IREP2 copy did not, so the
decay produced `&(if)[0]` and the pointer machinery met an `if` where it expects
an object. Ported, both rows agree.

**The pair that pins it needed a second measurement to be worth anything.** Arms
of *different* lengths each decay on their own — no array-typed conditional is
ever formed — so the first version of the test passed with the arm gated off.
Same-length arms (`"ab"` / `"xy"`) are what form the node, and then both halves
abort with the arm off. The condition is recorded in the test source, since it is
invisible from the construct.

## 8. The corpus re-swept: 3086 of 3095, and no crashes anywhere

Re-run end to end on the tree at this point, the same 3192 rows and the same
harness as §7:

| | §7 (before) | now |
|---|---:|---:|
| verdicts agree | 3033 | **3086** |
| diverge | 55 | **9** |
| hard failure under the flag | 25 | **0** |

The measurable set is 3095 either way (97 rows are `KNOWNBUG`/`FUTURE`, pin an
irep2 flag, or have no source), so agreement is **99.7%**, and not one row
crashes, aborts or fails to migrate. Five causes closed it: the `with_type` trait
on an assignment, the record alignment behind three `bad_optional_access` faces,
the bound-member type and its elimination, a primitive throw's exception ids, and
a dynamic exception specification — plus the conditional array decay of §7.8.

The nine that remain, by cause rather than by count:

| rows | what |
|---|---|
| `switch_declaration{,_1,_2}` | no verdict: `Couldn't convert expression in unrecognised format`, a migrate gap on a declaration inside a `switch` |
| `1032_POD_init`, `1043_POD_init` | POD aggregate initialisation |
| `CpyConstructorUnion`, `MoveConstructorUnion` | implicit union copy/move constructors — `gen_implicit_union_copy_move_constructor` is legacy-only, the same shape as §7.7's finaliser |
| `github_4317` | singleton |
| `vector_reserve_realloc_nested_fail` | the **flag** answers and the default path does not; the second such row, after `ch9_7` |

`switch_declaration` is the next task: three rows, one error message, and the only
group left that produces no verdict at all.

### 8.1 A switch condition that is a declaration

The three `switch_declaration` rows produced no verdict: the solver was handed a
`code_decl`.

```
ERROR: Couldn't convert expression in unrecognised format
code_decl
* value : c:main.cpp@42@F@main#@x
* init : constant_int …
```

C++ lets a switch condition be a declaration — `switch (int x = 0)` — and
`clang_cpp_adjust::adjust_switch` hoists it: the declaration goes ahead of the
switch, which then switches on the declared symbol. Without that the declaration
*is* the switched value, so it flows into the guard and reaches the SMT layer as a
statement.

Ported as an arm on `code_switch2t`. Two details: the seam flattens a
single-declaration `decl-block`, so the condition arrives either as the
declaration or as a block holding just it, and the rewrite **splices** rather than
nests — the declaration's scope is the switch statement, which is exactly what the
enclosing block gives it, and it is the shape legacy builds. `hoist_for_init`
documents the same choice for the other reason: a nested block would end the
scope too early (#4715).

All three rows agree. `irep2_switch_declaration{,_fail}` pins it; both halves fall
back to `unrecognised format` with the arm gated off.

### 8.2 The union copy/move constructor is generated, not converted

`CpyConstructorUnion` and `MoveConstructorUnion` diverged because a union's
implicitly-defined copy and move constructors have **no converted body at all**:
clang declares them, and `clang_cpp_adjust::gen_implicit_union_copy_move_constructor`
generates the one assignment that copies the object representation
([class.copy.ctor]/14). The IREP2 pass replaces that pass and generated only the
vptr initialisations, so both constructors ran with an empty body and the copy
never happened.

Generated in `gen_symbol_code`, before the value walk, so the assignment goes
through the arms like any other statement. Three things the port had to get
right, each found by measuring rather than reading:

1. A `sideeffect_assign2t` is an expression; a block's operands must be
   statements. `goto_convert: non-code operand` until it became a
   `code_assign2t`.
2. The marker is the converter's `#implicit_union_copy_move_constructor` on the
   constructor's return type, read from the symbol's legacy type — it is there,
   unlike the `constructor` spelling inside a *body*, which the write-back loses
   (§3.17).
3. Legacy's body is `*this = *ref`, **both** sides dereferenced: the second
   parameter is a reference, modelled as a pointer, and legacy reaches that shape
   through `adjust_assign`, which this pass's reference handling does not cover
   for a statement assignment. Assigning `*this = ref` instead handed the encoder
   a mismatch it has no handler for — a null function pointer in
   `convert_ast_node`, i.e. a SIGSEGV rather than a diagnosis.

Both rows agree, `ctest -R esbmc-cpp11` is 161 of 161, and
`irep2_union_copy_ctor{,_fail}` pins it over both the copy and the move
spelling — the passing half flips with the generator gated off.

### 8.3 The POD rows: a bitfield's flag does not cross the seam

`1032_POD_init` and `1043_POD_init` failed inside `memset`: *"memset of memory
segment of size 6 with 8 bytes"*. The type symbol is identical on both paths; the
global's initialiser is not:

```
legacy  { .a=0, .b=0, .c=0, .d=0, .anon_bit_field_pad#4=0, .anon_pad#5=0 }
flag    { .a=0, .b=0, .c=0, .d=0, .anon_pad#4=0 }
```

One pad short, so the object is 6 bytes where `sizeof` says 8. A bitfield
member's legacy type carries `#bitfield` **and** a subtype naming the underlying
type; `migrate_type` keeps the width and drops both. `adjust_struct` computed its
padded layout by back-migrating the literal's type and re-running `add_padding`,
which then saw three plain 2-bit integers rather than bitfields and inserted no
bit-field pad.

The fix is to stop round-tripping when there is no need to: the **tag symbol
already carries the padded layout**, so resolve it by name first and compute from
the type only when that lookup misses. §7.37 had replaced the lookup with the
computation because a struct declared inside a Solidity contract has a qualified
tag the literal's type does not name — so both paths stay, lookup first,
computation as the fallback. The Solidity corpus is still 507 of 507, which is
what keeps the fallback honest.

Both rows agree. `irep2_bitfield_global_init{,_fail}` pins it: with the lookup
forced to miss, the passing half fails on the same `memset` and the failing half's
violated property moves off its pinned line. Carrying `#bitfield` and its subtype
across the seam remains the more faithful fix, and is now the only known reason a
*computed* layout can differ from the tag's.

### 8.4 The last singleton: an array-typed element is not a decay

`github_4317` — a range-for over `S cases[][2]` — differed in one instruction:

```
legacy  ASSIGN __end1=&(*__range1)[0] + 1;
flag    ASSIGN __end1=&(*__range1)[0][0] + 1;
```

One index too many, so the end pointer is one *element* past the start rather than
one row, and the second row read is out of bounds. Instrumented, the decay fires
four times on an operand that is already an `index`, and twice on the symbol
itself, which is the correct one.

`&row` where `row` is a row of a 2-D array has type `S (*)[2]`: taking the address
of an array-typed *element* is not the array-to-pointer decay (C11 6.3.2.1p3),
and the arm now declines it. The legacy arm decays unconditionally too and gets
away with it because its converter leaves such an index typed as the element —
this pass types it as the row, correctly, and the decay was the only thing
relying on the old typing.

`irep2_nested_array_row_address{,_fail}` pins it; the passing half fails with an
array-bounds violation when the decay is forced on every array operand.

**Phase 7's census is now one row from exhausted**, and that row is not a defect:
`vector_reserve_realloc_nested_fail` is the second where the flag answers and the
default path does not.

### 8.5 Phase 7's census is exhausted

The last row was the harness, not the pass.
`vector_reserve_realloc_nested_fail` showed `none` for the *legacy* side, which
read as "the flag answers where the default path does not". Re-run with a real
budget both paths report `VERIFICATION FAILED`: the census caps each run at 45s
and that row needs more. `ch9_7`, the other row of that shape, is the same
artefact.

So over `regression/esbmc-cpp`, `esbmc-cpp11` and the standard-mode suites —
3192 rows, 3095 of them measurable — **every row agrees under
`--clang-cpp-irep2-adjust-only`**, and none crashes, aborts or fails to migrate.
The nine causes this took are §7.3 through §8.4, plus §7.6's throw ids and §7.7's
exception specifications.

Two things that number does *not* say, and both belong next to it:

- The cap is part of the measurement. Two rows were mis-read because of it, and a
  third pair (`github_2040`, `github_6368_insert`) was mis-read the same way in
  §7.1. A census row that reports no verdict is a row to re-run, not a defect.
- Agreement is not sufficiency. The flag replaces the legacy adjuster; it does
  not yet build IREP2 natively end to end, which is what §1's bar asks for. What
  agreement buys is the right to consider making the flag the default — and that
  needs an SV-COMP run, since it moves every C++ verdict path.

## 9. Twelve probes, written without reference to this branch

§7 and §8 swept the whole corpus, which is the broad measurement. This is the
narrow one, and it is worth having because the corpus was also what the branch was
developed against: a probe set written fresh, from the C++ feature list rather
than from the diff, and run on master first.

Twelve constructs: a virtual call through a base pointer, an lvalue and a const
reference, a function template at two instantiations, an overloaded `operator+`,
`throw`/`catch` of an `int`, a scoped constructor and destructor, `new`/`delete`,
a capturing lambda, multiple inheritance with a cast to the second base, a static
data member, a default argument, and `std::vector::push_back`.

| | master | this branch |
|---|---|---|
| agree | 10 | **12** |
| diverge | 2 | 0 |
| abort under asserts | 0 | 0 |

### 9.1 Which two, and why it matters that they were not chosen

On master the two that diverge are `v05_exception` and `v12_vector`, and their
shapes are exactly the families this branch addresses:

- the uncaught-exception assertion names the type `signed_int` on the default path
  and `signedbv` under the flag, and `__ESBMC_exc_site` differs -- the
  exception-id-at-conversion-time work;
- the vector model shows `val ? &"1"[0] : &"0"[0]` against
  `&(val ? "1" : "0")[0]`, the per-arm conditional decay, and
  `__refcnted_cstr(&this->msg)` against `this->msg=__refcnted_cstr()`, the
  constructor fold.

Nothing in the probe set was selected for those; the list is a textbook C++
feature enumeration. That the two failures land on the two families this branch
fixes, and that all twelve agree once it is applied, is the strongest evidence
available that its scope was the right one rather than a scope fitted to whatever
the corpus happened to contain.

### 9.2 What it says about the rest of C++

Ten agree on master already, including the constructs a reader would most expect
to be hard: virtual dispatch, multiple inheritance with a displaced second base,
templates, operator overloading, destructor scoping, `new`/`delete` and a
capturing lambda. So the C++ hop-off's remaining divergence is not spread thinly
across the language; before this branch it was two families, and after it, over
this probe set, none.

Run under `DebugOpt`, so with asserts: no probe aborts on either tree, which the
byte-identical comparison alone would not have shown
(`scope-jimple-irep2.md` §38.4a).

## 10. Fourteen harder probes, and what 26 agreeing rules out

§9's twelve were a textbook feature list. These fourteen are the constructs a
reader would reach for to break a C++ adjust pass, run on this branch under
`DebugOpt`:

virtual inheritance through a diamond with a cast to the shared base; an abstract
class called through its interface; a copy constructor with a side effect; an
overloaded assignment operator; a nested class; a friend function; `catch (...)`
selected over a non-matching `catch (int)`; a bare `throw;` rethrow from inside a
handler; `dynamic_cast` to a derived pointer; a template class with a member
function; a `const` member function beside a mutating one; `operator()` and
`operator[]`; `const_cast` followed by `static_cast`; and an array of objects with
a default constructor.

**All fourteen agree, and none aborts.** With §9 that is 26 independent probes
agreeing on this branch, against a corpus of 3 095 rows (§8) that also agrees.

### 10.1 What that does and does not establish

It rules out the failure mode §9 was written to test for: that the corpus
agreement reflects the corpus rather than the pass, since the pass was developed
against that corpus. Twenty-six constructs chosen from the language instead, two
of which fail on master and both of which this branch fixes (§9.1), is evidence
the agreement is a property of the pass.

It does not establish completeness. Every probe here is a *whole-program* check
of the emitted GOTO; a construct whose adjust arm is wrong in a way that cancels
out by the time goto_convert has run would pass. And the probes are small: none
mixes the features, where the corpus's 3 095 rows do.

What remains for this frontend is therefore not more probing of the adjust pass.
It is `clang_cpp_convert.cpp` and the 639 legacy type mentions
(`frontends-to-irep2.md` §43), which the phase list puts last for every frontend.

## 11. The exception specification crosses the seam, and the ctor/dtor marker still does not (2026-09-25)

`clang_cpp_adjust::adjust_symbol` and `clang_cpp_adjust_irep2::adjust_symbol_type` resolve a dynamic
exception specification (`throw(T...)`) and write the function type back. `migrate_type` dropped
`exception_spec_kind` and `exception_spec_types`, which `goto_convert_functions` decodes from the
symbol's type, so those writes had to stay legacy.

`code_type2t` now carries a resolved specification as two unreflected fields, `exception_kind` and
`exception_types`, both ways in `migrate_type`. An unresolved one (still holding
`exception_spec_decl`) is not carried: carrying only its kind would read back as `throw()`.
`finalize_exception_specification` now reports whether it resolved anything, so the write happens only
for a `throw(T...)` function; every other function's type was being rewritten unchanged.

The writes themselves stay legacy, for §57.1's other reason. `migrate_type` turns a constructor's or
destructor's pseudo return type (`"constructor"` / `"destructor"`) into `empty`, and the vptr
initialisation reads it right after this write. Stored IREP2-side, a `throw()` constructor lost its
vptr assignment: `regression/esbmc-cpp/cpp/throw_spec_ctor_dtor_vptr` became a false alarm and its
`_fail` twin (a destructor that must see the base's virtual, [class.cdtor]/4) a false proof. That pair
now pins it, and carrying the ctor/dtor marker is what converts these two writes. clang-cpp B-2*
stays 4.

With the carry disabled while the writes were IREP2-side, ten `regression/esbmc-cpp/try_catch/` tests
failed (e.g. `exception_spec_dynamic_violation_fail`); `unit/util/migrate.test.cpp` pins the round
trip, the unresolved case, that no specification adds no key, and that a specification is no part of
the type's identity.

## 12. The ctor/dtor marker crosses the seam (2026-09-25)

`frontends-to-irep2.md` §50.2 found the ctor/dtor function-type write blocked on the pseudo return
type: `migrate_type` maps `"constructor"` and `"destructor"` to `empty`, and vptr initialisation tests
for exactly those ids. §11 hit the same wall from the exception-specification side.

`code_type2t` now carries the marker (`return_marker`) and `#implicit_union_copy_move_constructor`
(`implicit_union_copy_move`) as unreflected fields, and `migrate_type_back` restores the pseudo return
type from them. With it, three writes store IREP2: `clang_cpp_convert.cpp`'s ctor/dtor
`fd_symb->set_type` (§50.2's 254 failures of 1 061) and §11's two exception-specification writes.
clang-cpp B-2* 4 -> 1; the last is `need_vptr_init` on the value.

Two further things the carry exposed, each measured:

- **An unresolved dynamic specification must cross too.** The converter's write runs before
  `finalize_exception_specification`, so a `throw(T...)` spec still holds `exception_spec_decl`.
  Dropping it made `X() throw() { throw 5; }` potentially throwing, a false proof in
  `try_catch/try-catch_decl_10_bug`. `exception_decl` now carries the declared types, and finalize
  resolves them after the round trip.
- **A constructor call that already passes its object.** In the IREP2 adjust pass a declined VLA
  construction keeps its initialiser, a constructor call whose first argument is the object.
  `goto_sideeffects` lowered every call to a `"constructor"`-typed callee by adding a temporary `this`,
  which it only ever saw once the marker survived; it now adds one only when the call is short of the
  constructor's parameters (`cpp/irep2_array_vla_construction` failed without it).

What the round trip still drops from a ctor/dtor type, from diffing `--symbol-table-only`: `#inlined`
on the code type, `#constant` on a reference parameter's pointee, and an implicit parameter's
`#location`, `name` and plain `identifier`. The one reader found is `c_link.cpp:193`, where `#inlined`
silences a duplicate-definition warning across translation units; a two-file probe with inline
ctors and dtors raised none. The argument-count rule in `goto_sideeffects` also cannot tell a variadic
constructor called without its object from one called with it; no probe reaches that shape.

`esbmc-cpp/{cpp,try_catch}` 1 333 of 1 339 pass, and the six failures (`github_7433*`, `ch8_5`) are the
exception-type spelling pins another branch updates; the rest of `esbmc-cpp` (1 927) passes apart from
seven tests at the 120 s cap that take the same time on the base binary.
