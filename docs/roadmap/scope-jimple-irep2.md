# Scope — jimple frontend → IREP2 construction

Opened 2026-08-09 as Phase 5 of `frontends-to-irep2.md`, which orders the
per-frontend migrations and puts jimple first: *"smallest surface, no
operational-model complication, lowest blast radius. The pathfinder for the
kit."* This document is the census and decomposition that phase asks each
frontend to open with.

**Build note (corrected 2026-08-09).** The first version of this document said
`ENABLE_JIMPLE_FRONTEND` is `OFF` locally and therefore *"every gate is a CI
round-trip"*. That was self-imposed and wrong: the frontend is gated only by a
`-D` define, needs **no JDK** (the tests feed `.jimple` files directly), and
`cmake -DENABLE_JIMPLE_FRONTEND=On .` plus a rebuild gives **17 tests passing in
3.5 s**. Every gate below is local.

## 1. Census

| measure | value |
|---|---|
| source | 3 259 LOC across 8 files (+ `AST/`) |
| legacy-type mentions (`*exprt`/`*typet`/`*codet`/`irept`) | **176** |
| IREP2 mentions (`expr2tc`/`type2tc`/`migrate_*`) | **0** |

By construct:

| construct | count |
|---|---:|
| `exprt` | 87 |
| `to_exprt` | 84 |
| `typet` | 20 |
| `to_typet` | 11 |
| `code_typet` | 9 |
| `pointer_typet` | 6 |
| `codet` | 6 |
| `empty_typet` | 5 |
| everything else (`struct_typet`, `nil_exprt`, `member_exprt`, `constant_exprt`, `array_typet`, `struct_union_typet`) | ≤2 each |

By file, the surface is concentrated:

| file | mentions |
|---|---:|
| `AST/jimple_expr.cpp` | 48 |
| `AST/jimple_statement.cpp` | 28 |
| `AST/jimple_expr.h` | 19 |
| `jimple-language.cpp` | 15 |
| `AST/jimple_method.cpp` | 10 |
| the remaining four | ≤9 each |

## 2. The shape this frontend has, and why it is the easy one

Almost all of it is **one interface, repeated**: a virtual
`to_exprt(contextt &, …)` on each AST node, with **27 overrides**, plus a single
`to_typet(const contextt &)`. There is no adjuster, no operational model, and no
`#`-attribute carriage of its own — the three things that made Python's
migration multi-quarter work.

That makes the decomposition unusually mechanical: the migration is changing one
virtual signature and its 27 implementations, not untangling a pass.

## 3. Phased decomposition

**J.1 — the type side first.** ~~`to_typet` is a single method and 20 `typet`
mentions. Migrating it to return `type2tc` is the smallest possible slice and
proves the seam before any expression work.~~ **Withdrawn — see §7.**

**J.2 — leaf expressions.** The `to_exprt` overrides that construct constants,
symbols and nil (`constant_exprt`, `nil_exprt`, and the symbol lookups). These
have no operands, so they can move without touching their callers' shape.

**J.3 — composite expressions.** The remaining overrides, inner-to-outer, so an
operand is already `expr2tc` when its parent migrates.

**J.4 — statements.** `jimple_statement.cpp`'s 28 mentions, which build `codet`
kinds. This is where §38.4's operand-surgery rule applies.

**J.5 — the seam.** `jimple-language.cpp`'s 15 mentions are the boundary to
`goto_convert`; once J.1-J.4 land, this is where the last `migrate_*` back-hop
is either removed or documented as remaining.

## 4. Gates

Inherited from `frontends-to-irep2.md` §7, plus one this frontend forces:

- **A build with `ENABLE_JIMPLE_FRONTEND=On` is a precondition for every gate.**
  Nothing in J.1-J.5 can be gated on this machine as configured.
- Byte-identity A/B over `regression/jimple` (15 tests), which §20 established at
  15/15 **non-trivially** — every one of those tests now takes the native
  dispatcher path end to end, so the baseline is meaningful rather than
  self-comparing.
- The decline census must stay at 0/15 (§20.1).

## 5. Risks

- **Small corpus.** 15 tests is the whole regression surface. §18.5's lesson —
  a sample that small cannot be the only evidence — applies with more force
  here than anywhere else in the program.
- **No local build.** Every gate is a CI round-trip, which is the slowest
  feedback loop of any frontend in the plan.
- **`to_exprt` takes `contextt &` by non-const reference** on most overrides,
  i.e. construction and symbol-table mutation are interleaved. Whether that
  survives the migration unchanged is the first design question J.1 has to
  answer, and it is not visible from the counts.

## 6. Status

Census and decomposition. No code has moved. J.1 as written is withdrawn (§7);
the replacement first slice is J.1'.

## 7. J.1 is the wrong first slice — checked before executing it

`to_typet` has nine call sites outside `jimple_type`, and **every one feeds its
result straight into a legacy API**:

| site | consumer |
|---|---|
| `jimple_class_field.cpp:10`, `jimple_declaration.cpp:8` | `symbolt::type` |
| `jimple_expr.cpp:204` | `c_typecast.implicit_typecast` |
| `jimple_expr.cpp:432` | a legacy `exprt` base type |
| `jimple_expr.cpp:535,581` | `gen_zero(typet)` |
| `jimple_expr.cpp:587` | `member_exprt` |
| `jimple_method.cpp:15,61` | `code_typet` |

So returning `type2tc` from `to_typet` would force a `migrate_type_back` at
**nine** sites and remove none. The slice makes the tree strictly worse until
J.2-J.4 land, which is the opposite of what a first slice is for.

The dependency runs the other way from what §3 assumed: in this frontend a type
has no independent consumer — it exists only to be handed to an expression, a
symbol or a code type. **Types can migrate only when their consumers do.**

### J.1' — one AST node, vertically (also withdrawn, see §8)

The replacement: migrate a single AST node kind **end to end** — its type, its
expression construction, and its operands — leaving exactly one `migrate_*` at
the boundary to its parent. One round-trip instead of nine, and the seam is
proved on a real node rather than on a method with no independent existence.

Pick the smallest leaf with its own type usage; `jimple_expr.cpp:535/581`'s
`gen_zero(type->to_typet(ctx))` constant nodes are the obvious candidates,
having no operands to thread.

This is the pattern Part V settled on for Python — *relax at construction,
re-enforce at the seam* — and it applies here for the same reason: a
horizontal slice through a construction tree has no cut that does not multiply
round-trips.

## 8. The slice is a virtual signature, not a node — so it needs a parallel method

§7 replaced J.1 with "migrate one AST node vertically". Measuring the 27
overrides to pick the right node killed that too, and found the real structure.

**The node exists — it just cannot be sliced.** Ranking every `to_exprt`
override by size and entanglement:

| node | lines | `ctx` uses | operand surgery |
|---|---:|---:|---:|
| **`jimple_constant`** | **15** | **0** | **0** |
| `jimple_nondet` | 21 | 0 | 0 |
| `jimple_cast` | 17 | 4 | 0 |
| `jimple_binop` | 21 | 3 | 0 |
| `jimple_deref` | 20 | 3 | 2 |
| `jimple_virtual_member` | 24 | 5 | 2 |
| `jimple_static_member` | 46 | 3 | 2 |
| `jimple_symbol` | 146 | 2 | 0 |
| the rest (`invoke`, `newarray`, `lengthof`, …) | 62-77 | 3-7 | 0-3 |

`jimple_constant` is a genuine leaf — four lines of body, no context, no
operands. It is exactly what §7 asked for. (Note in passing: the two nodes §7
*guessed* at, `jimple_static_member` and `jimple_virtual_member`, are among the
most entangled — 2 operand-surgery sites each. Guessing picked the worst
candidates available.)

**But `to_exprt` is a virtual declared on `jimple_expr`** (`jimple_expr.h:13-18`)
with a default body and 27 overrides. A return type is part of that signature.
You cannot migrate one override of it — the choice is all 27 at once, or none.

### 8.1 The technique this actually needs

Add a **parallel virtual**, do not change the existing one:

```cpp
virtual expr2tc to_expr2t(contextt &, const std::string &,
                          const std::string &) const
{
  // default: whatever has not migrated yet still builds legacy
  expr2tc e;
  migrate_expr(to_exprt(...), e);
  return e;
}
```

Overrides then move one at a time, each replacing the default for its own node,
with exactly **one** `migrate_expr` per un-migrated node and none per migrated
one. When the last override lands, the default and `to_exprt` both delete.

That gives the incremental path §7 was looking for, and it does it without the
round-trip multiplication that killed J.1 — because the round-trip lives in the
*default*, which shrinks as the migration proceeds, rather than at every call
site, which does not.

### 8.2 This is a decision for Phases 5-9, not just jimple

Every frontend in the plan converts through a similar interface — jimple's
`to_exprt`/`to_typet`, and the equivalent entry points in clang-c, clang-cpp,
solidity and python. If the parallel-method technique is right here, it is the
shape all five need, and §Phase 4's "reusable construction kit" (closed as
already-done in §38 of the parent document, on the grounds that its two named
helpers already exist) is missing this: **the kit's most important item is not a
helper, it is a migration technique for a virtual construction interface.**

That is worth settling before Phase 5 writes code, because getting it wrong
costs the same mistake five times — which is the exact failure Phase 4 was
written to prevent.

## 9. Status

Census, decomposition, and two withdrawn first slices. No code has moved. The
next decision is §8.1's technique, and it is a program-level one.

## 10. The seam is already IREP2-ready, and it is one line

§8.1's parallel-method technique has a precondition nobody had checked: it only
helps if *something* can consume `expr2tc`. If the top of the tree is still
legacy, migrating a leaf just moves the round-trip down rather than removing it
— which is the same objection that killed J.1.

**It is checked, and the answer is good.** `symbolt` already carries both sides
(`symbol.h:48-67`):

```cpp
const exprt   &get_value()  const;
const expr2tc &get_value2() const;
void set_value(const exprt &v);
void set_value(const expr2tc &v);   // <- the IREP2 setter exists
```

and the header states this is *"the end-state design, not transitional"*, with
the legacy side derived lazily and `migrate_expr_back` covering *"every expr2t
kind a symbol value may hold — including `code_block2t` for function bodies."*

**And jimple touches that seam in exactly one place:**

```
jimple_method.cpp:92   added_symbol.set_value(body->to_exprt(ctx, class_name, this->name));
```

One call site, and the setter it needs already exists beside the one it uses.

### 10.1 What that fixes about the decomposition

Migration should run **top-down from that line**, not bottom-up from a leaf:

| slice | change | round-trips after |
|---|---|---|
| K.1 | `jimple_method_body` gains `to_code2t`; `:92` calls `set_value(expr2tc)` | **one**, inside the new default |
| K.2 | statements override `to_code2t` | shrinks per override |
| K.3 | expressions override `to_expr2t` (§8.1) | shrinks per override |
| K.4 | `to_typet` → `type2tc`, last, when its consumers are gone (§7) | zero |

Each step *removes* a round-trip rather than adding one, because the boundary
starts at the top and moves down. Bottom-up had the opposite property, which is
why J.1 and J.1' both failed.

The caveat the header records applies and is worth carrying: the lazy split
tolerates *"latent holes"* in frontend-built legacy sub-expressions **as long as
nothing reads the IREP2 side**. K.1 makes something read it, so any such hole in
jimple's construction surfaces there — which is a feature for a migration, but
it means K.1's gate is the full 17-test suite, not a smoke test.

## 11. Status

Census; three withdrawn slices (J.1, J.1', both for reasons now understood); the
technique (§8.1); and the seam (§10). The decomposition is K.1-K.4 above. No
code has moved, and K.1 is the first executable step — one line at the seam plus
one new method with a migrating default.

## 12. K.1 shipped; K.2 attempted and blocked on decl-block flattening

**K.1 is PR #6851.** `jimple_method_body::to_code2t` with a migrating default,
and `jimple_method.cpp:92` handing the body over via `set_value(const expr2tc &)`.
GOTO output byte-identical across all 17 tests, captured before and after with a
stash-and-rebuild. The `symbol.h` "latent holes" caveat §10 flagged did not bite.

**K.2 was attempted and reverted.** The intended shape was a
`jimple_method_field::to_code2t(ctx, class, function, loc)` hook with a migrating
default, and `jimple_full_method_body::to_code2t` assembling a `code_block2t`
from it. Two things came out of trying it, one of which stops the design.

### 12.1 The location has to be a parameter

`jimple_full_method_body::to_exprt` stamps each statement's location *after*
building it (`expression.location() = l`). A `code_*2t` carries its location in
a non-reflected field, so it has to be set while the node is still a legacy
`exprt` — i.e. before migration, inside the hook. Hence the `const locationt &`
parameter. That part works and is worth keeping in any redesign.

### 12.2 The blocker: decl-block flattening is a legacy-side distinction

`migrate_expr`'s block arm (`util/irep/migrate.cpp`) does not migrate children
uniformly. It **splices** a child whose legacy statement is `decl-block`
directly into the parent's operand list, with a comment explaining why:
otherwise *"an extra code_block layer … would cause convert_block to emit DEAD
immediately after the initializer assignment instead of at scope end."*

A statement-level `to_code2t` returns an already-migrated `expr2tc`, and at that
point **the decl-block distinction is gone** — there is no `code_decl_block2t`
kind to test for, and a migrated decl-block is not reliably distinguishable from
an ordinary nested block. So the parent cannot decide whether to splice, and
reproducing `migrate_expr`'s behaviour through the hook is not possible as
designed.

### 12.3 What that leaves

Three options, none of them free:

1. **Keep the legacy read in the parent.** `to_code2t` on the body calls each
   statement's `to_exprt`, stamps the location, tests `statement() ==
   "decl-block"` itself, and migrates or splices accordingly. Correct, and it
   reproduces `migrate_expr` exactly — but it gives statements no hook, so K.3
   has nothing to override and the slice buys nothing.
2. **Give the hook a way to signal "splice me"** — return a small struct, or a
   distinct wrapper kind. Workable, but it puts a migration artefact into the
   AST interface.
3. **Stop emitting decl-blocks in the jimple frontend**, so the flattening has
   nothing to do. The cleanest end state, and the largest change: it means
   auditing every `jimple_declaration` site.

Option 3 is the one that leaves no residue, and it should be measured before
either of the others is built — the frontend has 3 259 LOC and the decl-block
may have few producers.

## 13. Status

K.1 shipped (#6851). K.2 blocked on §12.2 with three named options, the first
of which is a dead end for the slice and the third of which needs a census of
decl-block producers before it can be sized.

## 14. Phase 5 progress, and where the expression migration stops

Five PRs, each byte-identical across all 17 jimple tests, stacked in order:

| PR | slice |
|---|---|
| #6851 | K.1 — the seam: `set_value(const expr2tc &)`, `to_code2t` with a migrating default |
| #6853 | K.2 — the body assembles a `code_block2t` natively |
| #6854 | K.3 — `jimple_goto` |
| #6855 | K.3 — `jimple_label` |
| #6856 | K.4 — the parallel `to_expr2t` hook, with `jimple_constant` and `jimple_if` |
| #6858 | K.4 — `jimple_nondet` |

### 14.1 `jimple_binop` is where the ranking stops being a guide

§8's entanglement ranking puts `jimple_binop` next (21 lines, 3 `ctx`, no
operand surgery). It is not the next slice, for a reason the ranking cannot see:

```cpp
void jimple_binop::from_json(const json &j) { j.at("operator").get_to(binop); }
```

The operator is **a string taken straight from the input JSON**. Its domain is
whatever the Jimple producer emits, not anything closed by this repository —
grepping the frontend finds only the handful it special-cases (`==`, `+`, `*`,
`|`, `=`). Migrating it means mapping that string to an IREP2 kind, and an
unmapped operator would silently build the wrong node rather than fail.

The 17-test corpus cannot validate such a mapping: byte-identity only covers the
operators those tests happen to use. This is the same open-domain problem §32 of
the parent document found in `#cpp_type`, and it deserves the same treatment —
measure the domain before designing for it.

Two ways forward, neither guessed at here:

1. **Pin the operator set** from the Jimple specification or from the producer,
   and map exhaustively with a hard failure on anything unrecognised.
2. **Map the known operators and fall back** to the migrating default for the
   rest. The parallel-method design already allows a partial override, so this
   is expressible — but it needs the fallback to be deliberate and commented,
   not an accident of an incomplete `if` chain.

### 14.2 What is left

`jimple_symbol` (146 lines) and the invoke/member nodes carry the remaining
entanglement, and the operand-bearing statements — `invoke`, `return`,
`assignment`, `throw` — follow their expressions. `to_typet` stays last (§7).

## 15. The binop operator domain, measured (2026-08-09)

§14.1 said to measure the operator domain before designing for it. Measured
over the whole corpus:

| operator | occurrences |
|---|---:|
| `==` | 24 |
| `+` | 21 |
| `notequal` | 20 |
| `-` | 17 |
| `>=` | 14 |
| `>` | 9 |

Six distinct values, and the third is the one worth noticing: **`notequal` is a
word, not a symbol.** A mapping written from the C operator set would miss it,
and would do so silently.

`from_json` also rewrites `==` to `=` before anything sees it, with a
`// TODO, make hashmap for each operator` beside it — so the author already knew
this was the incomplete part.

### 15.1 The domain is bounded after all, and by something checkable

§14.1 called the domain open because the string comes from input JSON. That is
true of the *input* but not of the *effective* domain, which one more step
settles: `to_exprt` passes the string to `gen_binary`, which builds
`exprt(binop, …)` — a legacy irep id. The string is therefore only ever usable
if it already **is** a valid legacy binary-operator id, because `migrate_expr`
has to map it downstream. An operator outside that set is broken today, before
any migration.

So the bound is: *whatever `migrate_expr` maps for binary operators*. That is
enumerable from `migrate.cpp`, not from the Jimple producer.

### 15.2 What makes a partial mapping safe

Given that, the second option in §14.1 is the right one and can be made safe by
construction rather than by coverage: map the operators that are known, and
**fall back to the migrating default** for anything else. The parallel-method
design already permits a partial override, so an unrecognised operator takes the
same path it takes today rather than silently building the wrong node.

That turns the 17-test corpus from a validation problem into a sufficiency
question — the tests need only show that the mapped operators are mapped
correctly, because the unmapped ones are unchanged by construction. Byte
identity over the six above does exactly that.

## 16. `jimple_assignment` is gated on a typecast equivalence, not on jimple

#6860 migrated the relational operators but left `+` and `-` inert, because they
appear only in assignments. `jimple_assignment` is therefore the next slice by
value. It is not takeable yet, and the reason is not local to this frontend.

### 16.1 Three paths, two of which must fall back regardless

`jimple_assignment::to_exprt` has:

1. `is_skip` — returns a bare `code_skipt`.
2. Two `dynamic_pointer_cast` special cases for `jimple_expr_invoke` and
   `jimple_virtual_invoke`, which **mutate** the right-hand side
   (`dyn_expr->set_lhs(lhs_handle)`) and then return `rhs->to_exprt(...)` — a
   *statement*, not an assignment.
3. The plain path: build both sides, implicit-cast the right to the left's type,
   emit `code_assignt`.

Paths 1 and 2 fall back to the default cleanly; the partial-override design
already allows that. Path 3 is the one worth migrating.

### 16.2 The gate: two implementations of implicit_typecast, not one

Path 3 calls `c_typecastt::implicit_typecast(exprt &, const typet &)`. The IREP2
counterpart exists — `c_implicit_typecast(expr2tc &, const type2tc &, const
namespacet &)` — but it is **not a wrapper over the legacy one**:

```cpp
void c_typecastt::implicit_typecast(expr2tc &expr, const type2tc &type)
{
  ...
  implicit_typecast_followed(expr, src_type, dest_type);   // parallel impl
}
```

So migrating path 3 swaps one typecast implementation for another and *assumes*
they agree. Byte-identity over 17 jimple tests would exercise that assumption
only on the casts those tests happen to perform — the same insufficiency §15.2
solved for operators by falling back, and which does **not** apply here, because
there is no "unmapped" case to fall back on: every assignment takes the cast.

### 16.3 Why this is worth naming separately

Every frontend in Phases 5-9 performs implicit casts at assignment. If the two
implementations diverge anywhere, each migration inherits the divergence, and
the byte-identity gate will catch it only where the corpus happens to look.

That makes `c_typecastt::implicit_typecast(exprt&)` ≡ `(expr2tc&)` a
**program-level prerequisite**, in the same class as §8.1's parallel-method
technique: settle it once, or discover it five times. Establishing it is a
differential-testing question over the two implementations, not a jimple task,
and it belongs in the parent document rather than here.

### 16.4 Status

Seven slices shipped (#6851, #6853, #6854, #6855, #6856, #6858, #6860), all
byte-identical over the full suite. The next jimple slice by value is blocked on
§16.2; the next one that is *not* is `jimple_symbol` (146 lines, the largest of
the 27 overrides and the one the remaining statements all reach through).
