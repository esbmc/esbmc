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
§16.2; the next one that is *not* is `jimple_symbol` -- see §17.

## 17. `jimple_symbol` (#6865): a substitution, not a reimplementation

§16.4 called this the largest of the 27 overrides at 146 lines. That was wrong:
`jimple_symbol::to_exprt` is fifteen lines, and its body is a context lookup
followed by `symbol_expr(s)`.

That matters more than the correction does, because `symbol_expr` already has a
named IREP2 counterpart:

```cpp
expr2tc symbol_expr2tc(const symbolt &sym)
{
  return symbol2tc(migrate_symbol_type(sym), sym.id);
}
```

and `migrate_expr` routes level-0 symbols through the same construction
(`sym_name_to_symbol`, migrate.cpp:634). So unlike §16's assignment, this slice
does not swap one implementation for a parallel one — it calls the function the
migration path was already calling. The equivalence is by construction.

### 17.1 Why the mutant check was still worth running

Byte-identity over 17 tests proves nothing if the override never executes; a
`to_expr2t` that no caller reaches is trivially identical. Replacing the body
with `constant_int2tc(..., 4242)` changed the GOTO output of **all 17** tests,
which establishes that every test in the suite reaches this override — the
strongest exercise signal any slice in this stack has had, and unsurprising,
since every jimple statement that touches a variable goes through it.

### 17.2 Status

Eight slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865.
All byte-identical, all mutant-checked.

The remaining statement overrides (`invoke`, `return`, `throw`, `identity`,
`assertion`) reach operands through `jimple_symbol`, which is now native, so
they no longer inherit a migrating operand. `jimple_assignment` stays blocked on
§16.2. `to_typet` -> `type2tc` remains last (§7).

## 18. `jimple_return` (#6866): the valueless-value trap, and two mutants

`code_returnt` is not "zero or one operand". Its constructor resizes to one and
nils it:

```cpp
code_returnt() : codet("return") { operands().resize(1); op0().make_nil(); }
```

so `migrate_expr` takes the `operands().size() == 1` arm unconditionally
(migrate.cpp:2216) and recurses into a nil, which its first branch maps to a
null `expr2tc` (migrate.cpp:729). A native override that emitted "no operand"
for a valueless return would therefore be building a *different* node than the
one migration produces, even though both read as "return with no value".

This is the trap noted as low-priority during the goto_convert work; it is
cheap here because the correct native form is just a default-constructed
`expr2tc`.

### 18.1 One mutant was not enough

§17.1 used a single mutant, which sufficed because the override had one path.
`jimple_return` has two, and a single mutant cannot separate them:

| Mutant | Tests changed | What it establishes |
|---|---|---|
| M1 — drop the value, keep the return | 10 / 17 | 10 tests exercise the value-carrying path |
| M2 — replace the statement with a skip | 17 / 17 | every test reaches the override at all |

M1 alone would have left the other 7 ambiguous between "valueless return" and
"override never runs here". M2 resolves it: all 17 reach the override, so the 7
are genuine valueless returns and the nil-to-null mapping is covered by the
corpus rather than by argument.

**Rule for the remaining slices:** an override with N distinct output shapes
needs mutants enough to distinguish them, not one mutant to prove liveness.

### 18.2 Status

Nine slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865,
#6866. All byte-identical, all mutant-checked.

Remaining: `jimple_throw` (currently a bare `codet("cpp-throw")` with the body
commented out -- migrating it would pin an unfinished construct, so it is worth
checking whether it is reachable at all before taking it), `jimple_identity`,
`jimple_assertion`, and the two invoke forms. `jimple_assignment` stays blocked
on §16.2; `to_typet` -> `type2tc` remains last (§7).

## 19. Corpus census: three overrides can never be verified this way

Before taking `jimple_identity` I counted what the 17 tests actually contain,
by `object` key:

| Kind | Occurrences | State |
|---|---|---|
| `SetVariable` | 220 | blocked (§16) |
| `Variable` | 167 | #6868 |
| `Label` | 77 | #6855 |
| `If` | 69 | #6856 |
| `Return` | 53 | #6866 |
| `StaticInvoke` | 30 | open |
| `Goto` | 28 | #6854 |
| `Throw` | 14 | open, but see below |
| `SpecialInvoke` | 14 | open |
| **`identity`** | **0** | **unverifiable** |
| **`VirtualInvoke`** | **0** | **unverifiable** |

`jimple_assertion` is not in `from_map` at all, so nothing can construct it.

This is a hard limit, not a backlog. The method used by every slice in this
stack — byte-identity plus a mutant that must change the output — cannot say
anything about an override the corpus never reaches: identity holds vacuously
and no mutant moves. §18.1's rule makes that explicit, so `jimple_identity`,
`jimple_virtual_invoke` and `jimple_assertion` must not be migrated on the
strength of "it looks right".

`jimple_identity` would have been the worse trap of the three. Its right-hand
side is a `symbolt` constructed locally and never entered into the context, so
`migrate_expr`'s lookup fails and it falls through to the renaming parser,
which — finding no `?` or `!` — logs a warning and returns level0
(migrate.cpp:686). A native override calling `symbol_expr2tc` would produce a
level0 symbol too, but by a different route, and nothing in the corpus would
have caught a divergence.

### 19.1 Options for the three

Either extend the corpus so they become reachable, or leave them on the legacy
path indefinitely. Extending is the better answer and is not hypothetical —
§20 does exactly that for a *branch* rather than a statement.

## 20. `jimple_declaration` (#6868): a live arm the corpus did not reach

`jimple_declaration::to_exprt` ends with `decl.location() = get_location(...)`:
it sets its own location instead of taking the caller's. The migrating default
overwrites that only when `loc` is non-nil, so the override needs

```cpp
loc.is_nil() ? get_location(class_name, function_name) : loc
```

Mutating each arm to be unconditional changed **nothing**: both produce the
baseline across all 17 tests. On the usual reading that ternary is dead weight
and the "simplify aggressively" pass deletes it.

It is not. `jimple_label::to_code2t` passes nil to every nested member
(jimple_statement.cpp:124), so any declaration inside a label reaches the
fallback. A JSON walk over the corpus explains the measurement: 167
declarations, **0** of them nested in a label. The arm is demanded by a sibling
override's contract and simply never exercised.

### 20.1 The test, not the deletion

`regression/jimple/github_4715_label_scoped_decl_01` nests a declaration inside
a label. With the fallback the DECL carries
`file OriginalKt.jimple function main_0`; without it, that instruction and its
successor both print `no location`. The arm is now live, mutant-distinguished,
and pinned.

This is the shape §19.1 recommends for the unreachable overrides, and the
general lesson for the rest of the migration: **"no mutant moves" has two
causes — the code is dead, or the corpus is thin.** Deleting on the first
reading without checking the second silently drops behaviour that the language
permits and a sibling caller already relies on.

### 20.2 Status

Ten slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865,
#6866, #6868. Corpus now 18 tests, all byte-identical, all mutant-checked.

Remaining reachable: `StaticInvoke` (30), `SpecialInvoke` (14), `Throw` (14 --
but `jimple_throw::to_exprt` is a bare `codet("cpp-throw")` with its body
commented out, so migrating it would pin an unfinished construct rather than
preserve one). `jimple_assignment` stays blocked on §16.2; `to_typet` ->
`type2tc` remains last (§7).

## 21. `jimple_invoke` (#6870): the same gap, found by looking for it

§20 turned an unmoved mutant into a test. `jimple_invoke` has four distinct
shapes, so it got four mutants, and the pattern repeated:

| Mutant | Changed | Reading |
|---|---|---|
| M1 — intrinsic skip arm removed | 14 / 18 | heavily exercised |
| M3 — `@parameterN` assignments dropped | 3 / 18 | exercised |
| M2 — `@this` assignment dropped | **0 / 18** | see below |
| M4 — block `end_location` = loc, not nil | **0 / 18** | see §21.2 |

### 21.1 M2: reachable in principle, absent in practice

A census of every invoke in the corpus, keyed by `(object, base_class, has
variable)`:

| Count | Shape |
|---|---|
| 21 | `StaticInvoke` on `OriginalKt`, no variable |
| 14 | `SpecialInvoke` on `java.lang.AssertionError`, **with** variable |
| 7 | `StaticInvoke` on `kotlin.jvm.internal.Intrinsics`, no variable |
| 2 | `StaticInvoke` on `MainKt`, no variable |

Every invoke that carries a `variable` — the precondition for binding `@this` —
targets `java.lang.AssertionError`, which is on the intrinsic skip list and
returns before the binding is reached. The arm is not dead; the corpus simply
has no invoke of a non-static user method.

`regression/jimple/github_4715_invoke_this_binding_01` supplies one: a
non-static `setup` invoked via `SpecialInvoke` with a variable, which emits
`ASSIGN @this=$r0` ahead of the call (a non-static method gets its `@this`
symbol at jimple_method.cpp:32). Re-running M2 with that test present moves
1 / 19, and it is exactly the new test.

That is twice in two slices. **Treat an unmoved mutant as a question about the
corpus first and about the code second** — on this frontend the corpus has lost
that argument every time it has been asked.

### 21.2 M4: correct by construction, not by measurement

The block's `end_location` is nil rather than the statement location, because
`migrate_expr` reads `expr.end_location()` from a `code_blockt` that never had
one assigned (migrate.cpp:2375). The GOTO dump does not print it for a nested
block, so no mutant can move it and no test can pin it.

This is a genuinely different case from §20 and §21.1, and worth naming so it
is not mistaken for one: there, the arm was invisible because the corpus was
thin, and a test fixed it. Here it is invisible because the field is not
rendered at this position at all. The right response is to state the reasoning
and *not* claim the field as verified — the same nil-versus-empty distinction
that broke K.2 twice, where `end_location` was the visible one only because the
block terminated a function.

### 21.3 Status

Eleven slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865,
#6866, #6868, #6870. Corpus now 19 tests, all byte-identical, all
mutant-checked.

Remaining reachable: `SpecialInvoke` is already covered by this slice (it
constructs `jimple_invoke`); `Throw` (14) is a bare `codet("cpp-throw")` with
its body commented out, so migrating it would pin an unfinished construct.
`jimple_assignment` stays blocked on §16.2; `to_typet` -> `type2tc` remains
last (§7).

## 22. `jimple_assignment` (#6875): unblocked by one census, then by #6873

§16 blocked this slice on whether the two `c_typecast` copies agree.
`scope-coupled-arith-assign-conversion.md` §20 answered that: they did not, in
seven structural ways plus a constant-fold difference that reached every
frontend. #6873 fixed the fold. What remained was whether any of the seven can
arise here.

### 22.1 One census discharges all seven

`jimple_type::to_typet` produces exactly four shapes:

```cpp
case INT:     return int_type();
case BOOLEAN: return bool_type();
case _VOID:   return empty_typet();
default:      return pointer_typet(symbol->get_type());   // and arrays, as
                                                          // nested pointers
```

and `incomplete_array`, `cmt_constant`, `cmt_volatile` and `#reference` appear
**nowhere** in `src/jimple-frontend/`. So:

| Gap | Why it cannot arise |
|---|---|
| references (both directions) | needs `#reference`, never set |
| pointer-to-member | needs `to-member`, never set |
| `incomplete_array` source | jimple builds no array type at all |
| const/volatile warnings | no qualifier is ever set |
| `#reference` propagation | same |
| struct/union source to pointer | class types are already `pointer_typet(struct)`, never bare struct |
| string-constant to array | no array *destination* exists; `get_expression` also discards the string value |

This is a stronger argument than the one §20.1 made informally, and it is the
kind that generalises: the question "can this conversion arise?" is answered by
the frontend's *type constructor*, not by its statements.

### 22.2 The fold dependency, measured

Built against a base without #6873, **all 19** tests diverged, uniformly:

```
< ASSIGN $z0=1;
> ASSIGN $z0=(signed int)1;
```

With #6873 merged in, all 19 are byte-identical. That is a clean confirmation of
§20.2 from the other direction: the divergence was not hypothetical, and it hit
every single test the moment an assignment moved to the native path.

### 22.3 The first genuinely dead branch

§20 and §21 both turned an unmoved mutant into a test. Removing the `is_skip`
arm changed 0/19 — and this time the corpus is not the reason:

```cpp
bool is_skip = false;   // jimple_statement.h:164
```

It is assigned **nowhere** in the tree. `from_json` sets `lhs` and `rhs` only,
there is no setter, and no other file mentions it. The arm is unreachable by
construction, not by corpus, and no test could ever make it live.

So the override does not mirror it. Reproducing a provably-unreachable branch in
new code is dead instrumentation, which the C-Live obligation forbids; the
legacy arm in `to_exprt` is a dead-code candidate for its own PR, already marked
`//TODO: Remove this hack`.

**The rule from §21 survives with its exception now stated:** an unmoved mutant
is a question about the corpus first — but when the guard is a member that
nothing in the tree ever assigns, the answer really is dead code, and that is
provable statically rather than by adding a test.

### 22.4 Status

Twelve slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865,
#6866, #6868, #6870, #6875, plus #6873 in support. Every reachable statement
kind is now native.

Remaining: `jimple_throw` (14 occurrences, but `to_exprt` is a bare
`codet("cpp-throw")` with its body commented out); the three overrides §19 shows
are unreachable; and `to_typet` -> `type2tc`, which §7 keeps for last and which
§22.1 has now mapped in full.

## 23. `to_type2t` and the cast (#6877): introduce a helper with its consumer

§7 keeps `to_typet` -> `type2tc` for last. Taking it revealed that "migrate the
type helper" is not a slice on its own: every `to_typet` call site sits inside a
`to_exprt` that has not moved, so a `to_type2t` added by itself would have had
no caller -- the same dead instrumentation §22.3 refused for `is_skip`.

So it ships with its first consumer. An expression census picks that consumer:

| Count | Expression | State |
|---|---|---|
| 500 | symbol | #6865 |
| 183 | constant | #6858 |
| 110 | binop | #6860 |
| 47 | array_index | open |
| **33** | **cast** | **#6877, uses to_typet** |
| 21 | string_constant | maps to jimple_constant |
| 19 | static_invoke | open |
| 15 | static_member | open, uses to_typet |
| 14 | new | open, uses to_typet |
| 9 | newarray | open, uses to_typet |

### 23.1 A second provably-dead arm

`get_base_type` switches on `BASE_TYPES`, and `BASE_TYPES::BOOLEAN` has a case.
Nothing produces it: `from_map` maps `"boolean"` to `BASE_TYPES::INT`, and a
grep for `BASE_TYPES::BOOLEAN` finds only the two switch arms themselves, never
a mapping. So the mirror omits it, on the §22.3 rule.

Worth noting how the corpus census misleads here. All 33 casts take the INT arm,
including the 17 to `java.lang.String[]` -- because `java.lang.String` is mapped
to `BASE_TYPES::INT` too, with a `// TODO: handle this properly`. A mutant on
the pointer arm therefore moves nothing, and it would have been easy to call
that arm dead as well. It is not: it is reachable for any class name absent from
`from_map`, and will be exercised as soon as `new` or `newarray` migrates. Only
`BOOLEAN` is unreachable *by construction*.

### 23.2 A cast that nothing can observe

Dropping the cast's own conversion changed **0/19**. The first test written to
fix that -- a cast on an assignment's right-hand side -- changed **0/20**, for a
reason worth recording: `jimple_assignment` re-converts its source to the
target's type, so an enclosing assignment subsumes the cast entirely. The GOTO
is identical whether the cast converts or not.

`github_4715_cast_conversion_01` puts the cast in an **invoke argument**
instead, where `jimple_invoke` binds `@parameterN` with no typecast of its own:

```
ASSIGN @parameter0=(signed int *)$i0;
FUNCTION_CALL:  sink_1((signed int *)$i0)
```

With the conversion dropped, that test and only that test changes.

**Third addition to the §21 rule.** An unmoved mutant now has three causes, and
they need different responses: the corpus is thin (§20, §21 -- write a test);
the code is unreachable by construction (§22.3, §23.1 -- do not mirror it); or
**a caller downstream re-does the work, so the output cannot distinguish**
(here). The third is the subtlest, because the obvious test still shows nothing
-- the fix is to find a position where the redundancy does not apply, not to
write a bigger example of the same shape.

### 23.3 Status

Thirteen slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860,
#6865, #6866, #6868, #6870, #6875, #6877, plus #6873 in support. Corpus now 20
tests.

Next by value is `array_index` (47, `jimple_deref`), then `static_member` (15),
`new` (14) and `newarray` (9) -- the last three all consume `to_type2t`, which
now exists, and between them will exercise its pointer arm.

## 24. `jimple_deref` (#6880): the oracle's own blind spot

`array_index` was the largest remaining expression at 47 occurrences.
`to_exprt` builds it by assembling an `index_exprt` and then rewriting the node
in place:

```cpp
exprt &array_expr = index.op0();          // reference into the operand vector
...
addition.operands().swap(index.operands()); // vector now empty; the reference dangles
index.move_to_operands(addition);
index.type() = array_expr.type().subtype(); // reads the reference again
```

By the last line `array_expr` no longer designates the array -- the vector it
pointed into was swapped away and refilled, so it now names the *addition*.
The subtype happens to be the same either way, which is why this works. The
native override builds the intended result, `dereference(base + index)`,
without the rewrite.

### 24.1 A fourth cause for an unmoved mutant

| Mutant | Changed |
|---|---|
| index dropped | 7 / 20 |
| dereference dropped | 7 / 20 |
| result widened to the pointer type | 7 / 20 |
| **addition's operands swapped** | **0 / 20** |

Seven is every test that indexes an array, so the override is well covered. The
swap is different from all three earlier zeroes: the code is reachable, the
corpus is adequate, and nothing downstream re-does the work. The **oracle**
cannot see it. The GOTO printer renders `dereference(p + i)` in index notation:

```
ASSIGN r0[i2]=$i1;
```

so operand order is normalised away before the dump is written.

Nothing is at risk here -- pointer arithmetic identifies the pointer by type
rather than position, so both orders are semantically identical, and the
committed order matches `to_exprt` anyway. What matters is the general point:

**A/B byte-identity is an oracle over the *printed* GOTO, not over the IR.**
Any field the printer normalises or omits is outside its reach. Two such fields
are now known -- `end_location` on a nested block (§21.2) and commutative
operand order under index-notation printing (here) -- and for those the argument
has to be made from the migration source, not from the measurement.

So the four causes of an unmoved mutant, with their distinct responses:

| Cause | Response | Seen at |
|---|---|---|
| corpus is thin | write a test | §20, §21.1 |
| unreachable by construction | do not mirror the branch | §22.3, §23.1 |
| a caller downstream re-does the work | test in a position where it does not | §23.2 |
| the printer normalises the field away | argue from the source; do not claim it measured | §21.2, §24.1 |

### 24.2 Status

Fourteen slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860,
#6865, #6866, #6868, #6870, #6875, #6877, #6880, plus #6873 in support.

Remaining expressions: `static_member` (15), `new` (14), `newarray` (9) -- all
three consume `to_type2t` and between them will exercise the pointer arm §23.1
showed the cast corpus cannot reach -- and the expression form of
`static_invoke` (19). Statements are complete bar `jimple_throw`, which is an
unfinished construct.

## 25. `jimple_static_member` (#6882), and a correction to §24.2

### 25.1 The prediction that was wrong

§24.2 said `new` and `newarray` would exercise the pointer arm of `to_type2t`
that §23.1 showed the cast corpus cannot reach. A census says otherwise:

| Count | Shape |
|---|---|
| 14 | `new java.lang.AssertionError` |
| 5 | `newarray int` dims 0 |
| 2 | `newarray java.lang.Integer` dims 1 |
| 2 | `newarray java.lang.Integer` dims 0 |

`java.lang.AssertionError` and `java.lang.Integer` are both mapped to
`BASE_TYPES::INT` in `from_map`, alongside `java.lang.String`, `Main`,
`java.lang.Runtime` and `java.lang.Class` -- each with a `// TODO: handle this
properly`. So every one of these takes the INT arm too.

**The pointer arm of `get_base_type2` is reachable from no expression in the
corpus at all.** It fires only for a class name absent from `from_map`, and the
corpus has none. Any slice that wants it live has to bring its own test -- the
§20 response -- and until then the arm should not be claimed as verified.

### 25.2 Three arms, one reachable

All 15 `static_member` uses are `kotlin._Assertions.ENABLED`. The
`Main.$assertionsDisabled` arm and the member access proper are both unreached.
They are not equivalent cases, and the slice treats them differently:

- `Main.$assertionsDisabled` is trivially constructible, so
  `github_4715_static_member_intrinsic_01` constructs it. Flipping that arm now
  moves exactly one test.
- The member access is marked `// TODO: Needs OOP members` and rewrites a
  `member_exprt`'s base in place through a reference, much as §24 described.
  Reimplementing it with no test would be a guess. It stays on the migrating
  default.

This is the partial-override technique from §15.2 and §22 doing what it was
built for: the arm that can be verified moves, the arm that cannot stays on the
path that already works. A slice does not have to be all-or-nothing.

### 25.3 Discarded work in the legacy arms

`to_exprt` opens with `gen_zero(type->to_typet(ctx))` and then, on both
intrinsic arms, throws it away -- `make_true` is `*this = exprt(constant,
typet("bool"))`, a whole-node replacement. So the type computation, including a
symbol-table lookup when the type is a class, runs for nothing on 15 of the 15
corpus uses. The native arms return the constant directly.

Not a bug, and not worth a separate PR on its own, but it is the third instance
in this frontend of a node being built and then overwritten (§24's
`index_exprt`, §22's `code_returnt` operand, this). The pattern is worth
watching for in the remaining overrides.

### 25.4 Status

Fifteen slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865,
#6866, #6868, #6870, #6875, #6877, #6880, #6882, plus #6873 in support. Corpus
now 21 tests.

Remaining: `new` (14) and `newarray` (9) -- `newarray` is the most intricate
override left, allocating through a temp symbol and a synthesised call, with a
hardcoded 64-bit width fallback; the expression form of `static_invoke` (19);
`jimple_virtual_member`; and `jimple_throw`, still unfinished. Per §25.1 none of
these will reach the `to_type2t` pointer arm without a new test.

## 26. Allocation (#6884), and a mutant that tested the wrong copy

`jimple_new` derives from `jimple_newarray` and overrides only `from_json`
(setting `size` to the constant 1), so one override covers both -- 23 uses.

### 26.1 The fourth build-then-discard

`to_exprt` assembles a `code_function_callt`, sets its lhs to a fresh temp
symbol, and then never uses the call: it copies `function`, `arguments` and
`location` into a `side_effect_expr_function_callt` and returns that. The lhs,
and with it the temp symbol's only use, is dropped. `alloc_type` is computed
over two statements and never read at all.

That is the fourth instance in this frontend, after §22's `code_returnt`
operand, §24's `index_exprt` and §25.3's `gen_zero`. The native form uses the
existing `side_effect_function_call2tc` helper -- which already documents the
empty-not-nil alloctype trap -- and drops `alloc_type`.

### 26.2 An unobservable side effect that still has to be kept

Removing the temp symbol changes **0/22**. It is not dead, and it is not thin
corpus either:

```cpp
static symbolt get_temp_symbol(...)
{
  static unsigned int counter = 0;
  ... "return_value$tmp$" + std::to_string(counter++) ...
```

The counter is program-wide, so not calling it renames every later temp symbol.
The dump does not show that here only because these particular temps are
unused. This is §24.1's fourth cause again -- the oracle cannot see it -- but
with the opposite conclusion: there the invisible difference was harmless and
either choice was fine, here the invisible difference is real and the mirror
must be exact. **Oracle blindness cuts both ways; the source has to settle it.**

### 26.3 A mutant that silently tested the legacy copy

The first run of the width-fallback mutant reported 0/22 even against a test
whose dump plainly showed `MALLOC(signed char, 4 * 64)`. The cause was the
mutation itself:

```
525:  int type_width = 64;   <- to_exprt   (legacy)
573:  int type_width = 64;   <- to_expr2t  (native)
```

`str.replace(old, new, 1)` rewrote the **first** occurrence, so the mutant
perturbed the legacy path, which `--irep2-bodies` does not execute, and the
identical output was read as "the arm is unreachable."

This is a hazard specific to the parallel-method technique of §8.1: every
migrated override has a near-twin a few hundred lines away, and any
text-targeted mutant can hit the wrong one and return a false zero. Mutants
must be anchored to the native function -- slice the source at
`expr2tc <class>::to_expr2t` first, then assert the pattern occurs exactly once
in the tail:

```python
i = t.index("expr2tc jimple_newarray::to_expr2t")
head, tail = t[:i], t[i:]
assert tail.count(pattern) == 1
```

Re-run that way, the mutant moves 1/22 -- the new test. Earlier slices are not
affected: their mutant strings differed from the legacy text (`symbolt` versus
`auto`, `code_skip2tc` versus `code_skipt`), so they hit the native copy by
luck rather than by construction. From here they are anchored deliberately.

### 26.4 Status

Sixteen slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865,
#6866, #6868, #6870, #6875, #6877, #6880, #6882, #6884, plus #6873 in support.
Corpus now 22 tests, having grown by five written to make specific arms live
(§20.1, §21.1, §23.2, §25.2, §26.3).

Remaining: the expression form of `static_invoke` (19), `jimple_lengthof`,
`jimple_virtual_member`, `jimple_virtual_invoke`, and `jimple_throw`, still
unfinished. Per §25.1 none reaches the `to_type2t` pointer arm without a new
test.
