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

## 27. `jimple_expr_invoke` (#6885): remove-the-arm versus corrupt-the-arm

Only one of this class's five arms is reachable through `to_expr2t`, and
working out which took a chain of three facts rather than a census:

1. `jimple_assignment::to_code2t` (#6875) sends an invoke right-hand side to the
   migrating default *unless* it is nondet or intrinsic.
2. `is_nondet_call()` tests for `org.sosy_lab.sv_benchmarks.Verifier`, which
   never appears in the corpus.
3. `is_intrinsic_method` is set in exactly one place -- `java.lang.Integer` with
   `method == "valueOf_1"` -- and the corpus has 11 of those.

So `valueOf` is the arm that reaches `to_expr2t`; the Intrinsics and Runtime
skips and the main block path are reached only through the default, and the
main path returns a `code_blockt` -- a statement from an expression method --
which is why it belongs there anyway.

### 27.1 The mutant that proves nothing

Deleting the `valueOf` arm changes **0/22**, with the mutation correctly
anchored to the native copy per §26.3. The reason is not any of the four causes
in §24.1: the arm is reachable, the corpus covers it, the printer shows it, and
nothing downstream re-does it. What happens is that the *fallback itself* is
equivalent -- dropping to the migrating default reaches `to_exprt`'s own
`valueOf` arm, which returns the same operand.

That makes deletion the wrong probe. It measures whether the arm is
**necessary**, and a correct migration slice is never necessary: it produces
byte-identical output by construction, which is the whole premise of the A/B
gate. Corrupting the arm instead -- returning a constant where the argument
should go -- changes **2/22**, the two tests holding the 11 `valueOf` calls.

**Generalisation.** Two different questions have been conflated up to now
because for most slices one mutant answers both:

| Probe | Question | When it is the right one |
|---|---|---|
| delete the arm | is this arm *necessary*? | the fallback does something different |
| corrupt the arm | does this arm *execute*? | always |

Where an override shadows a fallback that already handles the case -- which the
partial-override technique produces by design (§15.2, §25.2, here) -- only the
corruption probe carries information. Earlier slices happened to use probes
that were corruptions in effect (§25's flipped constants, §26's altered width),
so their liveness claims stand; but §23.2's cast, where deletion *did* move a
test once the position was right, was a case where the fallback genuinely
differed.

### 27.2 Status

Seventeen slices shipped: #6851, #6853, #6854, #6855, #6856, #6858, #6860,
#6865, #6866, #6868, #6870, #6875, #6877, #6880, #6882, #6884, #6885, plus
#6873 in support.

Remaining: `jimple_lengthof`, `jimple_virtual_member`, `jimple_virtual_invoke`
(0 in the corpus, so §19's limit applies), and `jimple_throw`, still an
unfinished construct.

## 28. An audit of the slices that predate the mutant check (#6886)

Every expression kind the corpus reaches is now native. The census that
established this also turned up something about work already shipped:

| expr_type | Occurrences |
|---|---|
| symbol, constant, binop, array_index, cast, string_constant, static_invoke, static_member, new, newarray | migrated, all non-zero |
| **lengthof, local_member, virtual_invoke, nondet, class_reference** | **0** |

`nondet` is in the zero list -- and #6858 migrated `jimple_nondet`, reporting
"GOTO output is byte-identical across all 17 jimple tests."

### 28.1 The claim was true and worthless

The mutant check was introduced at §17.1, for #6865. Seven slices shipped before
it: #6851, #6853, #6854, #6855, #6856, #6858, #6860. Each reported byte
identity; none reported liveness.

For `jimple_nondet` that gap was real. Corrupting the override with the corpus
as it stood changes nothing, because `expr_type: "nondet"` never appears,
`is_nondet_call()` tests for `org.sosy_lab.sv_benchmarks.Verifier` which never
appears, and `java.util.Random` reaches only `jimple_virtual_invoke`, itself at
zero. Three routes into the class, none of them taken.

`github_4715_nondet_expr_01` supplies the direct one. Corrupting the override
now changes exactly that test and nothing else -- one measurement establishing
both that the new test covers the override and that the previous 22 did not.

### 28.2 The rest of the pre-check slices are fine

Not a general retraction. The other six operate on constructs with substantial
counts -- goto 28, label 81, if 72, assignment 226, constant 183, binop 110 --
and the method-body seam runs for every test in the suite. `jimple_nondet` was
the only one of the seven at zero, so the audit closes with one fix rather than
seven.

### 28.3 What this says about ordering

The mutant check was added when a slice happened to need it, and the six slices
already shipped were never revisited. The cost of that was one unverified
override surviving nine PRs. The general lesson for the remaining phases is
cheap to state: **when a verification step is added mid-campaign, re-run it over
what already shipped rather than only applying it forward.** A census of what
the corpus contains, done once at the start, would have flagged `jimple_nondet`
before it was written rather than nine slices later.

### 28.4 Status

Eighteen PRs: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865, #6866,
#6868, #6870, #6875, #6877, #6880, #6882, #6884, #6885, #6886, plus #6873 in
support. Corpus 23 tests, six of them written to make a specific arm live.

Remaining are the four other zero-count kinds -- `lengthof`, `local_member`,
`virtual_invoke`, `class_reference` -- plus `jimple_throw`. Each needs a test
written *before* its override, on the §28.3 lesson, and `jimple_throw` needs the
construct finished first.

## 29. `jimple_lengthof` (#6887): the §28.3 lesson applied

First slice taken test-first. `expr_type: "lengthof"` is one of the five
zero-count kinds §28 identified, so the order mattered: written the other way
round, the A/B gate would have passed on an override nothing executes, which is
precisely what #6886 had to go back and fix.

`github_4715_lengthof_01` allocates an array and reads its length. It was
written, run, and shown to reach the construct -- the GOTO carries the
`__ESBMC_get_object_size` call -- **before** the override existed. Only then was
`to_expr2t` added, and the corruption probe (§27.1) confirmed 1/24.

The override itself is unremarkable, which is the point: it is the fifth
instance of the build-then-discard shape (§22, §24, §25.3, §26.1), and
`side_effect_function_call2tc` already existed from #6884, so the migration was
a two-line mirror. **The test was the work.**

### 29.1 Cost comparison

| Order | Steps |
|---|---|
| override first (#6858 -> #6886) | write override, A/B passes vacuously, ship, census months later, discover the zero, write test, ship again -- two PRs, one false claim in between |
| test first (#6887) | write test, confirm it reaches, write override, A/B and mutant both meaningful -- one PR |

Nothing about the second order is harder. It only requires knowing the corpus
count first, which a single census gives for every construct at once.

### 29.2 Status

Nineteen PRs, corpus 24 tests -- seven of them written to make a specific
construct or arm live (§20.1, §21.1, §23.2, §25.2, §26.3, §28.1, §29).

Remaining zero-count kinds: `local_member` (`jimple_virtual_member`),
`virtual_invoke` (`jimple_virtual_invoke`), `class_reference`. The first two
have real overrides worth migrating and both need a test written first;
`class_reference` maps to `jimple_constant("-1")` in `get_expression` and has no
override of its own, so there is nothing to migrate for it. `jimple_throw`
remains an unfinished construct.

## 30. `jimple_virtual_member` (#6889), and a fifth cause

Test-first again. `local_member` is a zero-count kind, and
`github_4715_local_member_01` declares a class with a field, allocates an
instance and reads the member. `to_exprt` builds a `gen_zero` and a
`struct_type` local and reads neither -- the sixth build-then-discard -- so both
are dropped; the dereference arm is kept, since a class-typed variable is always
a pointer.

### 30.1 The pointer arm, at last

§23.1 and §25.1 both recorded that `to_type2t`'s pointer arm was reachable from
nothing in the corpus, because `from_map` maps every class name that appears --
`String`, `Integer`, `AssertionError`, `Main`, `Runtime`, `Class` -- to
`BASE_TYPES::INT`. A user class not in that map does reach it, and this test
declares one.

Getting there took three corrections to my own expectations:

1. `new OriginalKt` does not reach it -- #6884's override calls the **legacy**
   `to_typet` to build the symbol, not `to_type2t`.
2. A class-typed member does not reach it either -- the member's own type here
   is `int`.
3. A cast to the class type on an assignment right-hand side does not show it,
   because the enclosing assignment re-converts to the target type (§23.2).

What works is a cast to the class type in **argument** position, the same
position §23.2 needed for the same reason.

### 30.2 An unmoved mutant because the mutation is invalid

Even with the arm reached, three mutants reported 0/25. All three replaced
`pointer_type2tc(migrate_symbol_type(*symbol))` with the bare struct type, and
that is the flaw:

| | source type | destination | result |
|---|---|---|---|
| unmutated | `pointer(struct)` | `pointer(struct)` | legal, identical, **no cast emitted** |
| mutated | `pointer(struct)` | `struct` | **illegal**, error recorded, **no cast emitted** |

`c_implicit_typecast` on a rejected conversion pushes to `errors` and leaves the
expression untouched, so refusal and success-with-nothing-to-do print the same.
Substituting a *legal but different* type -- `pointer_type2tc(get_bool_type())`
-- emits `(bool *)$r0` and moves 1/25.

**Fifth cause for an unmoved mutant**, and the first that is a flaw in the probe
rather than a fact about the code:

| Cause | Response | Seen at |
|---|---|---|
| corpus is thin | write a test | §20, §21.1 |
| unreachable by construction | do not mirror the branch | §22.3, §23.1 |
| a caller downstream re-does the work | test where it does not | §23.2, §30.1 |
| the printer normalises the field away | argue from source | §21.2, §24.1 |
| **the mutation makes the operation invalid, and the error path is also a no-op** | **mutate to a valid alternative** | **§30.2** |

The pattern to watch for: any mutation feeding a function that validates its
input can be swallowed by that validation. Prefer mutants that stay inside the
valid domain and change the answer, over mutants that leave it.

### 30.3 Status

Twenty PRs, corpus 25 tests, eight of them written to make a construct or arm
live. Every expression kind that `get_expression` can build is now migrated or
deliberately on the default, and `to_type2t` has no unexercised arm left bar
`BOOLEAN`, which §23.1 proved unreachable by construction.

Remaining: `jimple_virtual_invoke` (zero-count, needs its own test) and
`jimple_throw`, still unfinished.

## 31. `jimple_virtual_invoke` (#6891): the expression migration is complete

The last zero-count kind, taken test-first.
`github_4715_virtual_invoke_nondet_01` assigns from a `java.util.Random`
virtual invoke, which is what `is_nondet_call()` recognises.

That arm is the only one reachable through `to_expr2t`, for the same reason as
§27: `jimple_assignment` sends an invoke right-hand side to the migrating
default unless it is nondet, and the three skip arms and the main block path all
produce *statements* from an expression method. Deletion proves nothing here
(§27.1), so liveness is the corruption probe: 1/26, the new test.

### 31.1 Where the frontend now stands

Every expression kind `jimple_expr::get_expression` can construct is migrated
or deliberately left on the migrating default, and every statement kind the
dispatcher can construct is migrated. `to_type2t` has no unexercised arm bar
`BOOLEAN`, which nothing can produce (§23.1).

Deliberately on the default, each with a stated reason:

| Construct | Reason |
|---|---|
| `jimple_identity`, `jimple_assertion` | unconstructible -- no `from_map` entry / not in the dispatcher (§19) |
| invoke main paths (`jimple_invoke` expr form, `jimple_virtual_invoke`) | return a `code_blockt`; the statement forms are already native (§27, §31) |
| `jimple_static_member` member access | marked "Needs OOP members"; rewrites a base in place (§25.2) |
| `jimple_assignment` invoke arms | rewrite their own lhs and lower to a call (§22) |
| `jimple_throw` | body commented out upstream; migrating it would pin an unfinished construct |

### 31.2 What the corpus cost

The suite went from 17 tests to 26. Nine were written to make something
measurable, and only one of those (§20.1) was prompted by a hazard I had
predicted -- the rest came from a mutant refusing to move and the reason having
to be chased down. That ratio is the honest summary of the campaign: the
migrations were mechanical, and the verification was the work.

### 31.3 Status

Twenty-one PRs: #6851, #6853, #6854, #6855, #6856, #6858, #6860, #6865, #6866,
#6868, #6870, #6875, #6877, #6880, #6882, #6884, #6885, #6886, #6887, #6889,
#6891, plus #6873 in support. All byte-identical, all mutant-checked bar the
arms listed in §31.1.

Next is not another jimple slice: it is `to_typet` -> `type2tc` at the
*declaration* sites (`create_jimple_symbolt` still takes a `typet`, §23), which
is a different shape of change, or a return to the parent document's Phase list.

## 32. The declaration sites: `create_jimple_symbolt` takes a `type2tc`

§31.3 named this as the next step, and it is a different shape of change from
the twenty-one slices before it. Those migrated a *body*: one `to_exprt`
override gained a `to_expr2t` twin, and the gate was that the GOTO stayed
byte-identical. This one migrates a *signature*. `create_jimple_symbolt` and
`get_temp_symbol` (`jimple_ast.h`) now take a `const type2tc &`, all nine call
sites were adjusted, and the two other symbol builders in that header
(`get_allocation_function`, `get_lengthof_function`) spell their signatures in
IREP2 instead of assembling a `code_typet`.

### 32.1 What `symbolt` actually does, which is not what this slice first assumed

The first draft of this section claimed `symbolt` holds the IREP2 type as its
source of truth, so `set_type(typet)` migrated on the way in and each
declaration site was building a `typet` only to have it converted. That is
wrong, and the review that caught it is worth recording. `set_type(const typet
&)` writes the legacy field and *invalidates* the IREP2 one
(`src/util/symtab/symbol.cpp:36-47`); the forward migration happens lazily on
the first `get_type2()` (`symbol.cpp:97-107`). The header says why: the lazy
split avoids forward-migrating a `typet` whose sub-expressions may not survive
a recursive descent.

So the honest accounting of this slice is the opposite of a saving. Three sites
now build the type natively and never touch the legacy form; six migrate at the
boundary, and for those the migration is work the old code did not do — at two
of them it is provably wasted, because the symbol's type is overwritten
legacy-side a few lines later (`jimple_file.cpp:159`, `jimple_method.cpp:92`),
which discards what the boundary migration produced. What the slice buys is
that the seam's *signature* no longer accepts a legacy type, so the three
natively-built sites cannot regress to one, and §32.5's blocker becomes visible
rather than latent.

### 32.2 Three sites convert natively, six migrate at the boundary

Native, via `jimple_type::to_type2t(ctx)`: both `jimple_declaration` sites
(`to_exprt` and `to_code2t`). The equivalence is not obvious, because the two
converters do not take the same route for a class type: `to_typet`'s default arm
is `pointer_typet(symbol->get_type())` and `to_type2t`'s is
`pointer_type2tc(migrate_symbol_type(*symbol))`. The argument is the write order,
not a round-trip property: the class symbol's legacy side is the last written
(`jimple_file.cpp:159`), so `get_type2()` *is* `migrate_type(get_type())` by
construction. The only window where the IREP2 side is last-written runs from
`jimple_file.cpp:130` to `:159`, where the struct is empty in both forms.

An earlier draft argued this from `migrate_symbol_type`'s round-trip assertion
instead. That argument is void in the configuration the gate ran in: the
assertion is inside `#ifndef NDEBUG` (`migrate.cpp:466-476`) and this build is
RelWithDebInfo, `-DNDEBUG`. `strings` on the binary finds no "not stable under
IREP2" message. An assertion compiled out is not evidence.

`jimple_assertion::to_exprt` converts natively too, by construction rather than
by a converter. A default `code_typet` sets only `id(code)`, so its argument list
is empty, `has_ellipsis()` is false, and its return type is *absent* — the const
accessor is `find_type`, so the code arm migrates an id-less `typet`, which
`migrate_type` maps to the empty type (`migrate.cpp:385-388`). The IREP2
spelling is therefore `code_type2tc({}, get_empty_type(), {}, /*ellipsis=*/false)`.
Two reviews reached opposite conclusions about this, one of them reading the nil
id as falling through to `migrate_type`'s throw, so it is now pinned in
`unit/util/migrate.test.cpp` ("a default code_typet migrates to a void
signature") rather than argued. A trap found while writing that test: probing
with `code_typet().return_type().is_nil()` returns false, because on a non-const
object `return_type()` is `add_type` and *creates* the sub-irep it is being asked
about. Probe absence with `find("return_type").is_nil()`.

The remaining six keep `migrate_type` at the call:

| Site | Type it assembles |
|---|---|
| `jimple_file::to_exprt` | `struct_typet`, tagged with the class name, still empty here |
| `jimple_method::to_exprt`, the method symbol | `code_typet`, arguments appended as `code_typet::argumentt` |
| `jimple_method::to_exprt`, the `this` parameter | `int_type()`, with a standing TODO to make it the struct |
| `jimple_method::to_exprt`, each declared parameter | from `jimple_type::to_typet` |
| `jimple_newarray::to_exprt` and `::to_expr2t`, the temp symbol | `pointer_typet(base_type)` |

These are not the same job. The first two build a type incrementally, so
converting them means converting the assembly — and §32.5 shows the second one
cannot be converted at all yet. The `newarray` sites keep the legacy `base_type`
regardless, because the allocation arithmetic reads a width off it.

### 32.3 Four of the nine sites are unreachable, and the dump gate reaches two

`ctest -R jimple` is 26/26 and every test's `--goto-functions-only` output is
byte-identical to the pre-change baseline, excluding the timing lines and the
version banner. Capture stderr: `--goto-functions-only` prints there, so a
baseline captured from stdout is empty and comparing against it proves nothing.

That gate is much weaker than it looks, in two independent ways.

First, four sites never execute. `jimple_assertion` is constructed nowhere in
`src/`: no key in `jimple_full_method_body::from_map` yields it, and Kotlin
assertions arrive as `java.lang.AssertionError` invokes that are skipped
(`jimple_statement.cpp:411-414`). `jimple_declaration::to_exprt` and
`jimple_newarray::to_exprt` sit in the legacy `to_exprt` subtree that the live
dispatch no longer enters, each shadowed by its own `to_code2t`/`to_expr2t`
override. A per-site breakpoint count over all 26 tests measures them at zero,
and the live counts reconcile exactly with the count at the shared body, so the
zeros are real and not an instrumentation artefact.

Second, the dump does not print what this slice changes. Types appear in a GOTO
dump at `DECL`, inside `NONDET`/`MALLOC`, and in casts; function signatures and
parameter symbols' types do not appear at all. So the dump constrains the
`to_code2t` declaration site directly and the class struct transitively, and
says nothing about the method signature, the parameter symbols, or the temp
symbol.

### 32.4 The test that does pin it, and three mutations that prove it

The instrument that observes these types is `--symbol-table-only`, which renders
each symbol's type and is validated like any other output — the harness matches
line 4+ regexes against stdout and stderr concatenated.
`github_4715_symbol_table_types_01` is one Jimple class carrying a field, a
non-static method with a declared parameter, a local, a `newarray` and a
`lengthof`, and it pins nine rendered types: the class struct, the method
signature, `@this`, `@parameter0`, the local, the array local, the discarded
temp symbol, and both converted helper signatures.

Pin each type to *its own* symbol. The obvious spelling — the symbol's name,
then a lazy gap, then the type line — does not bite: the gap happily runs past a
wrong type into the next symbol's block and matches there. The table has a fixed
layout, so `^Symbol\.+: X\n(?:.*\n){3}Type\.+: Y$` is the form that holds.

Three mutations, each rebuilt and measured:

| Mutation | New test | 26 old tests | 26 GOTO dumps |
|---|---|---|---|
| `malloc`'s argument `uint_type2()` → `int_type2()` | **FAILED** | pass | identical |
| `@this` symbol's type wrapped in a pointer | **FAILED** | pass | identical |
| `newarray` temp symbol loses a pointer level | **FAILED** | pass | identical |

Every one of the three is invisible to the byte-identical dump comparison *and*
to the whole pre-existing corpus. That is the measured answer to whether this
slice needed a test: the dump gate was never watching the sites the slice
changed, and one test that reads the symbol table is worth more here than any
number of verdict assertions. No `VERIFICATION FAILED` counterpart is added,
because nothing about the change moves a verdict — the failing halves of
`github_4715_irep2_bodies_jimple_01` and `_legacy_body_throw_01` already pin the
counterexample side of this seam.

### 32.5 Why the two completion sites cannot follow, measured

`jimple_file.cpp:159` and `jimple_method.cpp:92` overwrite the symbol's type
legacy-side once the struct's components and the method's arguments are known.
Converting them is the obvious next step and it does not work yet.

`jimple_file.cpp:158` sets a legacy `width` attribute on the struct, and
`migrate_type_back` does not restore it: the struct arm rebuilds components,
tag, `packed` and `alignment` and nothing else (`migrate.cpp:3141-3168`). So
making the class symbol IREP2-authoritative makes its derived legacy type lose
`width`, and `jimple_newarray` reads exactly that — `std::stoi(base_type.
subtype().width().as_string())` at `jimple_expr.cpp:575` and `:622`. The reader
has to stop asking a legacy attribute for the size before the writer can move.
That, not the `code_typet` assembly, is the next slice.

### 32.6 What this did to the parent document's bars, and a caveat on B-2

`frontends-to-irep2.md` §1 sets four bars per frontend. Measured on this branch
against its parent commit:

| Bar | Before | After |
|---|---|---|
| B-1 legacy type mentions | 202 | 190 |
| B-2 non-IREP2 symbol-table writes | 10 | 8 |

B-2's command counts the *spelling* of the argument, not its type:
`symbol.set_type(t)` with `t` a `type2tc` is exactly what the bar asks for and
the grep still counts it, because the token `2tc` is at the declaration and not
at the call. Read it as an upper bound whose lines each need inspecting. Of the
eight that remain, two are false positives (`jimple_ast.h:69`, and
`jimple_method.cpp:93`, whose argument is a `code_block2t` built by `to_code2t`),
two are the completion sites §32.5 blocks, and four are in
`jimple-language.cpp`.

### 32.7 Status, and what was deliberately left

Twenty-two PRs. The expression and statement migrations are complete (§31.1);
the declaration seam now takes an IREP2 type, natively built at three of nine
call sites and migrated at the boundary at the other six.

Reviewed and deliberately not done here, each its own change:

- Deleting `jimple_assertion`. It is orphaned scaffolding rather than a
  reference arm, so deletion is right, but it is a removal with its own
  justification and a unit test referencing the class.
- Deleting the unreachable `BASE_TYPES::BOOLEAN` arm. `get_base_type` has it and
  `get_base_type2` documents its absence (§23.1); the asymmetry is a drift trap
  now that `get_base_type2` is the declaration sites' only path, but removing an
  enumerator touches `from_map`.
- `jimple_expr.cpp`'s write-only `alloc_type` local, whose `is_nil()` guard is a
  branch: deleting a branch carries a proof obligation this slice has no reason
  to discharge.
- `get_temp_symbol`'s base name: `name += counter` with an `unsigned int`
  appends a character with that code point, not the digits, and `id` uses the
  pre-increment value while `name` uses the post-increment one. Invisible only
  because the temp symbol is never referenced (§26). A one-line fix, and
  `--symbol-table-only` can now pin it.

## 40. B-1's other half: one dead arm, and why the rest are not (2026-09-15)

B-2 for jimple has been met since §39. B-1 was 190 legacy mentions. This measures how
much of that is dead code rather than work.

### 40.1 The body's legacy builder is unreachable

`jimple_method.cpp:93` calls `body->to_code2t(...)`, so the IREP2 path is the live one
and `jimple_full_method_body::to_exprt` should never run. Instrumented with an
unconditional `fprintf` and swept over all 27 jimple tests: **0 observations**, the suite
green, so the runs happened. Deleted, with its override declaration -- the base's
`jimple_method_field::to_exprt` has a default body, so the class stays concrete.
B-1 190 -> 186.

### 40.2 The nine statement arms are *not* dead, measured twice

The obvious next step -- if the only entry is the body's `to_exprt`, the statements'
arms are unreachable too -- is wrong, and the suite says so twice:

- deleting all nine statement `to_exprt` overrides: **7 of 27 fail**, including
  `github_4715_irep2_bodies_jimple_01_fail` and `github_4715_legacy_body_throw_01_fail`,
  the two tests that exist to pin exactly this;
- deleting only the six whose class also declares a native `to_code2t`: **5 of 27
  fail**, the same two plus `kt-func-call-true` and `kt-func-call2-true`.

So a native `to_code2t` does not imply its class's `to_exprt` is unused: the arms are
reached from the statement builders themselves (`jimple_statement.cpp:197`, `:432`,
`:444` and others), where one statement's construction calls another node's legacy form.
Retiring them means converting those call sites first, one at a time, the way §32-§38
converted the expression arms -- not deleting the arm and seeing what breaks.

B-1 for jimple is therefore 186 of which the statements' share is real work, not dead
code. That is worth knowing before anyone reads the count as slack.

### 40.3 Which arms, measured instead of inferred

§40.2 drew a conclusion from two deletion failures. Instrumenting all nine arms with an
`fprintf` naming the class and sweeping the 27 tests replaces it with the answer:

```
14 jimple_throw
 8 jimple_assignment
```

**Two** of the nine are reached; the other seven never run. Both earlier attempts failed
because each deleted one of those two -- attempt 1 both, attempt 2 `jimple_assignment`,
which has a native `to_code2t` and is reached through its legacy arm anyway. That is the
fact neither inference could supply: a native IREP2 arm existing does not mean the legacy
one is unused.

The seven are deleted: `jimple_identity`, `jimple_invoke`, `jimple_return`,
`jimple_label`, `jimple_goto`, `jimple_if`, `jimple_assertion`. 27 of 27 jimple tests and
876 of 876 unit tests pass, and B-1 goes 186 -> **160**, a sixth of the phase's remaining
count removed as dead code.

What is left needs `jimple_throw` and `jimple_assignment` converted at their call sites
first, which is the §32-§38 shape and the next slice.

## 41. `jimple_throw` on a native arm (2026-09-15)

The first of §40.3's two. `jimple_throw::to_exprt` built a bare `codet("cpp-throw")` --
no operand and no exception list, because throw is not implemented -- so the native form
is `code_cpp_throw2tc(expr2tc(), {}, loc)` and takes the location as a parameter rather
than having it stamped afterwards, per K.2.

27 of 27 jimple and 876 of 876 unit tests, B-1 160 -> **157**.

Pinned rather than assumed: replacing the arm's body with a skip fails 7 of 27,
`github_4715_legacy_body_throw_01_fail` among them, so the arm is load-bearing and the
14 observations §40.3 measured are what exercises it.

`jimple_assignment` is the remaining one, and it is the harder half: it already has a
native `to_code2t` and its legacy arm is reached anyway, so the conversion is at the call
site rather than in the class.


## 42. The expression arms, probed the same way (2026-09-15)

§40.3's probe worked on the statements, so the same instrument was pointed at the twelve
`jimple_expr` legacy arms. Over the 27 tests:

```
10 jimple_symbol
 8 jimple_expr_invoke
 4 jimple_constant
```

**Three** of the twelve are reached. The nine that are not -- `jimple_binop`,
`jimple_cast`, `jimple_lengthof`, `jimple_virtual_invoke`, `jimple_newarray`,
`jimple_deref`, `jimple_nondet`, `jimple_static_member`, `jimple_virtual_member` -- are
deleted. 27 of 27 jimple and 876 of 876 unit tests, and B-1 157 -> **110**.

That is 310 lines of unreachable code, and §32-§38 had already converted each of those
nine to a native `to_expr2t`; what was left behind was the arm the conversion superseded.
Deleting them was never risky -- it only needed the measurement to say which.

### 42.1 What the three live arms need

| arm | why it is still reached |
|---|---|
| `jimple_symbol` | no native `to_expr2t`; the default migrates it |
| `jimple_constant` | same |
| `jimple_expr_invoke` | has a native arm, but it handles only the `valueOf_1` intrinsic and falls back for a real call |

The first two are ordinary conversions. The third is the chain §41 flagged:
`jimple_assignment::to_code2t` routes an invoke right-hand side to the migrating default
because `jimple_expr_invoke`'s native arm cannot build the call, and the invoke's `lhs` is
a legacy `exprt` set by `set_lhs`. Converting it means giving both invoke classes an IREP2
`lhs` and a native arm that builds `code_function_call2t` -- three classes, in that
order, and the only remaining B-1 work in this frontend that is not a one-liner.
## 33. `jimple_newarray::to_expr2t` goes native, and two defects it exposes

§32.5 named the width reader as the blocker on making the class symbol
IREP2-authoritative, so this slice takes it. `to_expr2t` now builds no legacy
type at all: the element type comes from `jimple_type::to_type2t`, the size type
from `uint_type2()`, and the callee's return type from
`to_code_type(alloca_symbol.get_type2()).ret_type` instead of a round trip
through the symbol's derived legacy signature.

The width itself comes from `type2t::get_width()`. That is sound here without
any new arithmetic: `struct_type2t::get_width()` sums its members' widths
(`irep2_type.cpp:199-209`), which is exactly what `jimple_file.cpp:158`
accumulates into the legacy `width` attribute the old code read. The IREP2 form
already carried the number.

### 33.1 The campaign's gate was never in the repository

No `test.desc` under `regression/jimple` had ever passed
`--goto-functions-only`. Twenty-two slices were gated on a byte-identical GOTO
comparison run out of a scratch directory, which is exactly the kind of claim the
PR conventions ask not to rely on: nobody else can reproduce it, and it vanishes
with the shell that produced it.

`github_4715_newarray_alloc_size_01` puts one construct's worth of it in the
repository, pinning both arms of the element-width choice from the dump itself:
`MALLOC(signed char, 2 * 32)` for an `int[]` and `MALLOC(signed char, 3 * 64)`
for a row of an `int[][]`. Two mutations, each rebuilt and measured, fail it and
nothing else:

| Mutation | New test | 27 other jimple tests |
|---|---|---|
| element width halved | FAILED | pass |
| pointer-row width 64 → 32 | FAILED | pass |

A trap on the way: the first version of the regex ended at `\)` and the dump
line ends `);`, so it could not match on *any* tree. It "failed under mutation"
and would have failed identically without one. A mutation check only means
something once the test is known to pass on the unmutated tree — run that
direction first.

### 33.2 Why halving the width changed no verdict: the allocation is 8x too big

The first mutation tried was the element width halved, and all 27 tests passed.
The reason is a pre-existing defect. `jimple_newarray` multiplies the element
count by the width **in bits** and hands that to `malloc`, whose argument is
bytes — the legacy arm even carries the comment `// we want bytes` next to the
bit width (`jimple_expr.cpp:577`). `new int[20]` allocates
`MALLOC(signed char, 20 * 32)`, 640 bytes for 80 bytes of array. The heap bounds
claim (`heap-array-bounds-violated`) is generated and passes, so nothing is
unsound; the object is simply 8x oversized, which is why the width can be halved
and even quartered without any access going out of bounds.

Not fixed here: it changes every allocation size in the frontend, so it needs
its own change, its own pair, and §33.3 settled first.

### 33.3 `lengthof` returns bytes, not elements

`jimple_lengthof` lowers to `__ESBMC_get_object_size`, which answers in bytes.
With §33.2's inflation, `new int[5]` followed by `arr.length` yields 160 where
Java and Kotlin both specify 5. `github_4715_lengthof_01` asserts only
`^VERIFICATION SUCCESSFUL$` and never reads the value, so it passes without
observing any of this.

The two defects interact, which is why neither should be fixed alone: correcting
the allocation to bytes alone would make `lengthof` answer 20 instead of 160,
still not 5. The lowering wants `get_object_size(p)` divided by the element
size, and a test that asserts the count rather than a verdict.

### 33.4 Status

Twenty-three PRs. `jimple_newarray::to_expr2t` builds no legacy type; the
remaining legacy surface is §32.5's two completion sites (still blocked on the
*legacy* `to_exprt` arm reading the `width` attribute, though that arm is
measured unreachable), `jimple_throw`, and the items in §32.7.

## 34. The symbol table's truth moves to IREP2

§32.5 recorded the blocker on the two completion sites and §33 removed half of
it. This slice removes the rest and takes both sites, in four measured steps.

1. `jimple_newarray::to_exprt` — the legacy arm, and the last reader of a class
   struct's legacy `width` attribute — takes the width off the IREP2 form the
   same way `to_expr2t` does.
2. `jimple_file.cpp` writes the completed class struct with
   `set_type(migrate_type(t))`, so the class symbol's IREP2 side is the one last
   written.
3. The `width` attribute and the `total_size` accumulation that fed it are
   removed: with step 1 done and step 2 storing IREP2, nothing reads it, and
   `migrate_type`'s struct arm never did.
4. `jimple_method.cpp` writes the completed method signature the same way.

### 34.1 The gate a slice like this needs, and the one it does not

A GOTO dump cannot see this change at all — it shows bodies, and §32.3 measured
that it prints no function signature and no parameter symbol's type. Nothing
about steps 2 and 4 is observable there, and indeed all 27 dumps are unchanged.

The instrument that can see it is the whole symbol table. Captured for all 28
tests before and after, `--symbol-table-only` output is byte-identical, which is
the claim this slice actually needs: after it, each symbol's legacy type is
*derived* through `migrate_type_back` rather than stored, and the question is
whether anything the pipeline renders differs. It does not, including the method
signatures — `migrate_type_back`'s code arm restores argument identifiers and the
ellipsis flag, and the argument `#base_name` it does not restore has no reader.

One gap in that instrument, found by probing for it: the rendered type does not
show a signature's ellipsis. Forcing `make_ellipsis()` unconditionally leaves
`signed int (signed int, signed int)` unchanged and passes all 28 tests. So the
symbol-table comparison covers argument and return types but not that flag; what
covers the flag is `migrate_type_back` restoring it, and
`unit/util/migrate.test.cpp`'s round-trip case over `make_func_type()`.

The class struct that step 2 now writes IREP2-side *is* pinned: dropping a
component fails both `github_4715_symbol_table_types_01` and
`github_4715_local_member_01`.

### 34.2 Status

Twenty-four PRs. Every symbol the jimple frontend creates now carries an IREP2
type, written IREP2-side. B-2 still counts 8 lines, and four of those are now
false positives — `jimple_ast.h:69`, `jimple_file.cpp:158`,
`jimple_method.cpp:92` and `:93` all pass an IREP2 argument, the first written as
a bare `t` and the rest through `migrate_type`/`to_code2t`, none of which spells
`2tc` on the call. B-1 is 189, from 202 before §32.

Remaining: `jimple-language.cpp`'s four `set_type`/`set_value` calls (the module
and `__ESBMC_main` symbols), `jimple_throw` (§31.1), §33.2's 8x over-allocation
and §33.3's `lengthof`, and the four items in §32.7.

## 35. B-2 is met: `jimple-language.cpp`, and a grep that cannot say so

§34.2 left four legacy symbol writes, all in `jimple-language.cpp`: the four
intrinsic globals `add_global_static_variable` creates, and `__ESBMC_main`'s type
and value. All four are converted here, in three measured steps -- the globals,
then `__ESBMC_main`'s type, then its value -- each gated on both the GOTO dumps
and the full symbol table over all 28 tests, each 0 of 28.

The globals build their type natively: `array_type2tc(get_bool_type(), expr2tc(),
true)` is the infinite array `migrate_type` produced from
`array_typet(bool_type(), exprt("infinity"))`, and `irep2_utils.h`'s
`gen_zero(const type2tc &, bool)` mirrors the legacy overload arm for arm --
`array_as_array_of` yields `constant_array_of2tc`, exactly what an
`array_of_exprt` migrates to.

### 35.1 A marker with no reader, checked rather than assumed

The legacy value carried `#zero_initializer`, and no IREP2 node models it, so the
conversion drops it. Markers dropped at this seam have bitten this campaign
before, so it was checked rather than assumed: the only readers of the attribute
in the tree are `solidity_convert_constructor.cpp:503` and `:516`, in Solidity's
own converter, which never sees a jimple symbol. Everything else only ever writes
it.

`__ESBMC_main`'s value was the one step expected to be awkward, because
`setup_main` resizes the call's arguments with *nil* ireps before migrating. It
is not: the symbol table has always migrated that value lazily on the first
`get_value2()`, so doing it eagerly reaches the same code.

### 35.2 B-2 is met, and its command reports 7

Every symbol-table write in the jimple frontend now carries an IREP2 argument,
which is what bar B-2 in `frontends-to-irep2.md` §1 asks for. Its command still
prints 7 lines, and all 7 are false positives:

| Line | Argument |
|---|---|
| `jimple_ast.h:69` | the `type2tc` parameter, as a bare `t` |
| `jimple_file.cpp:158`, `jimple_method.cpp:92` | `migrate_type(...)` |
| `jimple_method.cpp:93` | `to_code2t(...)`, a `code_block2t` |
| `jimple-language.cpp:99` | a `type2tc` local |
| `jimple-language.cpp:110` | `gen_zero(const type2tc &, bool)` |
| `jimple-language.cpp:198` | an `expr2tc` filled by `migrate_expr` |

A bar whose command cannot distinguish a met state from an unmet one is not a
bar. Either it needs the argument's type rather than its spelling -- which a grep
cannot get -- or B-2 should be restated as "no `set_type`/`set_value` call whose
argument is a `typet`/`exprt`", verified by inspection and recorded per frontend.
Jimple is the first frontend to reach it either way.

### 35.3 Status

Twenty-five PRs. B-1 is 183, from 202 at the start of §32. Remaining in this
frontend: `jimple_throw` (§31.1), the four items in §32.7, and §33.2/§33.3's two
defects -- none of which is a symbol-table write.

## 36. Retiring the dead legacy arms, and why most of them are not dead

B-2 is met (§35), and B-1 sits at 183 mentions. Three quarters of those are in
the legacy `to_exprt` overrides and their declarations, so the question this
slice answers is which of the 26 overrides the pipeline can still reach.

A reachability census answers the first half. Instrumenting every override with a
one-line print and running all 28 tests (breakpoints slide on inlined code; a
`fprintf` does not) gives 8 reached and 18 not:

| Reached | Hits | Not reached |
|---|---|---|
| `jimple_method` | 64 | `jimple_full_method_body`, `jimple_declaration` |
| `jimple_file` | 28 | `jimple_return`, `jimple_label`, `jimple_goto`, `jimple_if`, `jimple_invoke` |
| `jimple_throw` | 14 | `jimple_identity`, `jimple_assertion` |
| `jimple_symbol` | 10 | `jimple_binop`, `jimple_cast`, `jimple_lengthof` |
| `jimple_expr_invoke` | 8 | `jimple_virtual_invoke`, `jimple_newarray`, `jimple_deref` |
| `jimple_assignment` | 8 | `jimple_nondet`, `jimple_static_member` |
| `jimple_constant` | 4 | `jimple_virtual_member` |
| `jimple_class_field` | 2 | |

### 36.1 Not reached is not dead, and for the expression arms it is not even close

Eleven of the 18 are *expression* kinds, and they are reachable — the corpus
simply has no test that reaches them. `jimple_expr_invoke::to_exprt` converts
each of its parameters with `parameters[i]->to_exprt`, and
`jimple_assignment::to_exprt` converts its right-hand side the same way; both are
live (8 hits each, entered through the migrating default). An invoke parameter or
an assignment right-hand side can be any expression kind, so every expression
arm is one test away from being exercised. Deleting them would be deleting live
code on the evidence of an incomplete corpus.

That is the difference this section exists to record: a zero hit count is a
statement about the corpus. It becomes a statement about the program only with a
caller argument on top.

### 36.2 The statement arms do have that argument

Nothing calls `to_exprt` on a `jimple_method_body`. `jimple_method.cpp` calls
`to_code2t`, whose only other implementation is the base default, and that
default is reached solely by `jimple_empty_method_body`, which overrides neither.
So `jimple_full_method_body::to_exprt` is callerless — and it is the only caller
of a statement's `to_exprt` other than `jimple_method_field::to_code2t`'s
default, which a kind reaches only if it does not override `to_code2t`.

Three kinds do not override it, so their `to_exprt` stays: `jimple_identity` and
`jimple_assertion` (both unconstructible, §19) and `jimple_throw`, whose 14 hits
are real. `jimple_assignment` overrides `to_code2t` but its invoke arms delegate
to the default (§22), which is why it is reached.

That leaves seven provably callerless overrides, removed here:
`jimple_full_method_body`, `jimple_return`, `jimple_label`, `jimple_goto`,
`jimple_if`, `jimple_invoke` and `jimple_declaration` -- 234 lines, and B-1 from
183 to 160.

### 36.3 What discharges the removal

The obligation on removing a branch is that it was unreachable before. Here the
enclosing *function* has no caller, which is a compile-time fact rather than a
path condition, so a reachability query inside it would be answering a question
that cannot arise: there is no execution that reaches the function to reach a
branch within it. What discharges the removal is the caller argument in §36.2,
the census confirming 0 hits over 28 tests, and one property of the deletion
itself -- a statement that silently fell through to `jimple_method_field`'s base
`to_exprt` would become a `code_skipt` and change the GOTO. All 28 GOTO dumps and
all 28 symbol tables are byte-identical, so nothing did.

### 36.4 Status

Twenty-six PRs. B-1 is 160, from 202 at the start of §32; B-2 is met. §32 had
converted `jimple_declaration::to_exprt`, one of the sites deleted here — the
seam's signature was the point of that slice and this one does not undo it, but
the conversion at that particular site was incidental and is now gone.

### 36.5 What keeps the eleven expression arms alive, and how to kill it

B-1 cannot approach zero while the expression arms remain, and tracing why they
remain gives one concrete obstacle rather than the five separate design questions
§31.1 implies.

Exactly two live `to_exprt` bodies call an expression's `to_exprt`:
`jimple_assignment::to_exprt` (its left-hand side and right-hand side) and
`jimple_expr_invoke::to_exprt` (each parameter). Both are entered the same way:
`jimple_assignment::to_code2t` delegates to `jimple_method_field::to_code2t`'s
migrating default whenever the right-hand side is a non-nondet, non-intrinsic
invoke. `jimple_throw::to_exprt` is live too but is a dead end -- its operand
conversion is commented out upstream, so it reaches nothing.

The reason that delegation is still there is a single pattern:
`jimple_assignment::to_exprt` converts its left-hand side, calls `set_lhs` on the
right-hand side *invoke object* with the resulting legacy `exprt`, and then asks
that object to convert itself -- so the invoke lowers to a call with an already
built left-hand side rather than to an assignment. Porting it needs three things,
all mechanical:

1. `jimple_expr_invoke` and `jimple_virtual_invoke` to carry their injected
   left-hand side as an `expr2tc`.
2. Their `to_expr2t` to build the remaining arms: two `code_skip2t`s for the
   `Intrinsics` and `Runtime` skips, the nondet arm, and the main path's
   `code_block2t` of `@parameterN` assignments followed by the call.
3. `jimple_assignment::to_code2t` to stop delegating, setting the IREP2
   left-hand side and calling `to_expr2t`.

The payoff is the whole of `jimple_expr.cpp`: with those two callers gone, every
expression `to_exprt` becomes callerless -- the eleven untested arms plus
`jimple_symbol` and `jimple_constant`, which are only reached through them.

The corpus cannot verify steps 1-3 on its own: §19's census found the
`Intrinsics`, `Runtime` and nondet arms unreachable from any test in it, so each
needs a Jimple source written to reach it, the same way nine tests in this suite
already were.

## 37. The invoke expression forms, and the expression subtree goes quiet

§36.5 named one obstacle: `jimple_assignment::to_code2t` delegated to the
migrating default whenever its right-hand side was a non-nondet, non-intrinsic
invoke, because `jimple_assignment::to_exprt` converts its left-hand side, injects
it into the right-hand side *invoke object* with `set_lhs`, and lets that object
lower itself to a call rather than to an assignment. Both invoke forms now carry
that injected left-hand side as an `expr2tc`, both `to_expr2t`s cover every arm,
and the delegation is gone.

The block both forms produce -- one assignment per bound argument into the
callee's own `@this`/`@parameterN` symbol, then the call -- is now built once, in
`jimple_expr::lower_invoke2t`. The two legacy twins differ only in that the
virtual form binds `@this` and skips one more base class, so the helper takes the
`this` variable as a parameter and each caller keeps its own skip list.

### 37.1 The location tri-state, and a convention nothing pinned

The first version diverged on 8 of 28 dumps, all of the same shape: an instruction
the legacy path rendered `// 10 no location` came out as `// 10 ` instead.

`migrate_expr` reads a legacy statement's absent `#location` through the *const*
accessor, i.e. as nil, and `goto_programt::output_instruction` prints a nil
location as "no location". A default-constructed `locationt` is empty but **not**
nil, so it prints blank. `goto_convert_functions.cpp`'s `emitted_location`
documents the same distinction from the other side, where the legacy path
materialises the empty one. Building these statements natively therefore means
passing an explicitly nil location, which is what `no_location()` is for.

No test in the repository pinned that convention: the 8 dumps that caught it are
compared out of a scratch directory (§33.1). `github_4715_invoke_intrinsic_skip_01`
pins it now, together with the four arms the corpus could not reach.

### 37.2 One test, five things, four mutations

§19 found the `Intrinsics`, `Runtime`, `java.lang.Class` and nondet arms
unreachable from the corpus, so converting them needed a source written to reach
them. One `SetVariable` per arm, plus a real static invoke so the block path is
in the same dump, and a single regex over the instruction sequence pins all of
it: that nothing is emitted between the constant assignment and the nondet (the
three skips), the nondet itself, and the binding assignment and call with their
nil locations.

| Mutation | This test | 28 other jimple tests |
|---|---|---|
| block path takes a default `locationt` | **FAILED** | pass |
| `Runtime` dropped from the expression form's skips | **FAILED** | pass |
| `java.lang.Class` dropped from the virtual form's skips | **FAILED** | pass |
| nondet arm removed from the expression form | **FAILED** | pass |

A note on the third: reverting the second mutation with a one-shot text
replacement patched the *wrong function*, because both `to_expr2t`s contain the
same `base_class == "java.lang.Runtime")` line and the expression form comes
first in the file. The gate caught it immediately -- `java.lang.Class:getName_1`
is not a symbol, so the virtual form fell through to the block path and aborted.
When two functions differ only in a list, anchor an edit by line rather than by a
shared string.

### 37.3 The expression subtree is now callerless

Re-running §36's census: 4 of the remaining 19 `to_exprt` overrides are reached,
down from 8. `jimple_expr_invoke`, `jimple_assignment`, `jimple_symbol` and
`jimple_constant` are all at zero, and with them the eleven arms §36.1 had to keep
because those two could carry any expression.

What is left live, and what it reaches:

| Arm | Reaches |
|---|---|
| `jimple_method` | `body->to_code2t`, which is native |
| `jimple_file` | `field->to_exprt`, i.e. `jimple_class_field` |
| `jimple_class_field` | nothing -- it builds a struct component from a type |
| `jimple_throw` | nothing -- its operand conversion is commented out upstream |

So every expression `to_exprt` looked callerless, as did `jimple_assignment`'s.

**That conclusion was wrong, and §38 corrects it.** `jimple_binop::to_expr2t`
covered six operators and sent the rest to the migrating default, which converts
its operands with `to_exprt` -- so any expression kind was still reachable as the
operand of, say, a multiplication. The census read zero only because every binop
in the corpus happens to be one of the six. That is exactly §36.1's lesson
applied to a conclusion drawn one section after stating it.

### 37.4 Status

Twenty-seven PRs. B-1 is 154, from 160 -- the drop is the two `exprt lhs`
members and their setters. The slice mostly adds native code rather than removing
legacy code; the removal it unlocks is worth most of `jimple_expr.cpp`.


## 43. The invoke cluster, and an over-deletion §42 hid (2026-09-15)

§42.1 called `jimple_symbol` and `jimple_constant` "ordinary conversions". They are not:
both already have native `to_expr2t` arms, and their legacy arms were reached only from
inside the two remaining legacy consumers -- `jimple_assignment::to_exprt` and
`jimple_expr_invoke::to_exprt`. Grepping every direct `to_exprt` call confirms it: the
only ones left on expressions are at `jimple_statement.cpp:136,142,149,152` and
`jimple_expr.cpp:330,335,358`, all inside those two.

So the three live arms were one cluster rooted at the invoke lowering.
`jimple_expr_invoke` now builds the call natively -- a `code_block2tc` of the
`@parameter<i>` assignments followed by `code_function_call2tc`, with an `lhs2` the
assignment sets -- and `jimple_assignment::to_code2t` routes the non-virtual invoke there
instead of to the migrating default. Re-probing all four remaining legacy arms over the 27
tests gives **0 observations**: the cluster is retired.

### 43.1 What that probe also exposed

`jimple_virtual_invoke::to_exprt` was in the nine §42 deleted as unreached, and deleting
it was wrong. `jimple_assignment`'s virtual-invoke branch still delegates to the migrating
default, which reaches that arm; with the arm gone the base's `code_skipt` was returned
instead, so **a virtual-invoke assignment silently became a skip**.

Nothing in the corpus builds that shape -- which is why the deletion passed 27 of 27 and
why the probe read zero. The arm is restored here with a comment saying so. Two things
follow:

- the probe answers "is this reached by the corpus", not "is this dead". For a deletion
  the second question is the one that matters, and only a caller audit answers it. §42's
  other eight deletions are safe on that stricter test -- each class has a native
  `to_expr2t` and no remaining caller -- but that was luck rather than method;
- `regression/jimple` has no test for an assignment whose right-hand side is a virtual
  invoke. That gap let a silent semantic change through, and it is worth closing before
  `jimple_virtual_invoke` is converted for real.

### 43.2 The gap closed

`github_4715_virtual_invoke_assign_01{,_fail}` is that test. The jimple input is JSON, so
it is hand-authored rather than compiled: `kt-func-call-true`'s structure with its callee
made an instance method, a receiver local allocated with `new`, and the assignment's
right-hand side changed from `static_invoke` to `virtual_invoke`. The failing half changes
the callee's return value rather than the assertion, so both halves exercise the same
lowering.

It bites on exactly the over-deletion it exists to catch: delete
`jimple_virtual_invoke::to_exprt` again and **both halves fail**, where the 27 tests before
it all passed. `jimple` is 29 of 29 with it added.

B-1 is 115 rather than the 110 §42 reported, the difference being the restored arm.

## 44. `jimple_virtual_invoke` converted, and two more over-deletions found (2026-09-15)

With §43.2's test in place, `jimple_virtual_invoke` gets the native arm
`jimple_expr_invoke` got in §43: a `code_block2tc` of the `@this` and `@parameter<i>`
assignments followed by `code_function_call2tc`, and `jimple_assignment::to_code2t` now
routes both invoke forms to their own arm instead of to the migrating default. 29 of 29
jimple and 876 of 876 unit tests.

### 44.1 The caller audit §43.1 asked for, and what it found

§43.1 said a deletion needs a caller audit rather than the probe. Doing that audit across
both hierarchies -- which classes declare a native arm, and which inherit the base's
migrating default -- found **two more classes** in the same state
`jimple_virtual_invoke` was in:

| class | native `to_code2t` | legacy `to_exprt` |
|---|---|---|
| `jimple_identity` | no | deleted by §40.3 |
| `jimple_assertion` | no | deleted by §40.3 |

Both fall through `jimple_method_field::to_code2t` to the base's `to_exprt`, which returns
`code_skipt`. So since §40.3 an identity statement and an `Assertion` statement have
silently produced a skip. Both arms are restored here with a comment saying why.

No test builds either statement -- jimple tests express an assertion through the
`If`/`AssertionError` idiom rather than the `Assertion` object, and nothing in the corpus
emits `Identity` -- which is exactly why three deletions in a row passed 27 of 27.

### 44.2 The rule, stated properly this time

The probe answers *"does the corpus reach this?"*. A deletion needs *"can anything reach
this?"*, and for a virtual with a non-abstract base the answer is yes unless the class
overrides the replacement. So the criterion is:

> an arm may be deleted only if its class declares the native replacement.

That is checkable without running anything, it is what the table above applies, and it
would have prevented all three over-deletions. B-1 is 124: the true figure once the three
restorations are counted, against the 110 §42 claimed.

## 45. The legacy expression tree retired (2026-09-15)

§44.2's criterion -- an arm may be deleted only if its class declares the native
replacement -- applied to what was left. Five arms qualify and are deleted:
`jimple_assignment`, `jimple_constant`, `jimple_symbol`, `jimple_expr_invoke` and
`jimple_virtual_invoke`. 227 lines, 29 of 29 jimple and 876 of 876 unit tests, B-1
124 -> **103**.

The deletion is safe by inspection rather than by the suite: every direct `to_exprt` call
on an expression was inside one of those five arms, so they formed a closed cycle that the
`to_code2t`/`to_expr2t` conversions had already routed around. Re-auditing afterwards, the
only expression class still declaring a legacy arm is the base `jimple_expr` itself, whose
arm is the default the whole scheme hangs off.

### 45.1 What remains, and it is exactly two things

| class | why it still has a legacy arm |
|---|---|
| `jimple_identity` | no native `to_code2t`; restored in §44.1 |
| `jimple_assertion` | no native `to_code2t`; restored in §44.1 |

Both need converting before their arms can go, and neither is exercised by any test --
`regression/jimple` asserts through the `If`/`AssertionError` idiom rather than the
`Assertion` object, and nothing in the corpus emits `Identity`. So the next slice is a test
first, as §43.2 was for the virtual invoke, and only then the conversion. Converting them
blind is how §40.3's deletions passed while breaking both.

Jimple's B-1 over this run: **190 -> 103**, of which 87 is deleted dead code and three
restorations are the correction for measuring reachability where reachability was not the
question.

## 46. `jimple_assertion` is parse-only, and the audit has to include `unit/` (2026-09-15)

§45.1 named `jimple_identity` and `jimple_assertion` as the two classes still needing a
native arm. One of them does not need one at all.

There is no `statement::Assertion` enumerator (`jimple_method_body.h:108-122`) and no
`"Assertion"` entry in the JSON dispatcher's `from_map`, so the body walk can never build a
`jimple_assertion`. Its `to_exprt` has no production path, and §44.1's restoration of it was
unnecessary -- harmless, but it was restoring something unreachable. The arm is deleted
again, this time with the reason; the class stays. B-1 103 -> **97**.

### 46.1 How that was nearly got wrong, again

The first attempt deleted the whole class. The build failed:

```
unit/jimple-frontend/jimple_ast.test.cpp:165: 'jimple_assertion' was not declared
```

`unit/` constructs one and tests its `from_json`. So the class is live, only its lowering is
not -- and §44.2's criterion, applied to `src/` alone, would have missed that. The criterion
needs its scope stated:

> an arm may be deleted only if its class declares the native replacement, and a class only
> if nothing in `src/` **or `unit/`** names it.

The compiler enforces the second half for free, which is why this attempt cost a build
rather than a regression.

### 46.2 `jimple_identity` is the last one, and its arm looks broken

`Identity` *is* in the enum and the dispatcher, so that class is reachable and its arm has to
stay until it is converted. Reading it first, though:

```cpp
symbolt &added_symbol = *ctx.find_symbol(local_name);
```

`local_name` is the bare jimple local (`$i0`), not a qualified symbol id, and every other
lookup in this frontend goes through `get_symbol_name(class, function, name)`. So the arm
dereferences the result of a lookup that looks certain to miss. No test builds an `Identity`
statement, so nothing has exercised it either way.

That makes the next slice a probe rather than a conversion: author a jimple input with an
`Identity` statement and find out whether the existing arm works at all. Converting it first
would port a defect into IREP2 and call it a migration.

## 47. The identity statement crashes, and that is why it could not be converted (2026-09-15)

§46.2 suspected `jimple_identity::to_exprt` was broken and said to probe before converting.
Probing it took two attempts and both were informative.

The first used `"object": "Identity"` and got `ERROR: Unknown type`. The dispatcher's
`from_map` spells it **lowercase** -- `{"identity", statement::Identity}`
(`jimple_method_body.h:129`) -- while `to_map`, which is only used for printing, spells it
`"Identity"`. Reading the wrong one of the two is how the tag was got wrong.

With `"object": "identity"`, a three-statement method -- declare `$i0`, identify it as
`@parameter0`, return -- **SIGSEGVs during GOTO conversion**. The cause is the one §46.2
read off the source:

```cpp
symbolt &added_symbol = *ctx.find_symbol(local_name);
```

`local_name` is the bare local (`$i0`); every other lookup in this frontend goes through
`get_symbol_name(class, function, name)`. The lookup misses and the dereference is on null.

### 47.1 Pinned, not fixed, and why

`regression/jimple/github_4715_identity_crash_01` is the reproducer, as `KNOWNBUG`. It
passes ctest, which in this repo means the bug is still live.

It is not fixed here because the fix is not mechanical. The arm's right-hand side is a
`symbolt` that is never added to the context:

```cpp
symbolt rhs;
rhs.name = "@" + at_identifier;
rhs.id = "@" + at_identifier;
code_assignt assign(symbol_expr(added_symbol), symbol_expr(rhs));
```

so the statement was meant to assign from a symbol nothing declares. Since the arm has never
run -- it crashes first -- there is no observed behaviour to preserve, and choosing what
`$i0 = @parameter0` should lower to is a design decision about how this frontend binds
parameters, not a migration step. Converting it would be inventing semantics and calling it
IREP2.

So jimple's B-1 stops at 97 with one legacy statement arm left, and that arm is blocked on a
question about the frontend rather than about the migration.

## 48. B-2's residue, and two more spelling false positives (2026-09-15)

§39 recorded jimple's B-2 as met. `scripts/irep2/bars.py` reports 10 raw and 8 refined, so
the two disagree, and taking the eight one at a time explains why.

**Three convert**: `jimple_method.cpp:92` (a method's code type),
`jimple-language.cpp:97` and `:193` (a post-processed symbol's type and `main`'s). 30 of 30
jimple and 876 of 876 unit tests; B-2* 8 -> **5**.

**Two were never debt**, and both are cases the script's own caveat names:

- `jimple_ast.h:69` -- `create_jimple_symbolt` takes a `const type2tc &`, so
  `symbol.set_type(t)` already writes IREP2. The grep matched the spelling `set_type(t)` and
  the refinement could not tell, because the argument is a plain name.
- `jimple_method.cpp:93` -- `set_value(body->to_code2t(...))` passes an `expr2tc` returned by
  a method call, which is neither a `migrate_*` call nor a `*2tc` constructor.

Wrapping the first in `migrate_type` does not compile, which is how it was caught. §58's
`B-2*` is an upper bound for exactly this reason, and jimple is the frontend where the
remaining count is small enough for the residue to matter: 5 of the 8 are real.

**Three stay legacy, each for a reason already recorded**: `jimple_file.cpp:159` sets a
`width` attribute on the class struct type immediately before writing it, and that attribute
is read by this frontend's own `newarray` arms (§45.2 of the parent document);
`jimple-language.cpp:108` and `:194` are body writes, which §6.1 of
`scope-python-irep2.md` established must stay lazy until every symbol they name exists.

So jimple's B-2 residue is 5 reported, 3 of which are by design and 2 of which are miscounts.
That is what "met" in §39 meant, stated in numbers.
## 38. Every binop the frontend supports, and what a zero still hid

§37.3 concluded that the expression `to_exprt` arms were callerless. They were
not. `jimple_binop::to_expr2t` handled six operators and sent everything else to
the migrating default, whose legacy arm converts *its own operands* with
`to_exprt` -- so a multiplication with a cast operand would have reached
`jimple_cast::to_exprt`. The census read zero because the corpus's binops are all
among the six, which is §36.1's own lesson landing one section after it was
written down.

### 38.1 What the frontend actually supports

`jimple_binop::from_json` takes the operator string verbatim apart from mapping
`==` to `=`, and the legacy arm hands it to `gen_binary`, which builds an `exprt`
with that id whatever it is. So the supported set is not a list in the frontend at
all -- it is whatever `migrate_expr` knows. Probing 29 candidate spellings through
the frontend separates them cleanly:

| Converts | Rejected |
|---|---|
| `+ - * / mod < <= > >= = == notequal and or bitand bitor bitxor shl ashr lshr` | `% shr << >> >>> != & \| ^` |

Twenty work; nine produce `ERROR: migrate expr failed: <op>`. The rejected ones
are mostly the symbolic spellings of operators that *are* supported under a word
(`&` versus `bitand`), which is worth knowing before assuming a parser change is
safe.

The IREP2 arm covered 6 of the 20. The other 13 are added here, and all 20 emit a
byte-identical instruction. Two details fell out of the measurement rather than
from reading: the relational and logical kinds return a bool-typed node and the
*enclosing assignment* is what casts it, which is why the dump shows
`(signed int)($i1 < 2)`; and `and`/`or` take the operands as they come, not as
bools, because the legacy arm typed the node with the left-hand side's type too.

### 38.2 `ashr` and `lshr` cannot be told apart here

Both print `>>`, so the dump cannot separate them. Nor can a verdict: `-8 ashr 1`
is `-4` and `-8 lshr 1` is `2147483644`, but a program dividing by
`(x >> 1) + 4` reports division-by-zero either way. Swapping the two arms leaves
every dump identical and all 30 tests passing.

The reason is structural: `jimple_type` builds nothing but int, bool, void and
pointers (§23.1), so the left operand of a shift is always signed, and that is the
case where the two coincide in everything this frontend can observe. So the
mapping is pinned by mirroring `migrate_expr`'s arms (`migrate.cpp:1565` and
`:1725`) and by nothing else -- stated here rather than left as an unexamined pass
in the test log. The repo has hit this before from the C side, where unsigned types
make the difference visible.

### 38.3 The test, and three mutations

`github_4715_binop_kinds_01` puts all twenty operators in one method and pins the
emitted instructions as a single ordered sequence, so each mapping is pinned by
its own printed form and by its position. Three mutations, each failing that test
and nothing else: `mod` mapped to `div`, `<=` to `<`, and `bitxor` to `bitor`.

A regex trap, for the next person writing one of these: a raw-string `r";\n"` puts
a literal backslash and `n` in the pattern, which `rstrip("\n")` does not remove,
so the pattern ends up requiring a newline before `$` and cannot match anything.
Build the separators explicitly instead of trimming them off the end.

### 38.4 What still blocks the deletion

One thing. An unsupported operator reaches `jimple_expr::to_expr2t`, whose legacy
arm produces `ERROR: migrate expr failed: <op>` -- so `jimple_binop::to_exprt`
remains reachable purely as the error path, and its operand conversions keep every
expression arm reachable with it.

Closing that means rejecting an unsupported operator in the IREP2 arm, which
changes the message a user sees and so wants its own change and its own test. After
it, the twelve expression arms and `jimple_assignment`'s go the way the seven in
#7786 did.

### 38.4a `and` and `or` were invalid in both representations

CI found this, and no local run could have. `github_4715_binop_kinds_01` aborted
the assert-enabled `DebugOpt` build at `goto_check.cpp`'s `and_id`/`or_id` arm,
which asserts the node *and each operand* are bool. The arm here handed
`and2tc`/`or2tc` two `signed int` operands, because the legacy arm it mirrors
passed `gen_binary` the left-hand side's type.

The legacy spelling is no better: `migrate_expr`'s `and` arm asserts
`expr.type().id() == typet::t_bool`, so a jimple `and` over two ints aborted an
assert build on *either* path. The defect predates the IREP2 arm; what the new
test did was reach it for the first time, since no corpus program used a Boolean
binop.

Both operands are now converted with `c_implicit_typecast` and the node is bool,
so the dump reads `(signed int)((_Bool)$i1 && 1)` where it read
`(signed int)($i1 && 2)`. That is a deliberate change to the emitted GOTO rather
than a byte-identical port, and the only one in this slice: the constant folds to
`1` because `(_Bool)2` is `true`.

The lesson for the campaign's gates: a byte-identical dump comparison under
`NDEBUG` proves the two paths agree, not that either is *valid*. Two asserts that
both paths trip stay invisible to it.

### 38.4b Probing for the rest of that class

With the workspace switched to `DebugOpt` (`-O2 -g`, no `-DNDEBUG`, i.e. CI's
asserts locally) the obvious follow-up was to ask what else aborts. Seven jimple
probes over shapes the corpus has no instance of: an `If` on a bare `int` symbol,
a comparison stored into an `int` and then used as a condition, three-level nested
arithmetic, an `and` of two comparisons, an `and` inside an `If` condition, a
`bitand` over a comparison, and a shift by a comparison.

Two abort on master, both `and` with operands that are *already* bool, at
`migrate.cpp:1434` -- the same assertion as §38.4a, reached because master's arm
covers six operators and sends `and` to the legacy one. Both pass here.

They are kept as `github_4715_and_of_comparisons_01` and
`github_4715_and_in_condition_01`, and what they pin is worth stating exactly:
**not** the bool conversion -- their operands are bool already, so dropping the
conversion leaves them passing -- but that a Boolean binop is handled natively at
all. `github_4715_binop_kinds_01` is the one that pins the conversion, its `and`
having `int` operands. The three together cover both halves, and the mutation that
separates them is what showed which is which.

The other 126 probe runs from `scope-clang-c-irep2.md` §143 and
`frontends-to-irep2.md` §40-41, re-run under asserts, abort nowhere.

### 38.5 Status

Twenty-eight PRs. B-1 reads 155, one more than §37, and the extra hit is a comment
mentioning `to_exprt` by name -- the same class of false positive B-2's command has
(§35.2), now on B-1. The deletion §37.3 promised is one small slice further away
than that section claimed.

## 39. The expression subtree retires

Two changes, the second only possible because of the first.

### 39.1 Rejecting an unsupported operator where it is built

§38.4 left one caller for the expression `to_exprt` arms: an operator outside the
twenty reached `jimple_expr::to_expr2t`, whose legacy arm handed it to
`gen_binary` and then to `migrate_expr`, which rejected it. `jimple_binop::to_expr2t`
now rejects it directly, with `throw "Unsupported Jimple operator: " + binop` --
the same mechanism `jimple_type` already uses two files away, so the same handler
and the same exit code 6.

The user-visible output improves rather than merely moving. Before, `%` produced
three lines: the irep dump of the node, `migrate expr failed`, and
`ERROR: migrate expr failed: %`. Now it produces one, naming the operator the
source actually contained. The partition is unchanged, re-measured across all 29
candidate spellings: 20 convert, 9 are rejected, same nine.

`github_4715_binop_unsupported_01` pins the message; replacing the throw with a
silent `add2tc` fails it and nothing else.

### 39.2 Twelve arms, and the argument for each

With that, no live code calls an expression `to_exprt`. Removed:
`jimple_constant`, `jimple_symbol`, `jimple_binop`, `jimple_cast`,
`jimple_lengthof`, `jimple_expr_invoke`, `jimple_virtual_invoke`,
`jimple_newarray`, `jimple_deref`, `jimple_nondet`, `jimple_virtual_member`, and
`jimple_assignment`'s.

The caller argument is the same shape as §36.2's, and it is now complete because
the two escape hatches are closed: `jimple_binop`'s (§39.1) and
`jimple_assignment::to_code2t`'s delegation (§37). Every surviving `to_exprt` was
checked for operand conversions first -- `jimple_identity`, `jimple_assertion`,
`jimple_static_member`, `jimple_class_field` and `jimple_throw` are all leaves,
and `jimple_method` and `jimple_file` reach only `to_code2t` and
`jimple_class_field` respectively. So nothing that survives can call something
that was deleted, which matters here because virtual dispatch would have fallen
back to the base silently rather than failing to compile.

B-1 goes from 155 to 97.

### 39.3 What is left, and why each one is there

| Arm | Why it stays |
|---|---|
| `jimple_method`, `jimple_file`, `jimple_class_field` | the class and method conversion path, which is not an expression |
| `jimple_static_member` | its `to_expr2t` covers two intrinsics and leaves the member access on the default, still marked "Needs OOP members" (§25.2) |
| `jimple_throw` | live, but its operand conversion is commented out upstream (§31.1) |
| `jimple_identity`, `jimple_assertion` | unconstructible (§19) -- a different argument from callerless, so left alone as in #7786 |

Three of those seven are the honest remainder of the frontend: the static member
access, `jimple_throw`, and the two unconstructible classes. None is a mechanical
port.

### 39.4 Status

Twenty-nine PRs. B-1 is 97, from 202 when §32 opened; B-2 is met (§35). The
expression and statement migrations are complete and their legacy arms are gone
except for the four above.
