# Scope — jimple frontend → IREP2 construction

Opened 2026-08-09 as Phase 5 of `frontends-to-irep2.md`, which orders the
per-frontend migrations and puts jimple first: *"smallest surface, no
operational-model complication, lowest blast radius. The pathfinder for the
kit."* This document is the census and decomposition that phase asks each
frontend to open with.

**Build caveat, stated up front.** `ENABLE_JIMPLE_FRONTEND` is `OFF` in the
working build, so nothing here is measured by running jimple — the census is
static, and every gate below needs a build with the frontend enabled. The
byte-identity numbers §20 records for jimple (15/15, zero declines) came from CI
and the dispatcher work, not from this machine.

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

**J.1 — the type side first.** `to_typet` is a single method and 20 `typet`
mentions. Migrating it to return `type2tc` is the smallest possible slice and
proves the seam before any expression work.

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

Census only. No code has moved, and J.1 has not started.
