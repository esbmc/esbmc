# Scope — carrying a C type's spelling (W3/W4)

> Opened 2026-08-17. The scope document `frontends-to-irep2.md` §33.3 and
> `scope-clang-c-irep2.md` §102.2 both ask for. Parent: `frontends-to-irep2.md`
> (B-4, walls W3 and W4). Sibling: `scope-v2-w3-attribute-carriage.md`, which
> took Option D (seam, do not remove).

## 1. The problem, stated from its consumers

A C type's *spelling* — that it was written `char` rather than `int8_t`, `float`
rather than `signedbv` of 32 bits — is not part of the type in IREP2, and is
carried on legacy nodes by two `#`-attributes:

| attribute | written by | read by |
|---|---|---|
| `#cpp_type` | the clang converters | `clang_cpp_adjust_expr`'s exception-id builder (**semantics**); `cpp_expr2string`, `goto2c/expr2c` (**presentation**) |
| `#cformat` | `constant_exprt`'s `(BigInt, typet)` ctor (`std_expr.h:1102`); `string2array.cpp:25` for the char hint | `c_expr2stringt::convert_constant`, `goto2c/expr2c` (**presentation**); `padding.cpp`, `type_byte_size.cpp`, `goto_check.cpp` (**semantics** — §1.1) |

B-4 asks for no `#`-attribute escape hatch into a shared pass. Both attributes
are exactly that, and the two consumer classes are why this has resisted
closure: W3 is the semantics reader, W4 the printers, and they have been treated
as separable.

### 1.1 `#cformat` is not a presentation attribute — correction

The table above first listed `#cformat` as written by `string2array` and read only
by `convert_constant`. That was wrong, and it changes what a carriage decision has
to cover.

`constant_exprt(BigInt value, typet type)` sets **`value`** to
`integer2binary(value, bv_width(type))` and **`#cformat`** to
`integer2string(value)` — the same number twice, in binary and decimal. Three
semantics-bearing readers took the decimal one:

| site | what it decides |
|---|---|
| `clang-c-frontend/padding.cpp:276` | an explicit `__attribute__((aligned(N)))` for struct layout |
| `util/expr/type_byte_size.cpp:646` | the same alignment, for object size |
| `goto-programs/goto_check.cpp:560` | an array's width in an input-overflow check |

`#cformat` was therefore carrying a *value a consumer needed*, not a spelling a
printer preferred. The first two now call `to_integer` on the constant instead
(PR #7122), gated on byte-identical goto output over 70 tests; a mutant that
ignores the value trips `adjust_type`'s own `sz % a == 0` assertion, so the
readers are live.

`goto_check.cpp:560` is left: its expression is an array size, not a constant for
a VLA, so switching it needs a constant-or-not branch with no witness.

**Consequence.** Two of five `#cformat` readers are no longer readers, and the
remaining semantic one is a different problem (an array width, not a spelling).
What is left to decide about `#cformat` is only the char hint of §2 — which is
the presentation question this document is about. The correction narrows this
scope rather than widening it.

## 2. What is measured

Everything below is a recorded measurement, not an estimate.

| fact | source |
|---|---|
| **4** spellings reach catch-matching (`double`, `float`, `signed_char`, `signed_int`), over 949 C++ test directories | `frontends-to-irep2.md` §33.1 |
| **83** spellings exist in the domain; the other 79 reach only printers | §32.1, §33.2 |
| No vector spelling reaches catch-matching; the ACLE has **no** prohibition on sizeless types as exception objects, so this is evidence, not proof | §33.2 |
| Losing `#cformat` makes a char array print as integers, **14 tests** in `cstd` under `--clang-c-irep2-adjust-only` | `scope-clang-c-irep2.md` §102.1 |
| The IREP2 `c_typecastt` copy has no `string2array`, which is where the hint is set | `scope-coupled-arith-assign-conversion.md` §20.1 item 7 |

## 3. The coupling, which is the reason for this document

§33.3 concluded: go for the semantic half (a typed field carrying four scalar
spellings), not a B-4 closure, because 79 presentation-only spellings still need
a carrier. That reads as "do the easy half now, scope the rest later".

**The halves are not separable in that direction.** Consider the cheapest
imaginable fix to the W4 witness of §2 — teach `convert_constant` to render a
char-typed constant as `'T'` when no `#cformat` is present. It is additive: the
default path has the hint and cannot change. It still does not work, because a
legacy `typet` cannot distinguish `char` from `int8_t` — both are `signedbv` of
width 8 — and what distinguishes them is `#cpp_type`.

So the presentation half needs the spelling for the *same reason* the semantics
half does: to know which C type this is. A typed field that carries only the
four catch-matching spellings does not serve it; `signed_char` is in that set,
but the field would have to be populated for every char-typed constant, not only
for those that reach a throw.

That is a stronger statement than §33.3 makes, and it changes the sizing: the
"days for the semantic half" estimate holds only if the presentation half is
abandoned, not deferred.

## 4. Options

| # | Option | Verdict |
|---|---|---|
| A | typed field for the 4 scalar spellings only | serves catch-matching; leaves W4 and §2's 14 witnesses untouched (§3) |
| B | typed field for all 83 | a string field on `type2t` in all but name; this is Option C of `frontends-to-irep2.md` §5.1, already rejected as pushing presentation into the verifier IR |
| C | typed field for the spellings that *identify a C type* — the scalar set plus whatever the printers need to disambiguate | the honest middle; needs the printer-side domain measured, which §2 has not done |
| D | leave both attributes, close B-4 by re-scoping it | what §37 effectively concluded; this document exists because §102.2 shows the presentation half has a *correctness* consequence under the hop-off, not just a cosmetic one |

**Recommendation: measure before choosing.** Option C is the only one whose cost
is unknown, and it is the only one that closes both halves. The missing number is
the printer-side equivalent of §33.1: *which spellings does `convert_constant`
(and `cpp_expr2string`) actually need in order to render differently from what
the type alone would give?* If that set is also small and scalar, C collapses
into A and B-4 becomes closable. If it is the full 83, D is the answer and B-4
should be re-scoped explicitly rather than by attrition.

## 5. Phase 0 — the measurement this scope asks for

One `fprintf` at each presentation reader, over the C and C++ suites, recording
the spelling *and* whether the rendering differs from what the type alone would
produce. Method exactly as §33.1: replay `test.desc` flags, count distinct
values, and treat an unobserved value as unobserved rather than impossible
(§33.2's own caveat, and §30.2's).

Exit criterion: the size of the "needed to disambiguate" set. Nothing else in
this document should be built first.

## 6. Non-goals

- Removing the six dead `#implicit` writes (`scope-clang-c-irep2.md` §33.4) —
  adjacent, separately scoped.
- `#sol_type`, which has no consumer outside the Solidity frontend
  (`frontends-to-irep2.md` §37.1) and is not a B-4 item.
- The `string2array` gap itself, which is a `c_typecastt` divergence
  (`scope-coupled-arith-assign-conversion.md` §20.1) and closes W4's witness
  without deciding carriage.
