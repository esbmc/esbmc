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

**Consequence for sequencing.** Phase 7 cannot begin with an adjuster slice.
Porting the four C++ arms into the `expr2tc` overload is the first work item,
and `unit/util/c_typecast.test.cpp` — the differential harness #6873 added — is
where it is pinned. §20.3's lesson is the standing warning: a second
independently-written copy of a conversion is not a translation of the first,
and byte-identity on another frontend's corpus does not establish that it is.

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

1. Port §20.1 items 1, 2, 6, 7 into the `expr2tc` `implicit_typecast_followed`,
   pinned by `unit/util/c_typecast.test.cpp`'s differential harness.
2. Price option B in §3 against option A.
3. Add the C++ hop-off flag, then run the census by verdict.

Only then does a slice make sense.
