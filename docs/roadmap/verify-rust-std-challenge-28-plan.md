# Plan: Solving verify-rust-std Challenge 28 (flt2dec) with ESBMC

**Status (2026-11-24):** the Phase 1 gate failed (see §9). Per §5, we do not compete
for the bounty as planned. Phase 0 is half done: ESBMC v8.8 is released, and the
committee has not yet answered on #524.

**Related:** [`cprover-support-roadmap.md`](cprover-support-roadmap.md) (how ESBMC
reads Kani's CBMC goto-binaries), the challenge text
[`doc/src/challenges/0028-flt2dec.md`](https://github.com/model-checking/verify-rust-std/blob/main/doc/src/challenges/0028-flt2dec.md),
and the tracking issue
[model-checking/verify-rust-std#524](https://github.com/model-checking/verify-rust-std/issues/524).

Facts about verify-rust-std below come from main at `1feeadd937` and from its
GitHub issues and PRs on 2026-11-22. Bare `#N` numbers are verify-rust-std PRs
and issues; ESBMC ones are written `esbmc#N`.

---

## 1. What the challenge asks

Challenge 28 targets the float-to-decimal module, now at
`library/core/src/num/imp/flt2dec/`. Twelve safe functions with unsafe bodies
must be "proven unconditionally safe, or safety contracts should be added":

| Location | Functions |
|---|---|
| `flt2dec/mod.rs` | `digits_to_dec_str`, `digits_to_exp_str`, `to_shortest_str`, `to_shortest_exp_str`, `to_exact_exp_str`, `to_exact_fixed_str` |
| `strategy/grisu.rs` | `format_shortest_opt`, `format_shortest`, `format_exact_opt`, `format_exact` |
| `strategy/dragon.rs` | `format_shortest`, `format_exact` |

The proofs must rule out dangling or misaligned accesses, UB through intrinsics,
mutation of immutable bytes, and invalid values. Proofs over generic `T` may stop
at primitive types. The text has no unbounded-proof clause. The bounty is 5,000 USD.

The general rules add three conditions that shape this plan
(`doc/src/general-rules.md`): the solution is one PR that "should run as part of
the CI", every tool it uses must already be integrated in the repository, and it
"cannot impact the runtime logic of the standard library".

## 2. Where the competing solutions stand

Four PRs are open. The committee reviewed three of them on 2026-08-16 and again
on 2026-09-12; #760 has no review yet.

| PR | Author's claim | Review outcome |
|---|---|---|
| #596 | 12 of 12 | Changes requested, named the front-runner. The six `mod.rs` proofs are sound. The strategy proofs rest on disclosed compromises: `stub_div_rem_upto_16` assumes `d < 10`, a `CMP_CALLS` counter forces `Ordering::Less` after 12 `Big::cmp` calls, the `Fp::mul` stub is the identity on the mantissa, the Grisu shortest mantissa is limited to `2..=0xF`, and `round_and_weed` is stubbed away. Contracts are doc comments, not `#[safety::requires]` / `#[safety::ensures]` ("T7"). |
| #601 | 12 of 12 | Changes requested. `grisu::format_shortest_opt` and `format_exact_opt` are stubbed in every harness, so their digit loops are never checked. `dragon::format_shortest` assumes `i < MAX_SIG_DIGITS` before the buffer write, which the review calls "textbook assume-the-conclusion". |
| #606 | 12 of 12 | Changes requested. Three functions get separate `cfg(kani)` bodies, one with a different return type, and `assume(q < 10)` sits right before the digit write. |
| #760 | 12 of 12 | Opened 2026-11-15, review required. |

Every PR stops at the same place: the Grisu and Dragon digit-generation loops.
Each one stubs the arithmetic that bounds the digit value or the buffer index,
or assumes the bound outright. A solution that checks those loops over the real
bodies, with real bignum and `Fp` arithmetic and the full mantissa range, would
close the gap the reviews describe. That is the target of this plan.

## 3. Why ESBMC, and what is not yet known

ESBMC reads the GOTO that Kani generates, so it checks the same harnesses with
the same Rust semantics. What it adds is a different solver stack and k-induction.

On the 2026-11-22 sweep of all 1,888 verify-rust-std harnesses (900 s and 15 GB
per harness, ESBMC master `4757f8a6bd`), ESBMC agrees with Kani on 102 of 102
`num::verify` harnesses and on 190 of 190 float-to-int conversions. Those are the
closest modules to flt2dec. No flt2dec harness exists in main, so we have no
measurement on this module, and Phase 1 exists to get one.

The digit loops are bounded by `MAX_SIG_DIGITS = 17` (`flt2dec/mod.rs:143`), so
plain BMC with full unwinding covers them; k-induction is not the lever here. The
open question is solver cost. On the 64-bit target Kani uses, Dragon works on
`Big64x20` (20 `u64` limbs, `bignum.rs:396-401`), and an unstubbed run means
multiplying, comparing, and dividing those with a symbolic mantissa and exponent.
CBMC is also bit-precise, so any advantage over Kani on this code has to be
measured, not assumed.

## 4. Constraints outside verification

These decide whether a technically correct ESBMC proof can be submitted at all.

1. **Kani's CI runs every harness.** `scripts/run-kani.sh:191-233` verifies every
   harness `kani list` reports, with no skip list. An unstubbed Dragon harness
   that CBMC cannot finish would fail the Kani job and block the PR.
2. **ESBMC's CI runs almost nothing.** `.github/workflows/esbmc.yml:37` runs only
   `checked_unchecked.*.out`, and `scripts/run-esbmc.sh:13` pins ESBMC v8.7. No
   ESBMC release yet contains esbmc#10942 (k-induction default for Kani binaries)
   or the fixes merged since v8.7 (esbmc#10943, esbmc#10945, esbmc#10948).
3. **Contracts must be machine-checked.** The #596 review asks for
   `#[safety::requires]` / `#[safety::ensures]` with `#[kani::proof_for_contract]`.
   ESBMC already checks `requires` and `ensures` in Kani's generated GOTO; 1,236
   contract harnesses ran in the sweep. It does not see `modifies` clauses or
   loop invariants, because Kani's `--only-codegen` stops before goto-instrument.
   Any `stub_verified` replacement we rely on has to be sound without that frame
   check, or we do not use it.
4. **No runtime-logic changes.** Verification code stays additive and
   `cfg(kani)`-gated. The #601 review counted a behavior-preserving refactor of
   `dragon::format_exact` against it, and #606 lost credit for body swaps.
5. **The pending #733 ruling** ("mutating immutable bytes") binds every tool
   equally. It does not favor or block ESBMC.

## 5. Phases

### Phase 0: settle CI and release (week 1)

Constraint 1 is the blocker, so we start with the committee, before writing
harnesses. We ask on #524 how a harness that only ESBMC finishes should run.
Two options are workable:

- **Exclusion:** an attribute or naming convention that `run-kani.sh` skips and
  `run-esbmc.sh` selects. It needs a CI change in verify-rust-std.
- **Dual harnesses:** a Kani-sized harness, stubbed like #596, alongside an
  unstubbed ESBMC harness of the same function. It needs no exclusion, but the
  Kani copies would carry the compromises the reviews reject, so the committee
  has to accept that the ESBMC copy is the proof of record.

In parallel, cut an ESBMC release that includes esbmc#10942 and the fixes after
v8.7, so the CI job can move off v8.7.

**Exit:** a written answer from the committee on #524, and a release tag.

### Phase 1: measure before committing (weeks 1 to 2)

1. Generate GOTO for the harnesses in #596, #601, #606, and #760, rebased on
   current main, with the same Kani pin as the CI.
2. Run ESBMC with BMC, k-induction, and incremental BMC on each one at 900 s and
   15 GB, on the EC2 sweep machine.
3. Remove the #596 compromises one at a time and rerun: the `Fp::mul` identity
   stub, the `CMP_CALLS` counter, the `d < 10` assumption, and the narrow Grisu
   mantissa. Record time, memory, and verdict for each step.
4. Report cover checks with `--report-property-class` (esbmc#10877) so a green
   run is shown to reach the digit writes and is not vacuous.

**Gate:** `grisu::format_shortest_opt` and `format_exact_opt`, unstubbed, with a
symbolic `f64` input, finish inside the budget. If they do not, stop and write up
the measurements; the plan is not viable as stated. If Grisu passes and Dragon
does not, Phase 3 takes the compositional route for Dragon.

### Phase 2: ESBMC fixes (weeks 2 to 4)

Fix whatever Phase 1 turns up, each with the usual regression pair. Two known
issues are likely to show up here:

- **esbmc#10936 residual.** `fmt::num::verify::check_u64_fmt` still false-alarms
  through it. That harness formats integer digits into a buffer, the closest
  existing analogue to flt2dec's digit output.
- **esbmc#10940 / esbmc#10951.** `digits_to_dec_str` and `digits_to_exp_str` fill a
  `[MaybeUninit<Part>]` slice and return it through `assume_init_ref`
  (`flt2dec/mod.rs:181-304`). Whether this hits the union-read bug is inferred,
  not measured. Both issues are fixed in v8.8 (esbmc#10953).

### Phase 3: harnesses and contracts (weeks 3 to 6)

- **`mod.rs` wrappers.** Follow #596's approach, which the review judged sound:
  symbolic buffer contents, full-range `exp` and `frac_digits`.
- **Grisu.** Call `format_shortest_opt` and `format_exact_opt` directly, with no
  stubs, on inputs from `decode()` of symbolic `f32` and `f64` values, using the
  real `cached_power` and `Fp::mul`.
- **Dragon, compositional.** Write `requires` / `ensures` contracts on the
  `Big64x20` operations Dragon calls (`mul_pow2`, `mul_pow10`, `mul_small`,
  `mul_digits`, `add`, `sub`, `cmp`, `div_rem_small`, and `div_rem_upto_16`)
  and prove each with `proof_for_contract`. The loop then needs the bound that
  every competing PR assumed: each emitted digit is below 10, so `i` stays
  below `MAX_SIG_DIGITS`. We prove it from the scaling invariant the loop
  maintains (the remainder stays below the scale), as a contract on the loop
  body, never as an assumption on `i` or `d`.
- **Contracts in the required form.** Every precondition is a
  `#[safety::requires]`, every postcondition a `#[safety::ensures]`. No
  doc-comment contracts.

### Phase 4: CI and submission (week 7)

Wire the harnesses into `run-esbmc.sh` with an explicit allowlist and per-harness
timeouts, add `--force-malloc-success` where a harness allocates, and open the
solution PR. The PR body lists every assumption that remains, the way #596 does,
so reviewers do not have to find them.

## 6. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Solver cost of unstubbed `Big64x20` arithmetic | Dragon proofs do not finish | Compositional contracts in Phase 3; the Phase 1 gate catches it early |
| The digit bound is hard to prove without assuming the published correctness results for Dragon and Grisu | The proof keeps the compromise every other PR has | Prove the scaling invariant per iteration; if it cannot be done, say so in the PR and do not submit an assumption as a proof |
| The committee rejects ESBMC-only harnesses | No way to submit | Phase 0 settles this before harness work starts |
| #596 merges first (reviews go first come, first served) | The bounty goes to #596 | The unstubbed strategy proofs still add assurance #596 lacks; we offer them as a follow-up, or to the #596 author |
| esbmc#10936 or esbmc#10940 false alarms | Green proofs blocked | Phase 2 fixes |

## 7. Timeline

| Week | Work | Exit |
|---|---|---|
| 1 | Phase 0, start Phase 1 | Committee answer on #524; ESBMC release |
| 1 to 2 | Phase 1 measurements | Grisu gate passed or plan stopped |
| 2 to 4 | Phase 2 ESBMC fixes | No false alarm on the flt2dec harnesses |
| 3 to 6 | Phase 3 harnesses and contracts | All 12 functions proved in ESBMC |
| 7 | Phase 4 CI and PR | Solution PR open with CI green |

The estimate is 6 to 7 weeks if the Phase 1 gate passes cleanly and 9 weeks if
Dragon needs the compositional route. It does not include review rounds.

## 9. Phase 1 result (2026-11-24)

**The gate failed.** No unstubbed Grisu harness finished, even at four times the
900 s budget.

Setup: four harnesses on verify-rust-std main `1feeadd`, built with the CI's Kani pin
(`9834c96ce9`). They call `format_shortest_opt` and `format_exact_opt` on
`decode(kani::any::<f32 | f64>())`, restricted to finite values, with no stubs and no
assumption on a digit or an index. ESBMC v8.8 ran on the sweep machine (r7a.16xlarge),
one core and 15 GB per run.

| Round | Configuration | Shortest | Exact |
|---|---|---|---|
| 1 | BMC `--unwind 30` (multi-property), incremental BMC, k-induction; 900 s | all time out; per-claim solving at about 40 s a claim | symex does not finish (1024-byte buffer) |
| 2 | one combined solve, {Bitwuzla, Z3} x {plain, `--interval-analysis`}; 3,600 s | Bitwuzla times out; Z3 exceeds 15 GB | symex does not finish (17-byte buffer) |
| 3 | round 2, plus `round_and_weed`'s loop bounded at 10; 3,600 s | formula 3.4x smaller (14,932 VCCs, symex 40 s); both solvers time out | not run |

What we learned:

- In shortest mode, `round_and_weed`'s loop (`grisu.rs:422`) is inlined at every exit of
  the unwound digit loops. Its `debug_assert!` caps it at 8 iterations, so a bound of 10
  is sound with unwinding assertions on. That bound shrinks the formula, but the solve is
  still over an hour. What remains is the arithmetic every competing PR stubs or assumes:
  the 64x64-bit product in `Fp::mul` and the division by a symbolic `ten_kappa`.
- In exact mode, `round_up` calls `fill` at a symbolic offset with a symbolic length.
  ESBMC lowers that to the `__memset_impl` loop, which unwinds to the `--unwind` bound
  rather than to the object's size (esbmc#10991).
- `--show-loops` does not list the Rust loops of a Kani goto-binary, so they can only be
  bounded through `--unwindsetname` (esbmc#11008).

Following §8, the useful contributions are the ESBMC fixes (esbmc#10991, esbmc#11008)
and these measurements, which can go to the #596 author. A per-claim run would show
whether a few arithmetic claims carry the whole cost. If they do, Grisu could be done
compositionally, as §5 already plans for Dragon.

## 8. Decisions to take

1. **When to claim on #524.** Claim after the Phase 1 gate, not before. Three
   claims already sit on the issue; a fourth with no measurements adds nothing.
2. **Whether to compete with #596 or contribute to it.** I would compete only if
   Phase 1 shows the unstubbed Grisu proofs finish. If they do not, offering the
   ESBMC measurements to the #596 author is the more useful contribution.
3. **What counts as done for ESBMC** if the bounty goes elsewhere. The CI job
   running the flt2dec harnesses, and the ESBMC fixes from Phase 2, are worth
   keeping either way.
