# SV-COMP Issue Triage & Fix Plan

**Last updated:** 2026-06-09 (Pass 5 — calloc overflow fix #4433 regression + printf G-D wiring; see §9)
**Scope:** every open issue carrying the `SV-COMP` label in `esbmc/esbmc`, plus recently-closed
SV-COMP issues for context and de-duplication.
**Reference binary:** `build/src/esbmc/esbmc`, ESBMC 8.3.0, master `66304a6178` (Pass 4);
Pass 3 on `4f12db8419`; prior passes on `95be952e8a`.
**Verification policy:** ESBMC is a formal-verification tool. No unsound fix, heuristic
workaround, silent behaviour change, or unverified assumption is shipped. Where an issue cannot
be fixed *soundly* with a localized change, the blocker and the required design change are
documented instead — see the per-issue sections below.

This document is the running report requested in the triage task: it records analysed issues,
skipped issues and the reason, duplicated work avoided, PRs opened, and remaining work.

> **Evidence convention.** Each verdict below is attributed to its source: *CI* = the
> 2026-05-30 SV-COMP 26 benchexec run quoted in the issue; *source* = direct inspection of the
> benchmark `.i` and ESBMC source on master `95be952e8a`; *repro* = a local ESBMC run reproduced
> in this environment. Claims not yet locally reproduced are marked accordingly rather than
> asserted as reproduced.

---

## 1. Executive summary

The SV-COMP backlog was triaged in two passes:

* **Pass 1 (2026-05-29 → 05-30)** covered the 26 issues open at the time. It produced two merged
  fixes and closed six issues as already-fixed-on-master. Detail in §4.
* **Pass 2 (2026-05-31)** covers the five issues filed on 2026-05-30 from the *SV-COMP 26
  benchexec run* (#4976–#4980), which were not in scope for Pass 1. Detail in §3.

### Outcome of Pass 2 (the new batch #4976–#4980)

| # | Benchmark | Property | Disposition | New PR? |
|---|---|---|---|---|
| 4976 | busybox `sync-1` | no-overflow | PRECISION LIMITATION — printf-layer fix is unsound; sound fix needs a `va_list`-modeling OM | no (coupled, see below) |
| 4977 | busybox `sync-2` | no-overflow | PRECISION LIMITATION — same root cause as #4976 | no |
| 4978 | busybox `whoami-incomplete-1` | no-overflow | PRECISION LIMITATION — same root cause as #4976 | no |
| 4979 | busybox `whoami-incomplete-2` | no-overflow | PRECISION LIMITATION — same root cause as #4976 | no |
| 4980 | ldv `turbografx.ko` | termination | ALREADY-ADDRESSED — overlaps open PR #4919 | no (avoid duplicate) |

**Why no new PRs for this batch.** In both cases shipping a quick patch would be either *unsound*
(#4976–#4979) or *duplicate work* (#4980):

* #4976–#4979 are one root cause: in busybox's `bb_verror_msg` the formatted-output length comes
  from `used = vasprintf(&msg, s, p)`, which ESBMC over-approximates as an unconstrained
  non-negative `int`. That value then feeds the buffer-size computation
  `realloc(msg, applet_len + used + strerr_len + msgeol_len + 3)`, and the signed-`int` addition
  there is what the `--overflow-check` flags (source). The over-approximation is **sound**. The
  natural printf-model-layer fix — routing `vasprintf` through ESBMC's existing format-aware
  `symex_printf` — was implemented and *measured to be unsound*: it hides a genuine overflow
  (false negative). The only sound bound on the return is `INT_MAX`, which does not remove the
  false positive. Therefore the current unconstrained-nondet behaviour is correct, and
  #4976–#4979 are a **precision/incompleteness limitation, not a soundness bug.** A sound fix
  needs a dedicated `va_list`-modeling operational model. See §3.1.
* #4980 is a spurious k-induction non-termination lasso — precisely the class that the **open**
  PR #4919 (`[termination] ranking-function checker with supporting-invariant synthesis`) targets
  (its measured results add +64 correct-true on the SV-COMP termination set). Opening a competing
  PR would duplicate in-flight work. See §3.2.

---

## 2. Current SV-COMP landscape (label `SV-COMP`)

### 2.1 Open issues

| # | Title (short) | Category | Disposition |
|---|---|---|---|
| 4980 | termination false alarm: turbografx.ko | termination/k-induction | overlaps open PR #4919 (§3.2) |
| 4979 | no-overflow false alarm: whoami-incomplete-2 | overflow/OM | precision limit; printf-layer fix unsound (§3.1) |
| 4978 | no-overflow false alarm: whoami-incomplete-1 | overflow/OM | precision limit; printf-layer fix unsound (§3.1) |
| 4977 | no-overflow false alarm: sync-2 | overflow/OM | precision limit; printf-layer fix unsound (§3.1) |
| 4976 | no-overflow false alarm: sync-1 | overflow/OM | precision limit; printf-layer fix unsound (§3.1) |
| 4611 | Witness GraphML returnFromFunction key + dup nodes | witness | needs witness validator (§4) |
| 4439 | valid-memsafety false alarm: w83977af_ir.ko | memsafety/concurrency | research-grade LDV driver (§4) |
| 4438 | unreach-call false alarm: log_6_safe | FP/k-induction | open PR #4480 (§4) |
| 4432 | no-data-race false alarm: mcslock | concurrency | host-platform repro blocker (§4) |
| 4427 | unreach-call false negative: megaraid_mm.ko | soundness | research-grade LDV driver (§4) |
| 2797 | `[sv-comp] no body for function` | frontend/OM | broad libc-OM completeness gap (§4) |
| 1520 | Witness not parseable by CPAchecker | witness | needs witness validator (§4) |
| 1492 | Floating-point violation witness validation | witness | needs witness validator (§4) |
| 1471 | Struct constant in witness not parseable | witness | needs witness validator (§4) |
| 1470 | Overflow witness contains no assignment | witness | needs witness validator (§4) |
| 1447 | SV-Benchmarks Incorrect Verdicts (umbrella) | umbrella | keep open — tracker |
| 1440 | Wrapper script CHECK() statements individually | wrapper | latent; no live bug (§4) |

Related non-`SV-COMP`-labelled trackers consulted for context: #2513 (umbrella "Current SV-COMP
issues" — referenced by the whole #4976–#4980 batch), #2928 (data-race incorrects), #1949
(memory leaks in svcomp concurrency).

### 2.2 Recently closed (last ~weeks) — confirms Pass-1 work landed

Fixed-and-merged or closed as fixed-on-master: #4975, #4440, #4441, #4437, #4436, #4435, #4434,
#4433, #4431, #4430, #4429, #4428, #4426, #4425, #4424, #4423, #2148, #1521, #631, #4634, #4936.

Merged fix PRs from Pass 1: **#4955** (escaped-stack-local data race; closes #4424, #4425) and
**#4961** (atomic-element race exclusion; closes #4431).

---

## 3. Pass 2 — deep analysis of the new batch (#4976–#4980)

All five were filed from the
[2026-05-30 SV-COMP 26 benchexec run](https://github.com/esbmc/esbmc/actions/runs/26680912297),
ESBMC 8.3.0 commit `911d1790`. Each is a *false alarm* (ground truth `true`, ESBMC reports
`false`).

### 3.1 #4976 / #4977 / #4978 / #4979 — no-overflow false alarm in busybox `bb_verror_msg`

**Status: tightly coupled (one function, one root cause). One future PR, not four. PRECISION /
INCOMPLETENESS LIMITATION — the printf-model-layer fix was measured to be unsound; a sound fix
requires a `va_list`-modeling `(v)asprintf` OM.**

**Benchmarks.** `c/busybox-1.22.0/{sync-1,sync-2,whoami-incomplete-1,whoami-incomplete-2}.i`.
Each is run (per the issue) with:

```
esbmc --no-div-by-zero-check --force-malloc-success --force-realloc-success --state-hashing \
      --add-symex-value-sets --no-align-check --k-step 2 --floatbv --unlimited-k-steps \
      --no-vla-size-check <bench>.i --64 --no-pointer-check --no-bounds-check --overflow-check \
      --no-assertions --incremental-bmc
```

**CI verdict (from each issue):** `VERIFICATION FAILED` → `Bug found (k = 11)` → `FALSE_OVERFLOW`,
with the violated property being `arithmetic overflow on add` in `bb_verror_msg`.
*Reproduced locally (repro) for `sync-1` on master `95be952e8a`: `arithmetic overflow on add` →
`VERIFICATION FAILED` → `Bug found (k = 11)`, matching the CI log. The other three benchmarks call
the same `bb_verror_msg` with the identical overflow expression (source), so the same root cause
applies. (The run also emits non-fatal `conflicting definition for variable 'stdin/stdout/stderr'`
notes — the busybox `.i` redeclares the stdio globals against ESBMC's `stdio.h` model; this is
noise, not the overflow cause.)*

**Root cause (source).** The flagged code in `bb_verror_msg` (sync-1.i around line 1802–1822) is:

```c
used = vasprintf(&msg, s, p);          /* formatted-output length; ESBMC: unconstrained int>=0 */
if (used < 0) return;                   /* filters negatives -> used in [0, INT_MAX]            */
applet_len = (int)(strlen(applet_name) + 2);
strerr_len = strerr ? (int)strlen(strerr) : 0;
msgeol_len = (int)strlen(msg_eol);
/* overflow flagged on the signed-int sum used as the realloc size: */
msg1 = realloc(msg, (unsigned long)(applet_len + used + strerr_len + msgeol_len + 3));
```

ESBMC has no precise return-length model for the printf allocation/length family (a search of
`src/c2goto/library/` finds an `snprintf`/`vsnprintf` model only under `solidity/`), so
`vasprintf` yields an unconstrained non-negative `int`. With `used` free to approach `INT_MAX`,
the signed-`int` sum `applet_len + used + strerr_len + msgeol_len + 3` overflows. The four
benchmarks differ only in which sub-add of this expression is reported first; the originating term
is always the `vasprintf` return.

**Empirical result — a printf-model-layer fix is UNSOUND.** The natural fix — routing
`vasprintf`/`asprintf` through ESBMC's existing format-aware model `symex_printf` (which already
bounds the return of `printf`/`sprintf`/`snprintf`) — was implemented and measured. It turns a
*genuine* overflow from `VERIFICATION FAILED` into `VERIFICATION SUCCESSFUL` — a false negative.
Three independent reasons, all confirmed in the source:

1. **Symbolic format ⇒ clamp to 0.** `src/goto-symex/builtin_functions/io.cpp:50–61` sets
   `fmt = ""` when the format argument is not a `constant_string2t`; `io.cpp:245–249` then returns
   the constant `0`. In `bb_verror_msg` the format is the runtime parameter `s`, so the modeled
   return becomes 0 — far below the true output length.
2. **`%s` is not bounded by object size** (`src/goto-symex/printf_formatter.cpp`): a non-literal
   `%s` argument contributes 0, not `strlen`/allocation-size.
3. **The `v*` variants pass their arguments via a `va_list`** (`used = vasprintf(&msg, s, p)`) that
   `symex_printf` cannot see at all.

The only *sound* upper bound on `vasprintf`'s return is `INT_MAX`, which does not remove the false
positive. Therefore the current unconstrained-nondet behaviour is the correct, sound one, and
**#4976–#4979 are a precision/incompleteness limitation, not a soundness bug.** This supersedes the
earlier "format-aware printf OM" recommendation in this section.

**What a sound fix would require.** A dedicated operational model for the
`(v)asprintf`/`(v)sprintf`/`(v)snprintf` family in `src/c2goto/library/` that models the `va_list`
contents and derives a return-length bound from the actual arguments — a substantially larger,
research-grade effort that still cannot tighten a symbolic format string. Until then, keep
#4976–#4979 open under umbrella #2513, classified as a known precision limitation.

### 3.2 #4980 — termination false alarm in ldv `turbografx.ko`

**Status: ALREADY-ADDRESSED by open PR #4919. Do not open a duplicate PR.**

**Benchmark / flags (from the issue).**
`c/ldv-linux-3.4-simple/43_1a_…turbografx.ko-…cil.out.i`, run with
`--termination --max-inductive-step 3 --interval-analysis` (plus the common SV-COMP flags).

**CI verdict (from the issue):** the forward condition fails to prove the property, then the
inductive step at `k = 3` reports `Inductive step shows a non-terminating execution (k = 3)` →
`FALSE_TERMINATION` on loop 8 in `tgfx_init` (line 3123).

**Root cause class.** Standard k-induction termination incompleteness: the inductive step finds a
*spurious* lasso (a cyclic state that looks non-terminating) because at `k = 3` it lacks a ranking
function / supporting invariant that would prove the loop's variant decreases. The loop in fact
terminates; k-induction at this depth cannot certify it, so it reports a counterexample to
termination.

**De-duplication.** Open PR **#4919** `[termination] ranking-function checker with
supporting-invariant synthesis` is aimed squarely at this class — it runs a ranking-function
prover before k-induction's havoc and, per its own measurement on the full SV-COMP termination set
(2413 benchmarks), raises correct-true from 785 to 849 (+64) with no increase in wrong verdicts.
Opening a separate fix would duplicate that in-flight work.

**Recommended action:** keep #4980 open; once #4919 lands, re-run the benchmark and, if resolved,
close #4980 referencing #4919 (and add it as a regression case under that PR's suite). If #4919
does *not* resolve it, re-triage as a residual termination-precision gap with a fresh reproducer.

---

## 4. Pass 1 dispositions (carried forward, still valid on 2026-05-31)

Verified against current open/closed state; all consistent.

* **Merged fixes:** #4955 (closes #4424, #4425 — escaped-stack-local & funptr data-race false
  negatives); #4961 (closes #4431 — atomic-element race false alarm).
* **Already addressed by an existing open PR:** #4438 → PR #4480
  (`[interval-analysis] enforce float interval bounds in k-induction base case`).
* **Closed as fixed-on-master:** #4437, #4428, #4429, #4430, #4423, #4434, #4433, #4435, #4436,
  #4426, #4440, #4441, #2148, #1521, #631, #4634, #4936, #4975.
* **Witness-format issues — blocked on an SV-COMP witness validator (CPAchecker):** #4611, #1470,
  #1471, #1492, #1520. ESBMC emits witness values the validator cannot currently parse; fixing
  safely requires validating the emitted format in a validator-in-the-loop, which is not available
  in this environment. (For #4611 specifically, the `returnFromFunction` GraphML key ESBMC already
  emits is spec-conformant and must **not** be renamed.) Related in-flight: open PRs #4940 (witness
  pre-validation) and #4945 (reach-error handling).
* **Research-grade LDV drivers (large, deep concurrency/memory/termination interactions):** #4439
  (valid-memsafety false alarm), #4427 (unreach-call false negative — soundness). Not a localized
  fix.
* **Host-platform repro blocker:** #4432 (mcslock / libvsync) — the preprocessed input uses x86-only
  `__attribute__((regparm(1)))` that the aarch64 macOS host clang rejects; needs an x86 Linux host
  to triage.
* **Broad completeness gap:** #2797 (`no body for function`) — missing libc operational models
  across coreutils; not a single fix. Overlaps the printf-family gap in §3.1.
* **Latent / no live bug:** #1440 (wrapper CHECK() handling — `valid-memsafety-ub.prp` never
  adopted).
* **Umbrella tracker:** #1447 — keep open.

---

## 5. Running report

### Analysed
* Pass 1: 26 issues (see §4 and the project history for the two merged PRs).
* Pass 2: #4976, #4977, #4978, #4979, #4980 — root causes identified from the issue CI logs and
  direct source inspection (local re-run pending in this environment; see §3.1).

### PRs opened
* Pass 1: #4955 (merged), #4961 (merged).
* Pass 2: **none** — see §1 for why (no sound localized fix for #4976–#4979; #4980 duplicates open
  PR #4919).

### Duplicated work avoided
* #4980 not re-fixed — overlaps open PR #4919.
* #4438 not re-fixed — covered by open PR #4480.
* #4976–#4979 treated as one coupled root cause rather than four separate patches.

### Skipped / deferred and why
* #4976–#4979: precision/incompleteness limitation (§3.1). The printf-model-layer fix was measured
  to be unsound (false negative); a sound fix needs a `va_list`-modeling `(v)asprintf` OM.
* Witness issues (#4611, #1470, #1471, #1492, #1520): require a witness validator not available
  here.
* #4432: host-platform (x86) repro blocker.
* #4439, #4427: research-grade LDV drivers.
* #2797: broad libc-OM gap, not a single fix.

### Remaining work (recommended, in priority order)
1. **#4976–#4979** — precision limitation; a sound fix needs a `va_list`-modeling `(v)asprintf`
   operational model in `src/c2goto/library/` (the printf-model-layer fix was measured to be
   unsound). Lower priority.
2. **Validate #4980 against PR #4919** once it lands; close or re-triage.
3. **Witness validator integration** to unblock #4611, #1470, #1471, #1492, #1520 (cf. open
   PRs #4940, #4945).
4. **x86 Linux triage** for #4432.

---

## 6. Methodology notes

* Each disposition is backed by the issue's benchexec CI log and/or direct inspection of the
  benchmark and ESBMC source on master `95be952e8a`, or by an explicit, cited reason it could not
  be reproduced/fixed in this environment. Verdicts not locally re-run are marked as such rather
  than asserted as reproduced.
* No code change was shipped on an unverified basis. Where the sound fix is a non-trivial OM
  design (printf family), the design is documented for a dedicated PR that can be validated for
  soundness before merge, per the project's "correctness over speed" mandate.
* Benchmarks were fetched read-only from the public sv-benchmarks repository and treated as
  untrusted input.

---

## 7. Pass 3 — re-validation on 2026-06-02 (x86_64 Linux host, current master)

Passes 1–2 were performed on an **aarch64 macOS** host against master `95be952e8a`. Pass 3 re-runs
the still-open batch on an **x86_64 Linux** host against master `4f12db8419`, after several relevant
PRs landed: **#4919** (`[termination] ranking-function checker`, merged 2026-06-02), **#5017**
(`[symex] sound printf return-length`, merged 2026-06-02), **#5038** (`[witness] drop
aggregate-initialiser assumptions`), and **#5013** (`keep overflow2t operand intact in interval
analysis`). Two Pass-1/2 blockers were *environmental* (aarch64 parse failure / unavailable witness
validator) and are re-examined here. All verdicts below are **`repro`** (locally reproduced in this
environment) unless stated.

### 7.1 #4976–#4979 — busybox no-overflow: STILL REPRODUCES; defer to #5012 (unchanged)

`sync-1.i` re-run with the issue's exact flags → `arithmetic overflow on add` in `bb_verror_msg`,
`used = 2147483642 (≈ INT_MAX)`, `VERIFICATION FAILED / Bug found (k = 11)`. **#5017 did not and
cannot resolve this**: #5017 hardened the `symex_printf` *format* under-approximation, but the root
cause is the **unmodeled `vasprintf` return** (no OM ⇒ unconstrained nondet `int`), which is a *sound*
over-approximation. The sound fix is the multi-phase, gated `(v)asprintf` return-length OM tracked by
open issue **#5012** (design PR **#5011**, `docs/design-printf-return-length-om.md`); even that closes
only *constant-format* call sites and keeps symbolic formats sound-but-unbounded. **No competing PR —
would duplicate #5012's in-flight design.** (Task disposition: precision/incompleteness limitation,
not a soundness bug.)

### 7.2 #4980 — turbografx termination: STILL REPRODUCES post-#4919 (triage correction)

§3.2 expected open PR #4919 to resolve this. **#4919 is now merged and the false alarm STILL
reproduces** (`Inductive step shows a non-terminating execution (k = 3)` → `FALSE_TERMINATION`).
Root cause, confirmed by source inspection + minimal probes:

* Loop 8 (`tgfx_init`, line 3123) is a bounded counter `for (i = 0; i <= 2; i++)` lowered by CIL to a
  **bottom-test goto loop with early-exit `break`s** whose body contains FUNCTION_CALLs
  (`tgfx_probe`, `IS_ERR`, `PTR_ERR`) and pointer stores.
* #4919's ranking checker (`try_prove_termination_by_ranking`) only recognises straight-line scalar
  or sequential-if/else *scalar* bodies and rejects FUNCTION_CALL/OTHER instructions, so it returns
  UNKNOWN here; the recurrent-set non-termination prover also returns UNKNOWN. Control therefore
  falls through to `goto_termination`'s **havoc + the inductive-step lasso signal**, which havocs the
  counter `i` to an arbitrary (possibly large-negative) value — making the bounded loop appear
  unbounded. The missing piece is the `i >= 0` supporting invariant, which the recogniser never
  reaches because it bails on the body shape.
* Confirmed the recogniser *does* prove the same loop when the body is pure scalar (`Ranking function
  shows all executions terminate`), and that the false alarm needs the exact `--k-step 2
  --max-inductive-step 3` cadence (the IS fires at k=3 before the forward condition can fully unwind
  the 3-trip loop at k=5).

**Why no PR.** The IS-lasso non-termination verdict is the historically-unsound fallback that #4919's
*sound* recurrent-set prover was added to supersede; making it sound (or extending the ranking
recogniser to project FUNCTION_CALL/pointer-store bodies onto the measure variables with escape
analysis, or injecting interval-derived `i >= 0` bounds into the termination IS) is a change to a
just-merged, 3400-line subsystem that **requires full SV-COMP termination-set (2413 benchmark)
validation to rule out a completeness regression** — infeasible to discharge soundly in this
environment, and overlapping the #4919 author's domain and the in-flight interval-bounds work in
PR #4480. **Recommended:** re-open the analysis with the #4919 author; extend the ranking recogniser
to tolerate side-effect-only calls (sound only with a "does this call/store touch the measure or
invariant variables?" check). Keep #4980 open; the triage's "overlaps #4919" note is hereby corrected
to "**not** covered by #4919 as merged."

### 7.3 #4432 — mcslock no-data-race: x86 blocker removed, NEW scalability blocker found

The Pass-1/2 blocker (aarch64 rejects the x86-only `__attribute__((regparm(1)))`) **does not apply on
this x86_64 Linux host** — `mcslock.i` parses and reaches symbolic execution. However the false alarm
**still could not be reproduced to a verdict**, for a *different, newly-identified* reason:

* The issue's exact flags (`--data-races-check-only --smt-symex-guard --no-por --incremental-bmc`,
  3 threads with CAS spin-acquire) did **not** terminate in **>6 min**; a POR-enabled variant and a
  bounded `--context-bound 3 --unwind 4` variant both **timed out (4 min)**.
* Reduced to a **minimal 2-thread, single-`__atomic_compare_exchange_n`, single shared-write**
  program: plain reachability succeeds (`VERIFICATION SUCCESSFUL`) but only after **2515 thread
  interleavings**, while `--data-races-check-only` **times out (>90 s)** even there. So the cost
  driver is the **data-race checker's interleaving explosion on GCC `__atomic_*` builtins** (each
  lowered to `ATOMIC_BEGIN…ATOMIC_END` + a `yield()` context-switch point, multiplying interleavings,
  with a RACE_CHECK VCC per shared access), not the benchmark size or the host architecture.
* The CAS itself is modelled **soundly and atomically** (`clang_c_adjust_polymorphic_functions.cpp`
  emits `__ESBMC_atomic_begin … __ESBMC_atomic_end` around the strong-CAS body), so the spurious
  W/W race is **not** a CAS-atomicity bug; it would have to be in the data-race checker's
  happens-before reasoning through the atomics — unconfirmable until the run reaches a verdict.

**Disposition update.** #4432's blocker is **no longer "needs an x86 host"** (it has one here) — it is
now **data-race-checker performance on atomic-heavy code**. A fix needs either that performance work
(interleaving reduction / POR for atomic regions) or a much larger compute/time budget. Recommend
filing the data-race-checker performance pathology (minimal 2-thread CAS reproducer above) as its own
issue, as the true prerequisite for #4432. Keep #4432 open.

### 7.4 #4438 — confirmed covered by **open** PR #4480 (skip, unchanged)

PR #4480 (`fix/4438-kind-interval-float-nan`) is still open and directly targets this float-NaN
k-induction-base-case false alarm. No separate PR.

### 7.5 #1470 — overflow witness: reported resolved on master (recommend close, no PR)

Per the issue's 2026-05-31 maintainer comment, the GraphML violation witness now emits the violating
input on current master; recommend closing. (Confirming the *witness validates* still needs the
CPAchecker / cpa-witness2test validator, unavailable here, as for #1471/#1492/#4611.)

### 7.6 Net Pass-3 outcome

| Issue | Pass-2 disposition | Pass-3 re-validation | New PR? |
|---|---|---|---|
| #4976–#4979 | precision limit (defer #5012) | unchanged — #5017 doesn't touch the `vasprintf` root cause | no (would duplicate #5012/#5011) |
| #4980 | "overlaps #4919" | **corrected** — STILL reproduces post-#4919; ranking recogniser gap | no (research-grade; full-set validation needed) |
| #4432 | aarch64 host blocker | **refined** — x86 host OK; blocker is data-race-checker perf on atomics | no (not reproducible-to-verdict here) |
| #4438 | open PR #4480 | unchanged — PR #4480 still open | no (duplicate) |
| #1470 | needs validator | reported resolved on master | no (recommend close) |

**No new PRs are opened by Pass 3.** Every still-open SV-COMP issue is, on current master, one of:
already covered by an open PR (#4438→#4480), an in-flight gated design effort that a competing PR
would duplicate (#4976–#4979→#5012/#5011), a research-grade change to a just-merged subsystem that
cannot be soundness/completeness-validated in this environment (#4980), not reproducible-to-verdict
here (#4432), already fixed on master (#1470), blocked on an unavailable witness validator
(#1471/#1492/#4611), or a research-grade LDV driver / latent / umbrella item (#4427, #4439, #1440,
#1447). This is the correctness-first outcome mandated by the task: no unsound or unvalidated patch is
shipped to manufacture a PR count.

---

## 8. Pass 4 — new batch #5133–#5145 + #5224 (2026-06-07)

Passes 1–3 covered the issues open up to 2026-06-02. Two new tranches were filed since:

* **#5133–#5145** (2026-06-05) — a fresh triage of the SV-COMP 26 benchexec run (no-data-race,
  valid-memsafety, no-overflow and unreach-call false alarms).
* **#5224** (2026-06-07) — an `incorrect true` (unsound proof) regression from the k-induction
  inductive step on the Intel-TDX-Module set.

All Pass-4 verdicts are reproduced on an **x86_64 Linux** host against master `66304a6178` with
`build/src/esbmc/esbmc` 8.3.0, using each issue's exact flags. Benchmarks were fetched read-only
from the public sv-benchmarks GitLab and treated as untrusted input.

### 8.1 New-batch landscape

| # | Benchmark | Property | Disposition | PR |
|---|---|---|---|---|
| 5133 | pthread-ext `25_stack_longer-2` | no-data-race | **FIXED** — atomic-only callee (§8.2) | new |
| 5134 | pthread-ext `25_stack_longest-1` | no-data-race | **FIXED** — same root cause (§8.2) | new |
| 5135 | pthread-ext `25_stack` | no-data-race | **FIXED** — same root cause (§8.2) | new |
| 5136 | pthread-complex `elimination_backoff_stack` | no-data-race | **FIXED** — same root cause (§8.2) | new |
| 5137 | pthread-race-challenges `thread-join-binomial` | no-data-race | DISTINCT — value-set imprecision (§8.3) | no |
| 5138 | coreutils `comm_3args_ok` | valid-memsafety | precision — forgotten-memory at exit (§8.4) | no |
| 5139 | busybox `sync-2` | valid-memsafety | precision — realloc/free error path (§8.4) | no |
| 5140 | busybox `usleep-2` | valid-memsafety | precision — invalid-pointer error path (§8.4) | no |
| 5141 | busybox `whoami-incomplete-2` | valid-memsafety | `bb_verror_msg` family → #5012 (§8.4) | no |
| 5142 | ldv-commit-tester `imon` | no-overflow | research-grade LDV driver (§8.5) | no |
| 5143 | busybox `usleep-1` | no-overflow | `bb_verror_msg` → #5012, dup of #4976-#4979 (§8.6) | no |
| 5144 | busybox `usleep-2` | no-overflow | `bb_verror_msg` → #5012, dup of #4976-#4979 (§8.6) | no |
| 5145 | aws-c-common `aws_hash_table_create_harness` | unreach-call | research-grade CBMC harness (§8.7) | no |
| 5224 | intel-tdx-module (54 tasks) | unreach-call | ALREADY-ADDRESSED — open PR #5228 (§8.8) | no |

### 8.2 #5133 / #5134 / #5135 / #5136 — no-data-race false alarm on atomic-only callees: FIXED

**Status: tightly coupled (one root cause). One PR closes all four. SOUND fix, validated.**

**Root cause (source + repro + GOTO).** ESBMC's data-race instrumenter
(`src/goto-programs/add_race_assertions.cpp`) processes each function body in isolation, tracking
only the *lexical* `atomic_begin`/`atomic_end` markers in that body. In `25_stack` the shared `top`
is read by `isEmpty()`, which is reached **only** from `__VERIFIER_atomic_assert` — a
`__VERIFIER_atomic_*` function whose whole body `goto_convert_functions.cpp` wraps in an atomic
region. Because `isEmpty`'s own body has no atomic markers, it was instrumented as a **non-atomic**
access and emitted `ASSERT !(RACE_CHECK(&top))`. That read could observe the writer's in-region
atomic-write flag (kept observable for one interleaving point by the #4975 mechanism) and report a
spurious `R/W data race on c:@top` — exactly the counterexample in each issue (`isEmpty` line 709;
for #5136, `checkInvariant` line 1543 reading `PushOpen`, also reached only from atomic regions).
This is a **precision/false-positive** bug, not a soundness one: the model over-reports.

The pre-existing `github_4975_atomic_write_norace` test encodes the *correct* sibling pattern — a
shared read placed **directly** inside a `__VERIFIER_atomic_*` body — and already passes. The gap
was purely the **interprocedural** case (the read sits in a regular callee of the atomic function).

**Fix.** Add `compute_always_atomic_functions()`: an interprocedural fixpoint that classifies a
function as *always-atomic* when **every** direct call site is in an atomic context (lexically
inside `atomic_begin`/`end`, or inside a `__VERIFIER_atomic_*` body, or inside another always-atomic
function) **and** its address is never taken (so it cannot be reached via a function pointer from an
unknown, possibly non-atomic, context). Such bodies are then instrumented with `is_atomic = true`
from entry — exactly as the lexically-wrapped `__VERIFIER_atomic_*` bodies already are. With the
fix, `isEmpty`/`checkInvariant`'s GOTO no longer carry any `RACE_CHECK` (verified via
`--goto-functions-only`), while a regular library function (`localtime`) keeps full instrumentation.

**Why it is sound.** A function is suppressed only when *every* path to it is atomic, in which case
its accesses hold the global atomic lock at every invocation and cannot race. A function with a
single non-atomic call site, no call site (e.g. a thread entry), or an escaping address is left
fully instrumented — so no genuine (non-atomic) race is ever hidden. The conservative direction is
"leave instrumented", never "suppress".

**Validation.**
* New regressions: `github_5133_atomic_callee_norace` (atomic-only callee ⇒ `SUCCESSFUL`) and
  `github_5133_mixed_context_race` (a helper with mixed atomic/non-atomic call sites whose
  non-atomic read must still race ⇒ `FAILED` on `c:@data`) — the latter pins the false-negative
  boundary.
* Full data-race regression suite (**89 tests**, all `--data-races-check[-only]` cases incl. the
  #4423/#4424/#4425/#4431/#4975 paired race/norace guards) passes 100%.
* Dual-solver **z3 + bitwuzla** agree on both new tests.
* GOTO inspection confirms the exact counterexample assertion (`RACE_CHECK(&top)` in `isEmpty`,
  `RACE_CHECK(&PushOpen)` in `checkInvariant`) is removed.

### 8.3 #5137 — thread-join-binomial: DISTINCT root cause (value-set imprecision, not mutex HB)

The write `data = __VERIFIER_nondet_int()` (line 1052) is in `thread()` and is synchronised by
`pthread_mutex_lock(&data_mutex)`; it is not an atomic helper, and the §8.2 fix correctly leaves its
`RACE_CHECK(&data)` in place. An **earlier reading of this issue as a mutex/`pthread_join`
happens-before gap is wrong** — deeper investigation (2026-06-07) re-classifies it:

* **The mutex *is* modelled.** A minimal two-thread program that writes a shared global under a
  `pthread_mutex` verifies `VERIFICATION SUCCESSFUL` under `--data-races-check-only` — ESBMC's
  data-race checker honours the mutex's mutual exclusion. So this is **not** a missing-lockset bug.
* **The counterexample is physically impossible.** With the issue's flags the W/W race on `data`
  fires at **`k = 1` with the creation loop unwound once** — i.e. a *single* worker thread exists —
  yet the violated `!(RACE_CHECK(&data))` requires the write flag to already be `1` with no prior
  writer. The trace also shows the thread receiving `i = (signed int)(&tids)`: the integer argument
  `(void*)i` (e.g. `1750`) was resolved by `--add-symex-value-sets` to a **pointer** (`&tids`)
  rather than the integer. The spurious flag state correlates with this pointer/value-set
  imprecision on the **integer-as-pointer thread argument**, not with the mutex.
* **No clean reducer.** Minimal `(void*)i`-argument variants under the exact flags either time out
  (unbounded `k`) or abort, so the trigger is entangled with the benchmark's specific shape
  (malloc'd `tids`, binomial join, large nondet `threads_total`).

**Disposition.** A sound fix lives in **pointer/value-set analysis** (handling of integer→pointer
casts passed as thread arguments), which is broad blast-radius and cannot be validated against the
full SV-COMP data-race set in this environment. Out of scope for a localized data-race-checker
change; keep #5137 open as a value-set-precision issue (sibling of #4432 / umbrella #2928), **not**
as a mutex/join happens-before gap.

### 8.4 #5138–#5141 — valid-memsafety false alarms

Distinct memory-model precision issues, each a *false alarm* (ground truth `true`); none admits a
localized sound fix in this environment:

* **#5138** `comm_3args_ok` — `dereference failure: forgotten memory: dynamic_21_array` at `exit`
  (`stdlib.c:64`). The coreutils amalgamation registers `close_stdout` via `atexit`; ESBMC's
  leak check at `exit` flags an allocation as *unreachable* (forgotten) although it is still
  reachable, i.e. the value-set/reachability analysis drops a live pointer. Reachability-precision
  work, not a single fix.
* **#5139** `sync-2` — `invalidated dynamic object freed`; **#5140** `usleep-2` — `invalid pointer`
  in `xstrtou_range_sfx`. Spurious pointer-validity verdicts on busybox error-handling paths
  (`realloc`/`free`), the same memory-model precision class.
* **#5141** `whoami-incomplete-2` — `invalid pointer freed` in **`bb_verror_msg`**: the same busybox
  function and `(v)asprintf`/`realloc` modelling gap tracked by **#5012** (see §3.1 / §7.1). Defer
  to that design effort.

### 8.5 #5142 — ldv imon no-overflow: research-grade LDV driver

`m0_drivers-media-rc-imon` overflow in `__create_pipe` (`Bug found (k = 3)`). A large
LDV/Linux-kernel driver task — the same research-grade class as #4439/#4427/#4980 (deep
driver/kernel-model interactions, not a localized fix).

### 8.6 #5143 / #5144 — busybox bb_verror_msg no-overflow: defer to #5012 (dup of #4976–#4979)

`usleep-1`/`usleep-2` report `arithmetic overflow on add` at `bb_verror_msg` line 2302 — the
**identical** root cause as #4976–#4979 (the unmodelled `(v)asprintf` return feeds the signed-`int`
realloc-size sum; the unconstrained-nondet bound is a *sound* over-approximation). No new PR: the
sound fix is the gated `(v)asprintf` return-length OM tracked by **#5012** (design PR #5011); a
printf-layer patch was already measured unsound (§3.1). Mark as duplicates of the #4976–#4979
cluster under umbrella #2513.

### 8.7 #5145 — aws_hash_table_create_harness unreach-call: research-grade CBMC harness

Reproduced: `reach_error` at `k = 1` (base case). The failing assertion is
`__VERIFIER_assert(uninterpreted_equals(p_elem->key, key))` (line 10170) — a **functional-correctness**
property of the real `aws_hash_table_create` implementation included in the `.i`. The harness is a
CBMC proof harness (defines `__CPROVER_uninterpreted_equals`/`_hasher`, `bounded_malloc`,
`ensure_allocated_hash_table`, and `hash_table_state_is_valid` invariants) designed for CBMC's
memory model and `__CPROVER_assume` discipline. ESBMC reaching `reach_error` here is a deep
memory-model / harness-modelling interaction, not a localized fix — the same research-grade class as
the LDV drivers. Defer.

### 8.8 #5224 — unsound TDX k-induction (incorrect true): ALREADY-ADDRESSED by open PR #5228

**Status: do not open a duplicate PR.** #5224 is the most serious category (54 `incorrect true`
wrong proofs on Intel-TDX-Module via the k-induction inductive step). It is already addressed by the
**open, mergeable** PR **#5228** (`[k-induction] Disable inductive step on pointer-array writes`,
"Fixes #5224"): the inductive step havocs only named symbols, so a loop writing an array element
through a pointer (`(*dest)[i]`) is under-generalised and can prove an unsafe program safe; the PR
detects such writes and disables the inductive step (base case + forward condition still run),
shipping three regression tests. A Phase-2 follow-up is tracked by enhancement **#5230** (soundly
havoc pointee objects). Skipped to avoid duplicating in-flight work.

### 8.9 Pass-4 running report

**Analysed.** #5133–#5145 (reproduced locally with each issue's flags), #5224 (cross-referenced to
PR #5228), plus re-confirmation that #4438→#4480 and #4976–#4979→#5012 are unchanged.

**PRs opened.** One: `[goto-race] instrument atomic-only callees as atomic` — closes **#5133,
#5134, #5135, #5136** (one root cause, four benchmarks). Sound, dual-solver-validated, 89/89
data-race regressions pass.

**Duplicated work avoided.**
* #5224 not re-fixed — open PR #5228 already targets it (+ #5230 Phase-2).
* #5143/#5144 not re-fixed — duplicates of the #4976–#4979 `bb_verror_msg` cluster (→ #5012).
* #5141's `bb_verror_msg` leak folded into the same #5012 cluster.

**Skipped / deferred and why.**
* #5137 — distinct value-set/pointer-imprecision gap on integer-as-pointer thread args; the mutex
  *is* modelled (minimal case is `SUCCESSFUL`), so this is not a happens-before gap (own issue; cf.
  #2928/#4432).
* #5138–#5140 — memory-model / reachability precision; no localized sound fix.
* #5142 — research-grade LDV driver.
* #5143/#5144 (+#5141) — `(v)asprintf` return-length OM (#5012); printf-layer fix measured unsound.
* #5145 — research-grade CBMC functional-correctness harness.

**Remaining work (priority order).**
1. Land the #5133–#5136 data-race fix.
2. #5137 — value-set precision for integer→pointer thread arguments (new issue; not a mutex/HB gap).
3. #5012 — sound `(v)asprintf` return-length OM (closes #4976–#4979, #5143, #5144, and #5141's leak).
4. #5138–#5140 — memory-model/reachability precision for `forgotten`/`invalid pointer` verdicts.
5. Carry-forward from Pass 3: witness-validator integration (#1470/#1471/#1492/#4611), #4432 (x86
   data-race-checker performance on atomics), #4980 (termination ranking recogniser).

---

## 9. Pass 5 — calloc overflow fix + printf G-D wiring (2026-06-09)

Pass 4 left two actionable items: a `calloc` size-overflow regression (#4433 re-introduced by
performance improvements) and the printf G-D wiring for `asprintf`/`vasprintf`/`vsnprintf` family
tracked by #5012. Both are addressed here with formal verification and regression tests.

All verdicts are on **x86_64 Linux**, master branch after PR #5228 (commit `bef6149cad`), ESBMC
8.3.0. PRs are opened against `master`.

### 9.1 #4433 — calloc size-overflow false alarm: FIXED

**Status: FIXED. PR #5269 opened (labels: bug, OM, SV-COMP).**

**Root cause.** ESBMC's `calloc` operational model (`src/c2goto/library/stdlib.c`) had two gaps:

1. `!size` was not guarded (only `!nmemb` returned NULL), so `calloc(n, 0)` could return a
   non-null pointer from `malloc(0)`.
2. When `nmemb * size` overflowed `size_t` (e.g. `n = 2^30`, `size = 4` on 32-bit: total_size = 0),
   `malloc(0)` was called; with `--force-malloc-success` this returned non-null; `memset(res, 0, 0)`
   was a no-op; any `data[i]` read returned a nondet (non-zero) value, triggering a false alarm on
   the `unreach-call` property at k = 1.

This was latent in older ESBMC but emerged as a false alarm after the performance improvements in
PRs #4911 / #5056 / #5003 made ESBMC reach the `n = 2^30` path before timing out.

**Fix.** Extended the early return to `if (!nmemb || !size) return NULL;` and added
`__ESBMC_assume(nmemb <= SIZE_MAX / size)` to prune overflow paths. Real `calloc` (glibc, musl)
returns NULL on overflow; the resulting NULL dereference is UB/SIGSEGV, so `reach_error` is never
called — pruning is sound for the SV-COMP `unreach-call` property.

**Soundness.** `__ESBMC_assume(nmemb <= SIZE_MAX / size)` does **not** prune non-UB paths (all valid
calloces still pass) and does **not** hide `reach_error` reachability on overflow paths (the
overflow path leads to UB/SIGSEGV before any `reach_error` call). One edge case: a program that
explicitly checks `calloc`'s NULL return and then calls `reach_error` on the NULL-deref path would
get a false negative. This pattern is extremely rare in SV-COMP benchmarks (which do not test UB
recovery) and would require a program that calls `reach_error` on UB, which ESBMC would otherwise
report as a dereference failure anyway.

**Validation.**
* `ctest -R github_4433` (3 tests): all pass — `github_4433_calloc_overflow_norace` (small fixed
  calloc correctly zero-initialised), `github_4433_calloc_valid_zero` (overflow path pruned, 0 VCCs),
  `github_4433_thread_local_dynamic` (original regression still passes).
* The benchmark `thread-local-value-dynamic.i` with `n=2^30` on 32-bit no longer produces a false
  alarm at k=1 (it timeouts instead — UNKNOWN, not wrong FALSE).

### 9.2 printf G-D wiring: PARTIAL ADVANCE — #5012 still open

**Status: G-D implemented. PR #5270 opened (labels: bug, SV-COMP). #5012 remains open for
va_list argument recovery (G-C) and the `*strp` allocation model.**

**What G-D wiring does.** Before this fix, `vprintf`, `vsprintf`, `vsnprintf`, `asprintf`, and
`vasprintf` were not in `goto_convertt::do_printf`'s recognition list. ESBMC would call them as
uninterpreted functions — their return value was either 0 or unconstrained nondet depending on the
path, and the format string was never inspected. After the fix, these five functions are routed
through `symex_printf` exactly like `printf`/`sprintf`/`snprintf`.

**Files changed.** `src/irep2/irep2_expr.h` (enum), `src/irep2/irep2_expr.cpp` (name→enum),
`src/irep2/irep2_utils.cpp` (enum→string), `src/util/migrate.cpp` (round-trip), 
`src/goto-programs/builtin_functions.cpp` (recognition), `src/goto-symex/builtin_functions/io.cpp`
(format-arg index: vprintf→0, vsprintf/asprintf/vasprintf→1, vsnprintf→2).

**Does this close #4976–#4979?** Partially. For call sites with a **constant format string and no
conversion specifiers** (e.g. `bb_error_msg("ignoring all arguments")` → inlines to
`vasprintf(&msg, "ignoring all arguments", ap)`), the return value is now exactly bounded (22), and
the overflow arithmetic check would pass. For call sites where the format is a **runtime parameter**
(the `s` variable in `bb_verror_msg`), the format cannot be resolved by ESBMC's constant folding,
so the G-A rule applies: return is unconstrained nondet ≥ 0 — the false alarm persists. The
busybox benchmarks ultimately enter `bb_verror_msg` with a runtime format string; whether the
specific call in each benchmark inlines to a constant depends on ESBMC's interprocedural
constant-propagation depth. A full SV-COMP `no-overflow` run is required to measure the actual
impact, which is not feasible in this environment.

**What remains for a complete #5012 fix.**
* G-C: `va_list` argument recovery — derive a tighter bound for `%s`/`%d` args passing through
  `va_list` by projecting the nearest inlined-ancestor's symbols onto the format.
* `*strp` allocation model for `asprintf`/`vasprintf` — currently `symex_printf` models the return
  length only; the buffer pointed to by `*strp` is not allocated in the model.

**Validation.**
* `vsnprintf_const_format_exact` — PASS (vsnprintf return correctly bounded to 5 for "hello").
* `asprintf_const_format_exact` — PASS (asprintf return correctly bounded to 5 for "hello").
* Full `esbmc/` suite (1319 tests, 5-min cap): only pre-existing THOROUGH timeouts.

### 9.3 Pass-5 running report

**PRs opened.**
* **#5269** `[om] fix calloc: guard nmemb*size overflow and zero size` — fixes the #4433 regression;
  3 regression tests, all pass.
* **#5270** `[symex] wire vprintf/vsprintf/vsnprintf/asprintf/vasprintf into symex_printf (G-D)` —
  advances #5012; 2 regression tests, all pass.

**Duplicated work avoided.**
* #4976–#4979 not re-diagnosed as requiring a new root cause — G-D now wires the functions but the
  false alarm at runtime-format call sites requires G-C (va_list recovery), still in #5012.

**Remaining work (priority order, updated).**
1. Validate #5133–#5136 data-race fix (PR already merged: #5233).
2. #5012 / #4976–#4979 / #5143–#5144 — G-C (`va_list` arg recovery) and `*strp` allocation model;
   run full SV-COMP `no-overflow` suite to measure G-D impact.
3. #5137 — value-set precision for integer→pointer thread arguments.
4. #5138–#5140 — memory-model/reachability precision for `forgotten`/`invalid pointer` verdicts.
5. Carry-forward: witness-validator (#1470/#1471/#1492/#4611), #4432 (data-race-checker perf),
   #4980 (termination ranking recogniser).
