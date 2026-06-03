# SV-COMP Issue Triage & Fix Plan

**Last updated:** 2026-06-02 (Pass 3 re-validation — see §7)
**Scope:** every open issue carrying the `SV-COMP` label in `esbmc/esbmc`, plus recently-closed
SV-COMP issues for context and de-duplication.
**Reference binary:** `build/src/esbmc/esbmc`, ESBMC 8.3.0, master `4f12db8419` (Pass 3);
prior passes on `95be952e8a`.
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
