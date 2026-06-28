# SV-COMP Issue Triage & Fix Plan

**Last updated:** 2026-06-26 (Pass 9 — reconcile four closures since Pass 8 (#5395/#5399/#5400 and the
new #5593, an Intel-TDX `havoc_memory` recurrence of #5224 fixed by merged PR #5602); triage the new
live issue #5565 and **fix its first post-#5564 residual** — an unmodeled `getopt_long` left `optarg` an
invalid pointer, a sound localized OM fix; see §13. Pass 8 in §12, Pass 7 in §11)
**Scope:** every open issue carrying the `SV-COMP` label in `esbmc/esbmc`, plus recently-closed
SV-COMP issues for context and de-duplication.
**Reference binary:** `build/src/esbmc/esbmc`, ESBMC 8.3.0. Pass 8 on an **x86_64 Linux** host with
a binary built from branch `feat/python-object-heap-lifetime` (82 commits ahead of master `5046dc72a0`,
all Python-frontend- and FP-solver-scoped; the C integer/pointer verdicts triaged here are
unaffected — the one symex change is gated on the Python-only `__ESBMC_new_object` intrinsic and does
not touch `add_memory_leak_checks`). Pass 6/7 on master `74da7c0400` (aarch64 macOS); Pass 5 on
`bef6149cad`; Pass 4 on `66304a6178`; Pass 3 on `4f12db8419`; prior passes on `95be952e8a`.
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
| 4976 | busybox `sync-1` | no-overflow | RESOLVED — sound return-length model shipped (#5017/#5270/#5278/#5279); CLOSED 06-09. *Pass-2 view below was superseded; see §3.1 banner.* | yes (later) |
| 4977 | busybox `sync-2` | no-overflow | RESOLVED — same fix; CLOSED 06-09 (§3.1 banner) | yes (later) |
| 4978 | busybox `whoami-incomplete-1` | no-overflow | RESOLVED — same fix; CLOSED 06-09 (§3.1 banner) | yes (later) |
| 4979 | busybox `whoami-incomplete-2` | no-overflow | RESOLVED — same fix; CLOSED 06-09 (§3.1 banner) | yes (later) |
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
| 4979 | no-overflow false alarm: whoami-incomplete-2 | overflow/OM | RESOLVED — sound model shipped, CLOSED 06-09 (§3.1 banner) |
| 4978 | no-overflow false alarm: whoami-incomplete-1 | overflow/OM | RESOLVED — sound model shipped, CLOSED 06-09 (§3.1 banner) |
| 4977 | no-overflow false alarm: sync-2 | overflow/OM | RESOLVED — sound model shipped, CLOSED 06-09 (§3.1 banner) |
| 4976 | no-overflow false alarm: sync-1 | overflow/OM | RESOLVED — sound model shipped, CLOSED 06-09 (§3.1 banner) |
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

> **RESOLVED (2026-06-26 reconciliation) — supersedes the Pass-2 conclusion below.** A *sound*
> return-length model for the printf/(v)asprintf family was designed
> (`docs/design-printf-return-length-om.md`) and shipped: **PR #5017** (G-A, non-constant format
> no longer under-approximates), **PR #5270** (G-D, wire `(v)asprintf`/`(v)sprintf`/`vsnprintf`/
> `vprintf` into `symex_printf`), **PR #5278** (cap the unbounded return at `INT_MAX/2`), and
> **PR #5279** (model the `*strp` heap allocation). The key realisation that unblocked it: the
> earlier "unsound" measurement (PR #5009) was of a *naive* routing that clamped a symbolic
> format to length 0; the shipped model instead caps the return at a sound over-approximation
> (`INT_MAX/2`), which removes the false overflow *without* masking a genuine one. **All four
> issues #4976–#4979 were closed on 2026-06-09.** Regression coverage:
> `regression/esbmc/asprintf_const_format_exact`, `vasprintf_*`, and the reduced reproducers
> `github_4977`/`github_4978`/`github_4979` (+ `github_4978_fail`) — all green; the reproducers
> converge to `VERIFICATION SUCCESSFUL` at k=11 and the `_fail` variant still `FAILED`. Only the
> Phase-3 G-C precision pass (`va_list` `%s` recovery, symbolic-format tightening) remains open,
> tracked by #5012. The historical Pass-2 analysis is retained below for the record.

**Status (Pass 2, superseded — see banner above): tightly coupled (one function, one root cause).
One future PR, not four. PRECISION / INCOMPLETENESS LIMITATION — the printf-model-layer fix was
measured to be unsound; a sound fix requires a `va_list`-modeling `(v)asprintf` OM.**

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

---

## 10. Pass 6 — reconcile with master (2026-06-17)

Pass 5 left five actionable threads (its §9.3 list). Between 2026-06-09 and 2026-06-17 most of them
landed on master, and the deep aws-hash investigation (#5287/#5145) was carried to a conclusion. This
pass reconciles the report with the current open/closed state, re-validates the still-open set, and
records the one new empirical result that matters: **#5145 still reproduces on current master even
after the UF-modelling fixes** (#5364, #5382), confirming its residual is the documented memory-model
limitation, not the `__CPROVER_uninterpreted_*` gap.

Reference: master `74da7c0400`, ESBMC 8.3.0, aarch64 macOS host. Verdicts marked **repro** were
re-run locally this pass; closures are confirmed against the GitHub issue/PR state.

### 10.1 Closed since Pass 5 — Pass-5/Pass-4 remaining work landed

| # (cluster) | Property | Closed | Fixed by |
|---|---|---|---|
| #4976, #4977, #4978, #4979 | no-overflow (busybox `bb_verror_msg`) | 06-09 … 06-16 | **#5278** (cap `asprintf`/`vasprintf` return at `INT_MAX/2`) + **#5279** (model `*strp` heap allocation) |
| #5143, #5144 | no-overflow (busybox `usleep`) | 06-16 / 06-09 | same `bb_verror_msg` cluster (#5278/#5279) |
| #5133, #5134, #5135, #5136 | no-data-race (atomic-only callees) | 06-08 | **#5233** (`[goto-race] instrument atomic-only callees as atomic`, the Pass-4 PR) |
| #5139, #5140, #5141 | valid-memsafety (busybox) | 06-09 | closed (the `bb_verror_msg`/`realloc` precision class; #5141 folded into the printf cluster) |
| #5137 | no-data-race (thread-join-binomial) | 06-10 | closed — triaged to value-set int→ptr thread-arg imprecision (§8.3 / §10.4), not a localized fix |
| #5224 | unreach-call (Intel-TDX, 54 tasks) | 06-09 | **#5228** (`[k-induction] disable inductive step on pointer-array writes`) |

This discharges Pass-5 remaining-work items 1 (#5233 merged), 2 (the busybox no-overflow cluster — the
sound `INT_MAX`-bounded model the §3.1/§9.2 analysis called for, shipped as `INT_MAX/2` + a `*strp`
allocation model), and 4 (#5139/#5140 closed). **#5012 itself stays open** for the *symbolic-format*
residual (G-C `va_list` recovery), which #5278/#5279 deliberately do not attempt.

### 10.2 #5287 / #5145 — aws_hash_table_create false alarm: carried to conclusion

Pass 4 (§8.7) deferred #5145 as a "research-grade CBMC harness." A multi-day deep dive (2026-06-10 →
06-14) decomposed it into **two independent bugs** in the original `.i`, and resolved the first:

* **Bug A — function-pointer dispatch picks wrong-arity targets and drops arguments.** The heap
  function-pointer (`state->equals_fn`/`hash_fn`) had an over-approximated value-set with no arity
  filter, so a 1-arg call could dispatch to a 2-arg candidate and drop an argument
  (`missing argument … modelled as nondet`). **FIXED → PR #5317** (merged 06-13, `[symex] filter
  function-pointer call targets by signature` in `symex_function_call_deref`). A clean
  FAILED→SUCCESSFUL flip is not observable with plain functions (the dispatch guard stays UNSAT), so
  the standalone effect is a dropped spurious WARNING; it is a real soundness-relevant fix on its own.

* **Bug B — the reported false alarm.** The post-create assert
  `__VERIFIER_assert(uninterpreted_equals(p_elem->key, key))` computes `0` while `s_find_entry`
  matched the same slot — two reads of the *same* byte-reconstructed FAM pointer field
  (`dynamic_1_array`, `bounded_malloc`'d, symbolically indexed) yielding inconsistent pointer
  identity. The RCA traced this to **compound imprecision across multiple layers** (UF semantics;
  `int→ptr` cast minting fresh `int_to_ptr` per call in `convert_typecast_to_ptr`; `make_failed_symbol`
  minting a fresh `invalid_object` per fallback read; `construct_from_dyn_offset` byte reads;
  value-set dropping byte-*written* pointers in `get_value_set_rec`). Each individual layer is a *sound*
  over-approximation; no single localized fix closes the harness.

  - The UF half was modelled soundly and merged: **PR #5364** (`[symex] model __ESBMC_uninterpreted_* /
    __CPROVER_uninterpreted_* with functional congruence`, Option A, merged 06-16), refined by **#5382**
    (`don't model non-scalar uninterpreted functions as native UFs`, 06-17). #5364 **closed #5287**; the
    general pointer-identity mechanism was filed and closed as **#5369** (`[symex] reads through an
    unresolvable pointer mint inconsistent pointer identity`, closed 06-17 via #5382).

  - **New result this pass (repro).** The original harness `aws_hash_table_create_harness.i` (issue
    #5145, exact flags `--64 --k-induction --max-inductive-step 3 --no-pointer-check --no-bounds-check
    --interval-analysis --force-malloc-success --no-div-by-zero-check`) **still reports
    `VERIFICATION FAILED` / `Bug found (k = 1)`** on master `74da7c0400`. The trace now shows
    `uninterpreted_equals` as a *modelled function* (`State 101 … rval = 0`) rather than an inlined
    concrete body — i.e. #5364 changed the modelling but **not** the verdict, because the residual is the
    havoc'd-slot read-consistency (Bug B's pointer-identity layers), exactly as the RCA predicted.

**Disposition.** **#5145 stays open** as the SV-COMP tracker for this benchmark, now classified as a
*known memory-model limitation* (not a UF-modelling gap). #5287's RCA comment + `PLAN_5287_option_b.md`
are the durable handoff; a sound fix is a deliberate value-set/pointer-read-consistency project
(no localized patch). **No new PR** — #5283's `ASSUME(ret!=0)` was measured unsound (closed), and every
layer that can be patched in isolation has been (Bug A → #5317; UF → #5364/#5382) without flipping the
verdict.

### 10.3 Witness-format issues — pre-validation landed, end-to-end still blocked

The Pass-3 carry-forward witness PRs merged: **#4940** (`[CI] WitnessMap pre-validation`, 06-03) and
**#4945** (`[sv-comp] special treatment for reach error`, 06-04). These harden ESBMC's emitted witness
but do not provide a CPAchecker/cpa-witness2test loop, which is still unavailable in this environment.
So #1471, #1492, #4611 remain blocked on **end-to-end** validation; #1470 is reported resolved on master
(recommend close once an x86 CI witness run confirms). No change in disposition.

### 10.4 Still-open landscape (15 issues) — re-validated

| # | Property / area | Pass-6 disposition |
|---|---|---|
| 1440 | wrapper CHECK() handling | latent; no live bug |
| 1447 | incorrect-verdicts umbrella | keep open — tracker |
| 1470 | overflow witness assignment | resolved on master; recommend close (needs x86 witness CI) |
| 1471 | struct constant in witness | blocked on CPAchecker validator |
| 1492 | FP violation witness | blocked on CPAchecker validator |
| 4427 | unreach-call false negative (megaraid) | fixed-on-master by #4484 (fn-ptr inductive-step guard); recommend close after x86 CI confirm |
| 4432 | no-data-race (mcslock) | data-race-checker interleaving perf on `__atomic_*` (§7.3); not host-arch |
| 4438 | unreach-call false alarm (log_6_safe) | covered by **open** PR #4480; no separate PR |
| 4439 | valid-memsafety (irda LDV) | research-grade LDV driver |
| 4611 | witness `returnFromFunction`/dup nodes | blocked on validator; key is spec-conformant, must not rename |
| 4980 | termination (turbografx) | ranking recogniser bails on FUNCTION_CALL bodies (§7.2); **not** covered by merged #4919; needs full 2413-benchmark validation |
| 5012 | printf/(v)asprintf return-length OM | G-A/G-B + `*strp` shipped (#5270/#5278/#5279, closed the busybox cluster); open for G-C symbolic-format `va_list` recovery |
| 5138 | valid-memsafety (comm_3args_ok) | reachable-memleak precision; needs an `ESBMC_SVCOMP` build to reproduce (§8.4); research-grade |
| 5142 | no-overflow (imon LDV) | x86-only `.i` (parse-fails on aarch64); research-grade LDV driver |
| 5145 | unreach-call (aws_hash harness) | known memory-model limitation (§10.2); UF half fixed (#5317/#5364/#5382), Bug-B residual open |

### 10.5 Pass-6 running report

**Analysed / re-validated.** All 15 open SV-COMP issues; the six closed clusters in §10.1; the
aws-hash #5287/#5145 thread to conclusion (§10.2, incl. a fresh local repro of #5145 on master).

**PRs opened by Pass 6.** None — every still-open issue is blocked on an unavailable validator
(#1471/#1492/#4611), a research-grade subsystem change requiring full-set validation (#4980,
#4439/#5142, #5145 Bug-B), an `ESBMC_SVCOMP` build (#5138), perf work (#4432), an in-flight PR
(#4438→#4480), or already fixed-on-master pending CI confirm (#1470, #4427). This is the
correctness-first outcome: no unsound or unvalidated patch is shipped to manufacture a PR count.

**Duplicated work avoided.** #5145 not re-fixed at the UF layer — that half merged (#5317/#5364/#5382)
and the verdict is unchanged, so the residual is logged against the closed #5287/#5369 RCA, not
re-patched. #4438 not re-fixed (open PR #4480). The busybox no-overflow/memsafety clusters not
re-diagnosed — closed by #5278/#5279.

**Remaining work (priority order, updated 2026-06-17).**
1. **#5145 / Bug B** — value-set precision for byte-*written* pointers into `bounded_malloc`'d FAMs +
   dereference read-consistency (the closed #5369 RCA). Deliberate memory-model project; deepest blocker.
2. **#5012** — G-C `va_list` argument recovery for the symbolic-format printf return-length residual.
3. **#4980** — extend the termination ranking recogniser to tolerate side-effect-only call bodies
   (sound only with a "does this call/store touch the measure/invariant variables?" check); requires
   full SV-COMP termination-set (2413) validation.
4. **#4432** — data-race-checker interleaving reduction / POR for atomic regions (`__atomic_*` builtins).
5. **#5138** — reachable-memleak / `forgotten-memory` precision (reproduce under an `ESBMC_SVCOMP` build).
6. **Close-outs pending CI:** #1470 and #4427 (recommend close after an x86 witness/CI confirm);
   witness end-to-end validation for #1471/#1492/#4611 once a CPAchecker loop is available.

---

## 11. Pass 7 — new batch #5393–#5400 (2026-06-17)

A fresh batch of eight `SV-COMP` issues was filed on 2026-06-17 (all from the
[SV-COMP 26 benchexec run](https://github.com/esbmc/esbmc/actions), commit `ee2c67bf32`). This pass
triages them on the same-day master `74da7c0400` (ESBMC 8.3.0, aarch64 macOS). Benchmarks were
fetched read-only from the public sv-benchmarks GitLab and treated as untrusted input. **One sound
localized fix was produced (#5395); one soundness bug was reproduced and root-caused (#5400).**

### 11.1 New-batch landscape

| # | Benchmark | Property | Disposition | PR |
|---|---|---|---|---|
| 5393 | aws-c-common `aws_hash_table_init_bounded_harness` | unreach-call | aws-hash memory-model cluster → #5145/#5287 (§11.3) | no |
| 5394 | aws-c-common `aws_hash_table_init_unbounded_harness` | unreach-call | same cluster (§11.3) | no |
| 5395 | ldv-regression `rule57_ebda_blast` | unreach-call | **FIXED** — `calloc(_, 0)` ignored `--force-malloc-success` (§11.2) | new |
| 5396 | ldv-linux-3.14-races `nsc-ircc.ko.cil` | no-overflow | x86-only inline asm; aarch64 parse blocker (§11.4) | no |
| 5397 | ldv-linux-3.14-races `nsc-ircc.ko.cil` | valid-memsafety | same file, x86 parse blocker (§11.4) | no |
| 5398 | ldv-linux-3.14-races `cafe_ccic.ko.cil-1` | valid-memsafety | x86 LDV driver (§11.4) | no |
| 5399 | ldv-linux-3.14-races `cafe_ccic.ko.cil-2` | valid-memsafety | x86 LDV driver (§11.4) | no |
| 5400 | goblint-regression `09-regions_12-arraycollapse_rc` | valid-memsafety (memtrack) | **SOUNDNESS** — missed leak; value-set weak-update through a global pointer array (§11.5) | no (research-grade) |

### 11.2 #5395 — `rule57_ebda_blast` unreach-call false alarm: FIXED

**Status: FIXED. Sound localized OM fix + two regression tests.**

**Reproduced** (repro) at the BMC base case (`k = 1`, plain `--incremental-bmc` and `--k-induction`
agree, so it is a real path in ESBMC's model, not a k-induction artefact). The 139-line benchmark
asserts `used_tmp_slot==0 ⇒ freed_tmp_slot`. The only failing path returns from `ebda_rsrc_controller`
in the window where `freed_tmp_slot=0` and `used_tmp_slot=0`, reached when
`ibmphp_find_same_bus_num()` returns NULL.

**Root cause (source + trace).** That function does `return kzalloc(sizeof(struct bus_info), 0)`, i.e.
`calloc(1, sizeof(struct bus_info))`. `struct bus_info {}` is **empty ⇒ sizeof == 0**, so the call is
`calloc(1, 0)`. The `calloc` operational model (`src/c2goto/library/stdlib.c`), after PR #5269 added
`if (!nmemb || !size) return NULL;` for overflow handling, returned NULL **unconditionally** for a
zero-byte request — ignoring `--force-malloc-success`. A direct `malloc(0)` honours that option (the
symex layer at `memory_alloc.cpp:618-642`), so `calloc(n, 0)` was inconsistent with `malloc(0)`: the
trace shows `bus_info_ptr1 = NULL` despite `--force-malloc-success`, driving the error path and the
false alarm.

**Fix.** Route the zero-byte case to `malloc(0)` instead of returning NULL
(`return NULL` → `return malloc(0)`; the guard condition is unchanged, so the later
`__ESBMC_assume(nmemb <= SIZE_MAX / size)` overflow prune from #5269 is still protected from
div-by-zero). Zero-size `calloc` now mirrors `malloc(0)`: non-null under `--force-malloc-success`,
NULL under `--malloc-zero-is-null`. This is C-standard-conformant (C17 §7.22.3: `calloc(_, 0)` is
implementation-defined NULL-or-unique-pointer, exactly like `malloc(0)`), and the benchmark's
ground-truth `true` confirms the non-null intent.

**Validation.** `#5395` benchmark with the issue's exact flags → `VERIFICATION SUCCESSFUL`. The three
`github_4433` calloc-overflow regressions (#5269) still pass; all 49 calloc-using regression tests
pass. Two new tests: `regression/esbmc-unix/github_5395_calloc_empty_struct` (empty-struct zero-size
under `--force-malloc-success` ⇒ SUCCESSFUL) and `.../github_5395_calloc_zero_is_null_fail`
(`calloc(1,0)` under `--malloc-zero-is-null` ⇒ NULL ⇒ FAILED, pinning the option boundary). The change
is a single-statement-body change (condition unchanged) — Mode-C exempt per the triviality bar.

### 11.3 #5393 / #5394 — aws-c-common hash-table *init* harnesses: aws-hash cluster

Both are CBMC proof harnesses from `c/aws-c-common`, siblings of the `aws_hash_table_create_harness`
(#5145/#5287). They share the same byte-addressed `bounded_malloc`'d-state + `__CPROVER_uninterpreted_*`
modelling that the multi-day #5287 RCA decomposed into the compound pointer-read-consistency
limitation (§10.2). The UF half is now modelled (#5364/#5382); the residual is the same memory-model
project tracked by the closed #5287/#5369 RCA and `PLAN_5287_option_b.md`. No separate PR — folded into
the aws-hash cluster.

### 11.4 #5396–#5399 — ldv-linux-3.14-races drivers: x86 parse blockers

`nsc-ircc.ko.cil` (#5396/#5397) and `cafe_ccic.ko.cil-{1,2}` (#5398/#5399) are full Linux-3.14 kernel
driver CIL amalgamations containing **x86-only inline asm** (`"=a"`/`"a"` register constraints, `outb`,
paravirt `call *%cN`), which the aarch64 macOS clang frontend rejects at parse time
(`invalid output constraint '=a' in asm`). Same host-blocker / research-grade LDV-driver class as
#4439/#4427/#5142. Need an x86 Linux host to triage.

### 11.5 #5400 — `arraycollapse_rc` valid-memtrack: SOUNDNESS bug (missed leak), root-caused

**Status: SOUNDNESS bug (incorrect `true`) reproduced and root-caused. Research-grade — no sound
localized fix; same value-set-precision class as #5145/#4432/#5138.**

This is the most serious category: ESBMC reports `VERIFICATION SUCCESSFUL` on a benchmark whose ground
truth is `false(valid-memtrack)` — a **missed memory leak**. Reproduced (repro) with the issue's exact
flags on master `74da7c0400`.

**The leak is real.** In `main`, `p = new(3)` is inserted into `slot[j]` then `slot[k]`; the second
`list_add` overwrites `p->next`, **orphaning** the node that was `slot[j]`'s previous head (a `new(2)`
node) whenever `j != k`. That orphan is unreachable at exit = a `valid-memtrack` violation, which
`--no-reachable-memory-leak` is supposed to catch (it suppresses only *reachable* memory).

**Root cause (source + trace + reducer).** The bug is sequential (a minimal single-threaded reducer
reproduces it; the pthread thread is irrelevant) and survives concrete indices. Reduced to: a global
pointer **array** `slot[10]` where one node is orphaned via pointer reassignment while the others stay
reachable. The discriminator:

| reducer | `--no-reachable-memory-leak` | verdict |
|---|---|---|
| global pointer **array** `slot[10]` | on (SV-COMP config) | SUCCESSFUL ❌ (misses leak) |
| global pointer **array** `slot[10]` | off | FAILED ✓ |
| **scalar** globals `slot0`/`slot1` | on | FAILED ✓ |

`add_memory_leak_checks` (`symex_main.cpp:1292`) builds the leak VCC as
`alloc_guard ∧ ¬reachable_from_globals(obj)`, where `reachable_from_globals` is a BFS over the
value-set. The `memcleanup` debug log shows the BFS marks **all** dynamic objects reachable, including
the orphan: storing `d1.next` (reached through the global array `slot[]`) is a **weak value-set
update** that retains the stale target `{d2, d5}` instead of strong-updating to `{d5}`, so the orphan
`d2` keeps a spurious incoming edge and the leak VCC is killed. The scalar-global version
strong-updates and correctly drops `d2`. A may-points-to over-approximation cannot soundly *suppress*
leaks; doing so is what makes `--no-reachable-memory-leak` miss this one. (The code already documents
array reachability in a negated context as a known-incomplete "workaround" at `symex_main.cpp:1448-1472`.)

**Why no PR.** A sound fix requires either a strong-update-precise value-set for heap fields reached
through global arrays, or a must-reachable (under-approximation) for the suppression — both are
value-set/pointer-analysis changes with broad blast radius that cannot be validated against the full
SV-COMP memsafety set in this environment. Same research-grade class as #5145 (closed #5369 RCA),
#4432, #5138. Keep #5400 open as a soundness tracker with this RCA; do not ship an unvalidated
heuristic.

### 11.6 #5138 deep-dive — `comm_3args_ok` valid-memsafety false alarm (reproduced + root-caused)

Pass 4 (§8.4) deferred #5138 as needing an `ESBMC_SVCOMP` build. Done this pass: built ESBMC with
`-DESBMC_SVCOMP=On` (the `__ESBMC_SVCOMP` macro is a compile-time CMake option baked into the OM at
`clang_c_language.cpp:249`, **not** a user `-D` flag) and **reproduced the exact issue**:
`dereference failure: forgotten memory: dynamic_21_array` at `exit` (`stdlib.c:64`),
`VERIFICATION FAILED` / `Bug found (k = 11)` / `FALSE_MEMTRACK`. (On a non-`ESBMC_SVCOMP` build a
*different* artefact fires first — `io.c:113 fclose: invalid pointer freed`, the `fclose`→`free`
of the static stdout object that the SV-COMP OM path skips — which is why the issue is invisible to a
normal local build.)

**Root cause (repro + `memcleanup` debug).** Same subsystem as #5400 — `add_memory_leak_checks`
(`src/goto-symex/symex_main.cpp:1292`) under `--no-reachable-memory-leak` — but the **opposite horn**
of the value-set imprecision dilemma. The leak VCC is `alloc_guard ∧ ¬reachable_from_globals(obj)`,
where `reachable_from_globals` is a BFS over the value-set seeded from all global symbols. On this
benchmark the BFS **under-approximates**: the `memcleanup-skip` histogram for one base-case pass shows
only **21 targets followed** against **210 `unknown` value-set targets skipped** (plus 399 null, 525
constant-string, 84 code, 21 invalid — all correctly ignored). The skip of `unknown` is a *deliberate*
choice (`symex_main.cpp:1502-1515`: "Treating 'unknown' as could-point-anywhere generates too many
false positives... We ignore it"). Consequence: the globally-reachable set collapses to a single
object (`globals point to: c:5138.c@slotvec0_226539`), so any still-live dynamic object reached only
through an `unknown`-valued global pointer — here `dynamic_21` — is judged unreachable and reported as
a forgotten-memory leak, even though the program never lost it. Ground truth is `true`, so this is a
**false alarm** (precision/incompleteness), not a soundness bug.

**Relation to #5400.** Both are the value-set-based reachability in `add_memory_leak_checks` failing to
capture the exit-time heap precisely. #5400 *retains a stale edge* (over-approx ⇒ misses a real leak,
unsound); #5138 *drops a live edge* by skipping `unknown` (under-approx ⇒ false alarm). The
`unknown`-skip at 1502-1515 is the explicit knob that trades one for the other — treating `unknown` as
"reachable" instead would suppress #5138's false alarms but deepen the #5400 class of misses. A
minimal `global → malloc → reachable-at-exit` program is handled **correctly** (SUCCESSFUL), so the
trigger needs the coreutils structure where the value-set loses precision on a specific assignment
path (the slotvec/quotearg machinery).

**Disposition.** Research-grade — a sound fix needs precise exit-time heap reachability (or a sound,
non-explosive treatment of `unknown` targets), the same value-set-precision project as #5400 / #5145.
No localized fix; keep #5138 open as a precision tracker.

### 11.7 Pass-7 running report

**Analysed.** All eight #5393–#5400 (reproduced locally where the host allows; #5396–#5399 are x86
parse blockers).

**PRs opened.** One: `[om] calloc: route zero-size request to malloc(0), not NULL` — Fixes **#5395**.
Sound, code-reviewed, two regression tests, 49/49 calloc regressions pass.

**Duplicated work avoided.** #5393/#5394 not re-diagnosed — same aws-hash memory-model cluster as
#5145/#5287 (UF half already merged via #5364/#5382). #5396–#5399 not re-fixed — x86 host blocker.

**Skipped / deferred and why.** #5400 — soundness (missed leak); value-set weak-update through global
pointer arrays defeats `--no-reachable-memory-leak` suppression; research-grade, no sound localized
fix. #5138 — reproduced on an `ESBMC_SVCOMP` build and root-caused (§11.6): the mirror false-alarm of
#5400 (value-set BFS skips `unknown` targets ⇒ drops a live edge); research-grade. #5393/#5394 —
aws-hash pointer-read-consistency cluster (#5287/#5369 RCA). #5396–#5399 — x86-only inline asm parse
blockers.

**Remaining work (priority order, updated 2026-06-17).**
1. **#5400** — strong-update-precise value-set for heap fields through global pointer arrays, or a
   must-reachable suppression for `--no-reachable-memory-leak` (the `symex_main.cpp:1448-1472`
   workaround). Soundness; highest severity.
2. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980** — termination ranking recogniser for side-effect-only call bodies (full-set validation).
5. **#5138** — root-caused (§11.6): mirror of #5400; needs a sound treatment of `unknown` value-set
   targets in the `--no-reachable-memory-leak` BFS. Pair with the #5400 reachability work.
6. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
7. **x86 host triage** for #5396–#5399; close-outs #1470/#4427 pending CI.

---

## 12. Pass 8 — x86 host unblocks the ldv-linux-3.14-races drivers (2026-06-22)

Pass 7's #5396–#5399 dispositions were "x86-only inline asm; aarch64 parse blocker" — *deferred for
want of an x86 host*. This pass runs them on an **x86_64 Linux** host (binary version-stamped in the
header), turning that carry-forward item into actual reproductions and root causes. It also
reconciles the two Pass-7 fixes that have since closed (#5395, #5400). All verdicts below are
**repro** (locally reproduced this pass with each issue's exact flags) unless stated. Benchmarks were
fetched read-only from the public sv-benchmarks GitLab and treated as untrusted input.

### 12.1 Closed since Pass 7

| # | Property | Closed | Outcome |
|---|---|---|---|
| #5395 | unreach-call (`rule57_ebda_blast`) | 2026-06-19 | the Pass-7 fix landed — `calloc(_,0)` now routes to `malloc(0)` instead of returning NULL, honouring `--force-malloc-success` (§11.2). |
| #5400 | valid-memtrack (`arraycollapse_rc`) | 2026-06-19 | closed with a **refined public RCA** (no localized fix); see §12.2. |

### 12.2 #5400 — refined RCA confirms the soundness bug is allocation-site value-set imprecision

Pass 7 (§11.5) root-caused #5400 to a value-set weak-update through a global pointer array. The
issue's closing analysis sharpens this into an **engine-dimension** result worth recording, because
it pins the exact mechanism and rules out a localized patch:

* The orphaned node is wrongly kept in the `globals_point_to` reachable set computed by
  `add_memory_leak_checks` (`symex_main.cpp`), so the `forgotten memory` VCC
  `alloc_guard ∧ ¬reachable_from_globals(obj)` is killed (`targeted` becomes `same_object(d2,d2)=true`).
* **Engine split on the same program:** plain BMC (`--no-unwinding-assertions`) computes
  `globals point to {d1,d3,d4,d5}` — orphan **absent** → `FALSE` (leak correctly detected);
  `--incremental-bmc` / `--k-induction` compute `{d1,d2,d3,d4,d5}` — orphan **present** → `SUCCESSFUL`
  (leak wrongly suppressed). Distinct `malloc` call-sites (the R5 reducer) stay sound under **both**
  engines, confirming **allocation-site identity** is the discriminator: the value-set cannot
  distinguish heap instances minted at the same call-site, so a stale points-to edge keeps the orphan
  "reachable" via a live sibling.
* **Fix outlook (from the issue):** a sound `valid-memtrack` reachability needs an *under*-approximation
  of "reachable" (report a leak unless an object is *provably* reachable); the value-set provides the
  opposite (a *may*-points-to). The conservative variant ("don't suppress a leak for objects whose
  allocation site has multiple live instances") is expected to introduce false positives on the common
  SV-COMP pattern of a global linked list built in a loop, and the trade-off is being quantified on the
  `valid-memsafety` suite before any change. This is the same research-grade value-set/pointer-analysis
  project as #5145 (closed #5369 RCA) and #5138 (§11.6) — its mirror false-alarm horn.

**Disposition unchanged:** soundness tracker, no unvalidated heuristic. Closed-with-RCA, not fixed.

### 12.3 #5396 / #5397 — nsc-ircc: REPRODUCED on x86; one shared `platform_get_drvdata` gap

The Pass-7 blocker was a parse failure: the aarch64 macOS clang rejects the x86-only inline asm in
`nsc-ircc.ko.cil.i`. On **x86_64 Linux the file parses, reaches symbolic execution, and both issues
reproduce at the base case (`k = 1`)** — so they are real paths in ESBMC's model, not engine artefacts.

* **#5396 (no-overflow).** `arithmetic overflow on add` at `nsc_ircc_suspend` (`!overflow("+", iobase, 3)`),
  with the trace pinning `iobase = 2147483646` (`INT_MAX − 1`). Root cause (source + trace):
  `iobase = self->io.fir_base`, where `self = platform_get_drvdata(dev)` and `fir_base` is declared
  `int` (a device I/O-port base). ESBMC has no model for `platform_get_drvdata` / the device-resource
  API, so the returned struct's `int` field is an **unconstrained nondet `int`**; free to approach
  `INT_MAX`, the signed-`int` add `iobase + 3` overflows. This is the **same nondet-`int` → signed-add
  family as #4976–#4979** (busybox `bb_verror_msg`), but sourced from a *device-resource field* rather
  than a `(v)asprintf` return. The over-approximation is **sound**; ground truth is `true`, so this is a
  **precision/incompleteness limitation, not a soundness bug**.

* **#5397 (valid-memsafety).** Same `.i`, memory-leak flags. Reproduces as `dereference failure:
  invalid pointer` (trace: `self = (struct nsc_ircc_cb *)(invalid-object)`) — the *same* unmodeled
  `platform_get_drvdata` returns an over-approximated pointer that the value-set resolves to the failed
  `invalid-object`, so any dereference of `self` is flagged. (My repro fires at `nsc_ircc_resume`
  line 9676 before the issue's cited `nsc_ircc_interrupt` site; same root cause, earlier deref.)

**One root cause, two properties.** Both are the missing operational model for the Linux platform-device
API: `self` and its fields are unconstrained. A sound fix is a device-model that returns a *valid,
registered* driver-data pointer with bounded I/O-port fields — a research-grade LDV-driver change (the
same class as #4439/#4427/#5142), not a localized patch, and not validatable against the full
SV-COMP set in this environment. **No PR.** Disposition corrected from "x86 parse blocker" to
"reproduced on x86 — precision/incompleteness from the unmodeled `platform_get_drvdata` device API."

### 12.4 #5398 / #5399 — cafe_ccic: REPRODUCED on x86 (invalid-pointer-free in the LDV scenario)

`marvell-ccic--cafe_ccic.ko.cil-{1,2}.i` (valid-memsafety) parse on x86_64 Linux —
`--goto-functions-only` emits a complete GOTO program (no parse error), removing the Pass-7 aarch64
blocker — and, run to a verdict under the issue's exact flags (`--memory-leak-check
--no-reachable-memory-leak --malloc-zero-is-null --incremental-bmc`), **both reproduce identically**:

```
Violated property: … function ldv_free   dereference failure: invalid pointer freed
VERIFICATION FAILED / Bug found (k = 1)
```

at `ldv_free`/`kfree` (which both reduce to `free(s)`, cil-1 line 10798 / cil-2 line 10799), reached
through the multi-threaded LDV PCI/interrupt scenario (`ldv_pci_scenario_3`,
`ldv_dispatch_irq_register_*`, thread 2/3). **Exact mechanism (source-pinned on cil-2):**

* `cafe_pci_remove(pdev)` recovers the camera object from driver-data and frees it:
  ```c
  tmp  = ldv_dev_get_drvdata_84(&pdev->dev);     // pdev->dev.p ? pdev->dev.p->driver_data : NULL
  cam  = to_cam((struct v4l2_device *)tmp);        // container_of: cam = (char*)tmp - 0xD8
  if (cam == NULL) { /* print */ return; }         // guard cannot catch a wild offset pointer
  cafe_shutdown(cam);
  kfree((void const *)cam);                          // <-- "invalid pointer freed"
  ```
* The link that should make `dev_get_drvdata` return `&cam->v4l2_dev` is
  `v4l2_device_register(cam->dev, &cam->v4l2_dev)` in probe. **But the benchmark stubs that function
  as a pure no-op** (cil-2 line 12120: `int v4l2_device_register(...){ return __VERIFIER_nondet_int(); }`)
  — it never performs the kernel's implicit `dev_set_drvdata`. `pdev` is also allocated with the
  **non-zeroing** `ldv_xmalloc`, so `pdev->dev.p` is an uninitialised (nondet) field.
* ESBMC therefore finds a sound path with `pdev->dev.p == NULL` → `get_drvdata` returns `NULL` →
  `to_cam(NULL)` = `NULL − 0xD8` = a **wild, non-NULL** pointer (`0x…FF28`) that the `cam == NULL`
  guard does not catch → `kfree` of that wild pointer is a genuine invalid free *in the benchmark as
  written*. The same `container_of`-on-unmodelled-environment pattern recurs (e.g. the
  `mcam_vb_buffer` list walk, `__mptr − 840`, over `cam->buffers`; the run also names
  `invalid_object22` / `invalid_object45`), so it is a class, not a one-off.

**Why no sound localized fix.** ESBMC is here *more precise* than the benchmark's bundled environment
model supports: SV-COMP labels the task `true` only under the assumption of a faithful kernel
environment in which `v4l2_device_register` sets drvdata. Any ESBMC-side change that let this `kfree`
pass would be **unsound** (it would suppress a real wild-pointer free on the program as written), and an
ESBMC operational model cannot help because the benchmark *defines* `v4l2_device_register` itself and so
shadows any library model. The genuinely-correct fix is upstream — model `v4l2_device_register` to call
`dev_set_drvdata(arg0, arg1)` (and/or zero-initialise `pdev`) in `sosy-lab/sv-benchmarks` — so the
drvdata round-trips and the recovered `cam` is valid. This is the **same precision/incompleteness class
as #5397** and the research-grade LDV-environment modelling gap (#4439/#4427); umbrella #2513/#1447.
Ground truth is `true`, the over-approximation is **sound**. **No PR.** Disposition: "reproduced on
x86 — sound invalid-pointer-free from the benchmark's incomplete LDV environment model (no-op
`v4l2_device_register` + non-zeroed `pdev` + unguarded `to_cam` `container_of`); closed-with-RCA."

### 12.5 #5142 — imon: re-classified — clang `-Wint-conversion`, *not* inline asm

Pass 7 grouped `m0_drivers-media-rc-imon` with the inline-asm parse blockers. On x86_64 Linux it
**still fails to parse, but for a different reason**: clang rejects a CIL global initializer
`static struct mutex driver_lock = { …, 0xffffffffffffffffUL, … }` with
`error: incompatible integer to pointer conversion initializing 'void *' with an expression of type
'unsigned long' [-Wint-conversion]`. That is a **newer-clang strictness** issue on a CIL-emitted
sentinel pointer, independent of host architecture and distinct from the x86-only-asm class. Disposition
corrected: **frontend/clang-strictness parse blocker** (candidate for a `-Wno-int-conversion`-style
relaxation on CIL inputs, or a frontend fix-up of integer→pointer initializers), not an LDV inline-asm
blocker.

### 12.6 Pass-8 running report

**Analysed.** #5396, #5397, #5398, #5399 (all four ldv-linux-3.14-races drivers reproduced on x86),
#5142 (re-classified); plus reconciliation of #5395 and #5400 closures.

**PRs opened.** None code — every reproduced item is a sound over-approximation with no localized fix
(research-grade), consistent with the correctness-first mandate. This pass is the recurring triage-doc
update (the established cadence: Pass 4 → #5234, Pass 6 → #5389, Pass 7 → #5405).

**Duplicated work avoided.** #5396–#5399 not patched — the unmodeled-LDV-kernel-environment pointer/device
gap is the #4439/#4427 research-grade class. #5400 not re-fixed — closed-with-RCA; the value-set
under-/over-approx trade-off is the #5145/#5138 project.

**Remaining work (priority order, updated 2026-06-22).**
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack`: a sound under-approximation of
   "reachable" (the two horns of the same allocation-site/`unknown`-skip imprecision).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980** — termination ranking recogniser for side-effect-only call bodies (full-set validation).
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
6. **#5142** — clang `-Wint-conversion` relaxation / integer→pointer initializer fix-up for CIL inputs.
7. **Device-API / LDV-environment model** for `platform_get_drvdata` & the LDV interrupt-scenario
   pointers — the shared blocker behind all four #5396–#5399; research-grade, would close the
   ldv-linux-3.14-races overflow/memsafety cluster.
8. **Close-outs pending CI:** #1470, #4427; witness end-to-end validation for #1471/#1492/#4611.

---

## 13. Pass 9 — reconcile closures + comm_3args_ok getopt_long residual (2026-06-26)

Pass 8 closed on 2026-06-22. This pass reconciles the four issues that closed since, triages the one
new live issue (#5565), and produces **one sound localized OM fix** for it. Verdicts are **repro**
(locally reproduced this pass with each issue's exact flags) on an **x86_64 Linux** host. The reference
binary was rebuilt from master `9e79dbe179` — which **includes PR #5564** (`[symex] treat
stack-reachable heap as live in memory-leak check`, merged 2026-06-26), the fix for the #5138/#5565
forgotten-memory layer. (An earlier stale binary still carried the pre-#5564 leak and masked the
residual below — the rebuild was a prerequisite.)

### 13.1 Closed since Pass 8

| # | Property | Closed | Outcome |
|---|---|---|---|
| #5395 | unreach-call (`rule57_ebda_blast`) | 2026-06-18 | the Pass-7 fix landed (`calloc(_,0)` → `malloc(0)`); see §11.2. |
| #5399 | valid-memsafety (`cafe_ccic.ko.cil-2`) | 2026-06-22 | closed-with-RCA (§12.4). Its twin **#5398** (`cil-1`) stays **open** as the tracker — same root cause (no-op `v4l2_device_register` + non-zeroed `pdev` + unguarded `to_cam`); not re-triaged. |
| #5400 | valid-memtrack (`arraycollapse_rc`) | 2026-06-19 | closed-with-RCA (§12.2); allocation-site value-set imprecision. |
| #5593 | unreach-call (Intel-TDX `*_havoc_memory`, 27 tasks) | 2026-06-26 | **recurrence of #5224**; fixed by **merged PR #5602** (`[k-induction] disable inductive step on reachable __VERIFIER_nondet_memory`), extending the #5228/#5230 pointer-array havoc gate to the `__VERIFIER_nondet_memory` write idiom that #5268 missed. Two regression tests (`github_5593_nondet_memory_heap{,_fail}`). Sound Phase-1/2 gate extension — **no competing PR**. |

### 13.2 #5565 — `comm_3args_ok` residual after #5564: `getopt_long`/`optarg` false alarm — FIXED

**Status: FIXED. Sound localized OM fix (`getopt_long`/`getopt_long_only`) + two regression tests.
PR opened.**

#5565 was filed (2026-06-22) as the residual after #5564 resolved the `forgotten memory` leak on
`coreutils-v8.31/comm_3args_ok`, and *predicted* two residual model gaps: an `fclose(stdout)`
static-stream free and a `strcmp` invalid-pointer. **Reproduced on a post-#5564 x86_64 build, the
first-firing residual is neither of those** — it is an **unmodeled `getopt_long`**:

* Both the `--sv-comp` and the plain (`#5138`-command) runs report
  `dereference failure: invalid pointer` at `k = 11`: the `--sv-comp` run at `comm_3args_ok.c:51622`
  (`if (*optarg)` in `__main`), the plain run inside `strlen` (`string.c:92`, from
  `strlen(optarg)` two lines later). Ground truth is `true` ⇒ false alarm.
* **Root cause (trace + source).** The trace shows `WARNING: no body for function getopt_long` and
  `optarg = invalid-object`. ESBMC models `getopt` (`src/c2goto/library/getopt.c`: `optarg =
  argv[result_index]`, `result_index < argc`) but **not** `getopt_long`/`getopt_long_only`, so the
  global `optarg` is never assigned and stays its uninitialised (invalid/NULL) value; the benchmark
  then dereferences it. This is the missing-libc-OM class of §4 #2797, here narrowed to one function.

**Fix.** Add `getopt_long` and `getopt_long_only` to `getopt.c`, mirroring `getopt` via a shared
`__ESBMC_getopt_long(argc, argv)` helper: `__ESBMC_assume(result_index < argc); optarg =
argv[result_index]; return nondet_int();`. The nondet return (vs `getopt`'s constant `0`) keeps every
option-handling branch *and* loop termination via `-1` reachable. An incomplete `struct option`
forward declaration suffices (the long-option table is only forwarded, never read).

**Why it is sound.** `optarg` is bound to a real argv element (`result_index < argc`), exactly what the
real library does for an option that takes an argument, and exactly what the pre-existing `getopt`
model already does. It is an over-approximation in the same direction as `getopt`: `optarg` is always a
valid non-NULL pointer and `optind`/`*longindex` are not advanced (documented in the file comment) — so
no genuine race/leak/OOB is hidden. The esbmc-verifier agent confirmed the model does **not**
blanket-suppress pointer/bounds checks (a deref of `optarg` *without* calling `getopt_long` still fires
`invalid pointer`; an out-of-bounds write past the argv element still fires `array bounds violated`),
and that the positive test is **not** vacuous (the `optarg` branch is reachable and verified safe).

**Validation.**
* Minimal repro → `VERIFICATION SUCCESSFUL` (was `invalid pointer`/`NULL pointer` FAILED).
* Full `comm_3args_ok.c` with the issue's exact flags (`--sv-comp`): the spurious `FALSE` at `k = 11`
  is **gone** — ESBMC unwinds past it to `k ≥ 12` and times out (UNKNOWN, sound — no false positive),
  instead of reporting a false counterexample.
* Two regressions under `regression/esbmc-unix/`: `github_5565_getopt_long_optarg` (exercises both
  `getopt_long` and `getopt_long_only`; `optarg` deref safe ⇒ SUCCESSFUL) and
  `github_5565_getopt_long_optarg_oob_fail` (`optarg[100]` past a 4-byte argv element ⇒
  `array bounds violated` FAILED — pins that `optarg` is a *bounded* real object, not an invented
  infinite buffer).
* **Dual-solver z3 + bitwuzla agree** on both tests; esbmc-verifier (Mode A) reports
  `PATCH VERIFIED — no new errors`. The pre-existing `getopt_long` test
  (`instrumented_nohup_comb_overflow`) still passes (still finds its target overflow — the nondet
  return preserves bug-finding). Code-reviewed: no critical/major findings.

**Residual.** The `fclose` static-stream free the issue names is gated by the **compile-time
`ESBMC_SVCOMP`** build (`#5138` needs `-DESBMC_SVCOMP=On`, §11.6), and the `strcmp` invalid-pointer did
not surface as the first failure on a standard build — both are either that-build-gated or downstream
of the `optarg` layer fixed here. Keep #5565 open for those; the `getopt_long`/`optarg` layer is
closed by this PR.

### 13.3 Pass-9 running report

**Analysed.** The four closures (#5395/#5399/#5400/#5593) reconciled against current GitHub state;
#5565 reproduced and root-caused on a post-#5564 build.

**PRs opened.** One: `[om] model getopt_long/getopt_long_only so optarg is a valid pointer` — addresses
**#5565** (the `getopt_long`/`optarg` layer). Sound, dual-solver-validated, esbmc-verifier-confirmed,
code-reviewed, two regression tests.

**Duplicated work avoided.** #5593 not re-fixed — merged PR #5602 already targets it (sound gate
extension of #5228/#5230). #5398 not re-triaged — same RCA as its closed twin #5399 (§12.4). The
#5138/#5565 forgotten-memory layer not re-fixed — closed by #5564.

**Remaining work (priority order, updated 2026-06-26).** Unchanged from §12.6, with #5565 reduced to
its `fclose`/`strcmp` residual (lower priority, `ESBMC_SVCOMP`-gated):
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation of
   "reachable"); note #5564 advanced the live-stack-reachability half.
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980** — termination ranking recogniser for side-effect-only call bodies (full-set validation).
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
6. **#5142** — clang `-Wint-conversion` relaxation / integer→pointer initializer fix-up for CIL inputs.
7. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
8. **#5565 residual** — `ESBMC_SVCOMP`-gated `fclose(stdout)` static-stream free + `strcmp` argv-string
   pointer-validity; reproduce under a `-DESBMC_SVCOMP=On` build.
9. **Close-outs pending CI:** #1470, #4427; witness end-to-end validation for #1471/#1492/#4611.

---

## 14. Pass 10 — reconcile #5624 + universalise the #5142 int-conversion relaxation (2026-06-28)

Pass 9 closed on 2026-06-26. This pass reconciles the Pass-9 PR (now merged) and produces **one sound
localized frontend fix** for the most feasible remaining item, **#5142**. No new SV-COMP issues have
been filed since 2026-06-17 (all already triaged), so the priority list is otherwise unchanged. Host:
**x86_64-independent** — the fix and its tests are architecture-neutral and were validated on aarch64
macOS (ESBMC 8.3.0, master `f343663bde`).

### 14.1 Closed since Pass 9

| # | Property | Closed | Outcome |
|---|---|---|---|
| #5565 | valid-memsafety (`comm_3args_ok`, `getopt_long`/`optarg` layer) | merged | Pass-9 PR landed as **#5624** (`[om] model getopt_long/getopt_long_only so optarg is a valid pointer`). The `fclose`/`strcmp` residual (§13.2) stays open, `ESBMC_SVCOMP`-gated. |

### 14.2 #5142 — `m0_drivers-media-rc-imon`: int-conversion relaxation made universal — FIXED (parse layer)

**Status: FIXED (parse-blocker layer). Sound localized frontend change + two regression tests
(repurposed from #5530). PR opened.**

Pass 8 (§12.5) re-classified #5142 as a clang-strictness parse blocker: clang 15+ promotes
`-Wint-conversion` to a hard error, rejecting the CIL-emitted sentinel initialiser
`static struct mutex driver_lock = { …, 0xffffffffffffffffUL, … }`. ESBMC already carried
`-Wno-int-conversion`, **but only inside the `--sv-comp` block** of
`clang_c_languaget::build_compiler_args` (`src/clang-c-frontend/clang_c_language.cpp`). Immediately
below it, the sibling relaxation `-Wno-incompatible-pointer-types` is applied **universally** (same
class — "became a hard error … trips across all platforms"). The asymmetry meant a plain
`esbmc foo.c` on clang 15+ still hard-errored on GCC-acceptable implicit int↔pointer conversions that
ESBMC fully models.

**Fix.** Move `-Wno-int-conversion` out of the `--sv-comp` block to the universal section, beside
`-Wno-incompatible-pointer-types`. Reproduced and verified directly:

* `void *p = 0xdeadbeefUL;` (no cast) → plain run was `ERROR: PARSING ERROR`; now `VERIFICATION
  SUCCESSFUL`. `--sv-comp` already accepted it (proving the relaxation, not a semantics change, is the
  fix).
* The conversion is still **modeled, not rubber-stamped**: the same code with a wrong round-trip
  assertion reports `VERIFICATION FAILED` (value preserved through int→ptr→int).

**Why it is sound.** Purely a diagnostic-level change — it affects only whether clang refuses to
parse, not the AST/GOTO semantics. GCC (the compiler ESBMC targets) only *warns* on this construct, so
matching GCC is the correct behaviour; there is no program GCC rejects that ESBMC now silently accepts.
Inherited unchanged by `clang_cpp_languaget`, where the flag is a no-op (C++ int↔pointer is a genuine
type error not governed by `-Wint-conversion`), so C++ checking is not weakened. Code-reviewed: 0
critical/major/minor findings.

**Validation.** Two regressions under `regression/esbmc/`, repurposed from #5530 by dropping their
`--sv-comp` flag so they now pin the *universal* path: `github_5142_int_conversion` (implicit int→ptr
round-trip ⇒ SUCCESSFUL) and `github_5142_int_conversion_fail` (wrong round-trip value ⇒ FAILED, with
the violated-property message). Both pass; a 52-test conversion/cast regression sample
(`to_union*`, `*nondet_int_ptr*`, `github_382*`, `github_270*`, …) is green — no test depends on
int-conversion being a hard error.

**Residual.** This closes only the *parse-blocker* layer of #5142. The downstream no-overflow precision
layer (whether the benchmark verdict is correct once it parses) is x86 + full-benchmark-gated and not
reproducible in isolation here — it stays in the research-grade LDV-environment class. Keep #5142 open
for that layer.

### 14.3 Pass-10 running report

**Analysed.** Pass-9 closure (#5624) reconciled; GitHub SV-COMP issue list re-swept (no new issues
since 2026-06-17); #5142 parse layer reproduced and fixed.

**PRs opened.** One: `[clang-frontend] apply -Wno-int-conversion universally, not only under --sv-comp`
— addresses **#5142** (parse-blocker layer). Sound diagnostic-level change, code-reviewed, two
regression tests.

**Duplicated work avoided.** #5565 not re-fixed — its primary layer merged as #5624. The remaining
research-grade items (#5145/#5393/#5394 value-set, #5400/#5138 reachability, #5012 va_list, #4980
termination, #4432 atomics, device-API modelling) are unchanged — each requires a subsystem change
that cannot be validated against the full SV-COMP set in this environment, so none was patched with a
heuristic.

**Remaining work (priority order, updated 2026-06-28).** As §13.3, with #5142 reduced to its
no-overflow precision layer:
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980** — termination ranking recogniser for side-effect-only call bodies (full-set validation).
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
6. **#5142 residual** — no-overflow precision layer once parsed (x86 + full-benchmark-gated;
   research-grade LDV-environment class).
7. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
8. **#5565 residual** — `ESBMC_SVCOMP`-gated `fclose(stdout)` static-stream free + `strcmp` argv-string
   pointer-validity; reproduce under a `-DESBMC_SVCOMP=On` build.
9. **Close-outs pending CI:** #1470, #4427; witness end-to-end validation for #1471/#1492/#4611.

---

## 15. Pass 11 — reconcile + confirm the actionable backlog is at its research-grade floor (2026-06-28)

This pass re-swept GitHub immediately after Pass 10 and produced **no new code fix**, because the
re-sweep found no new issue and confirmed that every concretely-localizable item already has a fix in
mainline or in flight. It records the verifications that establish this, so a later pass does not
re-investigate the same residuals. Host: aarch64 macOS, ESBMC 8.3.0, master `f343663bde`.

### 15.1 Re-sweep result — no new work surfaced

* **No new SV-COMP issue** since 2026-06-17 (the four ldv-linux-3.14-races drivers + aws-hash pair);
  all are already triaged (§12, §13).
* **Nothing newly closed** since the Pass-9 reconciliation (#5565/#5593 closed 2026-06-26 are already
  in §13.1/§14.1).
* **Pass-10 PR (#5142 parse layer) in flight** as **#5660**
  (`[clang-frontend] apply -Wno-int-conversion universally, not only under --sv-comp`); awaiting review.

### 15.2 Residuals re-checked — already addressed in mainline

* **#5565 / #5138 `fclose(stdout)` static-stream free.** Re-verified this pass: a minimal
  `fclose(stdout)` under `--memory-leak-check` reports `VERIFICATION SUCCESSFUL`, not the issue's
  "invalid pointer freed". The guard already lives in `src/c2goto/library/io.c` `fclose`
  (`if (!__ESBMC_sv_comp() && __ESBMC_is_dynamic[__ESBMC_POINTER_OBJECT(stream)]) free(stream);`),
  which frees only heap streams from `fopen`/`fdopen` and leaves the static standard streams alone,
  while still catching a genuine double-`fclose` of an `fopen`'d stream. The remaining `strcmp`
  argv-string layer "did not surface as the first failure on a standard build" (§13.2) and is not
  independently reproducible here. The `fclose` residual of item 8 is therefore **discharged**; item 8
  reduces to the `strcmp`/argv layer only.

### 15.3 #4438 — log_6_safe.c-amalgamation (neural-network unreach-call false alarm): triaged research-grade

Not previously detailed in the priority list. `c/neural-networks/log_6_safe.c-amalgamation` is a TRUE
`unreach-call` task on which ESBMC reports `false(unreach-call)` under `kinduction` (SV-COMP 26 run,
ESBMC 8.2.0). It is a large floating-point neural-network amalgamation: the spurious counterexample is
a k-induction / float-precision interaction, not a localized frontend or OM gap, and is not
reproducible in isolation without the (large) benchmark and a full-set k-induction validation. Same
research-grade class as #4980 (k-induction) — added to the backlog rather than patched.

### 15.4 Pass-11 running report

**Analysed.** GitHub re-sweep (no new/newly-closed issues); #5565/#5138 `fclose` residual re-verified
discharged; #4438 triaged.

**PRs opened.** None code — the actionable, isolation-verifiable backlog is exhausted; every remaining
item is research-grade or full-benchmark-gated (consistent with the Pass-8 doc-only cadence and the
correctness-first mandate: no heuristic shipped for a soundness-sensitive subsystem that cannot be
validated against the full SV-COMP set in this environment). This pass is the recurring triage-doc
reconciliation, stacked on the #5660 (Pass-10) doc update.

**Duplicated work avoided.** #5565/#5138 `fclose` layer not re-fixed — already guarded in `io.c`
(re-verified §15.2). #4438 not patched — research-grade k-induction/float class. The #5142 parse layer
not re-touched — in flight as #5660.

**Remaining work (priority order, updated 2026-06-28, Pass 11).** As §14.3, with #4438 added and the
#5565 `fclose` half discharged:
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980 / #4438** — k-induction precision: termination ranking recogniser (#4980) and the
   neural-network float counterexample (#4438); both need full-set validation.
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
6. **#5142 residual** — no-overflow precision layer once parsed (x86 + full-benchmark-gated).
7. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
8. **#5565 residual** — `strcmp` argv-string pointer-validity only (the `fclose` half is discharged,
   §15.2); `ESBMC_SVCOMP`/`--sv-comp`-gated.
9. **Close-outs pending CI:** #1470, #4427; witness end-to-end validation for #1471/#1492/#4611.

---

## 16. Pass 12 — #1470 overflow-witness assignment verified emitted + pinned (2026-06-28)

This pass picks the most concretely-verifiable close-out from §15's item 9 — **#1470** (*overflow
witness contains no assignment*) — reproduces it in isolation, confirms the defect is resolved in
mainline, and **pins the fixed behaviour with two regression tests**. The missing-assignment defect is
directly observable by inspecting the emitted GraphML (an external validator is needed only for
end-to-end replay, not to see whether the assignment node is present), so #1470 is verifiable here even
though full UAutomizer/CPAchecker validation is not. Host: aarch64 macOS, ESBMC 8.3.0.

### 16.1 #1470 — overflow witness now records the overflowing variable — VERIFIED FIXED

**Status: defect resolved in mainline; pinned by two regression tests. Recommend closing #1470.**

#1470 (`CWE190_..._int64_t_fscanf_add_08_bad`) reported that ESBMC found the `data + 1` arithmetic
overflow but emitted a GraphML witness with **no assignment to `data`**, so UAutomizer/CPAchecker
could not replay it. Reproduced in isolation with the issue's overflow shape (nondet `int64_t data;
result = data + 1;`) under `--overflow-check --witness-output-graphml -`: the witness now contains

```xml
<edge ...>
  <data key="startline">5</data>
  <data key="assumption">data == 9223372036854775807;</data>
</edge>
```

on the assignment edge — `data` is pinned to `INT64_MAX` (the unique value that overflows signed
`+ 1`), exactly the assignment the issue said was missing. The missing-assignment defect is gone
(addressed by the intervening witness work, e.g. #5038 / the #4611 GraphML cleanup).

**Regression tests** (`regression/esbmc/`):
* `github_1470_overflow_witness_fail` — nondet `int64_t` add ⇒ `VERIFICATION FAILED`, and the stdout
  GraphML must contain `data == 9223372036854775807` (pins the assignment is emitted; the value is
  deterministic so the regex is stable across solvers/architectures).
* `github_1470_overflow_witness` — same shape but `__ESBMC_assume(data < INT64_MAX)` ⇒ `VERIFICATION
  SUCCESSFUL` (pins the overflow check is value-sensitive, not a blanket alarm on nondet operands).

Both pass.

### 16.2 Residual noted — scanf numeric-specifier buffer-overflow check (separate from #1470)

While reproducing, the literal `fscanf("%ld", &data)` form trips a **`buffer overflow on fscanf`**
before the arithmetic overflow: the scanf format parser in `src/goto-programs/goto_check.cpp`
(`fmt_idx`/`limits` loop, ~line 437) records every conversion whose `%` is not followed by a width
digit as `"INF"` length — including numeric conversions like `%ld`/`%d`/`%lld`, which write a single
scalar and cannot overflow a character buffer. Only `%s`/`%[...]` (string conversions) can. This is a
**separate scanf-format-parsing precision matter**, not #1470's witness defect; flagged here for a
future pass (a fix would restrict the INF-length treatment to string conversions, guarded so genuine
unbounded `%s` reads still alarm). The #1470 witness behaviour is correctly exercised via a clean
nondet input, which is the standard SV-COMP way to model `fscanf` input anyway.

### 16.3 Pass-12 running report

**Analysed.** #1470 reproduced, confirmed fixed, and pinned; scanf numeric-specifier INF-length
residual noted.

**PRs opened.** One: `[witness] regression-pin #1470 overflow-witness variable assignment` — two
regression tests pinning that an arithmetic-overflow GraphML witness records the overflowing
variable's assignment. No code change (the defect is already resolved); the tests guard against
regression in the actively-changing witness emitter. Stacked on the #5660/#5662 (Pass-10/11) doc
chain.

**Duplicated work avoided.** #1470 not re-fixed — already resolved; this pass pins and recommends
closure rather than re-patching. The scanf INF-length item is recorded, not speculatively patched
(soundness-sensitive: must not weaken genuine `%s` overflow detection).

**Remaining work (priority order, updated 2026-06-28, Pass 12).** As §15.4, with #1470 moved to
"verified fixed, pending closure" and the scanf numeric-specifier residual added:
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980 / #4438** — k-induction precision (termination recogniser + neural-network float CE);
   full-set validation.
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
6. **scanf numeric-specifier INF-length** — restrict scanf overflow-check INF length to string
   conversions only (§16.2); isolation-verifiable, soundness-sensitive.
7. **#5142 residual** — no-overflow precision layer once parsed (x86 + full-benchmark-gated).
8. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
9. **#5565 residual** — `strcmp` argv-string pointer-validity (`fclose` half discharged, §15.2).
10. **Close-outs:** #1470 (verified fixed §16.1, recommend closing), #4427; witness end-to-end
    validation for #1471/#1492/#4611.

---

## 17. Pass 13 — scanf numeric-specifier fix shipped + #4611 witness items verified resolved (2026-06-28)

This pass converts the §16.2 scanf residual into a shipped fix, and discharges the concrete,
isolation-verifiable half of the #4611 witness close-out. Host: aarch64 macOS, ESBMC 8.3.0.

### 17.1 scanf numeric-specifier INF-length (§16.2 item 6) — FIXED

`goto_checkt::input_overflow_check` (`src/goto-programs/goto_check.cpp`) treated *every* width-less
scanf conversion as `INF` → unconditional `buffer overflow on scanf`, including numeric/char/pointer
conversions (`%d`, `%ld`, `%f`, `%c`) that write a single fixed-size scalar and cannot overflow a
buffer. The spurious overflow preceded and masked the intended property (the #1470 CWE190 `data + 1`
overflow). **Fix:** restrict the `INF` rule to genuine unbounded string conversions — `%s`, the
`%[...]` scanset, and the wide-string `%S`/`%ls` — by skipping C length modifiers to reach the
conversion specifier; width-less non-string conversions become a `BOUNDED` sentinel that is skipped.
Sound (diagnostic-precision only; GCC/glibc semantics unchanged), code-reviewed (two findings applied:
the `%S` wide-string soundness edge, and pinning `scanf-false-3`'s genuine NULL-deref property after it
had relied on the removed false positive). Three new regression tests + one repaired. **PR #5666** (off
master, independent of the doc chain).

### 17.2 #4611 — witness GraphML concrete items verified already-resolved

#4611 lists four GraphML witness tweaks from the defunct `adjust_type` branch. Audited the current
emitter (`src/goto-symex/witnesses.cpp`) by inspecting `--witness-output-graphml -` output on a failing
assert (heap write) and a function-return counterexample:

* **Key naming (item 1).** Current output already emits the modern-spec `enterFunction` /
  `returnFromFunction` keys (the 2020-onwards format the issue identifies as conformant). No change
  needed.
* **No duplicate nodes (item 2).** Node/edge counts are minimal and consistent (4 nodes/3 edges for the
  heap-assert trace, 5/4 for the function-return trace); no repeated initial edge.
* **No out-of-file trace steps (item 3).** Despite `<assert.h>`/`<stdlib.h>` on the path, every emitted
  edge carries only user-file `startline`s — no system-header pollution.

So #4611's concrete, isolation-verifiable items are **already satisfied** in mainline; the only residual
is the external-validator audit (run a witness through CPAchecker/Ultimate to confirm key acceptance),
not reproducible in this environment (no validators). #4611 reduces to that validation-only residual,
the same class as #1471/#1492.

### 17.3 Pass-13 running report

**Analysed.** §16.2 scanf residual fixed (#5666); #4611 concrete items verified resolved; in-flight PR
states reconciled.

**PRs opened.** This doc reconciliation (stacked on the #5664 doc chain). The code deliverable of the
pass is **#5666** (scanf). In flight from earlier passes: **#5660** (#5142 parse layer), **#5664**
(#1470 witness-assignment regression tests).

**Duplicated work avoided.** #4611 not patched — concrete items already in mainline (§17.2). The
research-grade backlog (value-set #5145/#5393/#5394, reachability #5400/#5138, va_list #5012,
k-induction #4980/#4438, atomics #4432, device-API) is unchanged — none is isolation-verifiable here,
so none was patched with a heuristic.

**Remaining work (priority order, updated 2026-06-28, Pass 13).** Item 6 (scanf) closed; #4611 moved to
the validation-only residual:
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980 / #4438** — k-induction precision (termination recogniser + neural-network float CE);
   full-set validation.
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
6. **#5142 residual** — no-overflow precision layer once parsed (x86 + full-benchmark-gated).
7. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
8. **#5565 residual** — `strcmp` argv-string pointer-validity (`fclose` half discharged, §15.2).
9. **Validation-only close-outs (need CPAchecker/Ultimate):** witness end-to-end validation for
   #1471 / #1492 / #4611 (concrete items resolved §17.2); #4427 false-negative; #1470 (verified fixed
   §16.1, recommend closing).

---

## 18. Pass 14 — close the verified-fixed witness issues + scanf parser false-negative (2026-06-28)

This pass reconciles two concrete state changes since Pass 13: three witness issues were verified fixed
and **closed**, and the scanf overflow-check PR (#5666) was extended with a second, independent
correctness fix. No new SV-COMP issue has been filed (newest still 2026-06-17). Host: aarch64 macOS.

### 18.1 Issues closed (3)

A full validity re-audit of the 20 open SV-COMP issues confirmed three are effectively fixed in
mainline; they were closed with evidence-bearing comments:

| # | Property | Why closed |
|---|---|---|
| #1470 | overflow witness assignment | Witness now emits `data == INT64_MAX` on the assignment edge (§16.1); regression tests in #5664. |
| #4611 | witness GraphML keys / dup nodes | Concrete items already in mainline — modern `enterFunction`/`returnFromFunction` keys, no duplicate nodes, no system-header trace pollution (§17.2). Residual is validator-audit only. |
| #1471 | struct constant in witness not parseable | Struct constants are no longer emitted as unparseable brace literals (omitted instead, per #5038); the parse failure is gone. |

Open SV-COMP count: 20 → **17**.

### 18.2 scanf overflow-check — second fix: `%%`/`%*` false negative (extends #5666)

The §16.2 scanf work (PR #5666) initially fixed the *false positive* on width-less numeric
conversions (§17.1). Reviewing that change surfaced an independent *false negative* in the same parser:
a `%%` literal and a `%*...` assignment-suppressed directive consume **no** argument, but the format
parser pushed a phantom `limits[]` entry for each. That misaligned `limits[]` against the argument list
and tripped `too few arguments for format specifiers`, which **abandons the entire scanf overflow
check** — silently dropping a genuine overflow on a following `%s`. Reproduced: `scanf("%*d %s", buf)`
and `scanf("100%% %s", buf)` into `char[4]` were reported SUCCESSFUL. **Fix:** skip `%%`/`%*` during
parsing so real conversions line up with their arguments. Now correctly FAILED; no new false positive;
full `cstd` suite green. Added `scanf_suppressed_assignment_overflow_bug`. PR #5666 now bundles both the
false-positive and false-negative parser fixes.

### 18.3 Pass-14 running report

**Analysed.** Validity re-audit of all 20 open SV-COMP issues; three closed (§18.1); scanf parser
false-negative fixed and folded into #5666 (§18.2); in-flight PR states reconciled (no CI failures).

**PRs.** Code: **#5666** (scanf parser — both directions). In flight: **#5660** (#5142 parse layer),
**#5664** (#1470 regression tests). This doc update continues the #5667 reconciliation chain (Pass 13 +
14) rather than opening a fourth stacked doc PR.

**Duplicated work avoided.** The closed issues (#1470/#4611/#1471) are not re-touched. The research-grade
backlog is unchanged — none isolation-verifiable here, so none patched with a heuristic.

**Remaining work (priority order, updated 2026-06-28, Pass 14).** The witness close-outs #1470/#4611/
#1471 are gone (closed); the scanf parser is complete:
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
2. **#5400 / #5138** — value-set reachability for `valid-memtrack`; #5138 narrowed to the `strcmp`
   argv-string residual (other layers fixed by #5564/#5624 + the `fclose` guard).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980 / #4438** — k-induction precision (termination recogniser + neural-network float CE);
   full-set validation.
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*` (libvsync mcslock; large
   benchmark + concurrency k-induction, not isolation-reproducible here).
6. **#5142 residual** — no-overflow precision layer once parsed (x86 + full-benchmark-gated).
7. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399/#4439
   blocker.
8. **#4427** — false-negative on unreach-call (missed bug; hardest class).
9. **Validation-only (need CPAchecker/Ultimate):** #1492 FP witness validation; #1447 benchmark
   incorrect-verdicts umbrella; #1440 wrapper-script sub-property handling (speculative, no property
   file yet).
