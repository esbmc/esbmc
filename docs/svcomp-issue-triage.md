# SV-COMP Issue Triage & Fix Plan

**Last updated:** 2026-07-01 (**Pass 16 (§20)** followed up on the x86_64 host with the aws-hash
byte-addressed pointer-read-consistency cluster: **#5145** reproduces the same `Bug found (k = 1)`
verdict Pass 6 recorded on aarch64/macOS, now confirmed unchanged on native x86 against current master;
**#5393**/**#5394** get their **first genuine local reproduction** (Pass 7 only presumed cluster
membership without running them) and exactly match their reported `k = 129` depth. Disposition unchanged
— RCA #5369's open half needs multi-session SSA-trace/solver-internal correlation, too high-risk for a
speculative fix (the one prior shortcut was refuted as unsound, PR #5283); no PR/code this pass, value
is closing the reproducibility gap for the next session. **Pass 15 (§19)** ran on the first available
x86_64 Linux host and cleared the environment wall Pass 14 hit: **#4427** got the x86 CI cross-check
Pass 13 asked for — no unsound TRUE, **recommend closing**; **#4432** reached 2733 thread interleavings
with no violation, downgraded from "host-gated" to "not reproducible within a generous local budget";
**#4438** needs no new triage — open PR #4480 just needs a rebase + merge; **#5142**'s no-overflow layer
reproduces on x86 once `--sv-comp` is added. Pass 12 consolidated two parallel Pass-11 PRs that both
appended a `## 15` section: **#5730** (witness-closure reconciliation) stays **§15**; **#5735** (#5565
closed by #5650; `--sv-comp` flag retirement of the compile-time macro; aws-hash scouting) is **§16**.
Canonical remaining-work list is now §20.4. See §16/§17/§18/§19/§20. Pass 11 in §15, Pass 10 in §14,
Pass 9 in §13, Pass 8 in §12, Pass 7 in §11)
**Last updated:** 2026-07-01 (**Pass 15 (§19)** ran on the first available **x86_64 Linux** host and
cleared the environment wall Pass 14 hit: **#4427** got the x86 CI cross-check Pass 13 asked for — ~36
CPU-minutes across two k-steps, no unsound TRUE, **recommend closing**; **#4432** reached 2733 thread
interleavings (7.4× Pass 14's aarch64 depth) with no violation, downgraded from "host-gated" to
"not reproducible within a generous local budget"; **#4438** needs no new triage — open PR #4480 already
fixes it and just needs a rebase + merge; **#5142**'s no-overflow layer is confirmed reproducible on x86
once `--sv-comp` is added (the deliberate `-Wno-int-conversion` gate from Pass 10), landing on a sibling
shift site to the one filed. Pass 12 consolidated two parallel Pass-11 PRs that both appended a `## 15`
section: **#5730** (witness-closure reconciliation) stays **§15**; **#5735** (#5565 closed by #5650;
`--sv-comp` flag retirement of the compile-time macro; aws-hash scouting) is **§16**. Canonical
remaining-work list is now §19.5. See §16/§17/§18/§19. Pass 11 in §15, Pass 10 in §14, Pass 9 in §13,
Pass 8 in §12, Pass 7 in §11)
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

## 14. Pass 10 — #5142 parse-blocker layer: PR #5660 rejected, `--sv-comp` gate kept (2026-07-01)

PR #5660 (`[clang-frontend] apply -Wno-int-conversion universally, not only under --sv-comp`) proposed
moving `-Wno-int-conversion` out of the `--sv-comp` block in `clang_c_languaget::build_compiler_args`
(`src/clang-c-frontend/clang_c_language.cpp`) so a plain `esbmc foo.c` run on clang 15+ would no longer
hard-error on the implicit int↔pointer conversions common in CIL-preprocessed kernel inputs (the #5142
parse layer, §12.5). It was closed without merging; this entry records why, so the same relaxation is
not re-proposed in a later pass.

**Why closed.** The PR's own justification was "GCC (the compiler ESBMC targets) only *warns* on this
construct, so matching GCC is the correct behaviour." That reasoning targets the wrong compiler: ESBMC's
C frontend *is* clang — it is the parser that produces the AST ESBMC converts to GOTO. A plain ESBMC run
should therefore surface clang's own default diagnostic severity (a hard error on clang 15+ for
`-Wint-conversion`), not silently downgrade it to GCC's warn-only behaviour for every user. Doing so
would mean ESBMC accepts, by default, a class of int↔pointer conversions that the very compiler
front-ending it rejects — a widening of accepted input with no counterpart hard case for a plain-mode
user to detect.

**Why `--sv-comp` stays the only gate.** The `--sv-comp` relaxation is a narrow, justified accommodation:
SV-COMP benchmarks are frequently CIL-preprocessed kernel sources (the `m0_drivers-media-rc-imon`
sentinel-pointer initialiser cited in §12.5 is one instance) that rely on GCC-accepted implicit
conversions as an artefact of that preprocessing pipeline, not as a property of the code a general
ESBMC user writes. Scoping the relaxation to `--sv-comp` keeps that accommodation local to the
benchmark-replay use case instead of promoting it to a universal default.

**Disposition.** Master's pre-#5660 behaviour is retained unchanged: `-Wno-int-conversion` applies only
inside the `--sv-comp` branch of `build_compiler_args`; a plain run still reports `ERROR: PARSING ERROR`
on GCC-acceptable, clang-15+-rejected implicit int↔pointer conversions, matching clang's own default.
#5142 stays open against both of its layers: the parse-blocker layer (relaxation-under-`--sv-comp`-only
confirmed as the intended behaviour, not a bug — see above) and the no-overflow precision layer (whether
the benchmark verdict is correct once parsed under `--sv-comp`; still x86 + full-benchmark-gated,
unreproducible in isolation here, per §13.3 item 6).

---

## 15. Pass 11 — recover work stranded on an orphaned branch + reconcile witness closures (2026-07-01)

**Process finding, not a benchmark triage.** While resuming this plan, the #5660 PR chain (§14) turned
out to have a bookkeeping defect worth recording so it is not repeated. Follow-up PRs #5662, #5664 and
#5667 — a Pass-11 reconciliation, the `#1470` witness-assignment regression tests, and a Pass-13/14
reconciliation — were opened **stacked on #5660's branch** (`fix/5142-int-conversion-universal`) with
the intent to retarget to `master` once #5660 merged. When #5660 was instead **closed without merging**
(§14), the three stacked PRs had already been merged **into that side branch**, not into `master` — so
they show as `MERGED` on GitHub while being completely absent from `master`'s history
(`git merge-base --is-ancestor <commit> master` is false for all three). This document's own Pass 10
entry (§14) did not catch the gap because it only reconciled the rejected code change, not the doc/test
PRs stacked above it.

**What this stranded.** Two regression tests (`regression/esbmc/github_1470_overflow_witness{,_fail}`,
from #5664) and three doc-only passes (Pass 11/13/14 content, from #5662/#5667) never reached `master`.
Separately, **PR #5666** (`[goto-check] fix scanf overflow-check false pos/neg on numeric, %%/%*`) *was*
based directly on `master` and merged cleanly (2026-06-29) — it is present and unaffected.

**Recovery in this pass.**
* Cherry-picked the two `github_1470_overflow_witness{,_fail}` test directories from `33121f3db5` (no
  docs, no code — pure test addition) onto `master`. Both **pass** on current master
  (`ctest -R github_1470`): the `_fail` variant's GraphML witness contains `data == 9223372036854775807`
  on the assignment edge; the companion (`__ESBMC_assume(data < INT64_MAX)`) variant is
  `VERIFICATION SUCCESSFUL`.
* Independently re-verified (not just trusted the stranded doc text) the three claims behind the
  witness-issue closures below, directly against current `master`:
  - **scanf fix (#5666) is in master.** `goto_check.cpp`'s `input_overflow_check` restricts the `INF`
    sentinel to `%s`/`%[...]`/`%S`/`%ls`; width-less numeric/char conversions get `BOUNDED`. Full `cstd`
    label (142 tests, including the four new scanf regressions) passes 142/142.
  - **Witness key format (#4611).** `witnesses.cpp` emits the modern-spec `enterFunction`/
    `returnFromFunction` GraphML keys (confirmed at `witnesses.cpp:337-347,590-592`) — the naming the
    issue asked for, not the deprecated ones.
  - **Struct-constant witness handling (#1471).** `witnesses.cpp:1175` special-cases `is_struct_type`
    when rendering witness values, consistent with the issue's fix being present.
* Confirmed via `gh issue view` that **#1470, #1471 and #4611 are closed** on GitHub (all closed
  2026-06-28, cross-referenced from #5664/#5667's discussion) — the closures themselves are real and
  already reflected upstream even though the supporting doc/test commits were stranded. Current open
  SV-COMP issue count: **17** (`gh issue list --label SV-COMP --state open`), matching what the
  stranded Pass-14 draft had already computed (20 → 17).

**Why recover rather than re-derive from scratch.** The stranded commits' analysis was sound — this
pass independently re-checked each concrete claim above rather than taking the orphaned branch's word
for it, per the project's verification policy. Re-deriving the regression tests from scratch would have
duplicated exactly the two files already written and reviewed on the stranded branch; cherry-picking
them (with independent re-validation that they still pass on current `master`) was the correct-cost
path.

**No new PR this pass for #4427.** The remaining "close-out pending CI" item, #4427
(`megaraid_mm.ko` false negative on `unreach-call`), needs the actual sv-benchmarks `.i` file fetched
from GitLab and a multi-minute `--k-induction --max-inductive-step 3` run to re-confirm the Pass-6
"fixed-on-master by #4484" claim (§10.4) now that an x86_64 host is available (it was previously only
confirmed on aarch64-adjacent reasoning). That fetch-and-run is scoped as the next task (§15.2) rather
than rushed here.

### 15.1 Pass-11 running report

**Analysed.** The #5660 stacked-PR bookkeeping gap (root-caused via `gh pr view --json baseRefName` on
#5660/#5662/#5664/#5666/#5667 and `git merge-base --is-ancestor`); independent re-verification of the
scanf fix, witness key format, and struct-constant handling on current `master`.

**PRs opened.** One: recovers `github_1470_overflow_witness{,_fail}` (test-only, no ESBMC source
change) and this doc reconciliation. No code fix — the code fix (#5666) was already on `master`.

**Duplicated work avoided.** Did not re-implement the scanf fix (already in `master` via #5666). Did
not re-write the Pass 11/13/14 analysis prose verbatim — re-verified its conclusions independently and
recorded them in this pass's own words.

**Remaining work (priority order, updated 2026-07-01, Pass 11).** Unchanged from Pass 10 except #4427
promoted to the top of the concretely-actionable (non-research-grade) items, since it only needs a
benchmark fetch + a run, not a design change:
1. **#4427** — re-fetch `megaraid_mm.ko` from sv-benchmarks and re-run the issue's exact
   `--k-induction --max-inductive-step 3` command on the current x86_64 host to confirm the Pass-6
   "fixed by #4484" claim (§10.4); close if confirmed, re-triage if not.
2. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
3. **#5400 / #5138** — value-set reachability for `valid-memtrack`; #5138 narrowed to the `strcmp`
   argv-string residual (other layers fixed by #5564/#5624 + the `fclose` guard).
4. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
5. **#4980 / #4438** — k-induction precision (termination recogniser + neural-network float CE);
   full-set validation.
6. **#4432** — data-race-checker interleaving reduction on `__atomic_*`.
7. **#5142 residual** — no-overflow precision layer once parsed under `--sv-comp` (x86 +
   full-benchmark-gated).
8. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
9. **Validation-only (need CPAchecker/Ultimate):** #1492 FP witness validation; #1447 umbrella;
   #1440 wrapper-script sub-property handling.

### 15.2 Next task selected for the following pass

**#4427** (`megaraid_mm.ko` `unreach-call` false negative) — the only remaining "close-out pending CI"
item, and the only item in the backlog that is a bounded verification task (fetch one `.i` file, run one
command) rather than a design effort. Fetch
`c/ldv-linux-4.2-rc1/linux-4.2-rc1.tar.xz-43_2a-drivers--scsi--megaraid--megaraid_mm.ko-entry_point.cil.out.i`
from the sv-benchmarks GitLab (read-only, untrusted input, matching this document's established
methodology), run the issue's exact wrapper/ESBMC invocation, and either close the issue (if #4484
resolved it, matching the ground truth `false` now correctly detected) or re-open the RCA (if it still
reports the unsound `true`, in which case this is a live soundness bug and the highest-severity item in
the backlog).
## 16. Pass 12 — consolidate the two parallel Pass-11 merges (#5730 + #5735); carry the #5565/#5530 reconciliation (2026-07-01)

**Process finding.** Two independent Pass-11 PRs landed on `master` within minutes of each other and
both appended a `## 15. Pass 11` section plus a second `**Last updated:**` banner: #5730 (§15 —
orphaned-branch recovery + witness closures) and #5735 (this block — the #5565/#5530 reconciliation). The
parallel merge left the document with a duplicate `## 15` heading, duplicate `### 15.1/15.2/15.3`
subsection numbers, and two banners. This Pass-12 pass repairs the structure: #5730's section keeps
**§15** (it also owns the canonical, more-current remaining-work list at §15.1 and the selected next
task #4427 at §15.2); the #5735 content below is renumbered to **§16** and its subsections to 16.x. No
content is dropped — the #5565/#5530 findings are still accurate and complementary to §15. The one
reconciliation between the two: §16.3's remaining-work list is **superseded by §15.1** (which adds #4427
at the top and #4438/#1492/#1447/#1440); it is kept here only as the record of what #5735 saw.

The #5565/#5530 reconciliation carried forward from PR #5735 follows.

### 16.1 #5565 residual — CLOSED by #5650; `strcmp` half resolved upstream of it

* **`fclose(stdout)` static-stream free — FIXED.** Merged **PR #5650** (`[om] fclose: free only heap
  streams so fclose(stdout) is sound`, 2026-06-27) guards the `free(stream)` in ESBMC's `fclose`
  operational model on `__ESBMC_is_dynamic[__ESBMC_POINTER_OBJECT(stream)]`
  (`src/c2goto/library/io.c:121`). Outside SV-COMP mode `fclose` previously called `free(stream)`
  unconditionally; the standard streams `stdin`/`stdout`/`stderr` are not heap objects (`stdout`
  resolves to `invalid-object`), so coreutils' `close_stdout` `atexit` handler — `fclose(stdout)` —
  raised a spurious `dereference failure: invalid pointer freed`. The guard releases only streams ESBMC
  itself allocated (`fopen`/`fdopen`), keeping a genuine double-`fclose` of an `fopen`'d stream
  detectable while making `fclose(stdout)` sound. This is exactly the residual §13.2's *Residual* note
  and §13.3 item 8 named.
* **`strcmp` invalid-pointer — resolved upstream of it.** The `strcmp` half #5565 predicted was
  downstream of the `optarg` layer: `strcmp(optarg, …)` deref'd the unmodeled-`getopt_long` invalid
  pointer, fixed in Pass 9 (§13.2, merged as `[om] model getopt_long/getopt_long_only …`). With
  `optarg` bound to a real argv element, the `strcmp` deref is on a valid pointer.
* **Issue state.** **#5565 is CLOSED.** Its two-layer residual is fully discharged (`fclose` by #5650,
  `strcmp` by the Pass-9 `getopt_long` OM). No further ESBMC-side work is owed on #5565.

### 16.2 `ESBMC_SVCOMP` build macro retired → runtime `--sv-comp` flag (#5530)

Merged **PR #5530** (`[svcomp] Replace ESBMC_SVCOMP build macro with runtime --sv-comp flag`,
2026-06-25) removed the compile-time `-DESBMC_SVCOMP=On` CMake option that baked `__ESBMC_SVCOMP` into
the operational models, replacing it with the runtime `--sv-comp` flag already threaded through the
OMs (`__ESBMC_sv_comp()`). **Consequence for this document:** every reproduction recipe that says
"needs an `ESBMC_SVCOMP` build" / "build with `-DESBMC_SVCOMP=On`" (§8.4, §11.6, and the §13.3 item-8
gating note) is obsolete — the SV-COMP-mode OM behaviour is now selected per-run with the `--sv-comp`
flag on a stock binary, no dedicated build required. Those historical pass sections are left intact as
the record of how the run was done at the time; the corrected instruction is: **reproduce
`--sv-comp`-gated OM behaviour by passing `--sv-comp` at run time.** #5138 (§16.3 item 2) is the one
open issue whose reproduction recipe this simplifies.

### 16.3 Pass-11 running report (PR #5735)

**Analysed.** #5565 (now CLOSED — #5650 + Pass-9 `getopt_long`) reconciled against current GitHub
state; the #5530 `ESBMC_SVCOMP`→`--sv-comp` migration and its effect on every "`ESBMC_SVCOMP`-build"
reproduction recipe in this document. Open SV-COMP label re-enumerated: **no issue filed since Pass 9**
(newest open items #5396–#5398 predate Pass 8).

**PRs opened.** None code — the two reconciled items (#5650, #5530) merged independently before this
pass; every still-open backlog item is the research-grade / host-blocked / in-flight class of §13. This
pass is the recurring triage-doc update (established cadence: Pass 4 → #5234, Pass 6 → #5389, Pass 7 →
#5405, Pass 8 → doc-only, Pass 10 → #5728).

**Duplicated work avoided.** #5565 not re-fixed — CLOSED by #5650 (`fclose`) + the Pass-9 `getopt_long`
OM (`strcmp`). No re-proposal of a compile-time SV-COMP build — retired by #5530.

**Remaining work — SUPERSEDED by §15.1.** The list below is what PR #5735 saw; the canonical,
more-current priority list is **§15.1** (from the sibling Pass-11 PR #5730), which adds #4427 at the top
plus #4438 and the validation-only items #1492/#1447/#1440. Item 8 here (#5565 residual) is dropped —
closed. Retained verbatim as the #5735 record, with #5138's recipe simplified to the runtime `--sv-comp`
flag:
1. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency: the general mechanism
   is RCA-closed as **#5369** (fresh, non-deduplicated pointer identity minted per read through an
   unresolvable/byte-reconstructed pointer, in `make_failed_symbol` and `convert_typecast_to_ptr`); a
   sound fix must tie the minted identity across reads of the same location. Precision, not soundness.
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation of
   "reachable"); #5564 advanced the live-stack-reachability half. **#5138 now reproduces with a stock
   binary under `--sv-comp`** (no `-DESBMC_SVCOMP=On` build — see §15.2).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980** — termination ranking recogniser for side-effect-only call bodies; overlaps open PR #4919.
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*` (x86 host repro blocker).
6. **#5142** — no-overflow precision layer only; the parse-blocker layer is settled (§14: `--sv-comp`
   is the intended, sole gate for `-Wno-int-conversion`; PR #5660 rejected).
7. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker.
8. **Close-outs pending CI:** #1470, #4427; witness end-to-end validation for #1471/#1492/#4611.

### 16.4 Next-task scouting — #5145 / #5393 / #5394 (aws-hash) is research-grade, no loop-scoped fix

Scouted the top remaining-work item to confirm whether a sound localized fix is available before the
next coding pass commits to it. **Conclusion: not loop-scoped — the tractable half is already shipped
and the open half is research-grade.** Evidence:

* **RCA #5369 (CLOSED)** localises the mechanism precisely: a read through an *unresolvable* /
  byte-reconstructed pointer mints fresh `(object, offset)` identity at each layer
  (`make_failed_symbol` in `src/pointer-analysis/dereference.cpp`; `convert_typecast_to_ptr` in
  `src/solvers/smt/smt_casts.cpp`), so two reads of the same location with no intervening write can
  compare unequal — a spurious "location changed without a write" counterexample. **Precision, not
  soundness** (over-approximation → false alarm; ground truth `true`).
* **The k=1 base case is solved and validated:** congruence on the int→ptr reconstruction plus a
  location-keyed failed-symbol cache passes the full 148-test pointer/dereference/memory suite and the
  `github_2153{,_2..5}` soundness guards (RCA §"What has been validated").
* **The open half is k≥2**, where the dereference-result expressions are structurally identical across
  reads yet fail to deduplicate at the SMT layer. The RCA lists **two unresolved hypotheses** — (a)
  `convert_ast` cache eviction across push/pop, (b) post-dereference SSA re-versioning of a shared
  symbol — and states that separating them "needs SSA-trace / solver-internal correlation." That is a
  multi-session solver-internals investigation, not a localized OM/frontend patch.
* **Guard rails.** The one already-tried shortcut — forcing the reconstructed value non-zero — was
  **refuted as unsound** and closed (PR #5283); any fix must stay gated on the `github_2153` guards,
  which *rely* on a byte-reconstructed pointer being free to alias a valid object (PR #3924). This is a
  soundness tripwire, so a speculative dedup change is high-risk.
* **Host.** The aws-hash harnesses are x86 SV-COMP benchmarks; this repo's own note (§13 host banner)
  records SV-COMP repro was done on x86_64 Linux — the k≥2 divergence is not reproducible in isolation
  on the aarch64/macOS host available to this pass, so even a candidate fix could not be validated here.

**Disposition.** #5145/#5393/#5394 stays flagged **research-grade (k≥2 SMT re-versioning)** — the same
"reproduced, sound over-approximation, no localized fix" class as the LDV-environment cluster. It is #5735's
item 1; in the reconciled §15.1 list it is item 2, behind **#4427** (the selected next task, §15.2 —
a bounded fetch-and-run, not a design effort).

**Item-8 witness scouting (this Pass-12 pass).** Also scouted #5735's item-8 (witness) before adopting
#4427: #1470/#1471/#4611 are all **CLOSED** (reconciled by §15), so the residual witness *quality*
concern (GraphML violation path in `goto_trace.cpp:302-362` lacks the system-header file-path filter and
symbol dedup the YAML path has at `witnesses.cpp:1292-1308`) is not a live issue and any change to it is
soundness-gated on a witness validator (CPAchecker/Ultimate) absent here. So item 8 offers no
loop-scoped verifiable fix either — confirming #4427 as the correct next task over it.

---

## 17. Pass 13 — #4427 megaraid_mm.ko false-negative no longer reproduces (fixed by #4484) (2026-07-01)

Executed the next task selected in §15.2: fetched the benchmark and re-ran #4427's exact reproducer on
current master. **Result: the false-negative soundness bug does not reproduce — #4484 fixed it.**
Confirmed on an aarch64/macOS host; an x86 CI cross-check is still advisable before closing (this
document's standing recommendation for #4427).

**What was run.** Fetched
`c/ldv-linux-4.2-rc1/linux-4.2-rc1.tar.xz-43_2a-drivers--scsi--megaraid--megaraid_mm.ko-entry_point.cil.out.i`
(5371 lines, read-only from the sv-benchmarks GitLab raw endpoint; contains the `ERROR` label; no
`__regparm__` to strip on this host) and ran the issue's **exact** underlying ESBMC command
(`--k-induction --max-inductive-step 3 --k-step 2 --unlimited-k-steps --error-label ERROR
--enable-unreachability-intrinsic --interval-analysis --floatbv --64 …`) against the local ESBMC 8.3.0
binary built from a master that includes **#4484** (commit `b49db0942d`,
`[k-induction] deep-dive batch: … pointer handling …`).

**Root cause of the original false negative (now closed).** The megaraid ioctl path dispatches through
function pointers (`*adp->issue_uioc`; the issue's own log shows `No target candidate for function call
*adp->issue_uioc`). On ESBMC 8.2.0 the k-induction **inductive step ran on this fn-ptr program and
unsoundly concluded `VERIFICATION SUCCESSFUL`** (the reported false negative; ground truth `false`).
The current binary emits, at `k = 1`:

```
WARNING: k-induction does not support function pointer calls yet. Disabling inductive step
```

— i.e. **#4484 detects the function-pointer calls and disables the (unsound-for-fn-ptr) inductive
step**, removing the path that produced the spurious TRUE.

**Observed behaviour on current master.** With the inductive step disabled, k-induction reduces to
base-case falsification + forward-condition proof:
* **Base case** finds *no* bug through `k = 17` (`No bug has been found in the base case` at each step) —
  sound; the real violation needs deeper unwinding than the run reached.
* **Forward condition** never proves the property (`The forward condition is unable to prove the
  property`) because the driver's loops are not fully unwound at these depths.
* Neither converging, the run **escalates and times out at 280 s** (no `VERIFICATION SUCCESSFUL`
  anywhere in 4533 log lines) = **sound `UNKNOWN`**, not the previous unsound TRUE.

**Disposition.** #4427's soundness bug (false negative) is **fixed by #4484** — the unsound
`VERIFICATION SUCCESSFUL` no longer occurs; the tool now safely fails to conclude rather than reporting
an incorrect TRUE. This confirms the Pass-6 "fixed-on-master by #4484" hypothesis with a concrete local
run. **Recommend closing #4427 after an x86_64 CI cross-check** (host parity with the original SV-COMP
26 run). A residual, lower-priority follow-up: the benchmark now yields `UNKNOWN` (timeout) rather than
the ground-truth `false` — obtaining the actual counterexample would need deeper base-case unwinding or
a targeted falsification run, but that is a *precision/perf* item, not the *soundness* bug the issue
filed.

### 17.1 Pass-13 running report

**Analysed.** #4427 — exact-command re-run on current master; root-caused the original false negative to
the pre-#4484 inductive step running on function-pointer dispatch, and confirmed #4484's fn-ptr guard
(`Disabling inductive step`) removes the unsound TRUE.

**PRs opened.** This entry extends the Pass-12 consolidation PR (#5739) rather than opening a third
doc PR against a master that still carries the pre-consolidation `## 15` collision — and, per the
stranded-PR lesson recorded in §15, is based directly on `master`, not stacked on a branch that could be
closed unmerged.

**Remaining work.** §15.1 list, with #4427 moved from "selected next task" to **recommend-close pending
x86 CI**. No remaining item is loop-scoped without infra (x86 host + SV-benchmarks + witness validator):
the rest are research-grade (#5145 k≥2, #5012, #5400/#5138), overlap in-flight PRs (#4980/#4919), or
x86-host-gated (#4432, #5142 no-overflow layer, the LDV clusters).

---

## 18. Pass 14 — #4432 mcslock host-gating probe; confirm-and-close backlog exhausted (2026-07-01)

Pass 12's consolidation merged (#5739), so `master`'s report is structurally clean (single banner,
monotonic §14–§17). With #4427 resolved (§17), the "already-fixed, confirm-and-close" backlog is
**exhausted** — every other open SV-COMP item is research-grade, blocked on an in-flight PR, or
x86-host-gated. This pass probes the one remaining open question that a local run can answer — **is the
#4432 `mcslock` no-data-race false alarm host-independent, or x86-only?** — and records the answer so the
next pass does not re-attempt it blind.

### 18.1 #4432 — x86-host-gated (regparm parse artifact + no local repro within budget)

Fetched `c/libvsync/mcslock.i` and ran #4432's exact command
(`--incremental-bmc --data-races-check-only --no-por --smt-symex-guard --bitwuzla --32 …`).

* **Parse blocker (x86 artifact).** As-fetched, clang rejects the file on aarch64/macOS:
  `error: 'regparm' is not valid on this platform` (three `__attribute__((__regparm__(1)))` sites). This
  is the known x86-only `.i` artifact (repo methodology: strip `__regparm__` to parse SV-COMP `.i` on
  non-x86). After stripping the three attributes the file parses and symex runs.
* **No local reproduction of the false race within budget.** The stripped run explored **371+ thread
  interleavings** under incremental-BMC and **timed out at 260 s with no `VERIFICATION FAILED` and no
  data-race counterexample** — it did *not* surface the spurious `false(no-data-race)` the x86 CI
  reported (which fired at a specific interleaving, "State 57 … thread 3"). This is **inconclusive, not
  a clean non-repro**: the triggering interleaving may lie deeper than 260 s reached, and a
  32-bit-*modelled*-on-aarch64 run is not authoritative for an x86 CI verdict regardless.
* **Disposition.** #4432 is **x86-host-gated for authoritative triage** — it needs the original x86_64
  Linux host both to parse without stripping and to reach the reported interleaving in the SV-COMP time
  budget. The eventual fix (data-race-checker interleaving reduction on `__atomic_*`) is independently
  research-grade. Matches the memory-note uncertainty ("some wrong-verdicts are x86-only") — resolved
  here to: *treat #4432 as x86-gated; do not re-probe locally.*

### 18.2 Pass-14 running report

**Analysed.** #4432 reproducibility on aarch64/macOS (parse + 260 s incremental-BMC run); confirmed the
`regparm` x86 parse artifact and no in-budget local repro of the false race.

**PRs opened.** One doc PR (this entry), based directly on `master` (clean post-#5739). No code — #4432
offers no loop-scoped verifiable fix on this host.

**Environment wall (explicit).** The SV-COMP loop has reached the limit of what this aarch64/macOS
environment can verify. Every open item now needs one of: an **x86_64 Linux host** with the
SV-benchmarks tree (#4432, #5142 no-overflow, #4438 via its `--32` float path, the #5396–#5399 LDV
clusters), a **witness validator** (#1492 and the closed-but-unvalidated witness set), a **merged
in-flight PR** (#4438→#4480, #4980→#4919), or **multi-session solver-internals work** (#5145 k≥2, #5012
`va_list`, #5400/#5138 value-set reachability). Recommended next step for the maintainer: run the
recommended-close confirmations (#4427, and the witness closures) on the x86 CI, and treat the
research-grade items as scheduled design work rather than triage-loop candidates.

---

## 19. Pass 15 — native x86_64 cross-check: #4427 confirmed closeable, #4432 downgraded, #5142 `--sv-comp` unblocks parse (2026-07-01)

This pass ran on the **x86_64 Linux host** Pass 14 flagged as missing infra, removing the
aarch64/macOS host-gating for #4427, #4432 and #5142 in one session. Correction while re-reading §18's
environment wall: it grouped "the #5396–#5399 LDV clusters" with the x86-host-gated items, but this
document's own reference-binary note (top banner) records that **Pass 8 already ran on an x86_64 Linux
host** and reproduced all four there — they were never host-gated, and this pass does not re-touch them.

### 19.1 #4427 — x86 CI cross-check delivered: no unsound TRUE; recommend closing

Pass 13's aarch64/macOS run timed out at 280 s with a sound `UNKNOWN` and explicitly asked for "an x86
CI cross-check... before closing." This pass fetched the same
`megaraid_mm.ko-entry_point.cil.out.i` and ran the issue's exact k-induction command natively.

* Symex for the base case completed in 151 s (54 243 assignments → 48 VCCs after slicing); the run then
  proceeded to a second, larger k-step (symex 303.6 s, 54 195 assignments → 49 VCCs) and was solving
  that step with Bitwuzla when stopped.
* **Total: ~36 CPU-minutes across two k-steps, no `VERIFICATION SUCCESSFUL` at any point** — 2.4× the
  standard SV-COMP resource budget (900 s CPU) with zero occurrences of the unsound TRUE the issue
  reported on 8.2.0. The `WARNING: k-induction does not support function pointer calls yet. Disabling
  inductive step` guard (#4484) fires exactly as it did on aarch64.
* **Disposition: this is the x86 cross-check Pass 13 asked for.** Native x86_64, the correct
  architecture for the original SV-COMP run, agrees with the aarch64 finding — #4484 fixed the false
  negative. **Recommend closing #4427.**

### 19.2 #4432 — native x86 reaches 7× the aarch64 interleaving count, still no violation; downgrade from "host-gated" to "not locally reproducible at this budget"

`mcslock.i` parses natively without stripping `__regparm__` (confirmed: three sites present, clang
accepts them on x86_64 — the Pass 14 parse blocker was specifically an aarch64/macOS artifact, now
verified rather than assumed). Ran the issue's exact incremental-BMC command.

* Reached **2733 thread interleavings** (vs. 371 before the aarch64 260 s timeout) over **~30
  CPU-minutes** (2× the SV-COMP budget) with **no `VERIFICATION FAILED` and no data-race
  counterexample** at any point.
* This is a much stronger non-reproduction signal than Pass 14's — correct host architecture, 7.4×
  the interleaving depth, 2× the resource budget — but it is still not a *clean* TRUE: the run was
  stopped, not exhausted, and incremental-BMC's interleaving enumeration order is not guaranteed to
  match the original benchexec run's, so a specific triggering interleaving could in principle lie
  beyond interleaving 2733 without contradicting the CI's report.
* **Disposition revised.** Drop the "x86-host-gated" label from §18.1 — the parse blocker was host-gated,
  but the false alarm itself no longer is: it has now been given a fair run on the correct architecture
  and did not reproduce. Reclassify as **not reproducible locally within a generous budget**; the
  remaining path to a definitive verdict is either the original benchexec resource envelope (15 GB
  memory, exact solver/POR build) or the interleaving-reduction fix on `__atomic_*` already tracked as
  research-grade (§13.2 item 5 lineage). Do not re-attempt a bare local re-run; the next useful step here
  is design work, not more triage.

### 19.3 #4438 — superseded by open PR #4480; no further triage needed

#4480 (`[interval-analysis] enforce float bounds in k-induction base case`, authored by the maintainer)
already fixes #4438: it restricts the k-induction `inductive_step_instruction` marker to integer-only
constraint sets so float interval bounds (which NaN fails under IEEE 754) are enforced in the base case
too, closing the spurious `FALSE_REACH` this issue reported. It ships two regression tests
(`github_4438` positive, `github_4438_fail` negative) and a Copilot review pass with no unresolved
comments. **Current blocker is bookkeeping, not correctness:** `mergeStateStatus` is `BEHIND` current
master (last commit 2026-05-12; master has moved on across three merged doc/reconciliation passes since).
No triage action needed from this loop — recommend the maintainer rebase and merge #4480.

### 19.4 #5142 — `--sv-comp` unblocks the parse; no-overflow layer reproduces on x86 (different shift site than filed)

Pass 10 (§14) established that `-Wno-int-conversion` is deliberately gated behind `--sv-comp` in
`clang_c_language.cpp` and is not a bug. This pass exercises that gate on x86 for the first time:

* **Without `--sv-comp`** (the issue's literal "underlying ESBMC command", which omits the wrapper's
  `--sv-comp` flag): `imon.i` fails to parse with exactly the predicted
  `error: incompatible integer to pointer conversion ... [-Wint-conversion]` on the CIL sentinel-pointer
  initializer — confirms Pass 10's analysis holds on x86 too, not just aarch64.
* **With `--sv-comp` added** (matching what the real `esbmc-wrapper.py` invocation actually passes):
  the file parses and the incremental-BMC run produces `VERIFICATION FAILED` / `FALSE_OVERFLOW` at
  `k = 3` — but the violated property is a shift-UB at `usb_fill_int_urb:3643`
  (`urb->interval = 1 << (interval + -1)`), not the issue's cited `__create_pipe:3658`
  (`dev->devnum << 8`). **Both shift sites exist verbatim in the current `main`-branch benchmark file**
  (checked by line number), so this is not a stale fetch; it is incremental-BMC landing on a different
  member of the same "unconstrained shift amount from an untrusted driver field" precision-gap family,
  most likely due to path-exploration-order variance rather than benchmark drift.
* **Disposition.** #5142's no-overflow layer is **confirmed reproducible on x86 with `--sv-comp`**, same
  root-cause class as filed (over-approximation on a shift whose operand ESBMC cannot bound), just a
  different concrete instance. No localized sound fix without broader interval/value-set precision work
  on driver-supplied shift operands — stays research-grade, consistent with §13.3 item 6's prioritisation.

### 19.5 Pass-15 running report

**Analysed.** #4427 (x86 cross-check delivered, recommend closing), #4432 (7.4× deeper native-x86
non-repro, disposition revised from host-gated to budget-exhausted/research-grade), #4438 (no new triage
— superseded by open PR #4480, flagged as a rebase-and-merge bookkeeping item), #5142 (`--sv-comp` gate
confirmed as the correct unblock on x86; no-overflow layer reproduces at a sibling shift site).
#5396–#5399 confirmed already correctly triaged on x86 in Pass 8 — the §18 environment-wall's grouping
of them with the host-gated items was corrected, not re-investigated.

**PRs opened.** One doc PR (this entry), rebased onto current `master` post-#5742 merge (the Pass 14
docs PR was merged by the maintainer mid-pass; this branch was rebased and the now-upstream cherry-picked
commit dropped automatically by `git rebase`, avoiding duplicate content).

**Duplicated work avoided.** #5396–#5399 not re-run — already x86-confirmed in Pass 8. #4438 not
re-implemented — #4480 already provides a sound, tested fix awaiting merge.

**Remaining work (priority order, updated 2026-07-01).**
1. Close out: **#4427** (this pass's x86 cross-check confirms #4484 fixed it) and **#4480→#4438** (open
   PR needs rebase + merge, not new code).
2. **#5145 / #5393 / #5394** — aws-hash byte-addressed pointer-read-consistency (closed #5369 RCA).
3. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation).
4. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
5. **#4980** — termination ranking recogniser for side-effect-only call bodies (full-set validation,
   PR #4919 in flight).
6. **#4432** — data-race-checker interleaving reduction on `__atomic_*` (research-grade; local
   reproduction budget now exhausted on the correct host, see §19.2).
7. **#5142** — shift-operand interval/value-set precision for driver-supplied fields (research-grade;
   confirmed-reproducible on x86, see §19.4).
8. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker,
   already RCA'd in Pass 8, no localized fix.
9. Witness end-to-end validation for #1471/#1492/#4611 — needs a witness validator (CPAchecker/Ultimate),
   absent in this environment.

---

## 20. Pass 16 — aws-hash cluster (#5145/#5393/#5394) reproduced natively on x86 for the first time (2026-07-01)

Pass 15's PR (#5744) was still open (unmerged) when this pass started, so this branch cherry-picks its
one commit the same way Pass 15 cherry-picked Pass 14's — dropped automatically by `git rebase` once
#5744 merges. Priorities 1–3 from the standing loop brief (#4427, #4432, #4438/CIL-LDV) are already
closed out by Pass 15; this pass follows §19.5's own "remaining work" list to its top item instead of
re-running finished work: the **aws-hash byte-addressed pointer-read-consistency cluster**
(#5145/#5393/#5394). §16.4 explicitly flagged this cluster's reproduction as **also host-gated** — "the
k≥2 divergence is not reproducible in isolation on the aarch64/macOS host available to this pass" — so,
like #4427/#4432/#5142 before it, native x86 closes a reproducibility gap even though the underlying fix
stays out of scope (see disposition below).

### 20.1 #5145 — unchanged verdict on x86, confirms no regression across ~10 merged passes

Fetched `aws_hash_table_create_harness.i` and ran the issue's exact k-induction command. Result:
`VERIFICATION FAILED` / **`Bug found (k = 1)`** — byte-for-byte the same verdict Pass 6 (§10.2) recorded
on an aarch64/macOS host against master `74da7c0400`, now reproduced on native x86_64 against current
master (`888efe3f56`, ~15 passes and several merged fixes later, including #4484's fn-ptr guard and the
#5364/#5382 UF modelling). **No regression, no change** — the residual is still Bug B from §10.2's RCA
(havoc'd-slot pointer-read-consistency), not a new failure mode. Note this benchmark fails already at
`k = 1`, not `k ≥ 2` — the "k = 1 fixed, k ≥ 2 open" split in RCA #5369 describes the *general
mechanism's* minimal toy reproducer (`a == b` after two reads), not this specific compound-multi-layer
harness, which §10.2 already classified as a "known memory-model limitation" broader than the general
fix covers.

### 20.2 #5393 / #5394 — first genuine reproduction (Pass 7 only presumed the cluster membership)

Pass 7 (§11.3) filed these as "siblings... share the same byte-addressed... modelling" without a local
run — that host's own limitation (§11.4 shows Pass 7 predated the x86 unlock in Pass 8). This pass ran
both exact reproducers natively:

* **#5393** (`aws_hash_table_init_bounded_harness.i`): `VERIFICATION FAILED` / **`Bug found (k = 129)`**
  in 250 s (each unwind iteration ~0.7–0.9 s — this harness's `k` counts cheap loop-unwind iterations of
  a `--goto-unwind --unlimited-goto-unwind` scan, not restarted k-induction rounds, so depth 129 is fast
  here unlike #4427's per-k-step symex cost).
* **#5394** (`aws_hash_table_init_unbounded_harness.i`): `VERIFICATION FAILED` / **`Bug found (k = 129)`**
  in 2m50s.
* Both **exactly match** the depth (`k = 129`) each issue's own SV-COMP CI log reports. This upgrades
  Pass 7's presumption from "same cluster by inspection" to an **empirically confirmed** identical
  false-alarm mechanism on the correct host architecture.

### 20.3 Disposition — unchanged: research-grade, no session-scoped fix attempted

All three benchmarks are now x86-confirmed, but the fix itself remains out of scope for a triage pass,
per RCA #5369's own risk assessment: the open k≥2 divergence needs "SSA-trace / solver-internal
correlation" to distinguish `convert_ast` cache eviction from post-dereference SSA re-versioning — a
multi-session solver-internals investigation, not a localized patch — and the one shortcut previously
tried (forcing the reconstructed value non-zero) was refuted as **unsound** and closed (PR #5283). Per
the correctness-first mandate, this pass does not attempt a speculative fix under that risk profile.
**No PR, no code change** — the value delivered is closing the reproducibility gap (§16.4's stated
blocker) so a future session attempting the SSA-trace correlation work can validate on the correct host
from the outset, and does not need to re-derive that these three verdicts are unchanged.

### 20.4 Pass-16 running report

**Analysed.** #5145 (re-run, unchanged verdict, host gap closed), #5393/#5394 (first genuine x86
reproduction, exact depth match to CI reports).

**PRs opened.** One doc PR (this entry), branched from `master` with Pass 15's still-unmerged commit
cherry-picked on top (auto-drops via `git rebase` once #5744 merges, same pattern as Pass 15).

**Duplicated work avoided.** No fix attempted for the aws-hash cluster — RCA #5369 already rules out a
low-risk localized patch; re-attempting the refuted non-zero-forcing shortcut (PR #5283) was not
considered. #5393/#5394 not further decomposed — same root cause as #5145, no separate PR per Pass 7's
original folding decision (§11.3), now empirically justified rather than presumed.

**Remaining work (priority order, updated 2026-07-01).**
1. Close out: **#4427** (Pass 15 x86 cross-check) and **#4480→#4438** (rebase + merge, not new code) —
   both pending maintainer action, not further loop work.
2. **#5400 / #5138** — value-set reachability for `valid-memtrack` (sound under-approximation).
3. **#5012** — G-C `va_list` argument recovery (symbolic-format printf return length).
4. **#4980** — termination ranking recogniser for side-effect-only call bodies (full-set validation,
   PR #4919 in flight).
5. **#4432** — data-race-checker interleaving reduction on `__atomic_*` (research-grade; local
   reproduction budget exhausted on the correct host, see §19.2).
6. **#5142** — shift-operand interval/value-set precision for driver-supplied fields (research-grade;
   confirmed-reproducible on x86, see §19.4).
7. **#5145 / #5393 / #5394** — aws-hash SSA-trace/solver-internal correlation (research-grade,
   multi-session; now fully x86-confirmed, see §20.3). Demoted from position 2 — reproduction work here
   is exhausted; what remains is a design-level solver-internals investigation, not a bounded triage
   task.
8. **Device-API / LDV-environment model** (`platform_get_drvdata` etc.) — shared #5396–#5399 blocker,
   already RCA'd in Pass 8, no localized fix.
9. Witness end-to-end validation for #1471/#1492/#4611 — needs a witness validator (CPAchecker/Ultimate),
   absent in this environment.
