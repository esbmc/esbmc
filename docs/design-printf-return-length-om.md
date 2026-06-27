# Design Proposal: a sound return-length model for the printf / (v)asprintf family

**Status:** IMPLEMENTED — Phases 0–2 merged; Phase 3 Patch 1 (§4.2.1 object-size `%s` bound) in PR #5628; Phase 3 §4.4 va_list recovery deferred (optional precision)
**Tracking:** #4976, #4977, #4978, #4979 (busybox `no-overflow` false alarms) — **all CLOSED**; umbrella #2513; related #2797 (missing libc OM bodies); follow-on precision tracked by #5012.
**Author context:** follow-up to `docs/svcomp-issue-triage.md` §3.1 and the finding that wiring `vasprintf` into the *current* `symex_printf` is unsound (PR #5009).
**Date:** 2026-05-31 (proposal); 2026-06-26 (status update)

> **Implementation status (2026-06-26).** The sound model proposed here shipped across
> four PRs and closed the four busybox false alarms:
> - **Phase 1 / G-A** — non-constant format no longer under-approximates: **PR #5017**.
> - **Phase 1 / G-B** — non-literal `%s` contributes the sound *unbounded* fallback (never 0).
> - **Phase 2 / G-D** — `vprintf`/`vsprintf`/`vsnprintf`/`asprintf`/`vasprintf` wired into
>   `symex_printf`: **PR #5270**, with the unbounded-return cap at `INT_MAX/2` (**PR #5278**)
>   and the `*strp` heap-allocation model (**PR #5279**).
> - **Phase 3 Patch 1 / §4.2.1** — non-literal `%s` over a finite char array of `N` bytes
>   contributes the sound object-size bound `N − 1` instead of staying unbounded: **PR #5628**.
>   The `v*` variants keep the operand unreliable (the `%s` position is the `va_list` pointer),
>   so `%s` stays unbounded there until va_list recovery.
> - **Phase 3 §4.4 / G-C** — `va_list` `%s` argument recovery: **deferred** (precision-only;
>   correctness never depends on it). Tracked under #5012.
>
> Regression coverage lives in `regression/esbmc/`: `asprintf_const_format_exact`,
> `vasprintf_const_format_no_overflow`, `vasprintf_overflow_fail`,
> `vasprintf_unbounded_s_no_overflow`, `vasprintf_unbounded_s_overflow_fail`, and the
> reduced reproducers `github_4977`/`github_4978`/`github_4979` (+ `github_4978_fail`).
> Issues #4976–#4979 were closed on 2026-06-09. See `docs/svcomp-issue-triage.md` §3.1.

---

## 1. Motivation

ESBMC reports a spurious signed-`int` `arithmetic overflow on add` in busybox `bb_verror_msg`:

```c
used = vasprintf(&msg, s, p);              /* no model -> unconstrained nondet int */
if (used < 0) return;
applet_len = (int)(strlen(applet_name) + 2);
strerr_len = strerr ? (int)strlen(strerr) : 0;
msgeol_len = (int)strlen(msg_eol);
msg1 = realloc(msg, (unsigned long)(applet_len + used + strerr_len + msgeol_len + 3));
```

`vasprintf`/`asprintf` have **no operational model**, so symbolic execution assigns the return an
unconstrained nondet `int` (counterexample: `used ≈ INT_MAX`). The sum then overflows. Ground
truth is `true`: the real formatted output is short.

A previous attempt simply routed `vasprintf` through the existing `symex_printf` machinery. That is
**unsound** (it hid a genuine overflow — PR #5009 records the measurement). This proposal sets out
what a *sound* model actually requires, why it is achievable, and the exact gaps to close.

## 2. Soundness invariant (non-negotiable)

> For every execution, the modelled return value `R` of a printf-family call MUST satisfy
> `R ≥ (the true number of characters the call would produce)`.
> Equivalently: the model may **over**-approximate the output length (causing at worst a *false
> positive* / precision loss) but must **never under**-approximate (which would mask a real
> overflow — a false negative). For the allocating/length forms the relevant quantity is the full
> formatted length; for the bounded-write forms (`snprintf`) the return is still the *would-be*
> length, independent of the size cap.

Every design decision below is justified against this invariant.

## 3. Current state (grounded in source, master `53e3b9f5ac`)

Dispatch:
- `printf_kind_from_name()` — `src/irep2/irep2_expr.cpp:303-319` — maps names to `printf_kindt`
  (`src/irep2/irep2_expr.h:90`). Wired: `printf`, `fprintf`, `dprintf`, `sprintf`, `vfprintf`,
  `snprintf`. **Not** wired: `asprintf`, `vasprintf`, `vsprintf`, `vsnprintf` (and `vprintf`).
  Frontend gate: `src/goto-programs/builtin_functions.cpp:1075-1077`.
- `symex_printf()` — `src/goto-symex/builtin_functions/io.cpp:16` — picks the format-arg index per
  kind, parses the format with `printf_formattert`, and assigns the return (constant when
  `min_outlen == max_outlen`, else a nondet `assume`d into `[min,max]`).
- `printf_formattert` — `src/goto-symex/printf_formatter.{h,cpp}` — accumulates
  `min_outlen`/`max_outlen` per conversion.

There are **three** gaps, two of which are latent *soundness* bugs in the already-wired functions:

| # | Gap | Location | Nature |
|---|---|---|---|
| G-A | Non-constant format ⇒ `fmt = ""` ⇒ return **0** | `io.cpp:50-61`, `245-249` | **UNSOUND** (under-approx) |
| G-B | Non-literal `%s` arg contributes **0** (only string literals counted) | `printf_formatter.cpp:302-312` | **UNSOUND** (under-approx) |
| G-C | Variadic args passed through a `va_list` to an external callee are unrecoverable | `symex_function.cpp:214-251`, `builtin_functions/va_arg.cpp:10-38` | precision only |
| G-D | `(v)asprintf` / `(v)sprintf` / `(v)snprintf` not wired at all | dispatch above | missing model |

G-A and G-B already affect `sprintf`/`snprintf`: `n = sprintf(buf, sym_fmt, ...)` or a non-literal
`%s` makes the modelled return smaller than the truth. They are masked today only because most call
sites use literal formats. **They must be fixed before any new function is wired**, otherwise the
wiring inherits the unsoundness (exactly what PR #5009 measured).

### va_list mechanics (why G-C is "precision only", not fatal)

Variadic arguments are materialised as named symbols `<caller>::va_arg0,1,…` **at the call site**
(`symex_function.cpp:214-251`); `va_arg` fetches them by index (`va_arg.cpp`). Once a `va_list`
(a bare `char*`, `src/c2goto/headers/stdarg.h:21`) is handed to a *bodyless* callee like
`vasprintf`, those symbols are out of scope — the args are erased. **However:**

- Numeric/char conversions (`%d %u %x %c %f …`) are bounded by their **type/format alone**, not by
  the argument value (`%d` ≤ 11 chars for 32-bit, `%f` bounded by precision, etc. —
  `printf_formatter.cpp:225-268`). The arg value is not needed for a sound bound.
- `%n` produces **no** output.
- Only `%s`/`%p` need the argument to tighten beyond "unbounded". With G-B fixed to a *sound*
  fallback (object-size bound, or unconstrained when unknown), even an unrecoverable `%s` arg is
  handled soundly — just less precisely.

So va_list recovery is a **precision optimisation for `%s`**, not a soundness prerequisite.

### Decisive observation for #4976–#4979

In `sync-1.i` the *only* call into this family is `bb_error_msg("ignoring all arguments")` — a
string **literal**, no conversion specifiers, **no variadic arguments**. ESBMC inlines static
callees and propagates constants through parameter binds (`symex_function.cpp:74-212, 320`), so at
the `vasprintf` call the format operand resolves to the constant `"ignoring all arguments"` and the
true output length is exactly **22**. A constant-format model needs *no* va_list machinery here:
it just has to compute 22 instead of leaving `used` unconstrained. (The other benchmarks must be
checked individually — see §7.)

## 4. Proposed design

A staged set of changes, smallest-sound-first.

### 4.1 Fix G-A — non-constant format returns a sound over-approximation
`io.cpp`: when the format operand is not a `constant_string2t`, do **not** assign `0`. Assign an
unconstrained nondet `int` (the pre-model behaviour: `≥ 0`, or `≥ -1` for the allocating/`int`-error
variants). This removes a latent false-negative and makes it safe to wire more functions.

### 4.2 Fix G-B — `%s`/`%p` with a non-literal argument
`printf_formatter.cpp`: when the `%s` argument is not a string literal, contribute a **sound upper
bound** on its length:
1. If the pointer's pointed-to object has a known/derivable size `N` (ESBMC tracks object sizes;
   `__ESBMC_get_object_size` / value-set), contribute `N - 1` (max strlen of an `N`-byte buffer).
2. Otherwise, mark the whole result unbounded (fall through to the §4.1 unconstrained-nondet path).
Never contribute `0` for an unknown `%s`.
Apply the analogous rule to symbolic width/precision fields (`%*d`, `%.*s`): unknown field ⇒
unbounded (G-C-adjacent). A literal width/precision `W` still contributes its sound `max(W, …)`.

### 4.3 Wire the missing family members (G-D)
Add `ASPRINTF, VASPRINTF, VSPRINTF, VSNPRINTF` (and `VPRINTF`) to `printf_kindt`
(`irep2_expr.h`), `printf_kind_from_name` (`irep2_expr.cpp`), `type_to_string`
(`irep2_utils.cpp:482`), the `migrate.cpp:3460` switch, and the frontend recogniser
(`builtin_functions.cpp:1075`). Extend the `fmt_idx` table in `symex_printf` (`io.cpp:31-45`):

| function | signature | format-arg index | dest | return |
|---|---|---|---|---|
| `asprintf(char **p, fmt, …)` | varargs | 1 | `*p` (allocated) | length |
| `vasprintf(char **p, fmt, va_list)` | va_list | 1 | `*p` (allocated) | length |
| `vsprintf(char *s, fmt, va_list)` | va_list | 1 | `s` | length |
| `vsnprintf(char *s, size, fmt, va_list)` | va_list | 2 | `s` (≤size) | would-be length |

For the **allocating** forms the return value is the priority (it is what feeds overflow-prone
size arithmetic). Modelling the *written buffer contents* is secondary and may initially be left as
the existing havoc, provided the allocated object's size is made consistent with the returned
length (so a later `realloc`/index stays sound). This must be reviewed against
`--memory-leak-check` / pointer checks.

### 4.4 va_list argument recovery (optional precision pass, G-C)
For the `v*` variants, attempt to recover the backing `va_arg<N>` symbols of the *nearest inlined
variadic ancestor* (using the per-frame `va_index`, `goto_symex_state.h:210-213`) to tighten `%s`.
If recovery fails, fall back to §4.2's sound bound. This pass is **precision-only** and may be
deferred to a later phase; correctness never depends on it.

## 5. Soundness argument

The modelled return is `Σ contributions` over the format walk. Per §2 we need `Σ ≥ true length`.

- **Literal characters:** exact contribution. ✔
- **Numeric/char/`%%`:** contribution = type/format-derived max, independent of arg value; `≥` any
  concrete rendering. ✔ (existing `emit_int`/`emit_*`.)
- **`%s`/`%p`:** literal ⇒ exact; non-literal ⇒ object-size bound (`≥` true strlen since strlen of
  an `N`-byte object `< N`) or unbounded. Never 0. ✔ (G-B fix.)
- **`%n`:** 0 output, exact. ✔
- **Symbolic width/precision or non-constant format:** unbounded nondet, trivially `≥`. ✔
  (G-A / §4.2.)
- **Unrecoverable va_list `%s`:** falls to the non-literal `%s` rule ⇒ object-size or unbounded. ✔

Hence the return is a sound over-approximation in every case. The *only* behavioural change for the
already-wired functions is replacing two `0`-clamps (G-A, G-B) with over-approximations — strictly
in the safe direction.

## 6. Risks and limitations

1. **Precision regressions (not soundness).** Fixing G-A/G-B replaces `0` with larger/unbounded
   values, which could turn a few currently-passing benchmarks (that passed *because* of the unsound
   0) into **false positives**. This is acceptable per §2 but must be measured (§8); if the common
   constant-format path is unaffected, regressions should be rare.
2. **The overflow may shift, not vanish.** Pinning `used` to its true bound only removes the
   `bb_verror_msg` overflow if the *sibling* terms (`strlen(applet_name)`, `strlen(msg_eol)`) are
   themselves bounded. If ESBMC's `strlen` returns an unconstrained value for those globals, the
   solver will simply move the overflow there — a separate `strlen`/object-size modelling issue.
   **Phase 0 must confirm the benchmark actually flips to SUCCESSFUL**, not merely that `used` is
   bounded.
3. **va_list recovery is limited** to inlined ancestors; deep/opaque va_list flows stay imprecise
   (sound). 
4. **Allocating-buffer modelling** (`*p`) interacts with pointer/leak checks; must be validated.
5. **Performance:** constant-format walking is cheap; object-size queries for `%s` add some cost.

## 7. Effectiveness on #4976–#4979 (honest assessment)

- **sync-1:** format is a constant literal with no specifiers/varargs ⇒ `used` pinned to 22.
  **Likely fixed**, *iff* risk #6.2 does not bite (Phase-0 check required).
- **sync-2 / whoami-incomplete-1 / -2:** not yet traced to the same single literal. Each must be
  checked: if the violating-path format is constant (with bounded conversions), the model fixes it;
  if it is genuinely symbolic, the model keeps `used` unbounded (sound) and the benchmark stays a
  documented precision limitation.

This proposal therefore does **not** promise to close all four; it promises a **sound** model that
closes the constant-format cases and never introduces a false negative.

## 8. Validation plan (mandatory before merge)

1. **Soundness gate:** full SV-COMP `no-overflow` set, before vs after. **`wrong-false` (false
   negatives) MUST remain 0.** Any increase blocks the change outright.
2. **Precision delta:** report `correct-true` / `wrong-true` movement on the overflow set and the
   broader string-handling categories (to quantify risk #1).
3. **Negative test (kept):** the genuine-overflow program from PR #5009's measurement
   (`(INT_MAX-1) + asprintf(...)`) MUST stay `VERIFICATION FAILED`.
4. **The four benchmarks:** re-run with the exact issue flags; record SUCCESSFUL/FAILED honestly
   (per §7, some may remain FAILED — that is acceptable if sound).
5. **Mode-C dead-code discharge** for the new `printf_kindt` branches (C-Live: the new cases are
   reachable), per the repo's branch-change policy.
6. **clang-format / unit + regression subset** green.

## 9. Phased implementation plan

- **Phase 0 — feasibility spike (1 patch, no merge):** wire `vasprintf` minimally + a temporary
  log, confirm (a) the format resolves to the constant at the call, (b) pinning `used` flips sync-1
  to SUCCESSFUL (i.e. risk #6.2 does not bite). Decision point: continue only if (b) holds for at
  least one benchmark.
- **Phase 1 — soundness fixes (independent value):** G-A and G-B. Ship even if nothing else
  follows: removes latent false negatives in `sprintf`/`snprintf`. Full §8 soundness gate.
- **Phase 2 — wire the family (G-D):** constant-format path only; `(v)asprintf` return + consistent
  allocation size. Tests + §8.
- **Phase 3 — `%s` tightening:** precision pass; optional. Patch 1 (§4.2.1 object-size bound for a
  non-literal `%s` over a finite char array) shipped in PR #5628; Patch 2 (§4.4 va_list recovery,
  G-C) remains deferred.
- **Phase 4 — measurement & docs:** SV-COMP overflow run, update `svcomp-issue-triage.md` §3.1 with
  the outcome, close whichever of #4976–#4979 actually flip.

## 10. Test plan

Per phase, add `regression/esbmc/` pairs (test.desc format):
- `…_safe/` — constant-format `asprintf`/`vasprintf` whose return feeds a size sum that does **not**
  overflow ⇒ `VERIFICATION SUCCESSFUL` with `--overflow-check`.
- `…_fail/` — a reachable real overflow on a printf-family return ⇒ `VERIFICATION FAILED` (guards
  against re-introducing the false negative).
- A `sprintf`-with-non-literal-`%s` case pinning the G-B fix (sound bound, not 0).

## 11. Alternatives considered

- **Do nothing** (status quo): sound but imprecise; #4976–#4979 stay false positives. Baseline.
- **C operational model in `src/c2goto/library/`** that loops `va_arg` and delegates: blocked by the
  same va_list-erasure issue for the allocating return; offers no advantage over the symex-layer
  approach and is harder to keep sound for symbolic formats.
- **Full variadic redesign** (make va_list a first-class array of arg refs recoverable across
  calls): large, invasive; out of scope, but would also benefit other variadic models.

## 12. Decision / exit criteria

Proceed to Phase 1 regardless (it is a standalone soundness fix). Proceed to Phases 2–4 only if
Phase 0 shows at least one of #4976–#4979 genuinely flips to SUCCESSFUL under a sound model. If
Phase 0 shows the overflow merely shifts to `strlen` (risk #6.2), re-scope toward `strlen`/
object-size modelling and keep #4976–#4979 classified as precision limitations.
