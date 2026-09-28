# Plan — the per-task cost every SV-COMP run pays (issue #6831, cause 2)

**Status:** W0 closed (the term is a glibc secondary arena, not the library);
W1, W3, W4 and W5 shipped or in review; W2 closed unbuilt, its premise refuted
by measurement. Against the window's fast endpoint the oracle is back to
×0.999 from ×1.070 — see §9.
**Owner issue:** [#6831](https://github.com/esbmc/esbmc/issues/6831), *cause 2 —
a general slowdown tipping tasks already at the limit*, ~198 of 489 lost tasks,
led by 131 `Juliet_Test` no-overflow tasks at a median of 99.1 s of a 100 s
limit.
**Companion plan:** [`svcomp-6831-schedule-space-plan.md`](svcomp-6831-schedule-space-plan.md)
covers cause 1 (the schedule-space explosion). The two are independent.
**Last updated:** 2026-08-15.

**Measurement environment.** All numbers below were measured on an x86_64 Linux
host (Intel Xeon E5-2620 v4, 32 threads) against `build/src/esbmc/esbmc`, ESBMC
8.4.0, `RelWithDebInfo`, built from `4be7fbe015`. Solver: Bitwuzla 0.9.0. Phase
timings come from throwaway `std::chrono` instrumentation in
`add_cprover_library()` and `read_bin_goto_object()`, since at the time neither
phase was reported (W4 has since made `add_cprover_library()`'s four phases and
its two symbol counts available from a normal run under
`--verbosity c2goto:9`). Reproducer: `int main(void){return 0;}` —
deliberately the smallest possible program, so everything measured is cost the
input did not ask for.

The host drifts by ±15 % over tens of minutes, enough to invert a 10 % effect
measured sequentially. Every comparison below is therefore **interleaved** —
configurations alternate within one loop, 12–15 pairs, medians quoted — and
absolute seconds should be read as "this host, that hour". The ratios and the
phase split are the durable part.

---

## 1. Premise: two terms, only one of them explained

The issue decomposes the slowdown into **a fixed ≈0.15 s per task plus a ≈3.5 %
multiplicative term**, and attributes both to the operational-model (OM) library
growing (`src/c2goto/library` +9.8 KB, `src/cpp/library` +22.9 KB in the window).

That attribution is right for the fixed term and **unproven for the
multiplicative one**. The distinction decides what to work on, because the two
terms lose different tasks:

| term | what it costs a 99.1 s Juliet task | what it costs the whole run |
|---|---|---|
| fixed ≈0.15 s | +0.15 s — **does not tip it** | 0.15 s × 36,603 ≈ 1.5 CPU-hours |
| multiplicative ≈3.5 % | **+3.47 s — tips it over the 100 s limit** | ~7.9 % of total CPU |

So the 131 `Juliet_Test` losses this plan is named after are caused by the
**multiplicative** term, and no amount of on-demand library loading recovers
them. The issue's suggested action 3 ("loading the operational-model library on
demand would remove most of the fixed cost") is correct and worth doing — §2
shows the fixed cost is larger than the issue's +0.15 s delta suggests — but it
is not the fix for the tasks that were lost. **W0 is therefore attribution, not
optimisation.**

---

## 2. Measurements

### 2.1 What a trivial C program spends its time on

`esbmc trivial.c`, interleaved, 12 pairs:

| configuration | wall | GOTO creation |
|---|---|---|
| default | 0.46 s | 0.44 s |
| `--no-library` | **0.04 s** | 0.03 s |
| `--version` (process startup only) | 0.01 s | — |

**91 % of the run is the operational-model library**, on a program that uses
none of it. Instrumenting `add_cprover_library()` splits that (absolute times
from a quieter period on the same host; read the proportions):

| phase | time | what it does |
|---|---|---|
| deserialise blob | **0.21 s** | `read_bin_goto_object()` over all 3,847 symbols |
| build dependency map | **0.11 s** | `generate_symbol_deps()` over all of them |
| select symbols to keep | 0.003 s | |
| `c_link` into the context | 0.001 s | |
| goto-convert what was kept | ~0.04 s | (residual against `--no-library`) |

The two phases that scan *everything* cost 0.32 s — 80 % of GOTO creation; the
two that touch only what is used cost 0.004 s.

### 2.2 How little of the blob is used

`clib64_fp` holds 3,847 symbols in 3,369,011 bytes. Symbols actually linked into
the context:

| program | symbols kept | share |
|---|---|---|
| `int main(void){return 0;}` | 104 | 2.7 % |
| `regression/esbmc/00_big_endian_01` | 218 | 5.7 % |
| `regression/esbmc-unix/01_malloc_20` | 836 | 21.7 % |
| trivial C++ | 147 | 3.8 % |

Even the pthread-heavy reproducer from cause 1 uses under a quarter of the blob.

### 2.3 What is in the blob, by source tree

Attributing every symbol to the file it was defined in:

| tree | symbols | share |
|---|---|---|
| `src/c2goto/library/*.c` (libc models) | 1,206 | 31 % |
| `src/c2goto/library/libm/**` | 1,002 | 26 % |
| **`src/c2goto/library/python/*.c`** | **911** | **24 %** |
| system/bundled header declarations | ~647 | 17 % |
| `src/c2goto/library/cpp/*.cpp` | 81 | 2 % |

The largest single contributors are `python/list.c` (324) and
`python/string.c` (321) — larger than any libc model. **A C or C++ task
deserialises 911 Python-only symbols, 24 % of the blob, on every run.** The
Solidity models are already built as a separate `sol64.goto` "for faster runtime
loading" (`src/c2goto/CMakeLists.txt`); the Python models are not.

### 2.4 The cost is worse than linear in blob size

Truncating the symbol loop after *k* symbols (throwaway patch, timing probe
only):

| symbols read | deserialise | deps | µs/symbol |
|---|---|---|---|
| 100 | 0.003 s | 0.000 s | 30 |
| 1,000 | 0.030 s | 0.010 s | 30 |
| 2,000 | 0.094 s | 0.041 s | 47 |
| 3,000 | 0.150 s | 0.071 s | 50 |
| 3,847 (all) | 0.209 s | 0.119 s | 54 |

Per-symbol cost grows ~1.8× between 1k and 3.8k symbols. The mechanism is in
`irep_serializationt::reference_convert()`: the format stores back-references by
index into `ireps_on_read`, a `std::map<unsigned, irept>` that grows for the
whole read. Two consequences:

- **Removing symbols saves more than their share.** Dropping the 24 % of
  symbols that are Python-only should save appreciably more than 24 % of 0.32 s.
- **Adding OM files costs more than their share**, which is why an OM library
  that grew by 33 KB of source moved every task's baseline.

Caveat: the probe truncates the *tail* of the symbol stream, not a specific
tree, so the curve establishes superlinearity, not the exact saving from any
particular removal.

### 2.5 A measured, format-preserving prototype

The ids in `ireps_on_read` are assigned densely and in increasing order on the
write side (`reference_convert()` uses `ireps_on_write.size()`), so the
`std::map` is a red-black tree keyed by what is effectively a dense array index.
Replacing it with a `std::vector<std::pair<bool, irept>>` (~10 lines, no format
change):

Interleaved A/B between two binaries differing only in that container, 15 pairs:

| | wall | GOTO creation |
|---|---|---|
| baseline | 0.47 s | 0.436 s |
| vector-backed read table | **0.39 s** | **0.349 s** |
| ratio | 0.83 | **0.80** |

**20 % off GOTO program creation, 17 % off the wall clock of every run**, and
`ctest -R "regression/esbmc/0[0-3]"` passes 61/61. This was a prototype measured
for this plan and then reverted; it has since shipped as W3.1, which reproduced
the ratio (×0.805 on GOTO creation, 15 interleaved pairs). (Measured
sequentially rather than interleaved, the same change first appeared to be worth
15 % and then appeared to be a regression; hence the methodology note above.)

### 2.6 What the fixed cost is not

- **Not header extraction.** `--no-library` still extracts the bundled clang
  headers to `/tmp` and costs 0.04 s total.
- **Not process startup.** `--version` on the 100 MB binary is 0.01 s.
- **Not the `goto_functionst` the reader fills in.** `add_cprover_library()`
  declares `goto_functionst goto_functions` as a local, passes it to the reader,
  and never reads it; the reader populates 1,352 entries into it (with a
  `migrate_symbol_type()` per function symbol) and every one is discarded.
  Skipping it entirely measures at ~10 ms — real, free to take, not a lever.

---

## 3. Diagnosis

- **The fixed term is understood and large.** 0.32 s of a 0.44 s trivial run is
  two whole-blob scans over a blob of which 2.7–22 % is used. It is
  paid by all 36,603 tasks (~3.3 CPU-hours per SV-COMP run at current size), and
  it grows superlinearly as OM files are added — but it decides a verdict only
  for tasks within ~0.2 s of the limit.
- **The multiplicative term is not attributed.** Nothing measured here explains
  a uniform 3.5 % on tasks that run for 99 s: the library cost is a constant,
  and a 99 s task pays it once (0.3 % of its runtime). The issue reproduced the
  ratio locally between two builds on one machine, so it is real and it is in
  ESBMC — but "the OM library grew" is a hypothesis for it, not a finding. It is
  the term that lost the 131 `Juliet_Test` tasks.

---

## 4. Workstreams

### W0 — Attribute the 3.5 % multiplicative term (prerequisite, highest value)

The issue's local reproduction (`978a007e73` vs `7835797ebc`, 32 regression
tests, median 1.035×) is the oracle. Bisect it the way cause 1 was bisected, but
with a *long-running* task as the measurement — a `Juliet_Test` no-overflow task
or an equivalent regression test in the 10–100 s band, not a trivial program,
since a trivial program measures only the fixed term.

Hypotheses to separate, cheapest first:

1. ~~**A per-task constant misread as a ratio.**~~ **Refuted, from the issue's
   own data.** The runtime-bucket table needs no new measurement: median delta
   is 0.19 s and 0.18 s in the 0–1 s and 1–2 s buckets — that flat pair *is* the
   fixed term — and then climbs to **1.89 s** at 40–100 s. A fixed cost is
   constant in absolute terms and cannot grow 10×, so a runtime-proportional
   term does exist at long runtimes. Subtracting the ~0.18 s fixed part puts it
   nearer **2.4 %** than the headline 3.5 %.
2. ~~**More model code reaching symex.**~~ **Prime suspect refuted.** Rather
   than diff the two builds 183 commits apart, isolate the candidate: current
   master against current master with only the ten `__CPROVER_*` bodies of
   `5c9109d54c` (#6708) deleted from `builtin_libs.c`. On a 2,600-iteration C
   loop under `--unwind 2601 --overflow-check` (19.7 s wall, 18.0 s symex — in
   the band this workstream asks for), the two builds are indistinguishable
   where it counts:

   | | with #6708 | without |
   |---|---|---|
   | VCCs generated | 41,603 | 41,603 |
   | symex assignments | 39,028 | 39,028 |
   | remaining after simplification | 10,592 | 10,592 |
   | `Symex completed in` | 23.05 s | 23.29 s |

   Identical counts mean identical symex work, so #6708 contributes nothing to
   the multiplicative term. Some *other* commit in the window may still move
   what reaches symex; the obvious candidate no longer does.

   **Side finding, for W1.** All ten primitives are nonetheless linked *with
   bodies* into the GOTO program of `int main(void){return 0;}`, which calls
   none of them, and nothing else in the OM references them. The dependency
   closure is not excluding them, so they are dead weight in every C task —
   fixed cost, not multiplicative, but exactly the kind of thing W1 exists to
   stop paying for.
3. **Binary-size effects** (95.1 → 100.9 MB): i-cache/iTLB pressure. Now the
   leading unrefuted hypothesis, and the hardest to act on.

**Exit:** a named commit or a named mechanism for the 3.5 %, with before/after
timings on a task in the 10–100 s band, or a demonstration that the term is
hypothesis 1 and no multiplicative mechanism exists. Everything else in this
plan is worth doing regardless; **only W0 recovers the 131 Juliet tasks.**

**Still open.** The term is real (hypothesis 1 is refuted, so the exit cannot be
taken that way) and it is not #6708. What remains is the bisect this workstream
originally called for, over the 183 commits in the window, with a long-running
task as the oracle — but now with a sharper stopping rule than "the ratio moved":
VCC count and symex assignments, which the hypothesis-2 probe shows are stable
under a change that only adds unreferenced model code. A commit that moves those
counts is the mechanism; one that moves only wall time is hypothesis 3.

#### The window, measured end to end

Both endpoints built from one build directory (so ccache and the solver
dependencies are shared), `RelWithDebInfo`, Z3 + Bitwuzla, Python frontend on;
12 interleaved pairs on the oracle:

| metric | `978a007e73` | `7835797ebc` | B/A | IQR |
|---|---|---|---|---|
| wall | 10.001 s | 10.693 s | **1.063** | 0.048 |
| GOTO creation | 0.313 s | 0.399 s | **1.272** | 0.098 |
| symex | 0.752 s | 0.807 s | 1.056 | 0.094 |
| slicing | 0.682 s | 0.719 s | 1.074 | 0.049 |
| encoding | 4.452 s | 4.783 s | 1.053 | 0.055 |
| solving | 2.080 s | 2.202 s | 1.070 | 0.065 |
| VCCs / assignments | 79,992 / 120,003 | 79,992 / 120,003 | identical | |

**The term reproduces at ×1.063 on a 10 s task, five times the noise floor,
with the counts byte-identical.** Two things follow immediately:

- **Hypothesis 2 is refuted in general, not just for #6708.** Nothing in the
  window changed what symex produces on this input, so no amount of "more model
  code reaching symex" explains the 6 %.
- **Every phase moved by roughly the same factor** — and that includes
  `solving`, which is time inside Bitwuzla. Both builds link the *same* solver
  and hand it the *same* formula. A slowdown that reaches into an unchanged
  third-party library is not ESBMC doing more work; it is the process being
  slower at everything it does.

GOTO creation is the exception at ×1.272: that is cause 2's *fixed* term, the
blob having grown, and it is what W1–W3 address. Subtract it and the remaining
~6 % is still there.

**A mechanism this plan had not listed.** Hypothesis 3 named binary size
(i-cache/iTLB). There is a second process-wide candidate with the same
signature: the library load's *after-effects on the heap*. Deserialising 3,800
symbols allocates and frees millions of small `irept` nodes before verification
starts; a larger blob leaves the allocator with a bigger, more fragmented arena,
and every later allocation — in symex, in slicing, in the solver — pays for it.
That is exactly a uniform multiplicative term.

The two are separable in one experiment: **re-run the same A/B with
`--no-library`**. The oracle needs no models (identical VCCs either way), so if
the ratio collapses toward 1.0 the mechanism is the library load's residue, and
W1's blob split fixes the multiplicative term as a side effect. If the ratio
survives, it is layout and W1 does not help.

#### `--no-library`: the term is not the library, in any form

Same binaries, same oracle, `--no-library` added, 12 pairs:

| metric | `978a007e73` | `7835797ebc` | B/A | IQR |
|---|---|---|---|---|
| wall | 9.408 s | 10.005 s | **1.074** | 0.020 |
| GOTO creation | 0.024 s | 0.026 s | 1.082 | 0.050 |
| symex | 0.774 s | 0.845 s | 1.093 | 0.071 |
| encoding | 4.204 s | 4.537 s | 1.088 | 0.025 |
| solving | 2.065 s | 2.170 s | 1.054 | 0.022 |

**With the blob never read — GOTO creation down from 0.31 s to 0.024 s — the
slowdown is undiminished.** The heap-residue hypothesis is dead, and so is
every other library-mediated explanation: the two binaries are ~7 % apart on
work that touches no operational model at all.

This settles a question §7 could only raise as a risk. **W1 and W2 will not
recover the 131 Juliet tasks.** They remain worth doing for the fixed term —
1.5–3.3 CPU-hours a run, and §2.4's superlinearity means the next models cost
more than their share — but the plan should stop implying they might also
address the multiplicative one.

What is left is the binary. Between the two builds, `.text` grows 100,132,024 →
100,881,531 bytes (+0.75 %) and `.bss` 302,080 → 433,920 (+44 %). A 0.75 %
text growth is a thin explanation for a 7 % uniform slowdown on its own, so the
bisect is now about naming the commit and reading the mechanism off it, rather
than choosing between the hypotheses on the table. The stopping rule simplifies
accordingly: the counts do not move anywhere in this window, so the verdict is
the wall ratio against the fast endpoint.

#### Bisect log

Each row is 12 interleaved pairs of the oracle under `--no-library` against
`978a007e73`, wall ratio. "Slow" means the effect is at or before that commit.

| commit | # in window | date | B/A | IQR | verdict |
|---|---|---|---|---|---|
| `7835797ebc` | 183 | 2026-08-08 | 1.074 | 0.020 | slow (endpoint) |
| `98856b8c11` | 92 | 2026-08-04 | 1.073 | 0.035 | **slow** — the second half contributes nothing |
| `c8d4bf6f5c` | 46 | 2026-08-03 | 1.074 | 0.103 | **slow** |
| `bd54b099bc` | 23 | 2026-08-02 | 1.020 | 0.034 | **fast** — window is now 24–46 |
| `7202a4f52d` | 35 | 2026-08-02 | 1.015 | 0.064 | **fast** — window is now 36–46 |
| `b43bbd4ee5` | 41 | 2026-08-02 | 1.073 | 0.156 | **slow** — window is now 36–41 |

The IQR widens with host load (a large unrelated build was running for the
third and fourth rows); the median of 12 pairs is roughly IQR/4 of standard
error, so a ×1.07 verdict still clears ×1.00 by ~3σ. A row that lands near
×1.035 — half the effect — would not, and should be re-run with more pairs on
an idle host before it is believed.

Commit 23's ×1.020 sits just above the ±1.2 % floor. Read as "fast" here
because the step is deciding between ×1.00 and ×1.07, but if the final commit
does not account for the whole ×1.074, that residue is where to look for a
second contributor.

Ruled out by reading rather than building: **#6606** (`[build] Decouple
sanitizers from CMAKE_BUILD_TYPE`), the one commit in 1–46 that touches
compiler flags. With `ENABLE_SANITIZERS` empty and a `RelWithDebInfo` build the
new code computes an empty sanitizer list and adds no compile or link options,
so it cannot change codegen for these builds.

#### The mechanism: glibc gives the worker thread its own malloc arena

`45dae3ce88`, *[esbmc] run on a thread with a large stack* (#6618), sits at
position 38, inside the surviving 36–41 window. It moves the whole run off the
main thread onto a spawned one, to survive deep recursion in
`clang_c_convertert::get_expr` (#6617). It is the only change in the range that
could slow *everything at once*, and the reason is not the stack size:

**glibc allocates a secondary arena for a non-main thread.** The main thread
allocates from the main arena, which grows contiguously with `brk`. Every other
thread gets an mmap'd per-thread arena with different growth and trimming
behaviour. After #6618 the entire run — parsing, symex, slicing, encoding, and
the solver, which is called on that thread — allocates from a secondary arena.

That is testable without building anything, by re-running one binary under
`MALLOC_ARENA_MAX=1`, which collapses all arenas onto one. 12 pairs each:

| binary | wall B/A | symex | encoding | solving |
|---|---|---|---|---|
| `7835797ebc` (post-#6618) with `MALLOC_ARENA_MAX=1` | **0.946** | 0.889 | 0.933 | 0.952 |
| `978a007e73` (pre-#6618) with `MALLOC_ARENA_MAX=1` | 0.990 | — | — | — |

**A single arena buys back 5.4 % on the post-#6618 build and nothing (×0.990,
inside the noise floor) on the pre-#6618 one** — which is exactly what the
hypothesis predicts, since the older binary was already on the main arena. Of
the ×1.074 regression, ~5.4 points are the arena; the remaining ~2 points are
unattributed and may be the 512 MB mapping, or the ×1.020 already visible at
commit 23.

**This is W0's exit.** The mechanism is named, measured on a task in the
10–100 s band, and it is neither of the plan's standing hypotheses: not more
work reaching symex, not binary size. It is also *not a reason to revert
#6618* — the crash it fixes is real. The fix is to keep the thread and put the
allocator back: `mallopt(M_ARENA_MAX, 1)` before the worker starts, under
`__GLIBC__`. ESBMC allocates from one thread at a time, so collapsing the
arenas costs it no contention.

Commit 37 — the one immediately before #6618 — measures ×1.034 (IQR 0.056,
under build contention), which is neither verdict. Read with commit 23's ×1.020
and commit 35's ×1.015, the window looks like ~2–3 points of accumulated creep
plus the arena's ~5.4, not a single step.

#### The obvious fix does not work, and the reason matters

`mallopt(M_ARENA_MAX, 1)` before the worker starts was implemented and measured
against its exact master counterpart (12 pairs, oracle with the library loaded,
as SV-COMP runs it):

| | wall | symex | encoding | solving |
|---|---|---|---|---|
| master → master + `mallopt(M_ARENA_MAX, 1)` | **1.009** | 0.943 | **1.066** | 0.946 |
| master → master under `MALLOC_ARENA_MAX=1` | **0.999** | 0.932 | **1.046** | 0.955 |
| master → master under `MALLOC_ARENA_MAX=1`, `--no-library` | 0.977 | 0.878 | 1.009 | 0.961 |

The patch does what it says — symex and solving get their ~5 % back — but a
single arena makes **encoding 4–7 % slower whenever the operational-model
library was loaded**, and the two cancel. Under `--no-library` the encoding
penalty disappears and the win survives at ×0.977. The library's allocations
are freed back into the arena that encoding then allocates from; keeping the
worker on its own arena is what stops encoding from paying for that.

So: **the mechanism is confirmed, the intervention is not a fix, and it was not
committed.** On the workload SV-COMP actually runs, `M_ARENA_MAX=1` is worth
×0.999. Anyone reaching for it again should read this row first.

Two consequences for what is left:

- The measured gain also shrank between the window's slow endpoint (×0.946) and
  current master (×0.999) on the same experiment. Whether current master is
  still ~7 % slower than `978a007e73` at all is now the open question, and it
  needs current master built in the bisect's build directory — the numbers above
  come from two different build configurations and are not comparable across
  that line. **Answered below: yes, in full.**
- If the arena effect is real but masked by library-load fragmentation, then
  **W1 becomes interesting again for the multiplicative term after all** — not
  because loading is slow, but because what the load leaves in the arena makes
  the rest of the run slower. That is a different claim from the one refuted
  above, and it is testable the same way: W1's split blob, then this A/B.

#### Current master, one week on: the term is still there in full

`master` (`d72276d247`, 2026-08-15) built in the bisect's own build directory,
against the fast endpoint, 12 pairs on an idle host, library loaded:

| metric | `978a007e73` | `master` | B/A | IQR |
|---|---|---|---|---|
| wall | 9.890 s | 10.667 s | **1.070** | 0.016 |
| GOTO creation | 0.311 s | 0.317 s | **1.020** | 0.024 |
| symex | 0.764 s | 0.844 s | 1.101 | 0.072 |
| slicing | 0.673 s | 0.712 s | 1.057 | 0.061 |
| encoding | 4.226 s | 4.779 s | **1.115** | 0.047 |
| solving | 2.115 s | 2.203 s | 1.038 | 0.048 |

Two things at once.

**The fixed term is largely paid off.** GOTO creation was ×1.272 at the window's
slow endpoint and is ×1.020 today — W3.1 (the vector-backed read table) landed
in between, and this is an independent confirmation that it did what §2.5 said
it would, measured on a different oracle and against a different baseline.

**The multiplicative term is untouched**, a week after the window closed:
×1.070 on wall, and it is *not* spread evenly any more — encoding (×1.115) and
symex (×1.101) carry it, while solving has fallen back to ×1.038. Nobody fixed
this, and every SV-COMP task is still paying it.

#### The fix: one arena *and* a trim, which only work together

Both patches applied to `master` in the bisect's build directory, measured
against the `master` binary from the same directory, 12 pairs, library loaded:

| build | wall | goto | symex | encoding | solving |
|---|---|---|---|---|---|
| `mallopt(M_ARENA_MAX, 1)` only | 0.999 | — | 0.932 | 1.046 | 0.955 |
| `malloc_trim(0)` after GOTO creation only | **1.033** | 1.025 | 1.061 | 1.051 | 1.005 |
| **both** | **0.953** | 0.941 | 0.970 | **0.925** | 0.967 |

IQR on the winning row is 0.007 — the tightest measurement in this plan.

Neither half is a fix on its own; the trim alone is a 3.3 % *regression*. The
interaction is the point. A single arena puts the operational-model library's
freed blocks in the same arena the rest of the run allocates from, and the trim
hands them back before encoding starts. Without the arena change, the trim only
buys page faults on the way back up. With both, **every phase improves** and
encoding — the phase that carried the regression on master (×1.115) — turns
into a 7.5 % gain.

That is ~4.7 of the ~7.0 points of #6831's multiplicative term, on a change of
two lines under `__GLIBC__`, keeping #6618's crash fix intact.

**Shipped as [#7051](https://github.com/esbmc/esbmc/pull/7051).** Validation:

- **Short runs do not pay for the trim.** `int main(void){return 0;}`, 20
  pairs: ×0.977 wall, ×0.956 GOTO creation. The trim costs a 0.4 s run
  nothing, and the arena change helps the library load itself.
- **`--k-induction-parallel` is safe**: it `fork()`s (`k_induction.cpp:152`),
  so each process keeps its own arena configuration.
- **`--parallel-solving` is not.** It spawns one `std::thread` per claim job
  (`bmc.cpp:3023`), all allocating concurrently; capping the process at one
  arena would serialise them on a single malloc lock. The cap therefore has to
  be skipped for that mode — and since a thread's arena is chosen at its first
  allocation, before options are parsed, `main()` has to look for the flag in
  `argv` rather than wait for the option to be available. Not elegant; the
  alternative is leaving the ~5 % on the table for every sequential run.
- **The gain is ESBMC-bound time, not wall time in general.** On a
  solver-dominated workload — 200 iterations of `memset`/`memcpy` over a heap
  buffer, 14.7 s of which 10.9 s is inside Bitwuzla — the fix measures ×0.994,
  i.e. nothing. It recovers time in symex, encoding and GOTO creation, which is
  where the regression was, so the two are consistent; but a task that spends
  its 99 s in the solver is not saved by this.
- **Regression suite green** on the patch: 61 core C, 100 Python, 100 C++, 80
  `esbmc-unix` (the pointer- and malloc-heavy suite, the one most likely to
  notice an allocator change), plus a `--parallel-solving` run over the guarded
  path.

#### The bisect rig

The window is `978a007e73` (2026-08-01, fast) to `7835797ebc` (2026-08-08,
slow), 183 commits — about eight builds under binary search. Two pieces of it
are now in the tree rather than in someone's shell history:

- **`scripts/perf/ab_interleave.py`** — alternates the two binaries inside one
  loop, flips the order every pair, and reports the **median of the per-pair
  ratios** (not the ratio of two medians: the pairing is what cancels drift)
  with its IQR and sample count, per phase, alongside the count fingerprint.
  It says whether the VCC or symex-assignment counts moved, since that is the
  stopping rule above and not something to eyeball from two logs. A run that
  exits abnormally aborts the comparison with status 2 — silently scoring a
  crashed build as "fast, counts identical" is the one failure this tool must
  not have. `scripts/perf/test_ab_interleave.py` pins the regexes against
  captured ESBMC output.
- **`scripts/perf/oracles/loop10k.c`** — the oracle, `--unwind 10000
  --overflow-check --quiet`, ~11 s wall on the §2 host: 0.3 s GOTO creation,
  0.8 s symex, 0.2 s caching, 0.7 s slicing, **4.3 s encoding**, 2.1 s
  solving. Deterministic, so its 79,992 VCCs and 120,003 symex assignments are
  a fingerprint. `--quiet` matters: the 10,000 unwinding lines cost ~8 % of
  wall, common-mode work that dilutes the ratio being measured. A
  nondet-indexed array was tried first and rejected: it puts ~95 % of its time
  in the solver (49 s wall for 0.03 s of symex), which measures Bitwuzla
  rather than ESBMC.

**Noise floor: ±1.2 % on wall, ±4 % per phase.** Running the *same* binary as
both A and B, 8 pairs, gives wall ×0.988 but symex ×1.040 and GOTO ×1.033 —
the per-phase medians are noisier than the effect being hunted, because each
phase is a smaller number sampled the same number of times. The floor tightens
with pair count (the tool defaults to 12), so quote the count with the ratio.
Read the bisect verdict off wall time and the counts; use phase splits only to
explain a wall delta already established, and treat a per-phase difference
under ~5 % at this pair count as nothing.

The first runs on a cold cache are ~15 % slower and decay non-linearly, which
an order flip cannot cancel and which would land entirely on whichever binary
goes first; the tool spends one discarded run per binary on that transient
before measuring.

### W1 — Split the blob so a task loads only its language's models

`sol64.goto` is already a separate blob for exactly this reason. Do the same for
the Python models (911 symbols, 24 %), and evaluate the same for `libm` (1,002
symbols, 26 %) behind the existing `ENABLE_LIBM` split.

The C-side selection machinery already exists —
`goto_binary_reader::set_functions_to_read()` — but §2.1 shows it does not help
where it matters: the whitelist is applied *after* `symbolconverter.convert()`
has already deserialised the symbol, so Python's filter cuts the dependency
scan (0.11 s → 0.03 s) and leaves deserialisation untouched (0.20 s). Filtering
at read time cannot beat not shipping the bytes down that path at all.

**Exit:** a C task's deserialise+deps time down by ≥24 % (more, per §2.4), no
verdict change in any suite, and Python/Solidity/C++ regressions unchanged.

**Feasibility, measured.** The models compile standalone — `c2goto
library/python/*.c --64 --floatbv` produces a goto binary with warnings only,
no missing declarations — and it is **1,241,971 bytes against `clib64_fp`'s
3,369,011**, i.e. 37 % of the blob by size, more than the 24 % that the symbol
count suggested. So the split is buildable by mirroring what
`mangle_clib()` already does for `sol64`, and the prize is larger than §2.3
implies.

One thing does not mirror Solidity. `sol64` is self-contained — "sol64 holds
ONLY Solidity symbols, so callers need no whitelist" — but the Python models
call into libc and libm (`python_c_extern_deps` in `cprover_library.cpp` lists
`strncmp`, `ceil`, `fegetround`, the `__pyt_*` threading helpers, and more), so
a Python run has to read *both* blobs and resolve across them. The win is
therefore asymmetric by design: C, C++ and CHERI tasks stop paying for the
Python models entirely; Python tasks pay roughly what they do now, split over
two reads. That is the right trade — Python is a minority of any SV-COMP run —
but it means G2's "every frontend re-measured" is not a formality here.
**Shipped as [#7058](https://github.com/esbmc/esbmc/pull/7058)** (Python half;
`libm` not attempted). `py64` and `py64_fp` are built the way `sol64` is, and
`clib64_fp.goto` drops from 3,369,011 to 2,348,422 bytes — **−30 %**.
Interleaved A/B against the same build without the split:

| program | wall | GOTO creation |
|---|---|---|
| `int main(void){return 0;}` (14 pairs) | 0.760 | **0.720** |
| trivial Python (12 pairs) | 0.959 | **0.960** |

**Exit met, and G2's worry did not materialise**: the plan expected Python to
be flat at best, since it now reads two blobs, and it gains 4 %.

Two things this cost that the sketch above did not anticipate:

- **The split is not self-contained, so read order became load-bearing.**
  `sol64` holds only Solidity symbols; the Python models call libc, libm and
  the pthread helpers. `contextt::add` silently rejects duplicates, so clib is
  read *first* (its definitions win) and a nil declaration carried by the
  models' headers is dropped when clib holds the body in `ignored_ctx` — adding
  it would shadow the definition the dependency closure then cannot reach. Get
  that order backwards and libc bodies silently become declarations: a verdict
  change, not a crash, which is what G1 exists to catch.
- **Measurement discipline decided the outcome.** Single runs on a loaded host
  said the split made C 45 % *slower* and Python 2.2× slower. Interleaved on
  the same host at the same moment: ×0.72 and ×0.96. The numbers that looked
  like a reason to revert were an artefact of a load average of 33.

**Not done here:** the `libm` half (1,002 symbols, 26 %), which needs the same
treatment behind `ENABLE_LIBM` and is a bigger question — unlike the Python
models, libm is reachable from ordinary C, so it cannot be keyed off the
frontend and a wrong answer drops a model a task needed. That is a G1
soundness risk rather than a performance one.

#### Both fixes together, against the fast endpoint

`master` + #7051 + #7058 built in the bisect's build directory, 12 pairs on the
oracle, library loaded:

| metric | `978a007e73` | master+both | B/A |
|---|---|---|---|
| wall | 9.272 s | 10.398 s | **1.039** |
| GOTO creation | 0.293 s | 0.210 s | **0.690** |
| symex | 0.744 s | 0.782 s | 1.072 |
| encoding | 4.069 s | 4.617 s | 1.081 |
| solving | 2.026 s | 2.171 s | 1.012 |

**×1.070 → ×1.039: about half the regression recovered**, and GOTO creation is
now 31 % *below* the 2026-08-01 baseline rather than merely restored.

The residue is symex and encoding, and the shortfall against what the parts
predicted (×1.070 × ×0.953 ≈ ×1.02) is itself informative: **the two fixes
overlap**. The arena fix's value came from trimming what the library load left
in the arena; W1 makes that load smaller, so there is less left to trim. They
are not additive, and anyone re-measuring one of them after the other lands
should expect a smaller number than this plan quotes for it in isolation.

#### The residue is ten model bodies nothing calls — and §4 W0 got this wrong

The ~4 points left after both fixes are **entirely library-mediated**: the same
A/B under `--no-library` is ×0.994. So they are not code layout and not creep.

A trivial C program's GOTO program holds **6 function bodies on `978a007e73`
and 16 on master**. The ten are exactly the `__CPROVER_*` primitives #6708
added — `POINTER_OBJECT`, `POINTER_OFFSET`, `same_object`, `OBJECT_SIZE`,
`DYNAMIC_OBJECT`, `LIVE_OBJECT`, `WRITEABLE_OBJECT`, `r_ok`, `w_ok`, `rw_ok` —
linked with bodies into every C program, none of which calls them.

Deleting them from `builtin_libs.c` and re-measuring against the same build,
**20 pairs on an idle host** (a first pass at 12 pairs under load said ×0.958
and ×0.920; those numbers were too generous and are corrected here):

| metric | master+both | minus the ten | B/A |
|---|---|---|---|
| wall | 9.782 s | 9.625 s | **0.975** |
| encoding | 4.467 s | 4.212 s | **0.936** |
| symex | 0.790 s | 0.786 s | 0.999 |

On plain `master`, the same removal via the shipped filter is worth only
×0.994 (20 pairs). So the ten bodies cost **0.6–2.5 % depending on what else
the build carries** — real, concentrated in encoding, and an order of magnitude
short of "the rest of the regression".

**This contradicts W0's hypothesis-2 probe above**, which ran the same deletion,
found `Symex completed in` unchanged, and concluded "#6708 contributes nothing
to the multiplicative term". That reading was right about symex and wrong about
the total: unreferenced bodies cost nothing to *execute* and plenty to *encode*.
The probe measured the one phase where the effect could not appear. Corrected:
**#6708 is worth 0.6–2.5 % here, in encoding — real, but not the rest of the
regression.** The first version of this section claimed ~4 %, from 12 pairs on
a host at load average 33; 20 pairs on an idle host do not support it. Both
that claim and the probe it corrected failed the same way, in opposite
directions, and the fix for both is the same: measure the phase the mechanism
predicts, with enough pairs, on a quiet machine.

**So the regression's remaining ~2–3 points are still unattributed.** What is
ruled out: the library as a whole (`--no-library` A/B), the arena (#7051), the
blob size (#7058), and now these ten bodies at the scale first claimed.

The fix is not to remove the primitives — a program that uses them needs them.
It is to stop linking bodies nothing references. They arrive because
`esbmc_intrinsics.h` is force-included, so the context holds a nil-valued
declaration for each, and `add_cprover_library()`'s rule for the C path is
"declared here, empty value → link the body". Every intrinsic in that header
is therefore linked into every program whether or not it is mentioned. Options,
cheapest first: drop function bodies with no callers after goto-conversion;
or make the closure demand a reference rather than a declaration. Either is a
change to shared linking behaviour, so G1 applies in full.

### W2 — Make the blob indexable, so loading is O(used) not O(total)

The end state the issue asks for: seek to the symbols a task needs. Today this
is impossible without a format change — `reference_convert()` resolves
back-references against a table built in stream order, so record *N* may depend
on an irep first defined inside record *M* < *N*, and no record is independently
decodable.

Design sketch (one PR to specify, one to implement):

1. Bump `BINARY_VERSION` and write a **shared irep/string pool first**, then
   self-contained symbol records, then **a name → offset index**.
2. Read the index and the pool eagerly; deserialise a symbol record on first
   lookup.
3. `add_cprover_library()` becomes: link the frontend's undefined symbols,
   resolve their dependency closure through the index, and touch nothing else.

Risks are real: the pool may itself be most of the blob (measure before
committing to the design — if shared ireps dominate, the win collapses), and
every consumer of the format (`c2goto`, `--binary`, goto binary round-trips)
must move together. W1 delivers a large fraction of the win with none of this
risk, which is why it is sequenced first.

**Measured, and the risk is realised. Do not build this as sketched.**
A throwaway probe in `reference_convert()` charges every byte of the stream to
the record that owns it (a record's span minus the spans of records nested
inside it), reading `clib64_fp` for `int main(void){return 0;}`:

| | |
|---|---|
| distinct ireps (the pool) | **101,027** |
| back-references to them | 209,803 |
| bytes owned by distinct ireps | **3,363,256** |
| bytes in the stream | 3,378,683 |

**The pool is 99.5 % of the blob.** Two thirds of all irep slots are shared, so
the sharing is real and load-bearing — but what is left once you factor it out
is 15 KB of framing, not a set of substantial per-symbol records. Step 2 of the
sketch, "read the index and the pool eagerly, deserialise a symbol record on
first lookup", therefore reads 99.5 % of the bytes before it has done anything,
and there is no version of it that is O(used).

The alternative — self-contained records that duplicate whatever they share —
is a size trade, not a free one: with 209,803 of 310,830 slots being repeats,
independent records would inflate the blob severalfold, and §2.4 shows per-symbol
cost *rising* with blob size. That is the wrong direction.

**Recommendation: close W2.** What it was for is now largely delivered by other
means — W1 removed 30 % of the bytes for C tasks and W3.1 made the remaining
read ×0.805 — and §9 shows the fixed cost already a third below where the
regression started. If loading ever needs to be sublinear again, the question
to reopen is not indexing but whether the *frontend* can avoid asking for
symbols it will not use.

**Exit:** trivial-program library time proportional to symbols kept (≈100), not
symbols present (3,847); `esbmc --binary` round-trip regressions pass.

### W3 — Take the two free wins in the reader

Independent of W1/W2, no format change:

1. **Vector-backed `ireps_on_read`** (§2.5): **shipped.** Landed as a plain
   `std::vector<irept>` rather than the prototype's `pair<bool, irept>`: ids are
   dense, so a first occurrence is always exactly the next index, and that is
   checked (`id > size()` is a corrupt stream) instead of a presence flag. The
   slot is claimed before `read_irep()` recurses, since children are numbered
   after their parent. Re-measured interleaved on every frontend — GOTO creation
   ×0.805 (C, 15 pairs), ×0.883 (C++, 12), ×0.927 (Python, 12).
2. **Do not populate a discarded `goto_functionst`** (§2.6): **shipped**
   (`f6795ec90e`, #6914). The reader takes a nullable pointer and skips the
   work when a caller wants symbols only. −10 ms C, −13 ms C++ on GOTO
   creation, interleaved medians.

**Exit:** met — both landed with before/after timings, regression suite green,
`--binary` reading of externally-produced goto binaries unaffected.

### W4 — Report the cost, so it cannot silently regrow

Every number in §2 required a throwaway patch, because ESBMC reports "GOTO
program creation time" as a single figure that hides a 0.32 s library load
inside a 0.44 s total. That is why an OM library growing by 33 KB of source
reached SV-COMP before it reached anyone's attention.

Add to the existing timing output (and the JSON run summary): library
deserialisation time, dependency-scan time, symbols present, symbols kept. Then
add a CI check that fails when the fixed cost of a trivial program regresses
past a threshold — the metric this plan exists because nobody was watching.

**Shipped.** `add_cprover_library()` reports all four §2.1 phases it owns
(deserialise, dependency scan, select, link), the blob it read, and symbols
present and kept, on one `log_debug("c2goto", …)` line — visible under
`--verbosity c2goto:9`, silent by default, so no run's output changes. On the
§2 host, `int main(void){return 0;}` now says: 3,854 symbols present, 104 kept,
deserialise 0.149 s, dependency scan 0.091 s, select 0.001 s, link 0.000 s —
i.e. the blob has gained 7 symbols since §2.2 was measured, which is exactly
the drift this workstream exists to make visible.

The blob name is in the line because Solidity loads two (sol64 in `typecheck`,
clib in `final`), and because the report should say which one it measured.

**A phase boundary the throwaway instrumentation of §2 got wrong.** The
whitelist path scans dependencies a second time, inside the `to_include` loop,
for every symbol it pulls out of `ignored_ctx`. Billing that to "select" made
Python look like it scanned 3× faster than C (0.031 s vs 0.092 s) when it does
the same work: attributed correctly, Python's scan is 0.090 s and its select is
0.005 s. §2.1's C numbers are unaffected — C has no second scan — but any
future Python figure must use the corrected split.

Two deviations from the sketch above, both deliberate:

- **No JSON entry.** ESBMC's only JSON output is the per-counterexample report
  (`generate_json_report()`); there is no run summary to add a field to.
  Creating one is out of scope here.
- **The CI budget is a symbol ceiling, not a wall-clock threshold.** A time
  threshold on a shared runner is a flake generator. `symbols kept` is
  deterministic, drives every phase in §2.1, and is what actually grew.
  `regression/esbmc/library_fixed_cost_budget` caps a trivial C program at 199
  kept symbols (today 104); `regression/python/library_fixed_cost_budget` caps
  a trivial Python program at 1,999 (today 1,176) and also covers the whitelist
  path, where "present" is the `new_ctx` + `ignored_ctx` split. The bound is
  one-sided on purpose: W1 and W2 exist to *reduce* the closure, and a floor
  would make the metric's own guard fail on the improvement it is guarding.
  Both carry `REQUIRES bundled_libc`, since `ESBMC_BUNDLE_LIBC=OFF` parses the
  library from sources and leaves nothing to measure.

**Exit:** met — `esbmc trivial.c --verbosity c2goto:9` reports the §2.1
breakdown, and a regression test per frontend pins the trivial-program budget.

### W5 — Stop the 99 s/100 s band from generating false signals

The issue's suggested action 2: 131 tasks sitting within 1 % of the limit will
flip on any commit costing a few percent, in either direction, forever. Whatever
W0 finds, this band will keep producing score movements that read as regressions
and are not.

Options, in preference order: (a) speed the tasks up — W0's finding may do this
for free; (b) report them separately in the SV-COMP summary as *marginal* so a
flip is visibly a timing artefact; (c) accept them as lost and stop
re-litigating. This is a benchmarking-hygiene item, not a code change, and it
should not block W0–W4.

**Exit:** the SV-COMP run summary distinguishes tasks that timed out from tasks
that timed out *within 5 % of the limit*.

---

## 5. Sequencing

~~W0 first and alone~~ — **done**; its outcome did redirect the rest, as this
section anticipated:

- **W0 is closed.** The term is a glibc secondary arena, fixed in #7051. What
  it recovers is ESBMC-bound time (×0.953 on a 10 s task, ×0.994 on a
  solver-bound one), so the 131 Juliet tasks are recovered only to the extent
  they are ESBMC-bound — which is measurable now and worth measuring before
  anything else is planned around it.
- **W1 gained a second reason and is next.** Beyond the fixed term, W0 found
  that a single arena makes encoding slower *because of what the library load
  leaves free in it*; a smaller blob leaves less. The same A/B measures it.
- W2 only after W1's measurement, and after W2's pool-size question. W4 landed
  (#7039). W5 is independent of all of them.

---

## 6. Gates

Every workstream here changes what gets loaded or how, i.e. it can silently drop
a model a task needed and turn a `false` into a `true`.

- **G1 — no verdict change.** Full regression suite, all frontends (C, C++,
  Python, Solidity, CHERI), verdict-for-verdict identical. A partial load that
  drops a model shows up here or nowhere.
- **G2 — every frontend re-measured.** Python's path already behaves differently
  from C's (§2.1); a change tuned on C must be timed on Python and C++ too.
  Python's `select`+`c_link` cost 0.10 s where C's costs 0.004 s, so a "win" on
  C can be a loss on Python.
- **G3 — measured, not asserted.** Every claimed saving quoted as before/after
  wall and phase time on a named input, with the run count.
- **G4 — dual-solver agreement.** Bitwuzla and Z3 agree on the changed set.
- **G5 — format compatibility (W2 only).** `c2goto` output, `--binary` input,
  and CBMC goto-binary reading all round-trip after the version bump, with a
  regression test per direction.

---

## 7. Risks

- **Optimising the wrong term.** ~~The single largest risk in this plan:~~
  **Confirmed, not merely feared.** W1–W3 are tractable and satisfying and
  would not have saved one of the 131 Juliet tasks: W0's `--no-library` A/B
  shows the ~7 % is entirely present on runs that never touch the blob. They
  are still worth doing for the fixed term; they are not a fix for the losses.
- **A partial load that silently drops a model** turns unsound. G1 is not
  optional, and "the trivial program still verifies" is not evidence.
- **W2's win may be smaller than it looks** if the shared irep pool dominates
  the blob. Measure the pool before designing around it.
- **Single-machine measurement.** §2 is one host. Ratios are indicative;
  orderings are reliable.
- **The OM library will keep growing.** W4 now makes the growth visible from a
  normal run and fails a test when a trivial program's closure moves, so the
  next 33 KB of models is no longer silent — but visibility is not a fix, and
  the cost still lands on every task until W1/W2.

---

## 8. Non-goals

- **Cause 1 of #6831** (the schedule-space explosion behind #6607, 291 lost
  tasks). Separate defect, separate plan.
- **Removing or shrinking operational models to save load time.** The models
  earn their place; the loading strategy is the defect, not the models.
- **The `incorrect 6 → 8` delta.** The issue establishes it came from the
  benchmark repository moving between runs.
- **Pinning the sv-benchmarks revision** (suggested action 4) — benchmarking
  infrastructure, not this cost.

---

## 9. One-line summary

90 % of a trivial run is deserialising an operational-model blob of which 3 % is
used, superlinearly in its size, with 24 % of it Python models a C task can
never call — but that fixed cost is not what lost the 131 Juliet tasks sitting
at 99 s of a 100 s limit, so attribute the 3.5 % multiplicative term first (W0)
and only then stop paying for models nobody asked for (W1–W4).

**Outcome.** The multiplicative term was a glibc secondary arena, created the
moment the run moved onto a worker thread; the fixed term was the blob, and
neither was what the issue supposed. All three fixes together, 20 pairs on an
idle host against the window's fast endpoint:

| metric | `978a007e73` | with #7051 + #7058 + #7060 | B/A |
|---|---|---|---|
| wall | 9.794 s | 9.224 s | **0.999** |
| GOTO creation | 0.308 s | 0.200 s | **0.670** |
| symex | 0.748 s | 0.784 s | 1.065 |
| encoding | 4.182 s | 4.042 s | 0.995 |
| solving | 2.150 s | 1.965 s | 0.990 |

**×1.070 → ×0.999: the regression is closed on this oracle**, and GOTO creation
is a third faster than it was before the regression rather than merely
restored. `symex` is the one phase still above parity; at ×1.065 of a 0.75 s
phase it is ~50 ms of a 9.2 s run, and wall does not see it.

The caveat that matters for the 131 tasks: this is ESBMC-bound time. A task
that spends its 99 s inside the solver gains nothing from any of it (§W0's
solver-bound workload measured ×0.994), so the score should be re-measured
rather than predicted from these ratios.
