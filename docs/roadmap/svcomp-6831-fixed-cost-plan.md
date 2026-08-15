# Plan — the per-task cost every SV-COMP run pays (issue #6831, cause 2)

**Status:** in progress. W3.1 (vector-backed read table) is shipped; everything
else here is still plan only.
**Owner issue:** [#6831](https://github.com/esbmc/esbmc/issues/6831), *cause 2 —
a general slowdown tipping tasks already at the limit*, ~198 of 489 lost tasks,
led by 131 `Juliet_Test` no-overflow tasks at a median of 99.1 s of a 100 s
limit.
**Companion plan:** [`svcomp-6831-schedule-space-plan.md`](svcomp-6831-schedule-space-plan.md)
covers cause 1 (the schedule-space explosion). The two are independent.
**Last updated:** 2026-08-09.

**Measurement environment.** All numbers below were measured on an x86_64 Linux
host (Intel Xeon E5-2620 v4, 32 threads) against `build/src/esbmc/esbmc`, ESBMC
8.4.0, `RelWithDebInfo`, built from `4be7fbe015`. Solver: Bitwuzla 0.9.0. Phase
timings come from throwaway `std::chrono` instrumentation in
`add_cprover_library()` and `read_bin_goto_object()`, since neither phase is
reported today (that is W4). Reproducer: `int main(void){return 0;}` —
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
| `c8d4bf6f5c` | 46 | 2026-08-03 | | | running |

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
2. **Do not populate a discarded `goto_functionst`** (§2.6): ~10 ms. Not started.

**Exit:** both landed with before/after timings, full regression suite green,
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

**Exit:** `esbmc trivial.c --verbosity …` reports the §2.1 breakdown, and a
regression test pins the trivial-program budget.

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

W0 first and alone — it is the only workstream that recovers the lost tasks, and
its outcome may redirect the rest. W3 can land in parallel (it is small,
measured, and independent). W1 next; W2 only after W1's measurement shows how
much is left, and after the pool-size question in W2 is answered. W4 should land
before or with W1, so W1's claim is checkable from a normal run. W5 is
independent of all of them.

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
- **The OM library will keep growing.** Without W4 the next 33 KB of models
  costs the next SV-COMP run the same way, and the next investigation starts
  from zero again.

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
