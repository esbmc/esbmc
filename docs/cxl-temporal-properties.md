# Temporal properties for CXL: what ESBMC can and cannot check

An assessment of whether the LTL/CTL properties produced by the Seccom
intent-decomposition framework can be discharged with ESBMC, and what was done
about the part that can.

**Short answer:** the safety obligations transfer and are now checked
(`regression/cxl/cxl_fabric_lockdown_01/02`). The branching-time properties —
which are the large majority — do not transfer at all.

`--ltl` has since been wired up too (§7): `scripts/ltl_response_ba.py`
generates the Büchi monitor ESBMC needs, validated byte-equivalent against
ltl2ba's own output. Applied to a real temporal CXL property — the mailbox
doorbell wait — it produces **the same verdict for a correct driver and a
hanging one**, because bounded model checking cannot refute liveness. The
tooling works; the property class is out of reach.

---

## 1. What Seccom produces

An intent in natural language is decomposed into obligations, encoded as a
NuSMV model, and checked. For the CXL-relevant intent — *"Only permit hotplug
of USB HID devices on USB host controllers reachable via CXL/PCIe fabric; deny
any hot add of any new CXL endpoints at runtime"* — the decomposition yields
eight obligations, five predicates and three safety clauses:

| Obligation | `define_clause` | Spec |
|---|---|---|
| `ob_cxl_fabric` | `cxl_fabric = active` | CXL 4.0 §14.1 |
| `ob_deny_cxl_hotadd` | `!cxl_hot_add_event` | CXL 4.0 §14.7.5 |
| `ob_deny_binding` | `!switching_binding_event` | CXL 4.0 §14.7.7 |
| `ob_deny_runtime_cfg` | `!runtime_config_trigger` | CXL 4.0 §14.7 |

## 2. The property mix is not what "LTL formulas" suggests

Counting the generated SMV models:

| Form | Count per model | Example |
|---|---|---|
| CTL `SPEC` | 45–48 | `SPEC EF (sm_pcie_endpoint_msi_support = s4_violation_detected);` |
| `INVARSPEC` | 3–4 | `INVARSPEC !obl_hotplug_allow_network_pcie_only_violation;` |
| `LTLSPEC` | 1–2 | `LTLSPEC G (!obl_hotplug_allow_network_pcie_only_violation);` |

**Roughly 90% of the properties are branching-time CTL**, and every `LTLSPEC`
present is of the form `G (!violation)` or `G (a -> b)` — a safety invariant
written in temporal syntax, not a liveness property.

## 3. ESBMC's temporal capability

ESBMC has an LTL mode: `--ltl <buchi.c> -DLTL_PREFIX_BOUND=N`, exercised by
`regression/ltl/` (3 tests, passing). Its shape matters:

- It consumes a **precomputed Büchi automaton emitted as C**, not an LTL
  string. The `.ba-N.c` files in `regression/ltl/` are pre-generated; no
  `ltl2ba` ships with the repo or sits on `PATH`, so a new formula means
  obtaining that tool or writing the automaton by hand.
- Atomic propositions are **C expressions over program variables**
  (`{ pressed }`, `{ charge > min }`). There is no way to refer to anything
  that is not program state.
- Results are four-valued over a bounded prefix — `LTL_GOOD`,
  `LTL_SUCCEEDING`, `LTL_FAILING`, `LTL_BAD` — so a `G p` violation is found,
  but `G p` is not *proved* beyond the bound, and `F p` / `GF p` answer only
  "has not yet failed within N steps".

## 4. What transfers, and what does not

| Seccom form | ESBMC | Notes |
|---|---|---|
| `INVARSPEC !v` | **Yes** — `assert(!v)` | No LTL machinery needed |
| `LTLSPEC G (!v)` | **Yes** — `assert(!v)` at each point `v` could change | Identical to the above once the model is a program |
| `LTLSPEC G (a -> b)` | **Yes**, same way | Still an invariant |
| `SPEC EF p` | **Partly** — assert `!p` and read the counterexample as the witness | A reformulation, not a translation; a *failing* run is the positive result |
| `SPEC AG EF p` | **No** | Branching-time; no ESBMC equivalent |
| `SPEC AG (a -> EF b)` | **No** | Same |
| True liveness (`F p`, `GF p`) | **Bounded only** | `--ltl` gives "not violated within N", never a proof |

The `EF` reachability checks in the Seccom models are **non-vacuity** checks —
"the model still permits at least one valid event, so the safety property is
not satisfied by denying everything". That intent transfers even though the
operator does not: it is exactly the paired passing/failing test discipline
`regression/cxl/` already uses, where the failing partner shows the guard is
live rather than vacuous.

## 5. The code-grounding problem

Seccom's state machines carry a `linux_source_hint` per edge. For CXL, **352 of
360 hinted files do not exist in Linux 7.1.5**:

```
drivers/cxl/aes_gcm.c   drivers/cxl/arb_mux.c   drivers/cxl/bi_decoder.c
drivers/cxl/ats.c       drivers/cxl/bi_id.c     drivers/cxl/birsp_handling.c   ...
```

The eight that do exist are:

```
core/pci.c   core/hdm.c   core/port.c   core/ras.c
mem.c        pci.c        port.c        security.c
```

which are precisely the files `regression/cxl-linux/` already targets. So the
hints do not widen the reachable surface; anything grounded in real code is
grounded in those eight files. Treat the hints as unverified.

## 6. What was implemented

`regression/cxl/cxl_fabric_lockdown_01` and `_02`, with the supporting model in
`cxl_driver.c` (`cxl_fabric_init/lockdown/submit/bind_completed`).

The model encodes CXL 4.0 §14.7.7's bind sequence —
`bind_initiated → host_recognizes_hot_added_sld → host_enumerates_sld →
fm_indicates_vcs_binding → bind_complete` — behind a policy gate implementing
all three safety clauses. The gate is the only way to advance the state
machine, because a model that let callers assign the state directly would make
the obligations unfalsifiable.

- `_01` composes the fabric, locks it down, then submits every denied event
  class nondeterministically and asserts each is refused and the topology
  unchanged.
- `_02` is the policy bypass: the gate's `-EPERM` is discarded and the endpoint
  bound by hand. This is the same defect shape the rest of the suite keeps
  finding — **a dropped return value**.

`--ltl` was not used, and using it would have added a Büchi automaton and a
prefix bound to check something an assertion checks exactly.

## 7. Wiring in ltl2ba: what it cost and what it bought

`--ltl` needs a Büchi automaton emitted as C, and the patched `ltl2ba` that
produces that format is neither in this tree nor publicly available (checked:
no `esbmc/ltl2ba`, no `ssvlab/ltl2ba`). `scripts/ltl_response_ba.py` closes
that gap for one pattern:

    G (p -> F q)      the response pattern

It is **not** an LTL-to-Buchi translator — it emits one fixed two-state
automaton with the atomic propositions substituted in, and would silently
produce the wrong monitor for any other formula, because it does not parse a
formula at all. What makes it trustworthy is `--self-test`: it regenerates the
monitor for `regression/ltl/basic`'s formula and compares against the
committed ltl2ba output, which it reproduces **byte-equivalently modulo
whitespace**.

### The property

`cxl_pci_mbox_wait_for_doorbell()` (`drivers/cxl/pci.c:57`) submits a mailbox
command by setting the doorbell, polls until the device clears it, and gives
up after `CXL_MAILBOX_TIMEOUT_MS` (CXL 2.0 §8.2.8.4). The obligation is
genuinely temporal:

    G (doorbell_busy -> F mbox_settled)

No single-state assertion expresses it. This is the property I said in the
previous revision of this document did not exist in the Seccom output — it
does not; it comes from the driver source instead.

### The result, which is negative

`regression/cxl/cxl_ltl_doorbell_01` has the timeout. `_02` is the same driver
with the timeout removed — an unbounded wait on a device that may never answer,
a real bug. **ESBMC reports the same outcome for both: `LTL_FAILING`.**

The reason is structural, not a defect in the encoding. `G (p -> F q)` is pure
liveness: it has no finite counterexample, so the generated automaton's
`_ltl2ba_bad_prefix_states` is `{false, false}` and `LTL_BAD` is unreachable
whatever the program does. Bounded model checking cannot refute liveness, and
the four-valued output does not distinguish a correct driver from a hanging
one.

So the honest accounting of combining ltl2ba with this work:

| | |
|---|---|
| Generator, validated against ltl2ba | delivered, reusable |
| A real temporal CXL property, encoded | delivered |
| Ability to catch the bug it describes | **none** |
| Safety reformulation (`G !p`) as a way round it | tables correct, but transient propositions are missed |

The test pair is kept anyway, with both expectations set to `LTL_FAILING` and
headers saying why. It is a tripwire: if ESBMC's LTL support ever distinguishes
them, the pair fails and someone finds out.

### Follow-up: the concurrent invariant, and why it also fails

The obvious next move was a property that is temporal *but still safety*, so
that a finite counterexample exists and `LTL_BAD` becomes reachable — mailbox
mutual exclusion:

    G !(t1_inflight & t2_inflight)

`scripts/ltl_response_ba.py --absence` emits that automaton. Unlike the
response tables it is **constructed rather than transcribed**, so it is
validated empirically by `--self-test-absence`, and the validation passes:

| Program | `a && b` | Outcome |
|---|---|---|
| stably true | always | `LTL_BAD` |
| stably false | never | `LTL_SUCCEEDING` |

So the automaton is right. But the case that matters is neither of those:

| Program | `a && b` | Outcome |
|---|---|---|
| **transiently true** inside a loop | briefly | `LTL_SUCCEEDING` ✗ |

A proposition that holds only in the window between two assignments is
**missed**, and `--context-bound 2/4/8` does not change it. That is precisely
the mutual-exclusion case: `t1_inflight & t2_inflight` is true only in a
narrow interleaving window, which is the entire point of checking it.

This retracts the argument for `--ltl` I made in the previous revision of this
document. I claimed the monitor "samples at every interleaving point, unlike a
placed assert". The evidence says otherwise: `add_monitor_exprs()` does
instrument every instruction, but the monitor thread does not reliably observe
transient valuations, so an LTL monitor is a *worse* mutual-exclusion checker
than an assertion placed in the critical section. No concurrent LTL test was
added, because it would have passed while checking nothing.

### What this implies for the approach

Bounded LTL adds value over `assert()` only where a property is **temporal but
still safety** — has a finite bad prefix. Examples: bounded response ("q within
N steps"), ordering ("`INIT` never re-asserted before `ENABLE`"), and any
invariant over a *concurrent* program, where the monitor samples at every
interleaving point instead of only where an assertion was placed. Unbounded
liveness is out of reach, and the CXL obligations that matter most —
`!cxl_hot_add_event` and its siblings — are safety properties an assertion
already checks exactly.

## 8. Converting the FSMs to C: the vacuity, as a counterexample

`scripts/smv2c.py` translates a NuSMV model into C: enumerated variables become
enums, `next(x) := case ... esac` becomes a switch (with `{a, b}` sets as
nondeterministic choice), free variables are drawn fresh each step, and
`INVARSPEC` / `LTLSPEC G φ` become assertions. CTL is skipped with a comment,
since it has no ESBMC equivalent.

The interesting part is what it **cannot** carry. NuSMV's `INVAR p` and
`TRANS p` are *constraints* — they shrink the state space to the part
satisfying `p`. C has no such construct: a program's reachable states follow
from its statements. So the translation drops them, and lists them in a comment
at the top of the generated file.

That turns §2's argument into an experiment. Running it on
`intent-obligation-model.smv` (45 vars, 42 state machines, 2 constraints
dropped):

```
$ esbmc obl.c --unwind 5
Violated property:
  assertion !obl_hotplug_allow_network_pcie_only_violation
VERIFICATION FAILED
```

with the witness

| variable | value |
|---|---|
| `observed_event` | `hotplug_insert` |
| `endpoint_kind` | `root_complex_integrated_endpoint` |
| `endpoint_class` | `unknown_class` |

**The property NuSMV reports as PASS is false of the state machines.** A
hot-plug insert on a root-complex-integrated endpoint satisfies
`violation := hotplug_event & !(pcie_endpoint_context & network_class_context)`
in a single step. It passes in NuSMV only because `INVAR !violation` removed
those states from the model before checking whether they exist.

So the vacuity is no longer an argument from reading the source — it is a
counterexample trace. And note what the model *does* deny: the three
`INVARSPEC !(obl_hotplug_event & endpoint_kind = ...)` exclusion obligations
name `non_pcie_endpoint`, `root_complex_integrated_endpoint` and
`pcie_endpoint & other_class` — the very combination ESBMC just produced. Those
exclusions are also only true under the same constraint.

The comparison you would run against NuSMV itself needs the binary, which is
not installed here (`NuSMV`, `nusmv`, `nuXmv` all absent; Seccom shells out to
a path you supply). It is not needed for this result: the informative direction
is the unconstrained one, and that is the side ESBMC provides.

## 9. Linux Runtime Verifier: the one that converts cleanly

Linux RV (`kernel/trace/rv/`) specifies monitors as deterministic automata in
`.dot`, translated by `tools/verification/rvgen` into a C header holding an
enum of states, an enum of events, and a transition table with rejected entries
set to `INVALID_STATE`. The obligation is stated outright in
`include/rv/da_monitor.h:690`:

```c
next_state = model_get_next_state(curr_state, event);
if (next_state == INVALID_STATE) { react(curr_state, event); return false; }
```

That is `G (state != INVALID_STATE)` — safety, so an assertion discharges it
exactly. `scripts/rv2c.py` turns either an rvgen monitor header or plain
`dot2c` output into that assertion.

Compared with the Seccom models, everything that made those hard is absent:
the artefact is already C, the events are real tracepoints, and the property is
explicit rather than buried under an `INVAR` that assumes it.

**What this adds over RV itself.** RV observes the sequences a running kernel
happened to produce. Feeding the same automaton to ESBMC makes the event
producer *symbolic*, so every sequence the code could emit is covered at once.
Runtime verification cannot answer that question; bounded model checking can,
up to its bound.

### Demonstrated twice

`cxl_rv_snep_01/02` use **`snep`**, an unmodified kernel monitor
(`kernel/trace/rv/monitors/snep/`, 2 states, 4 events, 4 of 8 transitions
rejected). The passing test's producer respects the protocol; the failing one
toggles preemption inside scheduling context and trips
`assert(next != RV_SNEP_INVALID)` — the kernel's own obligation, checked
statically.

`cxl_rv_mbox_01/02` go the other way: `cxl_mbox.dot` is a **new** automaton for
the CXL mailbox doorbell protocol (CXL 2.0 §8.2.8.4, as transcribed at
`drivers/cxl/pci.c:210`), passed through the kernel's own `dot2c`. The same
`.dot` could generate a live kernel monitor and this static check. The failing
variant submits a second command without verifying the doorbell is clear —
step 1 of the spec sequence — which on hardware loses or corrupts a command
rather than crashing, and so survives testing.

This is the first artefact here shared between the kernel's runtime verifier
and ESBMC's static one.

### On the kernel's own ltl2ba

`tools/verification/rvgen/rvgen/ltl2ba.py` is a real LTL-to-Büchi translator —
568 lines implementing Gerth, Peled, Vardi and Wolper (1996), with a full
grammar (`always`, `eventually`, `next`, `until`, `imply`, …). Kernel models
use it: `models/rtapp/pagefault.ltl` is `RULE = always (RT imply not PAGEFAULT)`.

| | Linux `rvgen/ltl2ba.py` | ESBMC's ltl2ba |
|---|---|---|
| Available | **yes, in-tree, GPL-2.0** | no — absent and unpublished |
| Algorithm | Gerth et al. 1996, cited | unknown; only its output survives |
| Output | kernel monitor: `enum ltl_atom`, `enum ltl_buchi_state`, transition fn | C monitor *thread* + four-valued acceptance tables |
| Semantics | infinite-trace Büchi, judged at runtime | four-valued RV-LTL over bounded finite prefixes |

Adapting it to emit ESBMC's format is plausible — it already builds the
automaton, and `ltl2k.py` shows how a backend is written. The work is not the
automaton but the **four-valued tables** (`stutter_accept`,
`good_prefix_excluded`, `bad_prefix_states`), which encode finite-prefix
semantics the kernel version has no reason to compute. That, rather than my
hand-rolled `ltl_response_ba.py`, is where a general solution should start.

(It could not be run here: `rvgen` needs the `ply` package, which is not
installed. `dot2c` does not, which is why the automaton path above works.)

## 10. What would justify `--ltl`

A property that is genuinely temporal *and* grounded in code — an ordering or
eventuality constraint that no single-state assertion captures. Candidates
would look like:

- "a mailbox command, once issued, is eventually completed or timed out"
  (`G (issued -> F (done | timeout))`);
- "`INIT` is never asserted again before `ENABLE` is observed".

The second is expressible as an invariant over a monitor variable, so only the
first genuinely needs LTL — and it is a liveness property, which ESBMC answers
only within a bounded prefix. I did not find such a property in the Seccom
output: the eight CXL obligations are all invariants. Establishing whether one
exists among the 3,327 generated state machines is a separate exercise, and
should start by discarding the edges whose `linux_source_hint` is fabricated.

## 11. Reproducing

```sh
cmake -DENABLE_CXL_REGRESSION=On -Bbuild -S . && ctest -R cxl_fabric_lockdown
ctest -R cxl_ltl_doorbell                     # the LTL pair
python3 scripts/ltl_response_ba.py --self-test # generator vs ltl2ba output
python3 scripts/smv2c.py <model>.smv -o m.c && esbmc m.c --unwind 5
ctest -R 'regression/ltl/'                     # ESBMC's own LTL tests
```
