# Temporal properties for CXL: what ESBMC can and cannot check

An assessment of whether the LTL/CTL properties produced by the Seccom
intent-decomposition framework can be discharged with ESBMC, and what was done
about the part that can.

**Short answer:** the safety obligations transfer and are now checked
(`regression/cxl/cxl_fabric_lockdown_01/02`). The branching-time properties —
which are the large majority — do not transfer at all, and ESBMC's `--ltl` mode
was not needed for the part that does.

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

## 7. What would justify `--ltl`

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

## 8. Reproducing

```sh
cmake -DENABLE_CXL_REGRESSION=On -Bbuild -S . && ctest -R cxl_fabric_lockdown
ctest -R 'regression/ltl/'    # ESBMC's own LTL mode, for comparison
```
