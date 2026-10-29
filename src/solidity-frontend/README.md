# ESBMC Solidity Frontend

The ESBMC Solidity frontend verifies smart contracts written in
Solidity. It parses the Solidity AST (produced by `solc
--ast-compact-json`), lowers it to ESBMC's intermediate representation,
synthesises a verification harness, and hands the goto program to
ESBMC's bounded / k-induction engine and its SMT backends.

This README is the user-facing overview. Per-topic developer docs live
under `docs/claude/solidity/`.

## Table of contents

- [Quick start](#quick-start)
- [Solc invocation](#solc-invocation)
- [Supported Solidity versions](#supported-solidity-versions)
- [Choosing what to verify](#choosing-what-to-verify) — `--contract`, `--function`, `--focus-function`
- [External-call resolution](#external-call-resolution) — `--bound` vs default unbound
- [Transaction-sequence bound](#transaction-sequence-bound) — `--solidity-max-tx`, bounded-by-default harness, `--solidity-precise`
- [`assert` / `require` / `revert` semantics](#assert--require--revert-semantics) — what each lowers to, state-rollback on revert
- [Property selection](#property-selection) — overflow, reentry, multi-property, standard checks
- [Solver selection](#solver-selection) — automatic preference, `--16` word size, Z3/CVC5/Bitwuzla/Boolector
- [TOD detection](#tod-transaction-order-dependence-detection) — `--tod-race-check`, `--tod-balance-check`, `--dump-harness`
- [Structural coverage](#structural-coverage) — branch / condition / assertion coverage
- [Externally exposed intrinsics](#externally-exposed-intrinsics) — `__ESOL_deep_copy`, `__ESOL_nondet_state_forward`, `__ESBMC_assert`, `__ESBMC_assume`, `__ESBMC_nondet_*`
- [Approximations you should know about](#approximations-you-should-know-about)
- [Building](#building) — CMake options, Bitwuzla prerequisites, static release build
- [Developer notes](#developer-notes)

---

## Quick start

Pass a `.sol` file directly to ESBMC — it will automatically find and
invoke `solc`:

```sh
esbmc example.sol --contract MyContract
```

ESBMC searches for `solc` in this order: `--solc-bin <path>` > `$SOLC`
env var > `solc` in `$PATH`. You can pin a particular solc:

```sh
esbmc --solc-bin /path/to/solc example.sol --contract MyContract
```

### Manual AST generation (legacy)

You can also generate the AST yourself and pass both files. Use `--sol`
to tell ESBMC the original source (needed for pretty-printing and
source-location mapping):

```sh
solc --ast-compact-json example.sol > example.solast
esbmc --sol example.sol example.solast --contract MyContract
```

## Solc invocation

Auto-invoke (`.sol` input):

1. ESBMC resolves `solc` using the search order above.
2. ESBMC runs `solc --ast-compact-json <input>` and captures stdout to
   a temp AST file.
3. ESBMC prints the discovered path and version, e.g. `Compiling
   Solidity AST using: /usr/local/bin/solc (v0.8.30)`.
4. The temp AST is consumed like a manually-generated `.solast`.

On `solc` error, ESBMC prints the compiler's diagnostics and exits
without attempting verification.

## Supported Solidity versions

Version support is checked at the AST level via the top-level
`PragmaDirective`:

| pragma | Behaviour |
|---|---|
| Upper-bound strictly `< 0.5.0` (e.g. `pragma solidity ^0.4.24;`) | **Rejected.** Frontend prints `unsupported version` and exits. |
| Lower-bound `<0.5.0` but upper-bound admits `>=0.5.0` (e.g. `pragma solidity >=0.4.0 <0.9.0;`) | **Warning, accepted.** solc has already chosen a modern compiler for the file — ESBMC trusts solc's output. |
| `0.5.0 – 0.7.0` | **Warning.** May cause unexpected behaviour; AST/semantic shapes differ in minor ways from 0.8. |
| `>= 0.7.0` | **Fully supported.** Developed and tested against 0.8.x (current test baseline: solc 0.8.30). |

Implementation: `check_min_version` in `solidity_convert.cpp`. If a
contract in your codebase uses pre-0.5 syntax, upgrade the source and
recompile with a modern solc rather than falling back to `solc-select`
on an old compiler — see the user memory feedback
`feedback_upgrade_to_08.md`.

## Choosing what to verify

ESBMC Solidity has **two orthogonal axes** for scoping a verification
run:

1. What contract/function is the entry point?
2. How are external calls resolved?

This section covers axis 1. See [External-call resolution](#external-call-resolution) for axis 2.

### Default — verify every declared contract

```sh
esbmc contract.sol
```

ESBMC constructs a multi-contract wrapper `_ESBMC_Main` that drives
each contract's own `_ESBMC_Main_<C>` harness (constructor + nondet
public-method dispatch loop). Fastest smoke test; all soundness caveats
of `--contract` compound across contracts. Each per-contract harness is
bounded to two transactions by default — see
[Transaction-sequence bound](#transaction-sequence-bound).

### `--contract <ContractName>` (recommended)

```sh
esbmc contract.sol --contract MyContract
```

Narrow verification to a single contract. Constructor runs first;
state variables take their declared/initialiser values; every
public/external method is dispatched nondet-guarded inside a
transaction-sequence driver. **By default that driver is bounded to
two transactions** (a deterministic unroll) so that k-induction / BMC
converge — see [Transaction-sequence bound](#transaction-sequence-bound)
for what this means for soundness and how to restore the classic
unbounded `while (nondet_bool())` loop with `--solidity-precise`.

To target multiple contracts in one run, either repeat the flag or use
a space-separated list:

```sh
esbmc contract.sol --contract A --contract B
esbmc contract.sol --contract "A B"
```

### `--focus-function <name>` — narrow to one function, keep the harness

```sh
esbmc contract.sol --contract MyContract --focus-function myFunc
```

Same harness as `--contract MyContract`, but the nondet dispatch loop
inside `_ESBMC_Nondet_Extcall_MyContract` filters out every method
except `myFunc`. The constructor still runs; state variables still
receive their initialisers; `myFunc` can still be re-entered during its
own execution (via the filtered dispatch loop). This is a pure
verification-cost optimisation: the safety proof strength is the same
as `--contract MyContract` *for the paths through `myFunc`*.

Requires `--contract` when the source declares more than one
(non-library, non-interface) contract; auto-inferred when there is
exactly one.

**Use `--focus-function` instead of `--function` in regression tests.**
It preserves the full post-constructor harness, so adversarial tests
still exercise the dispatch logic and the frontend/solver weaknesses
they are designed to expose.

### `--function <name>` — single-function, nondet state

```sh
esbmc contract.sol --contract MyContract --function myFunc
```

Makes `myFunc` the GOTO entry point and calls it once with nondet
parameters. **The constructor does NOT run.** All state variables
start as unconstrained nondet of their declared type.

- **`VERIFICATION SUCCESSFUL`** under `--function` is a *stronger*
  safety proof than under `--contract` / `--focus-function`, because
  it proves the function safe under *every* conceivable initial state
  (strictly a superset of constructor-reachable states).
- **`VERIFICATION FAILED`** under `--function` may be a false positive
  — the counterexample could rely on a state unreachable from any
  `constructor() → (any tx sequence)`. Re-verify with `--focus-function`
  before trusting the trace.

Best for: quick safety checks on `pure` / `view` functions; over-
approximate proofs for one function. **Do not use `--function` in
regression tests** — it fabricates nondet state, which hides real
frontend/solver weaknesses. See
[`docs/claude/solidity/modes.md`](../../docs/claude/solidity/modes.md)
for the full soundness discussion.

Combining `--function` and `--focus-function` is rejected at
conversion time — the two flags express different intents.

### `--no-visibility`

Forces verification of *every* function, including internal/private
functions that would otherwise be unreachable from the nondet dispatch
harness. The frontend emits a warning when this is set, because driving
an internal/private function directly with nondet arguments **can
produce false positives** — the function may be unreachable from any
public entry under the states the constructor and dispatcher actually
allow. Rarely useful outside developer debugging.

### Summary — which flag to pick

| Goal | Flag combination |
|------|------------------|
| Whole contract, realistic state, all functions | `--contract C` |
| One function, realistic state | `--contract C --focus-function f` |
| One function, over-approximate state | `--contract C --function f` (not in tests) |
| Multi-contract system with known callees | `--contract A --contract B --bound` |
| Quick smoke test of all contracts | (no `--contract`, no `--function`) |
| Unbounded (sound) proof, any of the above | add `--solidity-precise` (or `--solidity-max-tx 0`) |
| Deeper bounded proof (N transactions) | add `--solidity-max-tx N` |

## External-call resolution

### Default: unbound

```sh
esbmc contract.sol --contract C --unbound    # or just --contract C
```

Each contract is verified in isolation. Low-level calls (`.call()`,
`.delegatecall()`, `.staticcall()`) return a nondet `(bool, bytes)`
tuple; their side effect is a nondet re-entry into the **current**
contract's dispatch loop; the target address is ignored. This is an
**over-approximation** for reentrancy detection (SWC-107), which is
what enables classic reentrancy-bug discovery even without a concrete
attacker contract.

### `--bound` — trusted closed-world

```sh
esbmc contract.sol --contract A --contract B --bound
```

Contracts declared in `contractNamesList` are linked together under a
**trusted closed-world assumption**, analogous to SMTChecker's
`--model-checker-ext-calls=trusted` mode. External calls to addresses
matching a known `_ESBMC_Object_X.$address` dispatch to that
contract's nondet dispatch; other addresses produce hard `return false`
(an **under-approximation** relative to real EVM, but sound under the
trusted assumption).

| Aspect | Unbound (default) | `--bound` |
|---|---|---|
| Reentrancy bugs (SWC-107) | ✓ (nondet self-reentry) | ✓ (plus concrete cross-contract dispatch) |
| Cross-contract state invariants | ✗ (contracts isolated) | ✓ |
| `delegatecall` storage context | nondet | ✓ via shadow fast path (literal `abi.encode*` only — see below) |
| `staticcall` read-only enforcement | nondet | ✓ via snapshot + rollback |
| Attacker-controlled arbitrary address | ✓ (nondet reentry models it) | ✗ (unknown-address path pruned) |
| Typical wall-time | Faster | Slower; more SMT constraints per contract |

**Low-level call semantic notes (bound mode):**
- `.call(abi.encodeWithSignature("f(T)", args))` — dispatches to the
  target's `f(T)` and returns `(true, BytesDynamic)`.
- `.delegatecall(...)` — fast path inlines the target body into the
  caller's storage context when the payload is
  `abi.encodeWithSignature` / `abi.encodeWithSelector` /
  `abi.encodeCall` **literal** and every state variable the target
  touches exists on the caller with the same name+typeString.
  Falls back to the generic `$delegatecall#0` dispatcher on any miss.
- `.staticcall(...)` — the target struct is snapshotted before
  dispatch and restored after, so any writes the target performs are
  silently invisible to the caller (correct under real EVM, which
  reverts on writes inside staticcall). Does not currently *surface*
  the "target tried to write" event — sound but loses that specific
  detection.

See [`docs/claude/solidity/modes.md`](../../docs/claude/solidity/modes.md)
for the full dispatch matrix and limitations.

## Transaction-sequence bound

A Solidity contract is a reactive system: after construction, the
outside world can call its public/external methods any number of times,
in any order. ESBMC models this with a **transaction-sequence driver**
that wraps the nondet method dispatcher. Historically that driver was
an unbounded `while (nondet_bool()) { reseed(); dispatch(); }` loop,
modelling an arbitrary-length transaction sequence.

The unbounded loop is the single biggest reason k-induction fails to
converge on Solidity contracts: the inductive step has to find a state
invariant strong enough to survive an arbitrary next transaction, which
is rarely automatic. In practice the suite worked around this by forcing
`--unwind N --no-unwinding-assertions`, which is itself an unsound
silent truncation.

### Bounded-by-default (the new behaviour)

As of `--solidity-max-tx`, the driver is **bounded to two transactions
by default**. Instead of the while-loop, the harness deterministically
emits the per-transaction body (`reseed(); dispatch();`) `N` times:

```c
// default N = 2, conceptually:
_sol_per_tx_reseed(); _ESBMC_Nondet_Extcall_C();   // tx 1
_sol_per_tx_reseed(); _ESBMC_Nondet_Extcall_C();   // tx 2
```

Each copy still dispatches a nondeterministically-chosen public method
with nondet arguments, and `_sol_per_tx_reseed()` re-havocs the ambient
`msg.*` / `tx.*` / `block.*` environment per transaction. So a bound of
`N` explores **every ordering of up to `N` method calls**, just not
sequences longer than `N`.

Because the driver is now a finite, loop-free unroll:
- k-induction / BMC **converge** without `--unwind` hacks;
- there is no transaction back-edge, so no spurious "unwinding
  assertion" noise from the harness loop itself.

**Forward-condition note.** Independently of the bound, the Solidity
dispatcher mode (any non-`--function` run) **auto-disables the
k-induction forward-condition phase**: the `while(nondet) dispatch()`
shape is unboundable, so forward condition can never prove and only
burns solver budget — only the inductive step (or, now, the bounded
unroll) closes the proof. Pass `--enable-forward-condition` to run it
anyway, e.g. for diagnostic comparison.

### Soundness posture (read this)

Bounding the transaction count is an **under-approximation**:

- **`VERIFICATION FAILED` is sound.** A counterexample found within `N`
  transactions is a real bug — bounding never invents behaviour.
- **`VERIFICATION SUCCESSFUL` is bounded.** It means *"no violation
  within `N` transactions"*, **not** an unbounded proof. A bug that
  requires a 3rd (or later) transaction to set up is invisible at the
  default `N = 2`.

To make this impossible to miss, the frontend emits a one-time
`log_warning` on every bounded run (the `VERIFICATION SUCCESSFUL` /
`FAILED` verdict line itself is left unchanged):

```
Solidity harness: transaction sequence bounded to 2 tx (default). A
VERIFICATION SUCCESSFUL result is bounded -- it means no violation
within 2 transaction(s), NOT an unbounded proof; bugs requiring more
transactions are not explored. Use --solidity-precise (or
--solidity-max-tx 0) for an unbounded proof.
```

### `--solidity-max-tx N` — choose the bound

```sh
esbmc contract.sol --contract C --solidity-max-tx 3   # unroll 3 transactions
esbmc contract.sol --contract C --solidity-max-tx 1   # single transaction
esbmc contract.sol --contract C --solidity-max-tx 0   # unbounded while-loop
```

| Value | Harness | Use when |
|---|---|---|
| `N > 0` | Deterministic unroll of `N` transactions | You want k-induction/BMC to converge and `N` transactions is enough to reach your bug / cover your property. Larger `N` ⇒ deeper sequences, more SMT cost. |
| `0` | Unbounded `while (nondet_bool())` loop | You want an unbounded proof and are prepared to manage k-induction convergence (or pass `--unwind`). Equivalent to `--solidity-precise`'s effect on the harness. |
| *(unset)* | Bounded to `2` (default), unless a mode below forces unbounded | Default — fast convergence, bounded guarantee. |

An explicit `--solidity-max-tx` **always wins**, even over the
mode-forced-unbounded rule below and over `--solidity-precise`.

The driver treats any value `<= 0` as unbounded (`emit_tx_driver` keys
on `bound <= 0`), so a negative `N` behaves like `0`. Only `0` is the
supported spelling for "unbounded" — negative values are not rejected
but are not a documented interface; use `0`.

### `--solidity-precise` restores the unbounded harness

`--solidity-precise` already controls precise address-uniqueness
modelling (see [Approximations](#approximations-you-should-know-about)).
It now *also* opts the transaction driver back into the unbounded
`while (nondet_bool())` loop — i.e. it is the single "I want a sound,
unbounded proof" knob:

```sh
esbmc contract.sol --contract C --solidity-precise            # unbounded harness
esbmc contract.sol --contract C --solidity-precise --unwind 5 # ...with a manual k cap
```

This is why several regression tests that genuinely need a 3rd+
transaction (or that exercise the precise address model) carry
`--solidity-precise` in their `test.desc`.

### Modes that force the unbounded harness automatically

Some analyses manage their own exploration depth and would be distorted
by a fixed transaction bound, so the frontend ignores the default bound
(treats it as `0`) whenever any of these is active — unless you override
with an explicit `--solidity-max-tx`:

- `--solidity-precise`
- TOD detection: `--tod-race-check`, `--tod-balance-check` (a fixed bound
  hides ordering races and the TOD pairing depends on the full driver)
- Structural coverage: every `--*-coverage` / `--*-coverage-claims`
  family option (a fixed bound distorts branch/condition counts; the
  coverage loop neutralisation handles the harness loop separately)

Implementation: `get_tx_bound()` and `emit_tx_driver()` in
`solidity_convert_contract.cpp`; the same driver is shared by the
single-contract harness (`multi_transaction_verification`), the
multi-contract wrapper, and the `--bound` peer driver
(`build_bound_drive_helper` in `solidity_convert_constructor.cpp`), so
the bound applies uniformly across `--contract`, default whole-program,
and `--bound` / `--unbound` modes.

### Interaction with `--bound` / `--unbound` and `--contract`

The transaction bound is **orthogonal** to external-call resolution:

- `--bound` / `--unbound` choose *how external calls are resolved*;
  `--solidity-max-tx` chooses *how many transactions the driver issues*.
  Both `--bound` and `--unbound` harnesses are bounded to 2 tx by
  default and both honour `--solidity-max-tx` / `--solidity-precise`.
- `--contract` selects the harness entry point; the bound then applies
  to that contract's transaction driver. With no `--contract`, every
  per-contract driver in the whole-program wrapper is bounded the same
  way.

## `assert` / `require` / `revert` semantics

Solidity's three failure primitives lower to **different** verification
constructs. Getting this distinction right is essential to reading an
ESBMC verdict, because only one of them is a checked property — the other
two are control-flow that *rolls state back*.

| Solidity | Lowers to | Effect on verification |
|---|---|---|
| `assert(cond)` | ESBMC assertion claim (`c:@F@assert`) | **A checked property.** If `cond` can be false on a reachable path, ESBMC reports `VERIFICATION FAILED` with a counterexample. This is the thing you are trying to prove. |
| `require(cond[, msg])` | `if (!cond) { rollback; return; }` (state-rollback) — legacy fallback: `__ESBMC_assume(cond)` | **A guard, not a property.** The `!cond` path reverts: it restores state and returns, so it never reaches downstream asserts. It cannot itself fail verification. |
| `revert([msg])` / `revert CustomError(...)` | `{ rollback; return; }` (state-rollback) — legacy fallback: `__ESBMC_assume(false)` | **An unconditional revert.** Same rollback shape as a failed `require`. |

### `assert` is a property, `require`/`revert` are rollbacks

- **`assert`** is the only one that can produce `VERIFICATION FAILED`.
  ESBMC models it as a plain checkable assertion. Note this is *stricter*
  than real EVM semantics: on-chain, a failing `assert` triggers a
  `Panic(0x01)` that **reverts** the transaction (state is rolled back,
  gas burned) rather than being a "bug". ESBMC deliberately treats it as
  a property to verify — a reachable failing `assert` is reported, not
  silently reverted. (Lowering: `solidity_convert_ref.cpp` maps `assert`
  → `c:@F@assert`; the claim is emitted in `solidity_convert_call.cpp`.)
- **`require` / `revert`** lower through `build_revert_rollback_block`
  (`solidity_convert_modifier.cpp`). The revert path stays **feasible**
  (it is *not* pruned), restores state, and returns a nondet value of the
  function's return type. The dispatch site is in
  `solidity_convert_expr.cpp` (`sol_name == "revert"` / `"require"`).

### Why the revert path restores state

A naive lowering of `revert` to `__ESBMC_assume(false)` *prunes* the
path — but it prunes it **with the pre-revert state writes still in the
SSA**. Under the [bounded multi-transaction
harness](#transaction-sequence-bound), transaction _k_ could write to a
state variable, revert, and have that write **leak** into transaction
_k+1_ — a false state the real EVM never reaches (a revert rolls back
*all* state changes of the transaction).

The current lowering instead makes the revert path feasible and rolls
state back to the function-entry snapshot:

- **`*this` struct fields** (scalars, structs, fixed arrays) are restored
  from a per-frame `_sol_save_this` snapshot taken at function entry.
- **Out-of-struct global stores** — mappings and state-variable dynamic
  arrays are lowered to file-local infinite-array globals *outside* the
  `*this` struct (keyed by `$address`), so `*this = save` alone does not
  reach them. `build_revert_rollback_block` snapshots each such store at
  entry and restores it alongside `*this` on revert
  (`collect_contract_global_stores`).
- Snapshots are **per call frame** (not static), so a recursive or
  reentrant call gets its own snapshot and an outer-frame revert restores
  to the outer entry state.

The caller's frame is **not** force-reverted when a callee reverts (we do
not propagate the revert up the stack). This is a sound
over-approximation for safety verification — it admits more paths, never
rules a real EVM path out.

### Worked example

`regression/esbmc-solidity/revert_rollback_mapping_rollback_pass`:

```solidity
contract C {
    mapping(uint => uint) m;

    function setAndRevert(uint k) public {
        m[k] = 1;     // mapping write...
        revert();     // ...then revert: EVM rolls m[k] back to 0
    }

    function checkClean(uint k) public view {
        assert(m[k] == 0);   // the property
    }
}
```

```sh
esbmc contract.sol --contract C --no-standard-checks --k-induction
# VERIFICATION SUCCESSFUL
```

Walk through it under the default 2-transaction harness:

1. The driver may pick `setAndRevert(k)` as transaction 1. It writes
   `m[k] = 1`, then `revert()` restores the mapping global, so `m[k]` is
   back to `0` when the transaction ends.
2. Transaction 2 calls `checkClean(k)`. Because the revert rolled the
   mapping back, `m[k] == 0` holds and the `assert` passes.

Drop the rollback (the old `assume(false)` lowering) and the `m[k] = 1`
write would survive into transaction 2, making `assert(m[k] == 0)` fail
**spuriously**. The negative control
`revert_rollback_mapping_persist_fail` performs a mapping write **without**
a `revert` and asserts the write did *not* leak; it correctly reports
`VERIFICATION FAILED` — confirming the assert is live and that the fix
rolls back only on the revert path, not on the normal committed path. The
`revert_rollback_dynarray_rollback_pass` /
`revert_rollback_scalar_revert_pass` /
`revert_rollback_reentrant_rollback_pass` tests extend this to dynamic
arrays, plain scalars, and the recursive-reentry case.

### Legacy fallback (when rollback does not apply)

In a few contexts there is no `_sol_save_this` snapshot to roll back to,
so `require` / `revert` fall back to the legacy
`__ESBMC_assume(cond)` / `__ESBMC_assume(false)` lowering (the path is
pruned, no state restore):

- constructors,
- `library` functions and free (file-level) functions,
- `event` / `error` definitions.

This is still **sound** (it never invents behaviour), but a reverted
write in one of these contexts is pruned rather than rolled back, so it
cannot leak — at the cost of dropping that path's coverage. Internal /
private helpers and ordinary public/external functions get the full
rollback form.

## Property selection

### Solidity defaults `--no-standard-checks` — most checks are opt-in

This is the single most important thing to know about Solidity property
selection. **Every `.sol` run implicitly enables `--no-standard-checks`**
(`esbmc_parseoptions.cpp`), because the C-level safety checks
(pointer / alignment / VLA / scanf) emit false positives on Yul-lowered
code. `--no-standard-checks` expands to disable
pointer / div-by-zero / bounds / narrowing / pointer-relation / VLA /
align / scanf checks — **unless** you re-enable the EVM-relevant ones
with their positive opt-in flag.

So for a meaningful Solidity scan you must explicitly turn checks **on**:

```sh
esbmc contract.sol --contract C \
  --overflow-check --div-by-zero-check --bounds-check --narrowing-check
```

| Flag | What it enables | Solidity default |
|------|-----------------|------------------|
| `--overflow-check` | Signed + unsigned overflow/underflow on all integer widths. | **OFF** (opt-in) |
| `--unsigned-overflow-check` | Unsigned-only variant. | OFF |
| `--div-by-zero-check` | Division by zero. Positive opt-in; overrides `--no-standard-checks`. | **OFF** (opt-in) |
| `--bounds-check` | Array-bounds checks. Positive opt-in; overrides `--no-standard-checks`. | **OFF** (opt-in) |
| `--narrowing-check` | Narrowing-typecast truncation overflow (e.g. `uint256 → uint8`) — this is how sub-256-bit overflow is caught despite C integer promotion. Positive opt-in. | **OFF** (opt-in) |
| `--no-bounds-check` / `--no-div-by-zero-check` / `--no-narrowing-check` / `--no-pointer-check` | Explicitly suppress the corresponding check (redundant on Solidity, where they are already off). | — |
| `--no-standard-checks` | Disable the default claim set. Implicit on every `.sol` run; listed here for completeness. | **ON** (implicit) |

Mechanism notes:
- **Sub-256-bit overflow** — `uint8`/`uint16`/`uint32` overflow is
  caught despite C integer promotion via a narrowing-cast check in
  `goto_check.cpp`. This fires **only when `--narrowing-check` is
  passed**; it is OFF by default on Solidity (part of the
  `--no-standard-checks` expansion). Note `--16` raises the minimum
  machine word to 16, so `uint8`/`int8` narrowing/overflow checks are
  unavailable under `--16`.
- **`unchecked { ... }` blocks** (Solidity 0.8+) — overflow assertions
  inside `unchecked` blocks are tagged with `#sol_unchecked` and
  skipped.

### `--reentry-check`

```sh
esbmc contract.sol --contract C --reentry-check
```

Instruments every `.call()` with assertions that capture potential
reentrant entry traces. A reported reentrancy is an **indicator**
(the trace shows a path where reentrancy is possible), not a
confirmed exploit. Inspect the counterexample to decide whether it
translates to an actual loss-of-funds scenario.

### `--reentry-balance-drain-check`

```sh
esbmc contract.sol --contract C --reentry-balance-drain-check --bound
```

Targets the DAO-style *balance drain* specifically: at every outbound
value-transfer call site (`transfer` / `send` / `call{value: V}`), it
asserts that the contract's `$balance` drops by **at most `V`** across
the call. A reentrant callback that re-enters and drains more than the
nominal transfer amount violates this assertion.

- Contracts with no outbound value-transfer call sites are skipped
  (nothing to instrument).
- Low-level `call{value:}` requires `--bound`: under the default
  unbound mode that call is special-cased to skip balance accounting,
  so the drain assertion would not see the transfer. `transfer` / `send`
  are instrumented in both modes.

Implementation: `solidity_convert_call.cpp` wraps each value-transfer
site with a pre-balance snapshot and a post-call `__ESBMC_assert`.

### `--multi-property`

Report **all** reachable property violations in a single run rather
than stopping at the first one. Useful for comprehensive scans.

```sh
esbmc contract.sol --contract C --overflow-check --multi-property
```

### `--negating-property [contract:]fn[:line]`

Rewrites `assert(cond)` to `assert(!cond)`, turning a safety check into
a reachability query (the negated assert fails iff the original
condition was reachable as true). The argument is more expressive than a
bare function name:

- `fn` — negate every assert in function `fn`.
- `contract:fn` — disambiguate same-named functions across contracts
  (Solidity, case-sensitive).
- `[contract:]fn:line` — restrict negation to asserts on that source
  line; falls back to the whole function if no assert matches the line.

```sh
esbmc contract.sol --contract C --negating-property "C:withdraw:42"
```

## Solver selection

When the user passes no explicit solver flag, the Solidity frontend
auto-selects a backend. It **first checks for three contract shapes
that force CVC5** (Bitwuzla aborts or balloons on them); if none match,
it falls back to a preference order.

**Forced-CVC5 shapes** (only when no `--<solver>` was given,
`esbmc_parseoptions.cpp`):

1. **Nested dynamic arrays** in storage — forces `--cvc5` **and**
   auto-injects `--cvc5-native-tuples`. Opt out of the native-tuples
   injection with `--no-cvc5-native-tuples` (plain CVC5 flattener
   instead); the flag is a no-op when CVC5 is not auto-selected.
2. **≥3-level nested mapping with a scalar leaf** — forces plain
   `--cvc5` (no native tuples). Bitwuzla aborts on the
   `CONST_ARRAY`-initialised infinite mapping array.
3. **k-induction over a multi-contract, value-call system** (i.e.
   `--k-induction` + `--bound`/`--reentry-check` + ≥2 contracts +
   a detected value call) — forces plain `--cvc5`; the linear address-
   equality if-chain balloons under Bitwuzla.

**Fallback preference order** (when none of the above fires):

1. Bitwuzla (fastest on 256-bit bitvectors and mapping chains)
2. CVC5
3. Boolector
4. Z3

If you need a specific backend (this also suppresses all the
forced-CVC5 logic above):

```sh
esbmc contract.sol --contract C --z3
esbmc contract.sol --contract C --cvc5
esbmc contract.sol --contract C --bitwuzla
esbmc contract.sol --contract C --boolector
```

**Empirical rules of thumb:**
- Z3 struggles with 256-bit QF_BV and often aborts with
  `datatype is not well-founded` on recursive Solidity structs. Avoid
  as a default for production scans.
- Bitwuzla is the fastest default but warns (and can abort) on
  equality over constant arrays. If a mapping-heavy contract aborts
  with `ERROR: SMT solver failed`, retry with `--cvc5`.
- CVC5 is the most consistent on the 1inch-scale benchmarks but
  slower.

### `--16` — reduce word size

```sh
esbmc contract.sol --contract C --16
```

Sets the machine word to 16 bits. Substantially faster on
mapping-heavy contracts (where keys would otherwise force 256-bit
expensive reasoning), at the cost of precision loss for addresses and
large uint arithmetic. **Not sound in general** — use only for
scaling triage, never for final verdicts on over/underflow properties.

Note: the minimum supported machine word is 16, so overflow/underflow
checks for `uint8` and `int8` are not available under `--16`.

## TOD (Transaction Order Dependence) detection

Detect pairs of functions that, when executed in different orders
from the same initial state, produce different final states — the
classic TOD vulnerability pattern.

```sh
# Auto-discover state-race pairs on a contract
esbmc contract.sol --contract C --tod-race-check

# Target a specific pair
esbmc contract.sol --contract C --tod-race-check=f1,f2

# Same two modes for balance-TOD (requires --bound)
esbmc contract.sol --contract C --tod-balance-check --bound
esbmc contract.sol --contract C --tod-balance-check=f1,f2 --bound
```

| Option | Modes | Purpose |
|--------|-------|---------|
| `--tod-race-check[=auto\|f1,f2]` | TransRacer-style storage-race TOD. Any public state variable in the pair's shared footprint is asserted equal across the two orderings. |
| `--tod-balance-check[=auto\|f1,f2]` | ETH-balance TOD. `address(this).balance` compared across the two orderings. Requires `--bound` + the EOA balance model. |
| `--dump-harness` | Print the generated Solidity harness to stdout and exit without verifying. Useful for inspecting / debugging the harness. |
| `--tod-jobs N` | Parallel ESBMC subprocesses in `auto` mode. Defaults to `min(hardware_concurrency, pair_count)`. Use `1` for sequential. |

The generator lives in
`src/solidity-frontend/solidity_tod_{analysis,harness}.{cpp,h}`.

Full pipeline (auto mode):
1. R/W footprint analysis per public function + call-graph closure.
2. Enumerate candidate pairs where footprints intersect.
3. Emit a multi-harness `.sol` with two **renamed** copies of the
   target contract (ESBMC's singleton aliasing requires separate
   names per copy); one `TOD_<a>_<b>` harness contract per pair.
4. Run ESBMC on each `TOD_<a>_<b>` contract in parallel.
5. Summarise: `N pair(s) — X clean, Y TOD found, Z error`, exit non-zero
   if any pair fails.

See [`docs/claude/solidity/tod.md`](../../docs/claude/solidity/tod.md)
for the algorithm details and the list of known limitations.

## Structural coverage

ESBMC supports all four coverage criteria on Solidity:

```sh
# Branch coverage (use --focus-function to narrow the instrumentation surface)
esbmc contract.sol --contract C --focus-function myFunc \
  --branch-coverage-claims --unwind 10 --no-unwinding-assertions

# Condition coverage (all user functions)
esbmc contract.sol --contract C \
  --condition-coverage-claims --unwind 10 --no-unwinding-assertions

# Assertion coverage
esbmc contract.sol --contract C --focus-function myFunc \
  --assertion-coverage-claims --unwind 10 --no-unwinding-assertions

# Branch + function entry
esbmc contract.sol --contract C \
  --branch-function-coverage-claims --unwind 10 --no-unwinding-assertions

# JSON report for CI or HTML rendering
esbmc contract.sol --contract C --branch-coverage --cov-report-json
python3 scripts/cov-report.py cov-report.json -o cov-report-html
```

### Additional coverage controls

| Flag | Purpose |
|------|---------|
| `--coverage-whole-unit` | With `--contract C`, keep C as the harness entry but count branch coverage over the **whole compilation unit** instead of scoping the denominator/numerator to C's own lexically-declared decisions (opt-out of per-contract "semantics A"). |
| `--coverage-covered-set <path>` | Cross-run persisted covered-set for `--branch-coverage`. Read at start (already-witnessed edges are not re-instrumented, cutting SMT cost) and merge-written at end. The denominator stays the full static universe, so skipping never inflates coverage. |
| `--coverage-exclude-contract <name>` | Exclude a contract's own decisions from branch coverage (repeatable). Those decisions count in **neither** denominator **nor** numerator. Pair with `--coverage-whole-unit` to drop dependency code (e.g. OpenZeppelin); a no-op in default per-contract mode. |
| `--no-cov-asserts` | Do not count the guard in assertions. |
| `--cov-assume-asserts` | Convert assertions to assumptions in coverage mode to preserve path constraints. |
| `--k-path-coverage[=N]` | k-path witness coverage (PathCrawler-style; Williams et al., EDCC 2005). `N` = prefix depth (1..30); if omitted, tied to `--unwind`, falling back to 4 when `--unwind` is unset. |
| `--k-path-coverage-claims` | Enable `--k-path-coverage` with default N and show all reached claims. |
| `--k-path-witness-depth D` | Cap on post-simplification witness expression depth; deeper witnesses are dropped (default 8). |
| `--k-path-max-goals M` | Per-function goal cap for `--k-path-coverage` (default 10000). |
| `--condition-coverage-rm` / `--condition-coverage-claims-rm` | `--condition-coverage[-claims]` variant that disables `--no-remove-unreachable`. |

These coverage families (like TOD) **force the unbounded transaction
harness** — see [Transaction-sequence bound](#transaction-sequence-bound).

Solidity-specific handling:
- The `_ESBMC_Main_*` multi-transaction loop is **automatically
  neutralised** in coverage mode (back-edges converted to SKIPs), so
  each user function executes exactly once.
- `__ESBMC_HIDE`-labelled auxiliary functions (~20 per contract:
  constructors, dispatchers, mapping helpers, ...) are excluded from
  the instrumentation surface.
- Coverage output is pretty-printed back to Solidity syntax: `(signed
  int)y < 50` displays as `y < 50`, `msg_sender` as `msg.sender`,
  `this->owner` as `owner`, etc.
- `require(cond)` is not counted as a branch (modelled as
  `__ESBMC_assume`).
- Modifier-renamed functions (e.g. `deposit_onlyPositive`) are
  prefix-matched, so `--focus-function deposit` targets the modified
  form.

## Externally exposed intrinsics

The Solidity frontend recognises several user-visible identifiers
that the verifier intercepts with special semantics. These are
intended to be used inside **harness contracts** or **test scaffolding**
that the author writes alongside the contract under verification.

| Identifier | Signature (conceptual) | What ESBMC does with it |
|---|---|---|
| `__ESOL_deep_copy(C src)` | `C → C` | Replaces with `_ESBMC_clone_<C>(src)` — allocates a fresh contract instance, copies every field (including mapping `.addr` retargeting, fixed-array slab reallocation, multi-dim arrays), mints a new `$address`. See [`docs/claude/solidity/operational-models.md`](../../docs/claude/solidity/operational-models.md) for per-field semantics. |
| `__ESOL_nondet_state_forward(C c)` | `C → void` | Drives `*c` through a nondet dispatch loop over `C`'s public/external methods. After the call, `*c` represents some reachable (not just initial) state. Internal / private functions are NOT invoked. |
| `__ESBMC_assume(expr)` | `bool → void` | Adds `expr` as a path constraint. Unreachable paths are pruned — behaves like Solidity's `require`, but without the revert semantics. Also a VSA hint for `p != 0` pointer guards. |
| `__ESBMC_assert(expr, msg)` | `bool, string → void` | User-level assertion. |
| `__ESBMC_reverted()` | `() → bool` | Returns `true` iff the most-recent external call reverted on the current path. The unit **opts in** by declaring `function __ESBMC_reverted() internal returns (bool) {}`; opting in also makes `try/catch` arm selection **revert-correlated** (the catch arm is entered iff the call actually reverted). Read it *immediately* after the call, before any other external call (the next external entry clears the flag), and run with `--bound`. Use for differential / mutation revert-rule tests. Constructor / library / `transfer` / `send` reverts are not captured — see [`docs/claude/solidity/revert-observation.md`](../../docs/claude/solidity/revert-observation.md). |
| `__VERIFIER_assume` / `__VERIFIER_assert` | same as `__ESBMC_*` | Accepted for compatibility with SV-COMP style stubs; identically handled. |
| `__ESBMC_nondet_*()` | `() → T` (T = declared return type) | Returns a fresh nondet value of the call's declared return type. Any name beginning with `__ESBMC_nondet_` is recognised — the suffix is for documentation only. Use when an instrumenter needs to inject a fresh nondet at a specific program point without changing function signatures (e.g. self-composition oracles for miner-timestamp / hyperproperty checks where neither a parameter nor a state variable is a viable injection site — parameters break internal callers, state variables start at the post-constructor default in `--contract` mode rather than being havoc'd). |

Users are expected to declare solc-compile-compatible stubs in the
source; ESBMC replaces the stub bodies at verification time. Example
stubs the TOD harness generator uses:

```solidity
function __ESOL_nondet_state_forward(C c) { assembly {} }
function __ESOL_deep_copy(C src) pure returns (C) { return src; }
function __ESBMC_nondet_uint() internal pure returns (uint256) {}
function __ESBMC_nondet_bool() internal pure returns (bool) {}
function __ESBMC_nondet_address() internal pure returns (address) {}
```

## Approximations you should know about

ESBMC Solidity makes deliberate soundness / completeness trade-offs.
Every in-source approximation carries an `[APPROX: OVER]` or
`[APPROX: UNDER]` marker; grep `\[APPROX:` to find them all.

Highlights (non-exhaustive, see
[`docs/claude/solidity/approximation-ledger.md`](../../docs/claude/solidity/approximation-ledger.md)
for all 21 entries):

| Feature | Direction | Consequence |
|---|---|---|
| `keccak256` / `sha256` / `ripemd160` / `ecrecover` | OVER+UNDER | Deterministic bijective (`~x`, `~(x+1)`, …). Same input → same output ✓, distinct inputs → distinct outputs ✓. Concrete hash values NOT computed; preimage/collision/signature-forgery properties NOT refutable. `ecrecover` ignores `(v,r,s)`. |
| `abi.encode*` | OVER+UNDER | Identity on the first argument; remaining arguments evaluated for side effects only. Packed layout properties unprovable; selector-dispatch checks with the same first arg look equal. |
| `abi.decode` | OVER | Nondet uint256. Round-trip `decode(encode(x)) == x` NOT provable. |
| `msg.*` / `tx.*` / `block.*` | OVER | All fields nondet per access. `block.number` not monotonic across reads. |
| Unbound external calls (`addr.call`, etc.) | OVER | Returns nondet; side-effect = nondet re-entry into current contract. Target address ignored. Enables reentrancy detection; cross-contract effects invisible. |
| Bound-mode `delegatecall` | UNDER (partial fix) | Fast path shadow-inlines target body into caller context, but only for literal `abi.encode*` payloads with matching state-var layouts. Proxy / UUPS / Diamond storage-slot patterns fall back to generic dispatcher. |
| `revert` / `require` / failed `transfer()` | UNDER (pruned contexts only) | Public / external / internal / private bodies and modifiers **roll state back** to the function-entry snapshot on revert (see [`assert` / `require` / `revert` semantics](#assert--require--revert-semantics)); constructors, libraries, and free functions fall back to `__ESBMC_assume(false)` (path pruned, no rollback). `try/catch` arm choice is **nondet by default** (the catch arm does not imply the call really reverted); declaring `__ESBMC_reverted` makes it **revert-correlated**. |
| `selfdestruct` | UNDER | Modelled as `exit(0)` — post-selfdestruct behaviour (address reuse, CREATE2 re-deploy) not explored. |
| Inline assembly / Yul | OVER | All externally-referenced variables havoc'd to nondet. Assembly-enforced invariants unprovable. |
| `IndexRangeAccess` slices (`b[s:e]`) | OVER | Fresh nondet value; no parent-array constraint, no `s <= e <= length`. |
| `type(I).interfaceId` / `creationCode` / `runtimeCode` | OVER | Nondet bytes4 / bytes. |
| `address(new C())` uniqueness | UNDER | 16-slot if-chain caps pairwise distinctness at the 16th allocation; the 17th `new C()` on a path is unconstrained vs prior addresses. Loose default; opt into precise modelling via `--solidity-precise` (see below). |

### Address uniqueness modelling — loose default vs `--solidity-precise`

`_ESBMC_get_unique_address` (the helper called from every contract
constructor to hand out a fresh address) ships in two variants:

- **Default — 16-slot unrolled if-chain.** Loop-free, so `--unwind`
  does not truncate the uniqueness constraints. Sound for the first
  16 contract instantiations on any path; the 17th is unconstrained
  (silent under-approximation).
- **`--solidity-precise` — `for`-loop linear scan over `sol_max_cnt`.**
  No slot cap. The loop is bounded by the runtime contract-allocation
  count, so the user must pass `--unwind N` with N ≥ the number of
  contract instantiations on any path. Without `--no-unwinding-assertions`,
  a too-low `--unwind` produces a visible "unwinding assertion loop
  &lt;id&gt;" failure that surfaces the bound explicitly. With
  `--no-unwinding-assertions`, the loop tail is silently truncated
  — same blind spot as `_ESBMC_get_addr_array_idx` and other linear
  scans in the address library.

**Why the loose form is the default.** The `--unwind` coupling of
the precise variant is awkward in practice: many tests legitimately
use `--unwind 1` or `--unwind 2` to control dispatcher exploration
depth, and bumping `--unwind` to satisfy address uniqueness changes
the entire test's path-exploration semantics. The 16-slot if-chain
decouples uniqueness from `--unwind` at the cost of a hard cap that
no regression test in the suite has ever exceeded.

A quantifier-based form (`__ESBMC_forall`) was investigated and
rejected — cvc5 returns `unknown (INCOMPLETE)` on every standard
quantifier strategy and only `--sygus-inst` succeeds (which conflicts
with incremental mode and is intractably slow on k-induction);
bitwuzla's BV-quantifier engine is also slow on related patterns.

**When to opt in.** Pass `--solidity-precise` if your contract
allocates more than 16 instances on any path AND your property
depends on address-distinctness. Pair it with `--unwind N` where N
covers your maximum on-path allocation count. Drop
`--no-unwinding-assertions` while debugging so insufficient unwind
shows up as a real warning instead of silent truncation.

**Future under-approximations bind to the same flag.** As we replace
more loose modellings with precise (sound) ones, they will be
controlled by `--solidity-precise` so users get one knob rather than
many. The first such addition is the transaction-sequence bound:
`--solidity-precise` also restores the unbounded
`while (nondet_bool())` harness loop (the default is bounded to two
transactions) — see
[Transaction-sequence bound](#transaction-sequence-bound).

**Consequences for review:**
- `VERIFICATION SUCCESSFUL` is a real safety proof *within* the
  approximation ledger **and within the default transaction bound**
  (two transactions — see
  [Transaction-sequence bound](#transaction-sequence-bound)). If the
  property depends on an approximation marked "False negatives", or on a
  transaction sequence longer than the bound, the proof is NOT a full
  guarantee — re-verify with `--solidity-precise` (unbounded), a deeper
  `--solidity-max-tx N`, a tighter model, or manual reasoning.
- `VERIFICATION FAILED` may be a spurious counterexample if it
  depends on an approximation marked "False positives" (e.g.
  row 7 — nondet `msg.value` at harness entry; row 5 — non-monotonic
  `block.number`; row 16 — cross-iteration state visibility across
  reverts). Check the counterexample trace against the ledger before
  filing a bug.

## Building

### Minimal Solidity build

```sh
cd build
cmake .. -DENABLE_SOLIDITY_FRONTEND=ON -DENABLE_REGRESSION=ON
cmake --build . -j$(nproc)
```

Both `ENABLE_SOLIDITY_FRONTEND` and `ENABLE_REGRESSION` must be ON.
The default `./scripts/build.sh` sets `ENABLE_REGRESSION=OFF`, so
regression tests won't appear in `ctest -N` unless explicitly
enabled.

### Bitwuzla (recommended default, auto-selected)

Required system packages on Ubuntu/WSL:

```sh
sudo apt install -y libgmp-dev libmpfr-dev
pip install --user --break-system-packages meson ninja
cmake .. -DENABLE_BITWUZLA=ON -DDOWNLOAD_DEPENDENCIES=ON
```

`libgmp-dev` and `libmpfr-dev` are required by Bitwuzla's upstream
build; `meson`/`ninja` drive the compile.

### CVC5 (needed for some 256-bit tests)

```sh
cmake .. -DENABLE_CVC5=ON -DDOWNLOAD_DEPENDENCIES=ON
```

### Static release build (shippable binary)

```sh
./scripts/build.sh -S ON -b Release build install
```

`-S ON` is the Linux default in `build.sh`; produces a statically
linked binary under `./release/bin/esbmc` that can be shipped to
machines without matching shared-library versions. Requires the build
dependencies above (`libgmp-dev`, `libmpfr-dev`, `meson`, `ninja`, plus
the ones `build.sh` installs via apt).

### Testing

```sh
cd build
# All Solidity tests
ctest -j$(nproc) -L esbmc-solidity --output-on-failure

# A specific test
ctest -R "regression/esbmc-solidity/address_1"

# Skip slow tests during development
ctest --timeout 60 -L esbmc-solidity
```

See [`docs/claude/solidity/testing.md`](../../docs/claude/solidity/testing.md)
for the test baseline, the list of slow THOROUGH tests, and the
adversarial / stress-test suites.

## Developer notes

- **Solidity → C IR**: Each contract is lowered to a C struct.
  Library contracts register as code-only symbols. Inheritance is
  handled via AST merging (C3 linearisation follows solc).
- **Blockchain state** is encoded as per-contract `$address`,
  `$balance`, `$bind_cname` struct fields. In the real EVM, balance
  is bound to an address; ESBMC binds balance to each contract
  instance.
- **Operational models** (`src/c2goto/library/solidity/`) are
  pre-compiled into a separate `sol64.goto` via the c2goto pipeline
  and embedded in the ESBMC binary. When adding a new `.c` model file,
  list its exported function names in `solidity_c_models` in
  `cprover_library.cpp`.
- **Pretty-printing** of intermediate C expressions back to Solidity
  syntax is applied at display time only (counterexamples, coverage
  output). Claim matching and internal reasoning use the original C
  strings.
- **Hook for debugging counterexamples**: dump the goto program with
  `--goto-functions-only` to see what the frontend actually emits;
  compare against an equivalent C program verified through ESBMC's C
  frontend to isolate frontend vs C-model vs backend bugs. See
  [`docs/claude/solidity/architecture.md`](../../docs/claude/solidity/architecture.md).
- **Contribute a regression test** whenever you fix a bug: every
  Solidity PR should add at least one PASS and one FAIL test case
  under `regression/esbmc-solidity/`.
