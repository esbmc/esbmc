# ESBMC Bug: Transfer to dynamic contract instance — credit mislanding + no-receive revert not modeled

**Filed:** 2026-06-24  
**Component:** `src/solidity-frontend/solidity_convert_call.cpp` — `get_transfer_definition()`  
**Severity:** Soundness (false proof — VERIFICATION SUCCESSFUL for property that is violated on real EVM)  
**Regression tests:** `transfer_dynamic_self_pass_knownbug`, `transfer_dynamic_cross_fail_knownbug`, `transfer_dynamic_self_no_receive_revert_pass_knownbug`

---

## Observable symptoms

```
contract C { constructor() payable {} function migrateTo(address to) public {
    payable(to).transfer(address(this).balance); } }

contract InvMutTest { C c;
    constructor() payable { c = new C{value: 1 ether}(); }
    function body() public {
        uint256 b0 = address(c).balance;  // = 1 ether
        c.migrateTo(address(c));           // self-transfer; C has no receive()
        assert(address(c).balance == b0);  // should be true (revert → unchanged)
    }
}
```

`--bound --k-induction --max-k-step 10 --solidity-max-tx 1`:

- **ESBMC (current, buggy):** VERIFICATION FAILED — balance is 0, not b0
- **Real EVM / correct ESBMC:** transfer reverts (no receive), path pruned, assert vacuously SUCCESSFUL

---

## Root cause (confirmed by code reading + empirical tests)

### The dispatch loop uses STATIC singleton addresses only

`get_transfer_definition()` (lines 3611–3778) generates a per-contract-type dispatch:

```cpp
for (auto str : contractNamesList) {
    get_static_contract_instance_ref(str, static_ins);        // _ESBMC_Object_<str>
    exprt mem_addr = member_exprt(static_ins, "$address", …); // static singleton address
    // if (_addr == _ESBMC_Object_<str>.$address) { debit; credit singleton; return; }
}
// EOA fallback — reached when no static singleton matched:
this.$balance -= val;              // debit (updates the object field correctly)
_ESBMC_eoa_credit(_addr, val);    // credit EOA map  ← WRONG for dynamic instances
```

### Dynamic `new C{value:…}()` assigns a DIFFERENT address from the static singleton

`_ESBMC_get_unique_address()` (solidity_address.c:92) produces a fresh nondet address
constrained to be distinct from all prior allocations. The static singleton
`_ESBMC_Object_C` gets one unique address; `new C{value:...}()` gets another.

Therefore:

```
address(c_dynamic) ≠ _ESBMC_Object_C.$address
```

The dispatch loop's equality check `_addr == _ESBMC_Object_C.$address` fails for `c_dynamic`.
Execution falls through to the EOA fallback unconditionally.

This is documented in a source comment (lines 3797–3802):

> "the multi-instance dispatch above matches only the static `_ESBMC_Object_<C>` per
>  contract type; new-allocated instances of the same type fall through to this EOA
>  branch even when the recipient is technically a tracked contract"

### Credit mislanding

EOA fallback credits `sol_eoa_balance_array[to_address]`.  
Reading `address(c).balance` where `c` is a known local variable routes through
`get_builtin_property_expr` → reads `c->$balance` (the object field).

These are **two different storage locations** — `c->$balance` and `sol_eoa_balance_array[idx]`
for the same logical account. The debit hits `c->$balance` (via `this.$balance -= val`), the
credit goes to the EOA map. Net: `c->$balance = 0`, EOA map holds the funds "in limbo".

### No-receive revert not modeled for dynamic instances

For static singletons, the dispatch loop only calls the receive/fallback callback; it does NOT
emit `__ESBMC_assume(false)` when there is no payable callback. For dynamic instances this is
moot because they fall to the EOA fallback, which also never emits the revert guard.

Real EVM: `transfer(to, v)` where `to` is a contract with no payable receive/fallback → REVERT.  
ESBMC model: "succeeds" (debit happens, credit mislanded) → no revert guard → property
after the call is NOT vacuously true.

---

## Two distinct bugs (same code path, separable fixes)

**Bug A — Credit mislanding (affects any dynamic-to-dynamic transfer):**  
`c1.migrate(address(c2))` where both are dynamic: c2's `$balance` is not updated; credit
goes to EOA map. `address(c2).balance` reads 0 credit (still sees the old `$balance`).

**Bug B — No-receive revert not modeled (affects any transfer to a contract without receive):**  
Should emit `__ESBMC_assume(false)` (path prune) when the recipient is a contract with no
payable receive/fallback. Currently the path "succeeds" with a mislanded credit.

Note: Bug B cannot be fixed purely at the EOA-fallback level without also fixing Bug A first,
because even if we add the revert guard in the EOA branch, the dispatch-loop branches
(for static singletons) also lack the no-receive revert guard.

---

## Fix direction (implemented) + Codex adversarial review

**CODEX ADVERSARIAL REVIEW (2026-06-24, second pass — read actual source code):**

The implemented patch inserts a dynamic dispatch loop between the static dispatch loop and the
EOA fallback (lines 3796–3933 of `solidity_convert_call.cpp`). Per-contract-type, it calls
`_ESBMC_get_obj(_addr, str)`, casts to `_ESBMC_Object_str*`, and on non-NULL: emits
`assume(false)` for no-receive (Bug B), debits sender, credits `__dyn_str->$balance` (Bug A).

### Q2 — Pointer aliasing (SOUND)

`void*` → `_ESBMC_Object_str*` typecast in ESBMC's byte-array memory model correctly aliases
`__dyn_str->$balance` with the original struct field. No TBAA in Solidity frontend.

### Q3 — Static/dynamic overlap (SOUND)

Static dispatch branches each end with `return true`. If `_addr` matches a static singleton,
the function returns before the dynamic dispatch loop is ever reached — no double-fire.

### Q1 — Bug B `assume(false)` placement (UNSOUND — partial coverage only)

`assume(false)` is inside `if (__dyn_str != NULL)`. If `_ESBMC_get_obj` returns NULL (e.g.,
k-induction inductive step with non-deterministic `sol_addr_array`, or instance at high index),
the path falls through to the EOA fallback WITHOUT the prune. Bug B fix is conditional on
`_ESBMC_get_obj` succeeding. Tests 1 and 3 pass because they each create only ONE dynamic
instance (at index 0) — k-induction base case's one loop iteration always finds it.

### Q4 — Static singletons missing Bug B (pre-existing incompleteness gap)

Static dispatch (lines 3611–3779) does NOT emit `assume(false)` when a static singleton has
no payable receive/fallback. The dynamic dispatch fixes this for dynamic instances only.
A transfer to a static singleton with no receive "succeeds" in ESBMC — false proof possible.
This is a pre-existing gap, not introduced by the current patch, but the patch is asymmetric.

### Q5 — Bug B under k-induction (UNCERTAIN — works for single-instance tests only)

Tests 1, 3 pass under k-induction because each creates exactly one dynamic instance (index 0).
k-induction base case's for-loop unwinds once → finds index 0 → `_ESBMC_get_obj` returns &c
→ `assume(false)` fires. For scenarios with multiple dynamic instances, the same loop-index
limitation as Bug A applies (see Q7).

### Q6 — Library mode (UNCERTAIN)

`is_library=true` skips the `assume(false)` even for no-receive contracts — semantic gap
(library delegatecall transfers should also model revert). The Bug A credit is structurally
correct for library mode. Library + no-receive is a known incompleteness.

### Q7 — Multiple instances + k-induction: correct diagnosis (SOUND)

k-induction inductive step: `sol_addr_array` and `sol_max_cnt` are non-deterministic in the
starting state. The for loop in `_ESBMC_get_addr_array_idx` is unwound k times — only the
first k slots are checked. A dynamic instance at slot k or higher is invisible. Result:
`_ESBMC_get_obj` returns NULL for c2 in the inductive step → Bug A credit not applied →
c2->$balance unchanged → inductive step succeeds spuriously.

### Q8 — Workaround vs. proper fix (WORKAROUND — introduces incompleteness)

Bug A is fixed for BMC (`--unwind 5`) but NOT for k-induction.
Bug B is fixed for BMC and for k-induction with single-instance scenarios.

The incompleteness (not unsoundness) means: ESBMC may miss real bugs in the inductive step
for multi-instance cross-transfer scenarios. For InvMut's mutation testing, incompleteness
lowers mutation score (mutants not killed → false equivalents), which is the more dangerous
failure mode compared to false counterexamples.

**Bottom line**: The patch is a correct partial fix. Bug B is properly fixed for the tested
scenarios. Bug A requires a deeper fix (loop-free lookup or per-type tracked arrays) to work
under k-induction. The test suite is appropriately split: Bug B tests use k-induction; Bug A
test uses BMC with explicit `--unwind 5`.

---

## Test cases (all three should flip from current to expected after fix)

| Test | Current | Expected after fix |
|---|---|---|
| `transfer_dynamic_self_pass_knownbug` | FAILED | SUCCESSFUL |
| `transfer_dynamic_cross_fail_knownbug` | SUCCESSFUL | FAILED |
| `transfer_dynamic_self_no_receive_revert_pass_knownbug` | FAILED | SUCCESSFUL |
