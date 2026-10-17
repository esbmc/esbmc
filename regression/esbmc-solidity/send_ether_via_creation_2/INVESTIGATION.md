# send_ether_via_creation_2 KNOWNBUG Investigation

## Summary

The test expects `VERIFICATION FAILED` but gets `VERIFICATION SUCCESSFUL` under `--bound` mode.
The root cause is **not** a Solidity frontend conversion bug — the GOTO and SSA are correct.
The issue is in the **`_ESBMC_get_unique_address` C model function** whose loop/array
interactions create formulas the solver cannot handle with unconstrained nondet parameters.

## Test Contract

```solidity
contract D {
    constructor(uint a) payable {
        payable(msg.sender).transfer(1 ether);  // transfers back to C
        x = a;
    }
}
contract C {
    function createAndEndowD(uint arg, uint amount) public payable {
        uint balancebefore = address(this).balance;
        D newD = new D{value: amount}(arg);
        uint balanceafter = address(this).balance;
        assert(balanceafter == balancebefore - amount);
        // Should FAIL: balanceafter = balancebefore - amount + 1 ether
    }
}
```

## GOTO Correctness

The frontend generates correct GOTO code:

1. `model_transaction` front_block: `msg_sender = C.$address`, `msg_value = amount`, `C.$balance -= amount`
2. D constructor: `this->$balance = msg_value` (payable init), then `transfer(this, msg_sender, 1 ether)`
3. Transfer function: matches `_addr == _ESBMC_Object_C.$address`, executes `_ESBMC_Object_C.$balance += 1 ether`
4. After return: `balanceafter = this->$balance` (reads `_ESBMC_Object_C.$balance`)

SSA trace confirms the phi-node correctly includes the `+1` update (version #3).

## Experimental Results — Solidity

| Scenario                          | Result                | Correct? |
|-----------------------------------|-----------------------|----------|
| Without `--bound`                 | VERIFICATION FAILED   | Misleading -- `address(this).balance` is modeled as independent nondet values, so any equality assertion trivially fails |
| `--bound`, nondet `amount`        | VERIFICATION SUCCESSFUL | **Wrong** -- should find violation |
| `--bound`, hardcoded `amount=100` | VERIFICATION FAILED   | Correct |
| `--bound`, `require(amount>=1)`   | VERIFICATION FAILED   | Correct |
| `--bound`, `require(amount>=2)`   | VERIFICATION FAILED   | Correct |

## C Equivalent PoC — All Pass

Created progressively more faithful C equivalents:

| C PoC Variant | Features | Result | Correct? |
|---------------|----------|--------|----------|
| Direct global access | Simple struct, constant addresses | VERIFICATION FAILED | Yes |
| Pointer indirection | `this_ptr->balance` (caller) vs `C_instance.balance` (transfer) | VERIFICATION FAILED | Yes |
| `_ExtInt(256)` | Full 256-bit types | VERIFICATION FAILED | Yes |
| Nondet addresses | `nondet_addr()` + uniqueness assumptions | VERIFICATION FAILED | Yes |
| Full harness model | msg_sender/msg_value context, C constructor calling D(4), nondet harness loop | VERIFICATION FAILED | Yes |

**All C equivalents produce correct VERIFICATION FAILED** with both Bitwuzla and Z3.
This means the ESBMC symex engine and solvers correctly handle:
- Pointer aliasing (this_ptr → global struct member)
- Global modification inside nested function calls
- 256-bit nondet with ASSUME-false path pruning
- Harness loop with nondet dispatch

## Root Cause: `_ESBMC_get_unique_address` Model

The only difference between the working C PoCs and the failing Solidity GOTO is the
**address model**: the Solidity version uses `_ESBMC_get_unique_address()` (defined in
`src/c2goto/library/solidity/solidity_address.c`), which:

1. Has an internal loop searching `sol_addr_array` for existing addresses
2. Generates nondet addresses with complex uniqueness constraints via array lookups
3. Creates deeply nested phi-nodes in the SSA from loop unwinding

The resulting formula is too complex for the solver to reason about address equality
(`_addr == _ESBMC_Object_C.$address`) when combined with nested function calls,
ASSUME-false pruning, and unconstrained 256-bit parameters.

When the address comparison is simplified (constant or simple nondet + assume),
the solver immediately finds the violation.

## Not a Frontend Conversion Bug

- GOTO generation is correct (verified via `--goto-functions-too`)
- SSA phi-nodes are correct (verified via `--ssa-trace`)
- The transfer function correctly matches C's address and updates `_ESBMC_Object_C.$balance`
- Both Z3 and Bitwuzla agree on the (incorrect) VERIFICATION SUCCESSFUL result
- Slicing is not the issue (`--no-slice` gives the same result)
- All equivalent C programs work correctly

## Potential Fix Direction

Simplify `_ESBMC_get_unique_address` to avoid loop-based array lookups.
For example, use a simple counter-based scheme:
```c
static addr_t next_addr = 1;
addr_t _ESBMC_get_unique_address(...) {
    return next_addr++;
}
```
This would produce concrete, distinct addresses without complex solver formulas.

## Date

2026-04-12
