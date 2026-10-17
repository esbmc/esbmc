// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Sound baseline / control for the dispatcher contract-param soundness
// pin (`dispatcher_contract_param_eq_state_var_unsound_knownbug`).
//
// Identical shape, but the parameter and state variable are `address`
// (uint160) instead of a contract / interface type. ESBMC correctly
// reports `VERIFICATION FAILED` here because address comparisons are
// nondet uint160 == nondet uint160 — the SAT solver finds the witness
// where the two values are equal, so the if-body is reachable.
//
// This locks in that the unsoundness is specific to contract-typed
// parameter modelling and not a generic if/state-var/dispatcher gap.

contract C {
    address public T;

    constructor(address _t) {
        T = _t;
    }

    function f(address x) public {
        if (x == T) {
            assert(false); // reachable: SAT solver picks x = T
        }
    }
}
