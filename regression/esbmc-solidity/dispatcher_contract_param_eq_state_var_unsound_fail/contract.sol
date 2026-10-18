// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// SOUNDNESS PIN: ESBMC's Solidity dispatcher allocates contract-typed
// function parameters as fresh heap instances (`new struct IERC20`).
// State-var contract pointers are allocated separately. The frontend
// lowers `==` on contract pointers to *pointer* equality, so the two
// `dynamic_*_value` symbols are statically distinct, the guard is
// statically false, the if-body is vacuously unreachable in symex,
// and `assert(false)` inside the body is reported safe.
//
// Real EVM execution: any external caller invoking `c.f(c.T())` makes
// `x == T` true at runtime and triggers the assertion. ESBMC's
// reporting `VERIFICATION SUCCESSFUL` is therefore a soundness
// violation (false negative on bug finding).
//
// Sister test `dispatcher_address_param_eq_state_var_baseline_fail`
// is the address-typed control: identical shape but `address` (uint160)
// instead of IERC20; ESBMC correctly reports VERIFICATION FAILED there.
// The differential isolates the bug to contract-typed parameter
// modelling, not k-induction, not the solver, not coverage mode.
//
// Flip target: once the fix lands, regex `^VERIFICATION FAILED$` matches
// and this test promotes from KNOWNBUG to CORE.

interface IERC20 {
    function balanceOf(address) external view returns (uint256);
}

contract C {
    IERC20 public T;

    constructor(IERC20 _t) {
        T = _t;
    }

    function f(IERC20 x) public {
        if (x == T) {
            assert(false); // reachable in real EVM via c.f(c.T())
        }
    }
}
