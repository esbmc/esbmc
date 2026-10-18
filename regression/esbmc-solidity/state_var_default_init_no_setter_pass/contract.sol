// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Pins the Solidity-frontend design choice (see commit 135c223362 / MODELING-2):
// state variables KEEP their constructor-assigned initial values (0 / default for
// unset slots) through the dispatcher loop. They are NOT havoc'd at function entry.
//
// For this contract there is no setter, so `a` and `b` remain 0 across every
// dispatched call to `f()`. `require(a > b)` is `require(0 > 0)` → unsat, the
// trace dies, and the downstream assert is unreachable. SUCCESSFUL is the sound
// verdict per Solidity post-deployment semantics.
//
// If a future change introduces implicit entry-havoc of state variables, this
// test will flip to FAILED and force the contributor to confirm intent. Users
// who need adversarial-state queries (self-composition, SWC-116
// timestamp-dependence, miner-manipulability) should declare the
// `__ESBMC_nondet_*` intrinsic inside the contract and assign into the state
// variable explicitly — see the dual partner
// `state_var_havoc_via_intrinsic_fail` for the FAILED counterpart.

contract Bug {
    uint256 public a;
    uint256 public b;

    function f() public view {
        require(a > b);
        assert(a == b);
    }
}
