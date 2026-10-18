// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Coverage: state_forward must be able to reach a state where x != 0.
// Asserting "x is always 0 after state_forward" must therefore FAIL —
// the failure trace is a counterexample where state_forward issued
// `set(nondet_v)` with nondet_v != 0.  If state_forward never invokes
// `set` (e.g. visibility-filter bug), this would falsely succeed.
function __ESOL_nondet_state_forward(C c) {}

contract C {
    uint256 public x;
    function set(uint256 v) public { x = v; }
}

contract H {
    function check() public {
        C c = new C();
        __ESOL_nondet_state_forward(c);
        // Wrong: state_forward can drive x to any nondet value via `set`.
        assert(c.x() == 0);
    }
}
