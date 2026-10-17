// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Invariant preservation under __ESOL_nondet_state_forward.  The only
// mutator (`step`) is monotonic: it can only set x = max(x, v).  After
// any sequence of nondet calls, x must remain >= the initial value.
// If state_forward leaks side-effects beyond the contract (e.g. wrong
// implicit-this binding), or if it bypasses function bodies, the
// invariant could fail.
function __ESOL_nondet_state_forward(C c) {}

contract C {
    uint256 public x;
    function step(uint256 v) public {
        if (v > x) x = v;
    }
}

contract H {
    function check() public {
        C c = new C();
        uint256 before = c.x(); // 0
        __ESOL_nondet_state_forward(c);
        assert(c.x() >= before);
    }
}
