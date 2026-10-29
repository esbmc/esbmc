// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Visibility: state_forward only invokes public/external methods.
// Here `priv` is internal and would set x = 42 if called.  The only
// public entry point is `touch`, which flips a flag but leaves x at 0.
// After state_forward, x must still be 0 because `priv` is
// unreachable from the dispatch loop.
function __ESOL_nondet_state_forward(C c) {}

contract C {
    uint256 public x;
    bool public touched;
    function priv() internal { x = 42; }
    function touch() public { touched = true; }
}

contract H {
    function check() public {
        C c = new C();
        __ESOL_nondet_state_forward(c);
        assert(c.x() == 0);
    }
}
