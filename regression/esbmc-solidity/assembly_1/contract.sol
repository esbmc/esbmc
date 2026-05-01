// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Test: inline assembly havoc over-approximation (VERIFICATION SUCCESSFUL)
// Assembly modifies x via nondet; asserting a tautology should pass.
contract AssemblyPass {
    function test() public pure {
        uint x = 5;
        assembly {
            x := add(x, 1)
        }
        // x is havoc'd to nondet, but x == x is always true
        assert(x == x);
    }
}
