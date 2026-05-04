// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Test: inline assembly havoc over-approximation (VERIFICATION FAILED)
// Assembly havocs x to nondet, so asserting a concrete value should fail.
contract AssemblyFail {
    function test() public pure {
        uint x = 5;
        assembly {
            x := add(x, 1)
        }
        // x is havoc'd to nondet, so x == 6 cannot be guaranteed
        assert(x == 6);
    }
}
