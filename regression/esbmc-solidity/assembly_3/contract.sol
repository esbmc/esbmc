// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: inline assembly inside a library function (havoc over-approximation)
library MathLib {
    function addMod(uint a, uint b, uint m) internal pure returns (uint) {
        uint result;
        assembly {
            result := addmod(a, b, m)
        }
        return result;
    }
}

contract UseMathLib {
    function test(uint x) public pure {
        uint r = MathLib.addMod(x, 1, 10);
        // r is havoc'd to nondet, so only tautologies hold
        assert(r == r);
    }
}
