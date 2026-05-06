// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// mulmod(MAX, MAX, MAX) = 0
// Tests that the 512-bit intermediate product (2^256-1)^2 is handled
// correctly by the constant evaluator during simplification.
contract MulmodOverflowMax {
    function test() public pure {
        uint256 a = 0;
        a -= 1; // a = MAX = 2^256 - 1
        uint256 result = mulmod(a, a, a);
        assert(result == 0);
    }
}
