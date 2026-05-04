// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test that addmod does NOT produce the wrapping result.
// Wrapping: (MAX + 2) wraps to 1, 1 % 3 = 1.
// Correct: (2^256 + 1) % 3 = 2.
// Asserting the wrong (wrapping) result must FAIL.
contract AddmodOverflowFail {
    function test() public pure {
        uint256 a = 0;
        a -= 1; // a = MAX
        uint256 result = addmod(a, 2, 3);
        assert(result == 1); // FAIL: wrapping result, not correct
    }
}
