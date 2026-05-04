// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test that mulmod does NOT produce the wrapping result.
// True: (2^256-1) * 2 % 3 = (2^257 - 2) % 3 = 0
// Wrapping: (MAX*2) mod 2^256 = MAX-1, (MAX-1) % 3 = 2.
// Asserting the wrong (wrapping) result must FAIL.
contract MulmodOverflowFail {
    function test() public pure {
        uint256 a = 0;
        a -= 1; // a = MAX
        uint256 result = mulmod(a, 2, 3);
        assert(result == 2); // FAIL: wrapping result, not correct
    }
}
