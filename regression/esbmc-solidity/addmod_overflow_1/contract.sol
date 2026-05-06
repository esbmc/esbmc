// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test addmod with arbitrary precision (no wrap at 2^256).
// MAX + 2 overflows uint256, but addmod must compute the true sum.
// True: (2^256 - 1 + 2) % 3 = (2^256 + 1) % 3 = (1 + 1) % 3 = 2
// Buggy (wrapping): (MAX + 2) wraps to 1, 1 % 3 = 1 (wrong)
contract AddmodOverflow {
    function test() public pure {
        uint256 a = 0;
        a -= 1; // a = MAX = 2^256 - 1
        uint256 result = addmod(a, 2, 3);
        assert(result == 2);
    }
}
