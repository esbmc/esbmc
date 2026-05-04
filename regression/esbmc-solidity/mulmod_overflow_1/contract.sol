// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test mulmod with arbitrary precision (no wrap at 2^256).
// MAX * 2 overflows uint256, but mulmod must compute the true product.
// True: (2^256-1) * 2 % 3 = (2^257 - 2) % 3 = 0
//   because 2^257 mod 3 = 2 (257 odd), so (2 - 2) % 3 = 0
// Buggy (wrapping): (MAX*2) mod 2^256 = MAX-1, (MAX-1) % 3 = 2 (wrong)
contract MulmodOverflow {
    function test() public pure {
        uint256 a = 0;
        a -= 1; // a = MAX = 2^256 - 1
        uint256 result = mulmod(a, 2, 3);
        assert(result == 0);
    }
}
