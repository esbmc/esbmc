// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Yul `sub` underflows silently to 2^256 - n. With --overflow-check,
    // ESBMC's automatic detector fires on `sub(0, 1)` and reports
    // "arithmetic overflow on sub" (the property name covers both over-
    // and underflow direction). No manual assert needed.
    function check() public pure {
        uint256 a = 0;
        uint256 b = 1;
        uint256 r;
        assembly {
            r := sub(a, b)
        }
    }
}
