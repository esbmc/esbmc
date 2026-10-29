// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Yul `mul` wraps mod 2^256. 2^128 * 2^128 == 2^256 wraps to 0.
    // With --overflow-check enabled, ESBMC's overflow detector fires
    // and reports "arithmetic overflow on mul".
    function check() public pure {
        uint256 r;
        assembly {
            let half := shl(128, 1)   // 2^128
            r := mul(half, half)       // overflows
        }
    }
}
