// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Same shape as yul_div_by_zero_fail: ESBMC's goto-check inserts
    // `assert(y != 0)` on the Yul `mod`'s divisor; symbolic y == 0 is
    // reachable; "division by zero" violation. (The check covers both
    // div and mod — same property name.)
    function check(uint256 y) public pure {
        uint256 r;
        assembly {
            r := mod(100, y)
        }
    }
}
