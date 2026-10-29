// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // PASS dual to yul_div_by_zero_fail. Symbolic y is constrained by
    // `require(y != 0)` BEFORE the assembly, so the path condition
    // reaching `div(100, y)` excludes y == 0. ESBMC's div-by-zero check
    // is satisfied (no reachable zero divisor), and verification passes.
    function check(uint256 y) public pure {
        require(y != 0);
        uint256 r;
        assembly {
            r := div(100, y)
        }
        // r == 100 / y >= 0; nothing further to assert.
    }
}
