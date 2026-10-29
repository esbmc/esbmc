// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // PASS dual to yul_mod_by_zero_fail. require(y != 0) prunes the
    // y == 0 path before the assembly, so ESBMC's div-by-zero check
    // on the Yul `mod` is satisfied.
    function check(uint256 y) public pure {
        require(y != 0);
        uint256 r;
        assembly {
            r := mod(100, y)
        }
    }
}
