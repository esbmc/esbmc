// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint r = 0;
        assembly {
            for { let i := 0 } lt(i, 10) { i := add(i, 1) } {
                if eq(i, 5) { break }
                r := add(r, 1)
            }
        }
        // break exits at i==5 BEFORE the body increments r, so r counts
        // iterations 0..4 only (5 increments).
        assert(r == 5);
    }
}
