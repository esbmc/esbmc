// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint r = 0;
        assembly {
            for { let outer := 0 } lt(outer, 3) { outer := add(outer, 1) } {
                for { let inner := 0 } lt(inner, 5) { inner := add(inner, 1) } {
                    if eq(inner, 2) { break }   // exits the INNER loop only
                    r := add(r, 1)
                }
                r := add(r, 100)                // confirms the outer continues
            }
        }
        // Each outer iter: inner contributes 2 (i=0,1) before break, then +100.
        // 3 outer iters: 3*(2+100) == 306.
        assert(r == 306);
    }
}
