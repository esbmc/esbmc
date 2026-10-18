// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint r = 0;
        assembly {
            for { let i := 0 } lt(i, 6) { i := add(i, 1) } {
                if iszero(mod(i, 2)) { continue }   // skip even i
                r := add(r, i)
            }
        }
        // i runs 0..5; odd-only sum: 1 + 3 + 5 = 9.
        assert(r == 9);
    }
}
