// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    function check() public pure {
        uint n = 5;
        uint r;
        assembly {
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                r := add(r, i)
            }
        }
        assert(r == 10);
    }
}
