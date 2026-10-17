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
        // r == 10 would only hold if `break` did nothing and the loop ran
        // through all 10 iterations. Precise lowering must violate this.
        assert(r == 10);
    }
}
