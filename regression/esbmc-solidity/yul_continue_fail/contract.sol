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
        // r == 15 would only hold if `continue` did nothing and every i in
        // 0..5 contributed (0+1+2+3+4+5 == 15). Precise lowering must
        // violate this.
        assert(r == 15);
    }
}
