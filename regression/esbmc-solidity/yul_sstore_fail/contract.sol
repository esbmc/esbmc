// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    uint256 x;

    // FAIL dual to yul_sstore_pass: same body but the wrong assertion.
    // After `sstore(x.slot, v)` precision lands, x equals v, so the
    // claim x == 0 is violable for any v != 0.
    function check(uint256 v) public {
        assembly {
            sstore(x.slot, v)
        }
        assert(x == 0);
    }
}
