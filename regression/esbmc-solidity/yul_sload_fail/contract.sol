// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    uint256 x;

    // FAIL dual to yul_sload_pass: same body, wrong assertion.
    // After precise sload, r equals v, so r == v + 1 is violable
    // for any v that does not overflow on +1 (and even at the wrap
    // boundary the universal claim still fails).
    function check(uint256 v) public {
        x = v;
        uint256 r;
        assembly {
            r := sload(x.slot)
        }
        assert(r == v + 1);
    }
}
