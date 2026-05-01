// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 1 adversarial: wrong assertion after tuple omission
contract TupleOmitFail {
    function getThree() public pure returns (uint, uint, uint) {
        return (10, 20, 30);
    }

    function test() public pure {
        uint a;
        uint c;
        (a, , c) = getThree();
        // a == 10, c == 30, but we claim c == 20 (wrong — 20 was the omitted middle)
        assert(c == 20);
    }
}
