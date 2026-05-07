// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 3 adversarial: wrong assertion on nested tuple element
contract NestedTupleFail {
    function getPair() public pure returns (uint, uint) {
        return (10, 20);
    }

    function test() public pure {
        uint a;
        uint b;
        uint c;
        ((a, b), c) = (getPair(), 30);
        // a should be 10, but we claim it's 20 (that's b's value)
        assert(a == 20);
    }
}
