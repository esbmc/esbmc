// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 3 basic: nested tuple destructuring with function return
contract NestedTuple {
    function getPair() public pure returns (uint, uint) {
        return (10, 20);
    }

    function test() public pure {
        uint a;
        uint b;
        uint c;
        ((a, b), c) = (getPair(), 30);
        assert(a == 10);
        assert(b == 20);
        assert(c == 30);
    }
}
