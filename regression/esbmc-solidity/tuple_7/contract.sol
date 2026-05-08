// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Phase 1 basic: middle omission in tuple assignment
contract TupleMiddleOmit {
    function getThree() public pure returns (uint, uint, uint) {
        return (10, 20, 30);
    }

    function test() public pure {
        uint a;
        uint c;
        // omit middle element
        (a, , c) = getThree();
        assert(a == 10);
        assert(c == 30);

        // omit first element
        uint y;
        uint z;
        (, y, z) = getThree();
        assert(y == 20);
        assert(z == 30);

        // omit last element
        uint p;
        uint q;
        (p, q, ) = getThree();
        assert(p == 10);
        assert(q == 20);
    }
}
