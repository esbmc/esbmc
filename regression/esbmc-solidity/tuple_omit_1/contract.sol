// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: tuple destructuring with omitted elements and swap
contract C {
    uint index;

    function f() public pure returns (uint, bool, uint) {
        return (7, true, 2);
    }

    function g() public {
        // Destructure with omitted middle element
        (uint x, , uint y) = f();
        assert(x == 7);
        assert(y == 2);

        // Swap via tuple assignment
        (x, y) = (y, x);
        assert(x == 2);
        assert(y == 7);

        // Partial assignment to state variable with trailing omissions
        (index, , ) = f();
        assert(index == 7);
    }
}
