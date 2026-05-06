// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: tuple return, destructuring, and swap
contract C {
    uint index;

    function f() public pure returns (uint, uint) {
        return (7, 2);
    }

    function g() public {
        // Destructure tuple return
        (uint x, uint y) = f();
        assert(x == 7);
        assert(y == 2);

        // Swap values via tuple assignment
        (x, y) = (y, x);
        assert(x == 2);
        assert(y == 7);

        // Assign to state variable via tuple
        (index, ) = f();
        assert(index == 7);
    }
}
