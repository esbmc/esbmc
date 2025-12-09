// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    uint x;
    uint y;

    function name() public returns (uint, uint) {
        return (1, 2);
    }

    function name2() public returns (uint, uint) {
        return name();
    }

    function test() public {
        (x, y) = name2();
        assert(x == 1);
        assert(y == 2);
    }
}

