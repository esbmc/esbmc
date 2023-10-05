// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract Base {
    uint x = 0;

    function test(uint y) public {
        assert(x == 0);
        x=y;
    }
}

contract Derived {
    uint x = 1;

    function test() public {
        assert(x == 0);
    }
}

