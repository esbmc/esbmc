// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract Base {
    uint x = 0;

    function test() public {
        assert(1 == 1);
    }
}

contract Derived {
    uint x = 1;

    function test() public {
        assert(0 == 1);
    }
}

