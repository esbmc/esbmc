// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract D {
    uint x;
    uint y;

    modifier check {
        _;
        assert(x == 0);
    }

    modifier check2 {
        _;
        x = 0;
    }

    function func2() public check check2{
        x = 1;

    }
}