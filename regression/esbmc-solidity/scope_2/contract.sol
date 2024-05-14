// // SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Creating a contract
contract Base {
    uint x;

    constructor() {
        x = 1;
    }

    function test() public {
        uint x = 10;
        {
            x = 2;
        }
        {
            uint x = 3;
        }
        assert(x == 10);
    }
}