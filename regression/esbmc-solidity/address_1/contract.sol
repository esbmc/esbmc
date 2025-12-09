// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    address x = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;
    address y = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;
    address z = 0x1f9840A85D5Af5bf1d1762f925BDAdDc4201f986;

    function comp() public {
        assert(x == y);
        assert(x < z);
    }
}
