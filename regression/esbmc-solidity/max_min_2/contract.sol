// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TypeExample {
    function getMaxInt256() public pure returns (int256) {
        return type(int256).max; // 2^255 - 1
    }
    function test() public pure {
        int x;
        x = getMaxInt256();
        x++;
    }
}
