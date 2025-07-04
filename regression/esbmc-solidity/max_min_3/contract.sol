// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TypeExample {
    function getMaxUint256() public pure returns (uint256) {
        return type(uint256).max; // 2^256 - 1
    }
    function test() public pure {
        uint x = getMaxUint256() + 1;
    }
}
