// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TypeExample {
    function max() public pure returns (uint) {
        return 1;
    }
    function min() public pure returns (uint) {
        return 0;
    }
    function getMaxUint256() public pure returns (uint256) {
        return type(uint256).max; // 2^256 - 1
    }

    function getMaxInt256() public pure returns (int256) {
        return type(int256).max; // 2^255 - 1
    }

    function getMinInt256() public pure returns (int256) {
        return type(int256).min; // -2^255
    }

    function getMinUint256() public pure returns (uint256) {
        return type(uint256).min; // -2^255
    }
}
