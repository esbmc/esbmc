// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TypeExample {
    function getMinInt160() public pure returns (int160) {
        return type(int160).min; // -2^255
    }
    function test() public pure {
        int160 x;
        x = getMinInt160() - 1;
    }
}
