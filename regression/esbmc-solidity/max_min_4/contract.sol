// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TypeExample {
    function getMinUint160() public pure returns (uint160) {
        return type(uint160).min; // -2^255
    }
    function test() public pure {
        uint160 x = getMinUint160() - 1;
    }
}
