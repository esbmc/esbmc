// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Overflow detection: adding 1 to uint8.max should trigger overflow check
contract IntOverflow {
    function test_overflow() public pure returns (uint8) {
        uint8 x = 255;
        uint8 y = x + 1; // overflow in Solidity 0.8+
        return y;
    }
}
