// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Compound assignment overflow: uint8 += should trigger overflow
contract CompoundOverflow {
    function test_overflow() public pure returns (uint8) {
        uint8 x = 250;
        x += 10; // 260 overflows uint8
        return x;
    }
}
