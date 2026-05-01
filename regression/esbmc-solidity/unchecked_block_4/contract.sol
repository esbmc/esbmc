// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Outside unchecked block, overflow SHOULD be caught
contract CheckedOverflow {
    function test_checked_overflow() public pure returns (uint8) {
        uint8 x = 255;
        uint8 y = x + 1; // this overflows WITHOUT unchecked — should fail
        return y;
    }
}
