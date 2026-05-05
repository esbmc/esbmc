// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Intentionally wrong bitwise assertions — verification should FAIL
contract BitwiseFail {
    function test_bitwise_fail() public pure {
        uint8 a = 0xF0;
        uint8 b = 0x0F;
        // XOR of complements should be 0xFF, not 0x00
        assert((a ^ b) == 0x00);
    }
}
