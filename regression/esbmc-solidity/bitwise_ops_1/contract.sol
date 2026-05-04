// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract BitwiseOps {
    function test_bitwise() public pure {
        uint8 a = 0xF0;
        uint8 b = 0x0F;

        // AND
        assert((a & b) == 0x00);
        assert((a & a) == 0xF0);

        // OR
        assert((a | b) == 0xFF);

        // XOR
        assert((a ^ b) == 0xFF);
        assert((a ^ a) == 0);

        // NOT
        assert(~a == 0x0F);
        assert(~b == 0xF0);

        // Shift
        uint8 c = 1;
        assert((c << 1) == 2);
        assert((c << 4) == 16);
        assert((c << 7) == 128);

        uint8 d = 128;
        assert((d >> 1) == 64);
        assert((d >> 7) == 1);
    }
}
