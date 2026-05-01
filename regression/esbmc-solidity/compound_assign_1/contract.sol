// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract CompoundAssign {
    function test_compound() public pure {
        uint256 x = 100;

        // +=
        x += 50;
        assert(x == 150);

        // -=
        x -= 30;
        assert(x == 120);

        // *=
        x *= 2;
        assert(x == 240);

        // /=
        x /= 4;
        assert(x == 60);

        // %=
        x %= 7;
        assert(x == 4); // 60 mod 7 = 4

        // <<=
        x <<= 3;
        assert(x == 32); // 4 * 8 = 32

        // >>=
        x >>= 2;
        assert(x == 8); // 32 / 4 = 8

        // &=
        x = 0xFF;
        x &= 0x0F;
        assert(x == 0x0F);

        // |=
        x |= 0xF0;
        assert(x == 0xFF);

        // ^=
        x ^= 0xFF;
        assert(x == 0);
    }
}
