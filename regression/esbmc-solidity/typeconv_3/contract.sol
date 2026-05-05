// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract TypeConv {
    function test_narrowing() public pure {
        // uint256 -> uint8: truncation
        uint256 big = 0x1FF; // 511
        uint8 small = uint8(big); // 511 mod 256 = 255
        assert(small == 255);

        // uint256 -> uint16
        uint256 val = 0x1ABCD;
        uint16 trunc = uint16(val); // 0xABCD = 43981
        assert(trunc == 0xABCD);
    }

    function test_widening() public pure {
        // uint8 -> uint256
        uint8 a = 0xFF;
        uint256 b = uint256(a);
        assert(b == 255);

        // int8 -> int256 (sign extension)
        int8 neg = -1;
        int256 wide = int256(neg);
        assert(wide == -1);

        int8 neg2 = -128;
        int256 wide2 = int256(neg2);
        assert(wide2 == -128);
    }

    function test_signed_unsigned() public pure {
        // uint8 -> int16 (safe: uint8 fits in int16)
        uint8 u = 200;
        int16 s = int16(uint16(u));
        assert(s == 200);

        // int8 -> uint8 (reinterpret: -1 becomes 255)
        int8 neg = -1;
        uint8 unsigned_val = uint8(neg);
        assert(unsigned_val == 255);

        // int8(-128) -> uint8 = 128
        int8 min_int8 = -128;
        uint8 as_uint = uint8(min_int8);
        assert(as_uint == 128);
    }
}
