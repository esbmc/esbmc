// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Verify unchecked block allows overflow wrapping
contract UncheckedWrap {
    function test_unchecked_overflow() public pure {
        uint8 result;
        unchecked {
            uint8 x = 255;
            result = x + 1; // wraps to 0
        }
        assert(result == 0);
    }

    function test_unchecked_underflow() public pure {
        uint8 result;
        unchecked {
            uint8 x = 0;
            result = x - 1; // wraps to 255
        }
        assert(result == 255);
    }

    function test_unchecked_mul() public pure {
        uint8 result;
        unchecked {
            uint8 x = 200;
            result = x * 2; // 400 mod 256 = 144
        }
        assert(result == 144);
    }
}
