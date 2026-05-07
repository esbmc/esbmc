// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract IntBoundary {
    function test_uint8_bounds() public pure {
        uint8 min_val = 0;
        uint8 max_val = 255;
        assert(min_val == type(uint8).min);
        assert(max_val == type(uint8).max);
        assert(max_val + 1 - 1 == max_val);
        assert(min_val + max_val == 255);
    }

    function test_uint256_bounds() public pure {
        uint256 max_val = type(uint256).max;
        assert(max_val > 0);
        assert(max_val + 1 - 1 == max_val);
        // 2^256 - 1 is all bits set
        assert(max_val == ~uint256(0));
    }

    function test_int8_bounds() public pure {
        int8 min_val = -128;
        int8 max_val = 127;
        assert(min_val == type(int8).min);
        assert(max_val == type(int8).max);
        assert(int8(0) - 1 == -1);
        assert(int16(max_val) - int16(min_val) == 255);
    }

    function test_int256_bounds() public pure {
        int256 min_val = type(int256).min;
        int256 max_val = type(int256).max;
        assert(min_val < 0);
        assert(max_val > 0);
        assert(min_val + max_val == -1);
    }

    function test_cross_width() public pure {
        // Assign small uint to larger — should preserve value
        uint8 small = 200;
        uint256 big = small;
        assert(big == 200);

        // Narrowing cast
        uint256 large = 300;
        uint8 narrow = uint8(large); // 300 mod 256 = 44
        assert(narrow == 44);
    }
}
