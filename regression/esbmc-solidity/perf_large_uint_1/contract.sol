// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Performance stress test: uint256 arithmetic with moderately large values
contract PerfLargeUint {
    function test_large_arithmetic() public pure {
        // Large values near uint128 boundaries
        uint256 a = 2**64;
        uint256 b = 2**64 - 1;

        // Addition
        uint256 sum = a + b;
        assert(sum == 2**65 - 1);

        // Multiplication of medium values
        uint256 c = 1000000; // 10^6
        uint256 d = 1000000;
        uint256 product = c * d;
        assert(product == 1000000000000); // 10^12

        // Division
        uint256 ratio = product / c;
        assert(ratio == d);

        // Modular arithmetic
        uint256 val = 1000000007;
        assert(val % 1000000000 == 7);
    }

    function test_chained_ops() public pure {
        uint256 x = 1;
        // Chain of operations
        x = x * 2 + 1;  // 3
        x = x * 2 + 1;  // 7
        x = x * 2 + 1;  // 15
        x = x * 2 + 1;  // 31
        x = x * 2 + 1;  // 63
        assert(x == 63);
    }

    function test_uint256_max() public pure {
        uint256 max_val = type(uint256).max;
        assert(max_val > 0);
        assert(max_val + 1 - 1 == max_val);
        // All bits set
        assert(max_val == ~uint256(0));
    }
}
