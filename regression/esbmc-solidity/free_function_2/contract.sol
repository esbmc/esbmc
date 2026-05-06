// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Free function with intentional bug
function safeDiv(uint256 a, uint256 b) pure returns (uint256) {
    return a / b;  // no zero-check
}

contract FreeFuncTest {
    function test() public pure {
        uint256 result = safeDiv(10, 2);
        assert(result == 5);
        // Division by zero should be caught
        uint256 bad = safeDiv(10, 0);
    }
}
