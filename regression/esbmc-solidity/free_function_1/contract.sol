// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Free function outside any contract
function add(uint256 a, uint256 b) pure returns (uint256) {
    return a + b;
}

function double(uint256 x) pure returns (uint256) {
    return x * 2;
}

contract FreeFuncTest {
    function test() public pure {
        assert(add(2, 3) == 5);
        assert(double(7) == 14);
        assert(add(double(3), 4) == 10);
    }
}
