// SPDX-License-Identifier: MIT
pragma solidity ^0.6.0;

library L {
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        if (a + b < a) return (false, 0);
        return (true, a + b);
    }
}

contract T {
    function f(uint256 x, uint256 y) public pure returns (uint256) {
        (bool ok, uint256 s) = L.tryAdd(x, y);
        assert(ok || s == 0);
        return s;
    }
}
