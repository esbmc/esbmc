// SPDX-License-Identifier: MIT
// #8707: ** still reports an overflow that happens, and wraps when unchecked.
pragma solidity ^0.8.0;
contract C {
    function u(uint256 a) public pure returns (uint256) {
        require(a == 2**200);
        return a ** 2;
    }
    function s(int256 a) public pure returns (int256) {
        require(a == -2);
        return a ** 256;
    }
    function w(uint256 a) public pure {
        require(a == 2**200);
        unchecked {
            assert(a ** 2 != 0);
        }
    }
}
