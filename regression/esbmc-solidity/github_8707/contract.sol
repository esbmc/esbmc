// SPDX-License-Identifier: MIT
// #8707: ** and type(int256).min reported overflow on values that fit.
pragma solidity ^0.8.0;
contract C {
    function u(uint256 a, uint256 b) public pure {
        require(a == 2**200 && b == 2**128 - 1);
        assert(a ** 1 == 2**200);
        assert(b ** 2 == 2**256 - 2**129 + 1);
    }
    function s(int256 a) public pure {
        require(a == -2);
        uint256 x = 255;
        assert(a ** 3 == -8);
        assert(a ** 255 == type(int256).min);
        assert((-2) ** x == type(int256).min);
    }
    function w(uint256 a, int256 b) public pure {
        require(a == 2**200 && b == -3);
        unchecked {
            assert(a ** 2 == 0);
            assert(b ** 3 == -27);
        }
    }
    function m() public pure {
        assert(type(int256).min == -2**255);
        assert(type(int8).min == -128);
    }
}
