// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

struct S {
    uint x;
    uint y;
}

library L {
    function reverse(S calldata _s) internal pure returns (uint, uint) {
        return (_s.y, _s.x);
    }
}

contract C {
    function test(uint, S calldata _s, uint) external pure returns (uint, uint) {
        return L.reverse(_s);
    }
}
