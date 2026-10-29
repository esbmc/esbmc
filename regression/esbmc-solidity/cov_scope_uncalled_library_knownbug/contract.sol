// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// L.f contains a branch but C NEVER calls it. Under --contract C the
// branch-coverage denominator must be C's branch only.
library L {
    function f(uint256 a) internal pure returns (uint256) {
        if (a > 5) {
            return a;
        }
        return 0;
    }
}

contract C {
    uint256 public x;
    function setX(uint256 v) public {
        if (v > 10) {
            x = v;
        }
    }
}
