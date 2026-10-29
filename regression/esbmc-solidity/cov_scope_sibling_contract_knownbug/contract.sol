// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Two independent contracts, no inheritance, C does not use Other.
// --contract C must not count Other's branch.
contract Other {
    uint256 public y;
    function setY(uint256 w) public {
        if (w > 7) {
            y = w;
        }
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
