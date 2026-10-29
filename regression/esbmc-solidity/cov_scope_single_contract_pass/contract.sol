// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Positive control: a single contract with one branch and NO
// out-of-scope code. --contract C must count exactly C's branch.
contract C {
    uint256 public x;
    function setX(uint256 v) public {
        if (v > 10) {
            x = v;
        } else {
            x = 1;
        }
    }
}
