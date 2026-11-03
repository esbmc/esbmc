// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// #8708: the shift is truncated, but the add after it still overflows.
contract C {
    function f(uint256 a) public pure {
        uint256 b = a << 255;
        uint256 c = b + b;
    }
}
