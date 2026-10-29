// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Base { constructor() { require(block.timestamp >= 1000000, "past"); } }
contract Derived is Base {
    uint256 public hit;
    function step(uint256 x) external { if (x >= 50) { hit = 1; } else { hit = 2; } }
}
