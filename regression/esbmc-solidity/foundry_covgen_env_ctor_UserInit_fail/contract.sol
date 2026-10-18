// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract UserInit {
    uint256 public hit;
    constructor() { initialize(); }
    function initialize() internal { require(block.timestamp >= 1000000, "past"); }
    function step(uint256 x) external { if (x >= 50) { hit = 1; } else { hit = 2; } }
}
