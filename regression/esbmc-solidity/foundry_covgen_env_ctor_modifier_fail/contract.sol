// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract ModCtor {
    uint256 public hit;
    modifier afterStart() { require(block.timestamp >= 1000000, "past"); _; }
    constructor() afterStart() {}
    function step(uint256 x) external { if (x >= 50) { hit = 1; } else { hit = 2; } }
}
