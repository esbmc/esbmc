// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract ModTime {
    uint256 public start;
    uint256 public v;
    constructor() { start = block.timestamp; }
    modifier gate() { require(block.timestamp >= start); _; }
    function act() external gate { if (block.timestamp > start + 100) v = 1; else v = 2; }
}
