// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract CtorTime {
    uint256 public startedAt;
    uint256 public hit;
    constructor() {
        require(block.timestamp >= 1000000, "OriginInThePast");
        startedAt = block.timestamp;
    }
    function step(uint256 x) external {
        if (x >= 50) { hit = 1; } else { hit = 2; }
    }
}
