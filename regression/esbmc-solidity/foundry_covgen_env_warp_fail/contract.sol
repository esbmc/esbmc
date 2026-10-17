// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Timelock {
    uint256 public unlockAt = 1000000;
    uint256 public hit;
    function claim() external {
        if (block.timestamp >= unlockAt) {
            hit = 1;   // reachable only when time advanced
        } else {
            hit = 2;
        }
    }
}
