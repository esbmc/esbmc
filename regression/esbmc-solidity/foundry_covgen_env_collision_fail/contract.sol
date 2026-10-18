// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Collide {
    uint256 public hit;
    // user variable deliberately named like the EVM global
    function f(uint256 block_timestamp) external {
        if (block_timestamp >= 100) { hit = 1; } else { hit = 2; }
    }
}
