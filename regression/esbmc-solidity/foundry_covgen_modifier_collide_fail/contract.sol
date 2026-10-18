// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Collide {
    uint256 public v;
    modifier b_mod() { require(true); _; }
    modifier mod() { require(true); _; }
    function a(uint256 x) external b_mod { if (x > 5) v = 1; else v = 2; }
    function a_b(uint256 x) external mod { if (x > 3) v = 3; else v = 4; }
}
