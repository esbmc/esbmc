// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Ovl {
    uint256 public x;
    function set(uint256 v) external { if (v > 100) x = 1; else x = 2; }
    function set(bool b) external { if (b) x = 3; else x = 4; }
}
