// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Mod {
    address owner;
    uint256 public v;
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner); _; }
    function bump(uint256 a) external onlyOwner { if (a > 5) v = 1; else v = 2; }
}
