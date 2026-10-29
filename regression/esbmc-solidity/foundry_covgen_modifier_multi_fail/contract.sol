// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Multi {
    address owner; uint256 public v;
    constructor() { owner = msg.sender; }
    modifier modA() { require(msg.sender == owner); _; }
    modifier modB() { require(v < 100); _; }
    function f(uint256 a) external modA modB { if (a > 5) v = 1; else v = 2; }
}
