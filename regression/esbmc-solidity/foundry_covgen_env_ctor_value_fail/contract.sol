// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract PayCtor {
    uint256 public v;
    constructor() payable { require(msg.value >= 1000); }
    function act(uint256 a) external { if (a > 5) v = 1; else v = 2; }
}
