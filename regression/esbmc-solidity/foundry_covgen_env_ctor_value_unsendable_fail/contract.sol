// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Base { constructor() payable { require(msg.value > 0); } }
contract Derived is Base {
    uint256 public v;
    function act(uint256 a) external { if (a > 5) v = 1; else v = 2; }
}
