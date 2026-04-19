// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./contract3.sol";

uint constant Value = 1;

contract C is B {
    uint public myCValue = myBValue + 1;
}