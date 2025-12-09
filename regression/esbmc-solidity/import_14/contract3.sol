// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./contract4.sol";

contract B is A {
    uint public myBValue = myValue + 1;
}