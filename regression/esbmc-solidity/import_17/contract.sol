// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./contract2.sol";

contract ContractB is ContractA {
    uint public number2;

    constructor() {
        number2 = 200;
    }
}