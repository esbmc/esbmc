// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./contract.sol";

contract ContractC is ContractB {
    constructor() {
        assert(number == number2);
    }
}