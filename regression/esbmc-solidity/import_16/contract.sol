// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract BaseContract {
    uint public value;

    constructor() {
        value = 42; 
    }

    function getValue() public view returns (uint) {
        return value;
    }
}