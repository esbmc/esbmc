// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract C {
    bytes1 public slot;

    constructor() {
        slot = bytes1(uint8(7));
    }

    function f() external view returns (bytes1) {
        return slot;
    }
}
