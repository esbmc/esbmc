// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.29;

// Test custom storage layout with assertion failure.
contract D layout at 0xFF {
    uint value;

    function test() public {
        value = 42;
        // FAILS: value is 42, not 0
        assert(value == 0);
    }
}
