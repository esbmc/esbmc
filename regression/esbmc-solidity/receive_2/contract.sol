// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test that receive() function body is correctly verified.
// The assertion inside receive() should fail because msg.value
// is nondeterministic and can be != 1 ether.
contract SinkFail {
    uint public totalReceived;

    receive() external payable {
        totalReceived += msg.value;
        // FAILS: msg.value is nondet, not necessarily 1 ether
        assert(msg.value == 1 ether);
    }
}
