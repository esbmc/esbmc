// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test basic fallback and receive function definitions.
// Verifies that contracts with fallback/receive parse and verify correctly.
contract Test {
    uint public x;
    fallback() external { x = 1; }
}

contract TestPayable {
    uint public x;
    uint public y;
    fallback() external payable { x = 1; y = msg.value; }
    receive() external payable { x = 2; y = msg.value; }
}
