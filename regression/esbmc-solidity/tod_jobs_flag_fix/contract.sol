// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// withdraw and setRecipient touch disjoint state -> --tod=auto should
// find zero candidates and exit cleanly.
contract Fund {
    address public recipient;
    uint public balance;

    constructor() {
        balance = 100;
        recipient = address(0x1);
    }

    function withdraw() public { balance = 0; }
    function setRecipient(address newRecipient) public { recipient = newRecipient; }
}
