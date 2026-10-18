// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// withdraw() touches only `balance`, setRecipient() touches only `recipient`.
// No shared state -> Phase 3 targeted assertions detect zero overlap and the
// harness has no equality check to violate.
contract Fund {
    address public recipient;
    uint public balance;

    constructor() {
        balance = 100;
        recipient = address(0x1);
    }

    function withdraw() public {
        balance = 0;
    }

    function setRecipient(address newRecipient) public {
        recipient = newRecipient;
    }
}
