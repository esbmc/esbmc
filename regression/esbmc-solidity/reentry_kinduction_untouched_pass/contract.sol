// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// `to.call("")` may re-enter Vault without bound, so k-induction has no
// finite unwinding to reach. The inductive step answers the re-entries past
// its bound with a havoc of what a dispatch may write: `deposits`, never
// `owner`, so it proves the assertion.
contract Vault {
    address owner;
    uint256 deposits;

    constructor() {
        owner = msg.sender;
    }

    function deposit(uint256 v) public {
        deposits = v;
    }

    function forward(address to) public {
        address o = owner;
        to.call("");
        assert(owner == o);
    }
}
