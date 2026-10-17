// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// EOA balance modelling — write-then-read round trip.
// Pin recipient initial balance to 0, transfer `amt`, observe that the
// recipient's `.balance` increased by exactly `amt`.  This proves the
// EOA-fallback in get_transfer_definition credits the global EOA
// balance map AND that get_aux_property_function reads the same map.
contract Bal {
    constructor() payable {}

    function check(address payable to, uint amt) public {
        // Skip self-transfer (uses the contract-tracked path, not the EOA path).
        if (to == address(this)) return;
        // Must have funds and a meaningful payment amount.
        if (address(this).balance < amt || amt == 0) return;
        // Pin the recipient's initial balance to 0 so the post-transfer
        // value equals exactly `amt`.  Reading `to.balance` here is
        // what triggers the slot insertion with nondet initial value;
        // the require collapses it to 0.
        if (to.balance != 0) return;
        to.transfer(amt);
        assert(to.balance == amt);
    }
}
