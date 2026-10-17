// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.4;

// Tests revert("string") and require(cond, "string") equivalence.
// Both patterns should make post-revert code unreachable.
contract VendingMachine {
    mapping(address => uint) public balances;

    function buy(uint amount) public payable {
        if (amount > msg.value / 2)
            revert("Not enough Ether provided.");
        // If we reach here, amount <= msg.value / 2
        assert(amount <= msg.value / 2);
    }

    function buyRequire(uint amount) public payable {
        require(
            amount <= msg.value / 2,
            "Not enough Ether provided."
        );
        // Same assertion should hold
        assert(amount <= msg.value / 2);
    }
}
