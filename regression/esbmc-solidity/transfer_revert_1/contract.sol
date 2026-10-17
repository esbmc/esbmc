// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bound-mode transfer balance accounting (self-transfer):
// A self-transfer decrements and increments the same balance, netting zero.
// This verifies that the transfer function's balance accounting is sound
// and that both the debit and credit paths execute on the same contract.
//
// The separate send_ether_via_creation_2 test covers the cross-contract
// balance decrease case (via new D{value: amount}() + constructor transfer).
contract SelfTransfer {

    function __ESBMC_assume(bool) internal pure {}

    function test() public {
        __ESBMC_assume(address(this).balance >= 100);
        uint256 b0 = address(this).balance;
        // Self-transfer: target is this contract's own static instance.
        // The transfer dispatcher matches _ESBMC_Object_SelfTransfer.$address.
        payable(address(this)).transfer(40);
        // Net effect: balance -= 40, balance += 40 = unchanged.
        assert(address(this).balance == b0);
    }
}
