// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bound-mode transfer revert semantics:
// Real Solidity transfer() reverts on insufficient balance. Any code
// immediately after the transfer is unreachable on the insufficient-
// balance path. The model uses __ESBMC_assume(false) for the revert,
// so the failing path is pruned and an assert(false) placed right
// after is vacuously true.
//
// Uses self-transfer (address(this)) so the target always matches the
// static instance address in the transfer dispatcher.
contract TransferRevert {

    function __ESBMC_assume(bool) internal pure {}

    function test() public {
        __ESBMC_assume(address(this).balance == 5);
        // Self-transfer with insufficient balance: must revert.
        payable(address(this)).transfer(10);
        // Unreachable under correct revert semantics.
        assert(false);
    }
}
