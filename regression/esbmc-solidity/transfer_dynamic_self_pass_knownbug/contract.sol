// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Regression test for transfer-to-self on a dynamically allocated contract instance.
//
// ROOT CAUSE (2026-06-24): the $transfer dispatch loop checks
//   `_addr == _ESBMC_Object_<C>.$address` for each contract type, using the
//   STATIC singleton address.  A dynamic instance created via `new C{value:...}()`
//   gets a different nondet address from `_ESBMC_get_unique_address`, so the loop
//   never matches, and the call falls through to the EOA fallback.
//
//   EOA fallback: debits `this.$balance` (correct) but credits
//   `sol_eoa_balance_array[to_address]` (wrong — that is a different storage
//   location from the recipient's `$balance` object field).
//
//   Consequence: after a self-transfer of the full balance
//   (`payable(address(c)).transfer(address(c).balance)`), ESBMC sets
//   `c.$balance = 0` (the debit) and puts the amount in the EOA map (the
//   credit), so a subsequent `address(c).balance` read returns 0.
//
// EXPECTED (correct) behaviour: `payable(address(c)).transfer(v)` where C has
//   no payable `receive()` or `fallback()` should REVERT (exactly like calling
//   `transfer` with insufficient balance), leaving the balance unchanged.  The
//   model should emit `__ESBMC_assume(false)` and prune this path, making any
//   assertion after the call vacuously true (VERIFICATION SUCCESSFUL).
//
// CURRENT (buggy) behaviour: ESBMC lets the path continue with c.balance = 0
//   and b0 = 1 ether, so `address(c).balance == b0` is falsified → VERIFICATION
//   FAILED (wrong).
//
// The companion test `transfer_dynamic_cross_fail_knownbug` exercises the same
//   EOA-credit mislanding on a cross-instance transfer (c1 → c2, both dynamic).

contract C {
    // No receive() or fallback() — any inbound transfer from another account
    // should revert on the real EVM.
    constructor() payable {}
    function migrateTo(address to) public {
        payable(to).transfer(address(this).balance);
    }
}

contract InvMutTest {
    C c;
    constructor() payable {
        // Dynamic allocation: address(c) ≠ _ESBMC_Object_C.$address (static singleton).
        c = new C{value: 1 ether}();
    }
    function body() public {
        uint256 b0 = address(c).balance;   // = 1 ether
        // Self-transfer: real EVM reverts (no receive); ESBMC should prune path.
        // On the pruned (only) path, b0 is the last-known balance, so b0 == b0.
        c.migrateTo(address(c));
        // After correct fix: this line is unreachable (path pruned) → vacuously true.
        // With current bug: balance = 0 ≠ b0 → assertion violated.
        assert(address(c).balance == b0);
    }
}
