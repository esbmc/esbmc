// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Reproduces the bug-report MODELING-2 example (temp_bug.txt §"TD guard-form
// manipulability"): a time-lock gate whose outcome can be flipped by a miner
// shifting `block.timestamp` by one. The natural oracle is self-composition:
// take a second, attacker-perturbed timestamp `t2` in the miner window and
// assert parity invariance.
//
// Pre-fix the user could not express this without either (a) breaking solc
// signatures by adding a parameter or (b) injecting a state variable, which
// fails because state vars start at their post-constructor default in
// --contract mode (not havoc'd). Both injection routes vacuously pass.
//
// Post-fix the `__ESBMC_nondet_uint` intrinsic synthesises a fresh nondet at
// the call site (see src/solidity-frontend/solidity_convert_expr.cpp).
// The miner-shift of 1 flips the parity ⇒ VERIFICATION FAILED.

contract Lock {
    uint256 public unlockTime;
    bool public unlocked;

    function __ESBMC_nondet_uint() internal pure returns (uint256) {}

    function unlock() public {
        uint256 t2 = __ESBMC_nondet_uint();
        require(t2 == block.timestamp + 1); // miner shift within window
        assert(block.timestamp % 2 == t2 % 2); // FAIL — parities differ
    }
}
