// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual PASS partner for `esbmc_nondet_intrinsic_td_manipulability_fail`.
// Same self-composition harness, but the perturbed timestamp `t2` is
// constrained to exactly match `block.timestamp` (no miner shift). The
// parity invariant then trivially holds ⇒ VERIFICATION SUCCESSFUL.
//
// Locks in that the `__ESBMC_nondet_uint` intrinsic does not over-approximate
// in the no-shift direction — when the require makes `t2` definite, the
// downstream assert is provable.

contract Lock {
    uint256 public unlockTime;
    bool public unlocked;

    function __ESBMC_nondet_uint() internal pure returns (uint256) {}

    function unlock() public {
        uint256 t2 = __ESBMC_nondet_uint();
        require(t2 == block.timestamp); // no shift
        assert(block.timestamp % 2 == t2 % 2); // trivially holds
    }
}
