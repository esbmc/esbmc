// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.3 — empirical test for block.number cross-iteration monotonicity.
// Iteration N: first() snapshots block.number into stored.
// Iteration M (M >= N): second() asserts block.number >= stored. Real
// EVM: block.number is monotonically non-decreasing across calls.
// _sol_per_tx_reseed (solidity_misc.c) already adds an assume for this;
// this test locks in the property and serves as a regression guard.
contract H {
    uint256 public stored;
    bool public stored_set;

    function first() external {
        stored = block.number;
        stored_set = true;
    }

    function second() external view {
        if (stored_set) {
            assert(block.number >= stored);
        }
    }
}
